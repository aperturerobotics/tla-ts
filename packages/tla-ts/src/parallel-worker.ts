import {
  createFingerprintTable,
  insertSharedFingerprint,
  isSharedFingerprintsFull,
  writeStateFingerprint,
} from "./fingerprints";
import type { FallbackReason } from "./check";
import type { CoordinatorMessage, StartMessage, WorkerMessage } from "./parallel";
import { createCheckEnvironment, createSearch } from "./search";
import { getSuccessors, type Successor } from "./successors";
import { OUT_DEGREE_BUCKETS } from "./tlc/summary";
import { createStateDecoder, createStateEncoder, type EncodedStates } from "./transfer";
import { createCanonicalizer, getUnorderedString, type Value } from "./values";
import { createBreak, getParentPort, type Port } from "./worker";

const LOCKED = -1;
const BREAK_INTERVAL = 100;

/** States a worker sends a thief, with their fingerprints in the same order. */
interface Loot {
  states: EncodedStates;
  fingerprints: BigInt64Array;
}

interface Expansion {
  states: Value[][];
  /** The fingerprint of each of states, as getStoredFingerprints returns it. */
  fingerprints: BigInt64Array;
  origin: number;
  end: number;
  successors: Successor[] | undefined;
  index: number;
  added: number;
  generated: number;
  outDegrees: number[];
  next: Value[][];
  fingerprintStates: Value[][];
  /** The fingerprint of the state each of next was found from. */
  parents: BigInt64Array;
  /** Whether the level stopped on a new state, which TLC counts as distinct though no one queues it. */
  isStoppedOnNew: boolean;
  breakAt: number;
}

const runWorker = (start: StartMessage, parent: Port) => {
  const modules = new Map(start.modules);
  const loadModule = (name: string) => modules.get(name);
  const classes = new Map(start.classes);
  const created = createCheckEnvironment(
    start.config,
    { ...start, loadModule, loadClass: (name) => classes.get(name), seed: undefined },
    () => {},
    true,
  );
  const specification = created.specification;
  const environment = specification.environment;
  const search = createSearch(start.config, environment, specification.variableLocations, () =>
    created.order.getDeclarationOrder("variable"),
  );
  const nextAction = search?.spec.next;
  if (!search || nextAction === undefined)
    return parent.postMessage({
      type: "abort",
      reason: { kind: "error", message: "the worker found no next-state action" },
    } satisfies WorkerMessage);
  const workerIndex = start.index;
  const canonicalize = createCanonicalizer();
  const decode = createStateDecoder(environment.variables.length);
  let sharedFingerprints = start.fingerprints;
  const requests = start.requests;
  const remaining = start.remaining;
  const generatedStates = search.isSymmetric ? createFingerprintTable() : undefined;
  const fingerprint = new Int32Array(2);
  let resumeLevel = () => {};
  let receiveLoot = (_loot: Loot) => {};
  start.ports.forEach((port) => {
    port?.addEventListener("message", (event) => receiveLoot(event.data as Loot));
    port?.start?.();
  });
  let frontier = decode(start.seed).map(canonicalize);
  let frontierFingerprints = search.getStoredFingerprints(
    frontier,
    frontier.map(search.getFingerprintState),
  );
  const takeBreak = createBreak();

  const answerRequest = (expansion: Expansion, request: number) => {
    const thief = Atomics.load(requests, workerIndex) - 1;
    const given = Math.floor((expansion.end - expansion.origin - 1) / 2);
    const encoder = createStateEncoder();
    const states = expansion.states.slice(expansion.end - given, expansion.end);
    const isEncoded = given > 0 && states.every(encoder.encode);
    const loot: Loot = {
      states: isEncoded ? encoder.finish() : createStateEncoder().finish(),
      fingerprints: isEncoded
        ? expansion.fingerprints.slice(expansion.end - given, expansion.end)
        : new BigInt64Array(0),
    };
    if (isEncoded) expansion.end -= given;
    start.ports[thief]?.postMessage(loot, [loot.states.words.buffer, loot.fingerprints.buffer]);
    Atomics.store(requests, workerIndex, request);
    Atomics.notify(requests, workerIndex);
  };

  const lockRequests = (expansion: Expansion) => {
    Atomics.store(remaining, workerIndex, 0);
    if (Atomics.compareExchange(requests, workerIndex, 0, LOCKED) === 0) return;
    answerRequest(expansion, LOCKED);
  };

  const unlockRequests = (expansion: Expansion) => {
    Atomics.store(remaining, workerIndex, expansion.end - expansion.origin);
    Atomics.store(requests, workerIndex, 0);
  };

  const expand = (expansion: Expansion) => {
    for (; expansion.origin < expansion.end; expansion.origin++) {
      const state = expansion.states[expansion.origin];
      if (!expansion.successors) {
        if (performance.now() >= expansion.breakAt) return "break";
        if (Atomics.load(requests, workerIndex) > 0) answerRequest(expansion, 0);
        Atomics.store(remaining, workerIndex, expansion.end - expansion.origin - 1);
        expansion.successors = getSuccessors(nextAction, state, environment, search.spec.scope);
        expansion.generated += expansion.successors.length;
        if (expansion.successors.length === 0 && start.config.checkDeadlock) return undefined;
      }
      const successors = expansion.successors;
      for (; expansion.index < successors.length; expansion.index++) {
        if (isSharedFingerprintsFull(sharedFingerprints)) return "paused";
        const successor = successors[expansion.index];
        if (search.findViolatedActionProperty(state, successor.state)) return undefined;
        if (!search.satisfiesActionConstraints(state, successor.state)) {
          // TLC checks the invariants of every successor it has not seen, even outside the model.
          if (search.findViolatedInvariant(successor.state, state)) return undefined;
          continue;
        }
        if (generatedStates) {
          writeStateFingerprint(successor.state, fingerprint);
          if (!generatedStates.insert(fingerprint[0], fingerprint[1])) continue;
        }
        canonicalize(successor.state);
        const fingerprintState = search.getFingerprintState(successor.state);
        writeStateFingerprint(fingerprintState, fingerprint);
        if (!insertSharedFingerprint(sharedFingerprints, fingerprint[0], fingerprint[1])) continue;
        expansion.isStoppedOnNew = true;
        if (search.findViolatedInvariant(successor.state, state)) return undefined;
        expansion.isStoppedOnNew = false;
        if (!search.satisfiesConstraints(successor.state, state)) continue;
        if (expansion.next.length === expansion.parents.length) {
          const grown = new BigInt64Array(2 * expansion.parents.length);
          grown.set(expansion.parents);
          expansion.parents = grown;
        }
        expansion.parents[expansion.next.length] = expansion.fingerprints[expansion.origin];
        expansion.next.push(successor.state);
        expansion.fingerprintStates.push(fingerprintState);
        expansion.added++;
      }
      expansion.outDegrees[Math.min(expansion.added, OUT_DEGREE_BUCKETS - 1)]++;
      expansion.successors = undefined;
      expansion.index = 0;
      expansion.added = 0;
    }
    return "done";
  };

  const steal = async (expansion: Expansion) => {
    const skipped = new Set<number>();
    for (;;) {
      const victim = start.ports.reduce<number | undefined>((best, port, index) => {
        if (!port || skipped.has(index) || Atomics.load(remaining, index) < 2) return best;
        return best === undefined || Atomics.load(remaining, index) > Atomics.load(remaining, best)
          ? index
          : best;
      }, undefined);
      if (victim === undefined) return false;
      const request = Atomics.compareExchange(requests, victim, 0, workerIndex + 1);
      if (request === LOCKED) skipped.add(victim);
      if (request > 0) Atomics.wait(requests, victim, request);
      if (request !== 0) continue;
      const loot = await new Promise<Loot>((resolve) => (receiveLoot = resolve));
      if (loot.states.count === 0) {
        skipped.add(victim);
        continue;
      }
      expansion.states = decode(loot.states).map(canonicalize);
      expansion.fingerprints = loot.fingerprints;
      expansion.origin = 0;
      expansion.end = expansion.states.length;
      unlockRequests(expansion);
      return true;
    }
  };

  const report = (message: WorkerMessage) =>
    parent.postMessage(
      message,
      message.type === "level" ? [message.fingerprints.buffer, message.parents.buffer] : [],
    );

  const runLevel = async (): Promise<WorkerMessage> => {
    const expansion: Expansion = {
      states: frontier,
      fingerprints: frontierFingerprints,
      origin: 0,
      end: frontier.length,
      successors: undefined,
      index: 0,
      added: 0,
      generated: 0,
      outDegrees: Array.from({ length: OUT_DEGREE_BUCKETS }, () => 0),
      next: [],
      fingerprintStates: [],
      parents: new BigInt64Array(1024),
      isStoppedOnNew: false,
      breakAt: performance.now() + BREAK_INTERVAL,
    };
    // A violation or an error stops the level on the state at origin, which the coordinator
    // finds again by its fingerprint to rebuild the trace.
    const stopAt = (reason: FallbackReason): WorkerMessage => ({
      type: "abort",
      reason,
      parent: expansion.fingerprints[expansion.origin],
      generated: expansion.generated,
      added: expansion.next.length + (expansion.isStoppedOnNew ? 1 : 0),
    });
    for (;;) {
      let status: ReturnType<typeof expand>;
      try {
        status = expand(expansion);
      } catch (error) {
        return stopAt(abortWith(error).reason);
      }
      if (status === undefined) return stopAt({ kind: "violation" });
      // HACK: Bun frees memory on event loop timers, so a long level returns to the loop.
      if (status === "break") {
        await takeBreak();
        expansion.breakAt = performance.now() + BREAK_INTERVAL;
        continue;
      }
      lockRequests(expansion);
      if (status === "paused") {
        const resumed = new Promise<void>((resolve) => (resumeLevel = resolve));
        report({ type: "paused" });
        await resumed;
        unlockRequests(expansion);
        continue;
      }
      if (!(await steal(expansion))) break;
    }
    const unorderedString = getUnorderedString();
    if (unorderedString !== undefined) {
      return { type: "abort", reason: { kind: "string", string: unorderedString } };
    }
    frontier = expansion.next;
    const fingerprints = search.getStoredFingerprints(expansion.next, expansion.fingerprintStates);
    frontierFingerprints = fingerprints.slice();
    Atomics.store(remaining, workerIndex, frontier.length);
    return {
      type: "level",
      generated: expansion.generated,
      added: expansion.next.length,
      outDegrees: expansion.outDegrees,
      fingerprints,
      parents: expansion.parents.slice(0, expansion.next.length),
    };
  };

  const startLevel = () => {
    runLevel().then(
      (message) => report(message),
      (error) => report(abortWith(error)),
    );
  };
  parent.addEventListener("message", (event) => {
    const message = event.data as CoordinatorMessage;
    if (message.type === "level") startLevel();
    if (message.type === "resize") {
      sharedFingerprints = message.fingerprints;
      resumeLevel();
    }
  });
  Atomics.store(remaining, workerIndex, frontier.length);
  report({ type: "ready" });
};

void getParentPort().then((parent) => {
  const onStart = (event: { data: unknown }) => {
    const message = event.data as StartMessage;
    if (message.type !== "start") return;
    parent.removeEventListener?.("message", onStart);
    try {
      runWorker(message, parent);
    } catch (error) {
      parent.postMessage(abortWith(error));
    }
  };
  parent.addEventListener("message", onStart);
  parent.start?.();
});

/** Returns the abort that carries a worker's failure to the coordinator. */
const abortWith = (error: unknown): Extract<WorkerMessage, { type: "abort" }> => ({
  type: "abort",
  reason: { kind: "error", message: error instanceof Error ? error.message : String(error) },
});
