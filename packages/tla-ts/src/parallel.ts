import type { Expression } from "./ast";
import {
  checkSequentially,
  type CheckOptions,
  type CheckProgress,
  type CheckResult,
  type FallbackReason,
  type InitialStates,
  PROGRESS_INTERVAL,
} from "./check";
import { parseConfig, type ModelConfig } from "./config";
import { TlaEvaluationError } from "./errors";
import { isImpure } from "./evaluate";
import {
  createSharedFingerprints,
  growSharedFingerprints,
  insertSharedFingerprint,
  isSharedFingerprintsFull,
  writeStateFingerprint,
  type SharedFingerprints,
} from "./fingerprints";
import {
  createCheckEnvironment,
  createSearch,
  findAssumptionViolation,
  reportInitialStatesProgress,
  reportStartup,
} from "./search";
import { getInitialStates } from "./successors";
import { getFingerprintDistance, OUT_DEGREE_BUCKETS } from "./tlc/summary";
import { createStateEncoder, type EncodedStates } from "./transfer";
import { createCanonicalizer, getUnorderedString, type Value } from "./values";
import { spawnWorker, type Port } from "./worker";

export interface StartMessage extends Pick<
  CheckOptions,
  | "source"
  | "isTool"
  | "traceSpec"
  | "isWarningEnabled"
  | "version"
  | "fingerprintPolynomial"
  | "maxSetSize"
  | "isLazyValueExpanded"
> {
  type: "start";
  config: ModelConfig;
  modules: [string, string | undefined][];
  classes: [string, Uint8Array | undefined][];
  index: number;
  ports: (Port | undefined)[];
  seed: EncodedStates;
  fingerprints: SharedFingerprints;
  requests: Int32Array;
  remaining: Int32Array;
}

interface LevelMessage {
  type: "level";
  generated: number;
  added: number;
  outDegrees: number[];
  /** The fingerprint of each state the level added. */
  fingerprints: BigInt64Array;
  /** The fingerprint of the state each of fingerprints was found from. */
  parents: BigInt64Array;
}

interface AbortMessage {
  type: "abort";
  reason: FallbackReason;
  /** The fingerprint of the state the worker was expanding when it stopped. */
  parent?: bigint;
  /** The states the stopped level generated and added before it stopped. */
  generated?: number;
  added?: number;
}

export type WorkerMessage = { type: "ready" } | AbortMessage | { type: "paused" } | LevelMessage;

export type CoordinatorMessage =
  | { type: "level" }
  | { type: "resize"; fingerprints: SharedFingerprints };

/**
 * Checks a model, with options.workers threads searching each level breadth first when the model
 * allows it. A success gives the same counts as one worker. A violation, a deadlock or an error
 * gives a shortest trace to it, which need not be the one a single worker finds first, with the
 * counts the workers reached when one stopped, as TLC's workers do.
 */
export const check = async (options: CheckOptions): Promise<CheckResult> => {
  const config =
    typeof options.config === "string"
      ? parseConfig(options.config, options.version)
      : options.config;
  const print = options.onPrint ?? console.log;
  let printedCount = 0;
  const getKey = (progress: CheckProgress) =>
    [
      progress.kind,
      "name" in progress ? progress.name : "",
      "count" in progress ? progress.count : "",
      "text" in progress ? progress.text : "",
    ].join(" ");
  const reported = new Set<string>();
  const report = (progress: CheckProgress) => {
    reported.add(getKey(progress));
    options.onProgress?.(progress);
  };
  // searchAgain checks on the calling thread from the initial states, storing only the states path
  // holds when it is set, and leaves out what this search already printed and reported.
  const searchAgain = (path?: Set<bigint>) => {
    let skippedCount = printedCount;
    return checkSequentially(
      {
        ...options,
        config,
        onPrint: (text) => (skippedCount-- > 0 ? undefined : print(text)),
        onProgress: (progress) => {
          if (!reported.has(getKey(progress))) options.onProgress?.(progress);
        },
      },
      path,
    );
  };
  const fallBack = (reason?: FallbackReason) => {
    if (reason !== undefined) options.onProgress?.({ kind: "sequential-fallback", reason });
    return searchAgain();
  };
  const workerCount = options.workers ?? 1;
  if (workerCount < 2) return fallBack();
  const unsupported = (feature: string) => fallBack({ kind: "unsupported", feature });
  if (typeof SharedArrayBuffer !== "function")
    return unsupported("a runtime without SharedArrayBuffer");
  if (options.dump !== undefined) return unsupported("-dump");
  if (options.coverage !== undefined) return unsupported("-coverage");
  if (options.isContinued === true) return unsupported("-continue");
  if (config.view !== undefined) return unsupported("VIEW");
  if (config.postconditions.length > 0) return unsupported("POSTCONDITION");
  if (config.possible.length > 0) return unsupported("possible states");
  if ((options.moduleOperators?.length ?? 0) > 0) return unsupported("module operators");
  const modules = new Map<string, string | undefined>();
  const loadModule = (name: string) => {
    const source = options.loadModule?.(name);
    modules.set(name, source);
    return source;
  };
  const classes = new Map<string, Uint8Array | undefined>();
  const loadClass = (name: string) => {
    const bytes = options.loadClass?.(name);
    classes.set(name, bytes);
    return bytes;
  };
  const created = createCheckEnvironment(
    config,
    { ...options, loadModule, loadClass, seed: undefined, moduleOperators: undefined },
    (text) => {
      printedCount++;
      print(text);
    },
    true,
  );
  const specification = created.specification;
  reportStartup(specification, config, report);
  const environment = specification.environment;
  const search = createSearch(
    config,
    environment,
    specification.variableLocations,
    () => created.order.getDeclarationOrder("variable"),
    report,
  );
  if (findAssumptionViolation(specification.assumptions, environment)) return fallBack();
  if (!search || search.spec.init === undefined || search.spec.next === undefined)
    return fallBack();
  if (search.properties.liveness.length > 0) return unsupported("temporal properties");
  if (isImpure(environment)) return unsupported("operators with side effects (such as TLCSet)");

  const canonicalize = createCanonicalizer();
  let sharedFingerprints = createSharedFingerprints();
  const fingerprint = new Int32Array(2);
  const seeds = Array.from({ length: workerCount }, createStateEncoder);
  const statistics = {
    generatedStates: 0,
    distinctStates: 0,
    queuedStates: 0,
    depth: 0,
    initialStates: undefined as InitialStates | undefined,
    outDegrees: Array.from({ length: OUT_DEGREE_BUCKETS }, () => 0),
  };
  const tlcFingerprints: BigInt64Array[] = [];
  // The parent log: for each level after the first, each worker's added fingerprints and the
  // fingerprint of the state each was found from.
  const levelLogs: LevelMessage[][] = [];
  const getDistance = () =>
    getFingerprintDistance(statistics.distinctStates, statistics.generatedStates, (visit) => {
      const fingerprints = new BigInt64Array(statistics.distinctStates);
      tlcFingerprints.reduce((offset, level) => {
        fingerprints.set(level, offset);
        return offset + level.length;
      }, 0);
      visit(fingerprints);
    });
  const createSuccess = (): CheckResult => ({
    status: "success",
    violation: undefined,
    failure: undefined,
    ...statistics,
    fingerprintDistance: getDistance(),
  });
  // findPath returns the fingerprints of the states from an initial state to the state with
  // fingerprint last, which the last level added, by walking the parent log back.
  const findPath = (last: bigint) => {
    const path = new Set([last]);
    let target = last;
    for (let level = levelLogs.length - 1; level >= 0; level--) {
      const parent = findParent(levelLogs[level], target);
      if (parent === undefined) return undefined;
      path.add(parent);
      target = parent;
    }
    return path;
  };
  // stop rebuilds the trace of a worker that stopped on a state: the calling thread searches
  // again along the path to it, which reaches the same violation or error in a few levels, and
  // reports this search's counts. It searches every state again only if the path doesn't.
  const stop = (abort: AbortMessage) => {
    const path = abort.parent === undefined ? undefined : findPath(abort.parent);
    const guided = path && searchAgain(path);
    if (!guided || guided.status === "success") return fallBack(abort.reason);
    return {
      ...guided,
      generatedStates: statistics.generatedStates + (abort.generated ?? 0),
      distinctStates: statistics.distinctStates + (abort.added ?? 0),
      queuedStates: statistics.queuedStates,
      initialStates: statistics.initialStates,
      outDegrees: guided.outDegrees && statistics.outDegrees,
      fingerprintDistance: guided.fingerprintDistance && getDistance(),
    };
  };
  const seedInitialStates = (init: Expression) => {
    const seeded: Value[][] = [];
    for (const state of getInitialStates(init, environment, search.spec.scope)) {
      reportInitialStatesProgress(statistics.generatedStates, report);
      canonicalize(state);
      statistics.generatedStates++;
      writeStateFingerprint(search.getFingerprintState(state), fingerprint);
      if (isSharedFingerprintsFull(sharedFingerprints)) {
        sharedFingerprints = growSharedFingerprints(sharedFingerprints);
      }
      if (!insertSharedFingerprint(sharedFingerprints, fingerprint[0], fingerprint[1])) continue;
      if (search.findViolatedInvariant(state) || search.findFailedInitialPredicate(state)) {
        return false;
      }
      if (!search.satisfiesConstraints(state)) continue;
      if (!seeds[seeded.length % workerCount].encode(state)) return false;
      seeded.push(state);
      statistics.distinctStates++;
    }
    tlcFingerprints.push(
      BigInt64Array.from(seeded, (state) => environment.runtime.getFingerprint(state)),
    );
    return true;
  };
  report({ kind: "initial-states-started" });
  try {
    if (!seedInitialStates(search.spec.init)) return fallBack();
  } catch (error) {
    if (!(error instanceof TlaEvaluationError)) throw error;
    return fallBack();
  }
  const unorderedString = getUnorderedString();
  if (unorderedString !== undefined) {
    return fallBack({ kind: "string", string: unorderedString });
  }
  statistics.depth = statistics.generatedStates === 0 ? 0 : 1;
  statistics.initialStates = {
    generated: statistics.generatedStates,
    distinct: statistics.distinctStates,
  };
  report({ kind: "initial-states-computed", initialStates: statistics.initialStates });
  if (statistics.distinctStates === 0) return createSuccess();

  const reports: WorkerMessage[] = [];
  let pendingCount = workerCount;
  let resolvePending = (_reports: WorkerMessage[]) => {};
  const onMessage = (message: WorkerMessage) => {
    reports.push(message);
    if (message.type === "abort" || reports.length === pendingCount) {
      resolvePending(reports.splice(0));
    }
  };
  const waitForReports = (expected: number) =>
    new Promise<WorkerMessage[]>((resolve) => {
      pendingCount = expected;
      resolvePending = resolve;
    });
  const ready = waitForReports(workerCount);
  const requests = new Int32Array(new SharedArrayBuffer(workerCount * 4));
  const remaining = new Int32Array(new SharedArrayBuffer(workerCount * 4));
  const workers = await Promise.all(
    Array.from({ length: workerCount }, () =>
      spawnWorker(
        new URL("./parallel-worker.js", import.meta.url),
        (message) => onMessage(message as WorkerMessage),
        (error) =>
          onMessage({ type: "abort", reason: { kind: "error", message: describe(error) } }),
      ),
    ),
  );
  const terminate = () => workers.forEach((worker) => worker.terminate());
  const channels = new Map<string, MessageChannel>();
  const getPort = (from: number, to: number) => {
    const key = from < to ? `${from}:${to}` : `${to}:${from}`;
    const channel = channels.get(key) ?? new MessageChannel();
    channels.set(key, channel);
    return (from < to ? channel.port1 : channel.port2) as unknown as Port;
  };
  workers.forEach((worker, index) => {
    const ports = Array.from({ length: workerCount }, (_value, peer) =>
      peer === index ? undefined : getPort(index, peer),
    );
    const seed = seeds[index].finish();
    const message: StartMessage = {
      type: "start",
      source: options.source,
      config,
      modules: [...modules],
      classes: [...classes],
      index,
      ports,
      seed,
      isTool: options.isTool,
      traceSpec: options.traceSpec,
      isWarningEnabled: options.isWarningEnabled,
      version: options.version,
      fingerprintPolynomial: options.fingerprintPolynomial,
      maxSetSize: options.maxSetSize,
      isLazyValueExpanded: options.isLazyValueExpanded,
      fingerprints: sharedFingerprints,
      requests,
      remaining,
    };
    worker.postMessage(message, [...ports.filter((port) => port !== undefined), seed.words.buffer]);
  });
  const findAbort = (messages: WorkerMessage[]) =>
    messages.find((message) => message.type === "abort");
  const send = (message: CoordinatorMessage) =>
    workers.forEach((worker) => worker.postMessage(message));
  const collectLevels = async (messages: WorkerMessage[]): Promise<WorkerMessage[]> => {
    const pausedCount = messages.filter((message) => message.type === "paused").length;
    if (pausedCount === 0 || findAbort(messages)) return messages;
    sharedFingerprints = growSharedFingerprints(sharedFingerprints);
    const resumed = waitForReports(pausedCount);
    send({ type: "resize", fingerprints: sharedFingerprints });
    const unpaused = messages.filter((message) => message.type !== "paused");
    return [...unpaused, ...(await collectLevels(await resumed))];
  };
  const startAbort = findAbort(await ready);
  if (startAbort) {
    terminate();
    return fallBack(startAbort.reason);
  }
  let progressAt = performance.now() + PROGRESS_INTERVAL;
  for (;;) {
    const levelReports = waitForReports(workerCount);
    requests.forEach((_request, index) => Atomics.store(requests, index, 0));
    send({ type: "level" });
    const messages = await collectLevels(await levelReports);
    const abort = findAbort(messages);
    if (abort) {
      terminate();
      return stop(abort);
    }
    const levels = messages.flatMap((message) => (message.type === "level" ? [message] : []));
    const added = levels.reduce((sum, message) => sum + message.added, 0);
    statistics.generatedStates += levels.reduce((sum, message) => sum + message.generated, 0);
    statistics.distinctStates += added;
    statistics.queuedStates = added;
    levelLogs.push(levels);
    levels.forEach((message) => {
      message.outDegrees.forEach(
        (degreeCount, degree) => (statistics.outDegrees[degree] += degreeCount),
      );
      tlcFingerprints.push(message.fingerprints);
    });
    if (added === 0) {
      terminate();
      return createSuccess();
    }
    statistics.depth++;
    if (performance.now() >= progressAt) {
      progressAt = performance.now() + PROGRESS_INTERVAL;
      // Progress goes straight to the caller: report() would mark its kind as seen and hide the
      // progress of a later fallback search.
      options.onProgress?.({
        kind: "progress",
        statistics: {
          generatedStates: statistics.generatedStates,
          distinctStates: statistics.distinctStates,
          queuedStates: added,
          depth: statistics.depth,
          coverage: undefined,
        },
      });
    }
  }
};

/** Returns the fingerprint of the parent of the state with fingerprint child in one level's log. */
const findParent = (level: LevelMessage[], child: bigint) => {
  for (const log of level) {
    const index = log.fingerprints.indexOf(child);
    if (index !== -1) return log.parents[index];
  }
  return undefined;
};

/** Returns the message of a worker's error event or thrown value. */
const describe = (error: unknown) =>
  error instanceof Error
    ? error.message
    : String((error as { message?: unknown } | undefined)?.message ?? error);
