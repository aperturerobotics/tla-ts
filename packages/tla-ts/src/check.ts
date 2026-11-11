import type { Expression } from "./ast";
import { parseConfig, type ModelConfig } from "./config";
import {
  addCoverageRelation,
  buildCoverage,
  countCoverageVariables,
  createCoverage,
  getCoverageLabel,
  getCoverageReport,
  type CoverageReport,
} from "./coverage";
import { createSeenStates } from "./seen-states";
import { createStateQueue } from "./state-queue";
import {
  createStackOverflowError,
  isStackOverflow,
  TlaEvaluationError,
  TlaIllegalStateError,
  TlaIncompleteSuccessorError,
  TlaModuleError,
  TlaStackOverflowError,
  UNEXPECTED_EVAL_EXCEPTION,
  UNEXPECTED_EXCEPTION,
  formatRange,
} from "./errors";
import {
  describeRange,
  getDefinition,
  isImpure,
  type CallStack,
  type Environment,
  type FingerprintFailure,
} from "./evaluate";
import { EXIT_STATUS, setExitCode, withLivenessExitCode } from "./exit-codes";
import {
  fingerprintReplayed,
  getFailureExitCode,
  getReplayFailure,
  type FailureOptions,
} from "./check-failure";
import {
  createTraceBuilder,
  type Behavior,
  type SearchPosition,
  type TracedStep,
} from "./check-trace";
import { isSameState } from "./fingerprints";
import type { JavaOverride } from "./java-overrides";
import {
  createLivenessChecker,
  type Lasso,
  type LivenessChecker,
  type LiveSuccessor,
} from "./liveness-checker";
import {
  createCheckEnvironment,
  createMissingOperatorError,
  createSearch,
  getIncompleteSuccessorMessage,
  findAssumptionViolation,
  reportInitialStatesProgress,
  reportStartup,
  withNextStateExitCode,
} from "./search";
import {
  getActionLabel,
  getActionSignature,
  getSpecDescription,
  getSpecificationValue,
  toActionValue,
  type SpecAction,
  type SpecDescription,
} from "./spec-description";
import {
  createStateWriter,
  type GraphState,
  type StateDump,
  type StateWriter,
} from "./state-graph";
import {
  forEachInitialState,
  forEachSuccessor,
  getInitialStates,
  getSuccessors,
  hasIncompleteSuccessor,
  type Successor,
} from "./successors";
import {
  getFailureMessage,
  getWrappedException,
  isEvalException,
  isHeadedError,
  MESSAGE_CODE,
  withUnexpectedException,
  type FailureMessage,
} from "./tlc/messages";
import { createTautologyError } from "./liveness";
import { KEY_SEPARATOR } from "./resolve";
import { writeTraceSpec } from "./tlc/trace-spec";
import {
  getPrimedRanges,
  replayInitialStates,
  replaySuccessors,
  walkSuccessors,
} from "./tlc/replay";
import { getFingerprintDistance, OUT_DEGREE_BUCKETS } from "./tlc/summary";
import { createActionRecord, createCounterExample } from "./tlc/tlcext-operators";
import type { TlcVersion } from "./tlc/version";
import { runTypedSearch, type TypedResult } from "./typed-tier";
import {
  createCanonicalizer,
  createModelValue,
  formatState,
  getFingerprintError,
  getModelValues,
  isSameValue,
  markFingerprinted,
  type Value,
} from "./values";

export interface CheckOptions {
  source: string;
  config: string | ModelConfig;
  loadModule?: (name: string) => string | undefined;
  loadClass?: (name: string) => Uint8Array | undefined;
  onPrint?: (text: string) => void;
  onPause?: () => void;
  readFile?: (path: string) => string | undefined;
  writeFile?: (path: string, content: string | Uint8Array) => void;
  makeDirectory?: (path: string) => void;
  seed?: bigint;
  workers?: number;
  /** TLC's -lncheck: "final" checks liveness only on the complete graph, "off" never looks for a cycle. */
  livenessCheck?: "default" | "final" | "off";
  /** TLC's -fp: the index of the polynomial in TLC's FP64.Polys that fingerprints states. */
  fingerprintPolynomial?: number;
  /** False for TLC's -terse, which prints a lazy value, such as SUBSET S, without enumerating it. */
  isLazyValueExpanded?: boolean;
  /** TLC's -view: prints each state as the value of the configuration's VIEW. */
  isViewPrinted?: boolean;
  /** TLC's -maxSetSize: the most elements TLC enumerates into a set, or states into a list of them. */
  maxSetSize?: number;
  /** TLC's -dump: writes the reachable states, or with `dot` the state graph, through writeFile. */
  dump?: StateDump;
  /** TLC's -tool, which changes the messages of TLC's errors. */
  isTool?: boolean;
  /** TLC's -continue: reports each violation through onProgress and goes on. */
  isContinued?: boolean;
  /** Skips fingerprintDistance, which takes fingerprinting every state again after the search. */
  isFingerprintDistanceSkipped?: boolean;
  moduleOperators?: ModuleOperator[];
  /** TLC's trace exploration spec, and the binary trace TLC records for it unless -noTEBin. */
  traceSpec?: TraceSpecOptions;
  /** Whether the caller printed SANY's report, its warnings included, before the check. */
  isSanyReported?: boolean;
  /** False for TLC's -nowarning, which changes how TLC computes some successors. */
  isWarningEnabled?: boolean;
  /** TLC's -coverage, in minutes. */
  coverage?: number;
  version?: TlcVersion;
  onProgress?: (progress: CheckProgress) => void;
  /**
   * The directory the search's queue writes its encoded states to, in a new directory it removes
   * when the search ends; the runtime's temporary directory, which follows TMPDIR, by default.
   */
  spillDirectory?: string;
}

export interface TraceSpecOptions {
  module: string;
  traceFile?: string;
}

export interface ModuleOperator {
  kind: "postcondition" | "constraint" | "view";
  module: string;
  operator: string;
  constants?: Record<string, string>;
}

export interface InitialStatesProgress {
  kind: "initial-states-progress";
  count: number;
}

export interface InitialStateViolation {
  kind: "initial-state-violation";
  status: "invariant-violation" | "liveness-violation";
  name: string;
  state: Record<string, Value>;
}

export type OutputLine = string | InitialStatesProgress | InitialStateViolation;

/**
 * Why a search asked for several workers checks the model on the calling thread instead: before
 * it starts them, or after it stops them, from the initial states again.
 */
export type FallbackReason =
  /** The search ordered a string it created against another; workers don't share that order. */
  | { kind: "string"; string: string }
  /** A worker found a violation or a deadlock; one thread rebuilds its trace. */
  | { kind: "violation" }
  /** A worker failed, such as by running out of memory. */
  | { kind: "error"; message: string }
  /** The model or the options use a feature only the calling thread checks. */
  | { kind: "unsupported"; feature: string };

export type CheckProgress =
  | { kind: "parsed" }
  | { kind: "constant-level-invariant"; name: string }
  | { kind: "state-level-property"; name: string }
  | { kind: "liveness-initializing"; isSymmetric: boolean; isConstrained: boolean }
  | { kind: "liveness-prepared"; branches: number }
  | { kind: "initial-states-started" }
  | InitialStatesProgress
  | { kind: "initial-states-computed"; initialStates: InitialStates }
  | { kind: "progress"; statistics: SearchStatistics }
  | {
      kind: "liveness-started";
      branches: number;
      /** Whether TLC checks the complete state graph. */
      isComplete: boolean;
      /** How many states of the graph each branch has. */
      states: number;
      statistics: SearchStatistics;
    }
  | { kind: "liveness-finished" }
  | { kind: "sequential-fallback"; reason: FallbackReason }
  | { kind: "warning"; code: number; text: string }
  | { kind: "violation"; status: CheckStatus; violation: Violation }
  | ({ kind: "java-override" } & JavaOverride);

export interface SearchStatistics extends Pick<
  CheckResult,
  "generatedStates" | "distinctStates" | "queuedStates" | "depth" | "coverage"
> {
  /** In a breadth-first search's progress, the bytes the seen states' columns and index take. */
  seenBytes?: number;
  /** In a breadth-first search's progress, the bytes of the queue's chunk files on disk. */
  spillBytes?: number;
  /** In a breadth-first search's progress, the queued states those chunk files hold. */
  spillStates?: number;
}

export type CheckStatus =
  | "success"
  | "assumption-violation"
  | "invariant-violation"
  | "property-violation"
  | "liveness-violation"
  | "postcondition-violation"
  | "deadlock"
  | "error";

export interface TraceStep {
  action: string | undefined;
  /** TLC's name, parameters and location for the action that took this step. */
  label: string | undefined;
  state: Record<string, Value>;
  /** What the spec prints before this step as TLC rebuilds the trace, and before it for the first. */
  output?: OutputLine[];
}

export interface Violation {
  name: string;
  /** Every temporal property that a liveness counterexample violates, in the configuration's order. */
  properties?: string[];
  trace: TraceStep[];
  /** What the spec prints after the trace, as TLC finds the lasso's step back or rebuilds the trace. */
  traceOutput?: OutputLine[];
  /** What the postconditions print, which TLC checks after the lasso's step back. */
  postconditionOutput?: OutputLine[];
  loopStart: number | undefined;
  loopLabel?: string;
  message?: string;
  counts?: string;
  unwitnessed?: Postcondition;
  /** A POSTCONDITION that is false or fails, which TLC checks as it prints the counterexample. */
  postcondition?: Postcondition;
  /** What TLC warns about after a counterexample, with its code. */
  warning?: { code: number; text: string };
}

export interface Postcondition {
  name: string;
  code: number;
  message: string;
  counts?: string;
}

/** An evaluation that failed during the search, with the behavior up to the state TLC was working on. */
export interface EvaluationFailure {
  code: number;
  message: string;
  toolMessage: string;
  trace: TraceStep[];
  traceOutput?: OutputLine[];
  initialState: Record<string, Value> | undefined;
  exitCode: number;
  /** The expressions TLC was evaluating, outermost first, when its replay finds them. */
  callStack: Expression[] | undefined;
  /** Whether TLC prints the call stack before the error, as it does for a failed liveness check. */
  isCallStackFirst?: boolean;
  /** The FingerprintException the replay raises, which TLC reports in place of the call stack. */
  fingerprint?: FingerprintFailure;
  /** How TLC reports failing again as it prints the trace, since it can't fingerprint the state. */
  traceFailure?: FailureMessage;
  /** A POSTCONDITION that is false or fails after the trace, which TLC checks as it prints one. */
  postcondition?: Postcondition;
  /** Whether TLC stops before it prints the statistics, as it does for an illegal initial state. */
  isWithoutStatistics?: boolean;
}

export interface InitialStates {
  generated: number;
  distinct: number;
}

export interface CheckResult {
  status: CheckStatus;
  violation: Violation | undefined;
  failure: EvaluationFailure | undefined;
  generatedStates: number;
  distinctStates: number;
  queuedStates: number;
  depth: number;
  /** Once TLC computed all the initial states without a violation or an error. */
  initialStates: InitialStates | undefined;
  /** How many expanded states found each number of new states, when TLC prints the outdegree. */
  outDegrees: number[] | undefined;
  /** The smallest distance between two fingerprints, when TLC estimates a collision from them. */
  fingerprintDistance: bigint | undefined;
  /** A temporal property TLC finds violated as it adds a deadlocked state, reported after. */
  additionalViolation?: Violation;
  /** Where TLC's worker fails to add the state to the behavior graph after it reports a violation. */
  liveCallStack?: Expression[];
  /** The outdegrees when TLC's workers stop on a violation that TLC 1.7.4 reports as a success. */
  violationOutDegrees?: number[];
  traceSpec?: string;
  coverage?: CoverageReport;
}

interface CoveredAction extends SpecAction {
  node: number;
}

interface ReportedViolation extends Omit<Violation, "loopStart"> {
  loopStart?: number;
}

const appendTraceOutput = <Traced extends Behavior>(behavior: Traced, lines: OutputLine[]) => ({
  ...behavior,
  traceOutput: [...(behavior.traceOutput ?? []), ...lines],
});

const STOP_REPLAY = new Error("The replay stopped");
const STOP_WALK = new Error("The walk stopped");

export const checkSequentially = (options: CheckOptions): CheckResult => {
  const stateWriter =
    options.dump && createStateWriter(options.dump, options.writeFile, options.version);
  const closers: (() => void)[] = [];
  try {
    return searchStates(options, stateWriter, closers);
  } catch (error) {
    throw isStackOverflow(error) ? createStackOverflowError() : error;
  } finally {
    stateWriter?.close();
    closers.forEach((close) => close());
  }
};

const PERIODIC_START = 3000;
/** How often a breadth-first search reports its progress, in milliseconds. */
export const PROGRESS_INTERVAL = 60_000;
const LIVENESS_GROWTH = 0.1;
const LIVENESS_RATIO = 0.2;

const searchStates = (
  options: CheckOptions,
  stateWriter: StateWriter | undefined,
  closers: (() => void)[],
): CheckResult => {
  const config =
    typeof options.config === "string"
      ? parseConfig(options.config, options.version)
      : options.config;
  const onPrint = options.onPrint ?? console.log;
  let capturedLines: OutputLine[] | undefined;
  const print = (text: string) => {
    if (capturedLines) capturedLines.push(text);
    else onPrint(text);
  };
  const captureOutput = <Result>(rebuild: () => Result) => {
    const outer = capturedLines;
    const lines: OutputLine[] = [];
    capturedLines = lines;
    try {
      return { result: rebuild(), lines };
    } finally {
      capturedLines = outer;
    }
  };
  const created = createCheckEnvironment(config, options, print);
  const specification = created.specification;
  reportStartup(specification, config, options.onProgress);
  const coverage =
    options.coverage === undefined
      ? undefined
      : createCoverage(specification.environment.variables.length);
  const environment: Environment = coverage
    ? {
        ...specification.environment,
        evaluators: new WeakMap(),
        entryEvaluators: new WeakMap(),
        definitionEvaluators: new WeakMap(),
        coverage,
      }
    : specification.environment;
  const runtime = environment.runtime;
  const is174 = runtime.version === "1.7.4";
  runtime.pause = options.onPause ?? (() => {});
  let description: SpecDescription | undefined;
  const getDescription = () =>
    (description ??= getSpecDescription(config, environment, specification.instanceActions));
  runtime.getSpecification = () =>
    getSpecificationValue(getDescription(), config, environment, specification.variableLocations);
  let searchDepth = 0;
  // Every state the search has seen, by fingerprint; see SeenStates.
  const seen = createSeenStates();
  let computedInitialStates: InitialStates | undefined;
  const getTraceSpec = (trace: TraceStep[], loopStart: number | undefined) =>
    options.traceSpec &&
    writeTraceSpec({
      module: options.traceSpec.module,
      specName: specification.rootName,
      rawConstants: config.rawConstants,
      states: trace.map((step) => getStepState(step)),
      loopStart,
      modelValues: getModelValues().map((value) => value.name),
      isDefined: (name) =>
        environment.definitions.has(name) ||
        environment.constants.has(name) ||
        [...environment.definitions.keys()].some((key) =>
          key.startsWith(`${name}${KEY_SEPARATOR}`),
        ),
    });
  const createResult = (
    status: CheckStatus,
    violation?: ReportedViolation,
    failure?: EvaluationFailure,
  ): CheckResult => {
    const traceSpec = getTraceSpec(
      violation?.trace ?? failure?.trace ?? [],
      status === "property-violation" ? undefined : violation?.loopStart,
    );
    return {
      status,
      violation: violation && { ...violation, loopStart: violation.loopStart },
      failure,
      generatedStates: runtime.generatedStates,
      distinctStates: runtime.distinctStates,
      queuedStates: runtime.queueSize,
      depth: runtime.generatedStates === 0 ? 0 : Math.max(searchDepth, 1),
      initialStates: computedInitialStates,
      outDegrees: undefined,
      fingerprintDistance: undefined,
      coverage: coverage?.relations.some((relation) => relation.kind === "next")
        ? getCoverageReport(
            coverage,
            environment,
            specification.variableLocations,
            search?.getTlcOrder() ?? [],
          )
        : undefined,
      ...(traceSpec !== undefined && { traceSpec }),
    };
  };
  const outDegrees = Array.from({ length: OUT_DEGREE_BUCKETS }, () => 0);
  let newStateCount = 0;
  const getOutDegreeBucket = () => Math.min(newStateCount, OUT_DEGREE_BUCKETS - 1);
  const getPartialOutDegrees = () =>
    outDegrees.map((count, degree) => (degree === getOutDegreeBucket() ? count + 1 : count));
  const getFailureOutDegrees = () => (is174 ? [...outDegrees] : getPartialOutDegrees());
  const getViolatedProperties = (checker: LivenessChecker, path: Value[][], cycleStart: number) =>
    is174 ? [] : checker.getViolatedProperties(path, cycleStart);
  const withOutDegrees = (result: CheckResult, degrees = [...outDegrees]): CheckResult => ({
    ...result,
    violationOutDegrees: degrees,
  });
  const createSuccess = (): CheckResult => ({
    ...createResult("success"),
    outDegrees: [...outDegrees],
    fingerprintDistance: options.isFingerprintDistanceSkipped
      ? undefined
      : getFingerprintDistance(runtime.distinctStates, runtime.generatedStates, (visit) =>
          visit(seen.getFingerprints()),
        ),
  });

  const search = createSearch(
    config,
    environment,
    specification.variableLocations,
    () => created.order.getDeclarationOrder("variable"),
    options.onProgress,
    specification.moduleOperators,
  );
  const findViolatedAssumption = () =>
    findAssumptionViolation(specification.assumptions, environment);
  if (!search) {
    runtime.isStarted = true;
    const assumptionViolation = findViolatedAssumption();
    if (assumptionViolation) return createResult("assumption-violation", assumptionViolation);
    computedInitialStates = { generated: 0, distinct: 0 };
    options.onProgress?.({ kind: "initial-states-started" });
    options.onProgress?.({
      kind: "initial-states-computed",
      initialStates: computedInitialStates,
    });
    return createSuccess();
  }
  const hasPostconditions =
    config.postconditions.length > 0 ||
    config.possible.length > 0 ||
    specification.moduleOperators.some((operator) => operator.kind === "postcondition");
  const tracedSteps = new WeakMap<TraceStep, TracedStep>();
  const getStepState = (step: TraceStep) => tracedSteps.get(step)?.state ?? step.state;
  const getStepAction = (step: TraceStep) => {
    const action = tracedSteps.get(step)?.action;
    return action ? toActionValue(action) : createActionRecord(step.action);
  };
  const checkTracePostconditions = (
    trace: TraceStep[],
    loop: { start: number; action: Value } | undefined,
  ) => {
    if (hasPostconditions) {
      runtime.counterExample = createCounterExample(
        trace.map(getStepState),
        trace.slice(1).map(getStepAction),
        loop,
      );
    }
    const unwitnessed = search.findUnwitnessedPossible();
    const checked = unwitnessed
      ? { result: undefined, lines: [] }
      : captureOutput(() => search.findPostconditionViolation(true));
    return { unwitnessed, postcondition: checked.result, lines: checked.lines };
  };
  const reportViolation = (
    status: CheckStatus,
    violation: ReportedViolation,
    getLoopAction?: () => Value,
  ) => {
    const start = violation.loopStart;
    const checked = checkTracePostconditions(
      violation.trace,
      start === undefined || !getLoopAction ? undefined : { start, action: getLoopAction() },
    );
    return createResult(status, {
      ...violation,
      ...(checked.lines.length > 0 && { postconditionOutput: checked.lines }),
      ...(checked.unwitnessed && { unwitnessed: checked.unwitnessed }),
      ...(checked.postcondition && { postcondition: checked.postcondition }),
    });
  };
  // Deviation from TLC: -continue ends with the status of the first violation, not a success.
  let continuedStatus: CheckStatus | undefined;
  const reportContinuedViolation = (
    status: CheckStatus,
    violation: Omit<Violation, "loopStart">,
  ) => {
    continuedStatus ??= status;
    options.onProgress?.({
      kind: "violation",
      status,
      violation: { ...violation, loopStart: undefined },
    });
  };
  const spec = search.spec;
  const getFingerprintState = search.getFingerprintState;
  // Tool's constructor splits the next-state relation into actions before TLC checks anything.
  const nextActions = getDescription().actions;
  runtime.getSuccessorAction = (state, next) =>
    nextActions.find((action) =>
      getSuccessors(action.expression, state, environment, action.scope).some((successor) =>
        successor.state.every((value, index) => isSameValue(value, next[index])),
      ),
    );
  const addCoverageRelations = (kind: "init" | "next", actions: SpecAction[]) =>
    actions.map((action): CoveredAction => ({
      ...action,
      node: coverage
        ? addCoverageRelation(
            coverage,
            kind,
            getCoverageLabel(action.name, action.expression, environment),
            action.expression,
            action.scope,
          )
        : -1,
    }));
  const coverageInits = addCoverageRelations("init", coverage ? getDescription().inits : []);
  const coverageActions = addCoverageRelations("next", coverage ? nextActions : []);
  if (coverage) buildCoverage(coverage, environment, getPrimedRanges(coverageActions, environment));
  const walkCoveredSuccessors = (
    action: CoveredAction,
    state: Value[],
    onState: (successor: Value[]) => void,
  ) =>
    walkSuccessors(
      action,
      action.node,
      state,
      environment,
      (successor) => {
        if (coverage) coverage.primary[action.node]++;
        onState(successor);
      },
      (partial) => {
        const missing = environment.variables.filter(
          (_variable, index) => partial[index] === undefined,
        );
        throw new TlaIncompleteSuccessorError(
          `Successor state is not completely specified by action ${action.name}. The following variables are not assigned: ${missing.join(", ")}`,
          partial,
          action.name,
          missing,
        );
      },
    );
  const getCoveredSuccessors = (action: CoveredAction, state: Value[]) => {
    const successors: Successor[] = [];
    walkCoveredSuccessors(action, state, (successor) =>
      successors.push({ state: successor, action: action.name }),
    );
    return successors;
  };
  const rejectIncompleteInitialState = (partial: (Value | undefined)[]) => {
    throw setExitCode(new TlaIllegalStateError([...partial]), EXIT_STATUS.error);
  };
  const getCoveredInitialStates = () => {
    const initialStates: Value[][] = [];
    replayInitialStates(
      coverageInits,
      environment,
      (state) => initialStates.push(state),
      rejectIncompleteInitialState,
    );
    return initialStates;
  };
  const createReplayedStates = () => {
    const replayed = unstoredFingerprint ? [unstoredFingerprint] : [];
    return (state: Value[], callStack: CallStack | undefined) => {
      fingerprintReplayed(state, getFingerprintState(state), callStack);
      const fingerprintState = getFingerprintState(state);
      const isSeen =
        findSeen(fingerprintState) !== -1 ||
        replayed.some((seen) => isSameState(seen, fingerprintState));
      if (isSeen) return false;
      replayed.push(fingerprintState);
      runtime.distinctStates++;
      return true;
    };
  };
  const replayInit = () => {
    let replayedCount = 0;
    const deferred: unknown[] = [];
    const addReplayed = createReplayedStates();
    return getReplayFailure(environment, (tracing) => {
      replayInitialStates(getDescription().inits, tracing, (state) => {
        reportInitialStatesProgress(replayedCount, (progress) => capturedLines?.push(progress));
        replayedCount++;
        runtime.generatedStates = replayedCount;
        if (deferred.length > 0) return;
        try {
          const addFingerprint = () => {
            if (!addReplayed(state, tracing.callStack)) return;
            runtime.queueSize++;
            liveness?.addInitialState(seen.count, state, tracing);
          };
          const violation = search.checkReplayedInitialState(state, tracing, addFingerprint);
          if (violation) {
            const traced = traces.toTrace([{ action: undefined, state }], []).trace[0].state;
            capturedLines?.push({
              kind: "initial-state-violation",
              ...violation,
              state: traced,
            });
            throw STOP_REPLAY;
          }
        } catch (error) {
          const isFingerprintFailure =
            error instanceof TlaEvaluationError && tracing.callStack?.fingerprint !== undefined;
          if (!isFingerprintFailure && !isStackOverflow(error)) throw error;
          deferred.push(error);
        }
      });
      if (deferred.length > 0) throw deferred[0];
    });
  };
  const replayNext = (index: number) =>
    getReplayFailure(environment, (tracing) => {
      const state = getState(index);
      const source = stateWriter && getGraphState(state, seen.getDepth(index));
      const addReplayed = createReplayedStates();
      replaySuccessors(nextActions, state, tracing, (next, action) => {
        const addFingerprint = () => {
          const isUnseen = addReplayed(next, tracing.callStack);
          if (source) {
            stateWriter?.writeSuccessor(
              source,
              getGraphState(next, seen.getDepth(index) + 1),
              !isUnseen,
              isNamingEdges ? getGraphAction(action) : undefined,
            );
          }
          return isUnseen;
        };
        if (!search.checkReplayedSuccessor(state, next, tracing, addFingerprint)) {
          throw STOP_REPLAY;
        }
      });
    });
  // The low word of each state's TLC fingerprint, by index, which the liveness checker orders
  // successors by; hasTlcFingerprint marks the ones computed. A successor the search never
  // stored has no index, and its fingerprint is computed each time.
  let tlcFingerprints = new Int32Array(0);
  let hasTlcFingerprint = new Uint8Array(0);
  const getTlcFingerprint = (successor: LiveSuccessor) => {
    const index = successor.index;
    if (index === -1) return Number(BigInt.asIntN(32, runtime.getFingerprint(successor.state)));
    if (index >= tlcFingerprints.length) {
      const length = Math.max(1024, 2 * tlcFingerprints.length, index + 1);
      const grown = new Int32Array(length);
      grown.set(tlcFingerprints);
      tlcFingerprints = grown;
      const marks = new Uint8Array(length);
      marks.set(hasTlcFingerprint);
      hasTlcFingerprint = marks;
    }
    if (hasTlcFingerprint[index] === 0) {
      tlcFingerprints[index] = Number(BigInt.asIntN(32, runtime.getFingerprint(successor.state)));
      hasTlcFingerprint[index] = 1;
    }
    return tlcFingerprints[index];
  };
  if (search.properties.liveness.length > 0) {
    options.onProgress?.({
      kind: "liveness-initializing",
      isSymmetric: search.isSymmetric,
      isConstrained: search.isConstrained,
    });
  }
  const liveness = withLivenessExitCode(() =>
    withUnexpectedException(() =>
      createLivenessChecker(search.properties.liveness, spec.temporals, environment, {
        getFingerprint: getTlcFingerprint,
        getSuccessors: (source, state) => {
          const successors: LiveSuccessor[] = [];
          if (spec.next === undefined) return successors;
          const livenessSource = current.livenessSource;
          const livenessState = current.livenessState;
          current.livenessSource = source;
          current.livenessState = state;
          try {
            forEachSuccessor(
              spec.next,
              state,
              environment,
              (successor) => {
                canonicalizeState(successor.state);
                const isInModel =
                  search.satisfiesConstraints(successor.state, state) &&
                  search.satisfiesActionConstraints(state, successor.state);
                if (!isInModel) return false;
                const fingerprintState = getFingerprintState(successor.state);
                const index = findSeen(fingerprintState);
                successors.push({ index, state: successor.state });
                return false;
              },
              spec.scope,
            );
          } finally {
            current.livenessSource = livenessSource;
            current.livenessState = livenessState;
          }
          return successors;
        },
      }),
    ),
  );
  if (liveness?.branches === 0) throw createTautologyError();
  runtime.isStarted = true;
  const isCanonicalizedFirst = search.isSymmetric || liveness !== undefined;

  const canonicalizeState = createCanonicalizer();
  // The search keeps a state's values only while the state waits in the queue or is being
  // expanded, at current.index; a trace rebuilds the states on its path by replaying it from an
  // initial state, matching each step's fingerprint. The liveness checker keeps only indices and
  // the truth of its predicates, evaluated while a state is expanded, so a lasso is rebuilt the
  // same way. A search over an impure spec, which a replay cannot repeat, keeps every state in
  // retained.
  const retained: Value[][] | undefined = isImpure(environment) ? [] : undefined;
  const queue = createStateQueue(
    environment.variables.length,
    canonicalizeState,
    retained === undefined,
    options.spillDirectory,
  );
  closers.push(queue.close);
  let expanding: Value[] | undefined;
  const fingerprint = new Int32Array(2);

  const toRecord = search.toRecord;
  const getSpecInitialStates = () => {
    if (coverage) return runtime.replayInitialStates(getCoveredInitialStates);
    const init = spec.init;
    if (init === undefined) return [];
    return runtime.replayInitialStates(() => getInitialStates(init, environment, spec.scope));
  };
  const isUnfingerprintable = (state: Value[]) => {
    try {
      return getFingerprintState(state).some((value) => getFingerprintError(value) !== undefined);
    } catch (error) {
      if (!(error instanceof TlaEvaluationError)) throw error;
      return true;
    }
  };
  const createFailureResult = (
    failed: unknown,
    behavior: Behavior,
    failureOptions: FailureOptions = {},
  ) => {
    const error = isStackOverflow(failed) ? new TlaStackOverflowError() : failed;
    if (!(error instanceof TlaEvaluationError)) throw error;
    const checked =
      behavior.trace.length > 0 && !failureOptions.traceFailure
        ? checkTracePostconditions(behavior.trace, undefined)
        : undefined;
    const postcondition = checked?.unwitnessed ?? checked?.postcondition;
    const traceOutput = [...(behavior.traceOutput ?? []), ...(checked?.lines ?? [])];
    return {
      ...createResult("error", undefined, {
        ...getFailureMessage(
          error,
          failureOptions.phase !== undefined && failureOptions.phase !== "initial",
        ),
        ...behavior,
        ...(traceOutput.length > 0 && { traceOutput }),
        ...(postcondition && { postcondition }),
        initialState:
          failureOptions.initialState && !(error instanceof TlaModuleError)
            ? toRecord(failureOptions.initialState)
            : undefined,
        exitCode: getFailureExitCode(error, failureOptions, is174),
        callStack: failureOptions.callStack,
        ...(failureOptions.phase === "liveness" && { isCallStackFirst: true }),
        ...(failureOptions.fingerprint && { fingerprint: failureOptions.fingerprint }),
        ...(failureOptions.traceFailure && { traceFailure: failureOptions.traceFailure }),
      }),
      outDegrees: failureOptions.outDegrees,
    };
  };
  const reportIncompleteSuccessor = (index: number, error: TlaIncompleteSuccessorError) => {
    runtime.generatedStates++;
    const named =
      nextActions.length === 1
        ? nextActions
        : nextActions.filter((item) => item.name === error.action);
    const action =
      named.length === 1
        ? named[0]
        : captureOutput(() =>
            named.find((item) =>
              hasIncompleteSuccessor(item.expression, getState(index), environment, item.scope),
            ),
          ).result;
    const message = getIncompleteSuccessorMessage(
      error.variables,
      error.action,
      nextActions.length,
      options.version,
    );
    const undefinedValue = createModelValue("null");
    const partial = error.state.map((value) => value ?? undefinedValue);
    const failure = createFailureResult(error, traces.getStateTrace(index));
    return withOutDegrees({
      ...failure,
      failure: failure.failure && {
        ...failure.failure,
        code: MESSAGE_CODE.incompleteSuccessor,
        message,
        toolMessage: message,
        exitCode: EXIT_STATUS.specEvaluationFailure,
        trace: [
          ...failure.failure.trace,
          {
            action: error.action,
            label: action && getActionLabel(action, options.version),
            state: toRecord(partial),
          },
        ],
      },
    });
  };
  const fingerprintModelState = (state: Value[]) => {
    for (const value of state) {
      const error = getFingerprintError(value);
      if (error) throw error;
    }
  };
  // findSeen returns the index of the state whose fingerprint state is fingerprintState, or -1,
  // and leaves its fingerprint in fingerprint.
  const findSeen = (fingerprintState: Value[]) => {
    search.writeFingerprint(fingerprintState, fingerprint);
    return seen.find(fingerprint[0], fingerprint[1]);
  };
  let unstoredFingerprint: Value[] | undefined;
  const countDistinctState = (depth: number) => {
    runtime.distinctStates = seen.count + 1;
    searchDepth = Math.max(searchDepth, depth);
    runtime.diameter = Math.max(runtime.diameter, depth);
  };
  // addState stores a new state whose fingerprint words are low and high and queues it.
  const addState = (
    state: Value[],
    low: number,
    high: number,
    parent: number,
    action: string | undefined,
  ) => {
    seen.add(low, high, parent, action, parent === -1 ? 1 : seen.getDepth(parent) + 1);
    retained?.push(state);
    queue.push(state);
  };

  const current: SearchPosition = {
    index: -1,
    isExpanding: false,
    candidate: undefined,
    candidateLevel: 0,
    livenessSource: -1,
    livenessState: undefined,
    rebuilt: undefined,
  };
  const traces = createTraceBuilder({
    config,
    options,
    environment,
    search,
    nextActions,
    current,
    seen,
    retained,
    getExpanding: () => expanding,
    findSeen: (state) => (isUnfingerprintable(state) ? -1 : findSeen(getFingerprintState(state))),
    tracedSteps,
    captureOutput,
    getInitialStates: getSpecInitialStates,
    findSuccessorAction: (state, isMatch) =>
      coverage
        ? coverageActions.find((item) => getCoveredSuccessors(item, state).some(isMatch))
        : nextActions.find((item) =>
            getSuccessors(item.expression, state, environment, item.scope).some(isMatch),
          ),
  });
  const getState = traces.getState;
  const getPath = (index: number): Value[][] => {
    if (index === -1) return [];
    const path = traces.getStoredSteps(traces.getPathTo(index)).map((step) => step.state);
    if (index === current.index && expanding) path[path.length - 1] = expanding;
    return path;
  };
  runtime.isSeen = (state) =>
    !isUnfingerprintable(state) && findSeen(getFingerprintState(state)) !== -1;
  runtime.getBehavior = (state) => {
    if (state === current.candidate) return [...getPath(current.index), state];
    if (state === expanding) return getPath(current.index);
    const source = current.livenessSource;
    if (source !== -1) {
      return state === current.livenessState ? getPath(source) : [...getPath(source), state];
    }
    const index = retained?.indexOf(state) ?? -1;
    return index === -1 ? [...getPath(current.index), state] : getPath(index);
  };
  // While a trace rebuilds a step, the state it replays from is current, so a random choice in
  // Next draws what it drew when the search expanded that state.
  runtime.getCurrentState = () =>
    current.rebuilt?.state ?? (current.index === -1 ? undefined : expanding);
  runtime.setLivenessSource = (index, state) => {
    current.livenessSource = index ?? -1;
    current.livenessState = index === undefined ? undefined : state;
  };
  runtime.getLevel = (state) => {
    const source = current.livenessSource;
    if (is174) {
      const explored = source !== -1 ? source : current.isExpanding ? current.index : -1;
      return explored === -1 ? 0 : seen.getDepth(explored);
    }
    const rebuilt = current.rebuilt;
    if (rebuilt) return state === rebuilt.state ? rebuilt.level : rebuilt.level + 1;
    if (source !== -1) {
      return state === current.livenessState ? seen.getDepth(source) : seen.getDepth(source) + 1;
    }
    if (state === current.candidate) return current.candidateLevel;
    if (current.index === -1) return 0;
    if (state === expanding) return seen.getDepth(current.index);
    return current.isExpanding ? seen.getDepth(current.index) + 1 : 0;
  };

  const getGraphAction = (action: SpecAction) => ({
    name: action.name,
    signature: getActionSignature(action),
  });
  const getGraphState = (state: Value[], level: number): GraphState => ({
    fingerprint: BigInt.asIntN(64, runtime.getFingerprint(state)),
    level,
    getLabel: () =>
      formatState(
        config.alias === undefined
          ? toRecord(state)
          : search.evaluateAlias(config.alias, state, undefined, options.isTool ?? false),
      ),
    getText: () => formatState(toRecord(state)),
  });

  const visitInitialState = (state: Value[]) => {
    canonicalizeState(state);
    runtime.generatedStates++;
    runtime.initialStates++;
    current.candidate = state;
    current.candidateLevel = 1;
    const fingerprintState = getFingerprintState(state);
    const isInModel = search.satisfiesConstraints(state);
    if (isInModel) {
      search.trackStatePossible(state);
      fingerprintModelState(fingerprintState);
      state.forEach(markFingerprinted);
    }
    const isSeen = isInModel && findSeen(fingerprintState) !== -1;
    if (isSeen) return undefined;
    const [low, high] = fingerprint;
    unstoredFingerprint = isInModel ? fingerprintState : undefined;
    if (isInModel) {
      countDistinctState(1);
      runtime.queueSize++;
      liveness?.addInitialState(seen.count, state);
      stateWriter?.writeInitialState(getGraphState(state, 1));
    }
    if (options.isContinued) {
      search.getViolatedInvariants(state).forEach((invariant) =>
        reportContinuedViolation("invariant-violation", {
          name: invariant.name,
          ...traces.toTrace([{ action: undefined, state }], []),
        }),
      );
    }
    const violated = options.isContinued ? undefined : search.findViolatedInvariant(state);
    if (violated) {
      return reportViolation("invariant-violation", {
        name: violated.name,
        ...traces.toTrace([{ action: undefined, state }], []),
      });
    }
    const failedPredicate = search.findFailedInitialPredicate(state);
    if (failedPredicate) {
      return reportViolation("liveness-violation", {
        name: failedPredicate.name,
        ...traces.toTrace([{ action: undefined, state }], []),
      });
    }
    if (!isInModel) return undefined;
    addState(state, low, high, -1, undefined);
    return undefined;
  };
  // Looks a successor up, where the constraints keep it in the model, as the search does before it
  // checks the successor.
  const lookUpSuccessor = (state: Value[], successor: Successor) => {
    const fingerprintState = getFingerprintState(successor.state);
    const isStateInModel = search.satisfiesConstraints(successor.state, state);
    if (isStateInModel) search.trackStatePossible(successor.state);
    const isInModel = isStateInModel && search.satisfiesActionConstraints(state, successor.state);
    if (isInModel) {
      search.trackActionPossible(state, successor.state);
      fingerprintModelState(fingerprintState);
      successor.state.forEach(markFingerprinted);
    }
    // findSeen leaves the successor's fingerprint words in fingerprint.
    return { fingerprintState, isInModel, existing: isInModel ? findSeen(fingerprintState) : -1 };
  };
  const visitSuccessor = (
    index: number,
    successor: Successor,
    liveSuccessors: LiveSuccessor[],
    source: GraphState | undefined,
    action: SpecAction | undefined,
  ) => {
    const state = getState(index);
    const depth = seen.getDepth(index);
    runtime.generatedStates++;
    current.candidate = successor.state;
    current.candidateLevel = depth + 1;
    if (isCanonicalizedFirst) canonicalizeState(successor.state);
    const found = lookUpSuccessor(state, successor);
    const [low, high] = fingerprint;
    if (source && found.isInModel) {
      stateWriter?.writeSuccessor(
        source,
        getGraphState(successor.state, depth + 1),
        found.existing !== -1,
        action && getGraphAction(action),
      );
    }
    if (source && !found.isInModel && options.dump?.constrained) {
      const excluded = getGraphState(successor.state, depth + 1);
      search
        .getExcludingConstraints(state, successor.state)
        .forEach((constraint) =>
          stateWriter?.writeConstrained(
            source,
            excluded,
            action && getGraphAction(action),
            constraint,
          ),
        );
    }
    if (found.existing !== -1 && liveness) {
      // The stored values, where the search holds them, keep their identity for the liveness
      // checker's caches; a state it no longer holds is not rebuilt for that.
      const stored = found.existing === index ? state : retained?.[found.existing];
      const isStored = stored?.every((value, position) => value === successor.state[position]);
      liveSuccessors.push({
        index: found.existing,
        state: stored && isStored ? stored : successor.state,
      });
    }
    const isNew = found.isInModel && found.existing === -1;
    if (isNew && liveness) liveSuccessors.push({ index: seen.count, state: successor.state });
    unstoredFingerprint = isNew ? found.fingerprintState : undefined;
    if (isNew) {
      countDistinctState(depth + 1);
      newStateCount++;
      if (coverage) coverage.secondary[coveredNode]++;
    }
    if (found.existing === -1) {
      try {
        const violated = search.findViolatedInvariant(successor.state, state);
        const violation = violated && { name: violated.name, ...traces.getTrace(index, successor) };
        if (violation && !options.isContinued) {
          return withOutDegrees(reportViolation("invariant-violation", violation));
        }
        if (violation) reportContinuedViolation("invariant-violation", violation);
      } catch (error) {
        if (isStackOverflow(error)) throw error;
        const replay = captureOutput(() => replayNext(index));
        return createFailureResult(
          error,
          appendTraceOutput(traces.getFailedTrace(index, successor), replay.lines),
          replay.result ?? { outDegrees: getFailureOutDegrees() },
        );
      }
    }
    const violatedProperty = search.findViolatedActionProperty(state, successor.state);
    const propertyViolation = violatedProperty && {
      name: violatedProperty.name,
      ...traces.getTrace(index, successor),
    };
    if (propertyViolation && !options.isContinued) {
      return withOutDegrees(reportViolation("liveness-violation", propertyViolation));
    }
    if (propertyViolation) reportContinuedViolation("liveness-violation", propertyViolation);
    if (!isNew) return undefined;
    if (coverage) countCoverageVariables(coverage, successor.state);
    if (!isCanonicalizedFirst) canonicalizeState(successor.state);
    addState(successor.state, low, high, index, successor.action);
    runtime.queueSize++;
    return undefined;
  };

  let outcome: CheckResult | undefined;
  let coveredNode = -1;
  if (liveness) {
    options.onProgress?.({ kind: "liveness-prepared", branches: liveness.branches });
  }
  const assumptionViolation = findViolatedAssumption();
  if (assumptionViolation) return createResult("assumption-violation", assumptionViolation);
  options.onProgress?.({ kind: "initial-states-started" });
  const typedChecks =
    !options.isContinued &&
    options.dump === undefined &&
    // A set of small integers on the tier is a 31-bit mask, which a lower bound could overflow.
    options.maxSetSize === undefined &&
    !hasPostconditions
      ? search.getTypedChecks(liveness?.solutions)
      : undefined;
  const typed =
    typedChecks &&
    runTypedSearch(environment, {
      ...typedChecks,
      isDeadlockChecked: config.checkDeadlock,
      isFingerprinted: !options.isFingerprintDistanceSkipped,
      livenessCheck: options.livenessCheck ?? "default",
    });
  if (typed && !typed.deadlock && !typed.violation && !typed.lasso) {
    for (let count = 2; count < typed.initialStates.generated; count *= 2) {
      reportInitialStatesProgress(count, options.onProgress);
    }
    options.onProgress?.({ kind: "initial-states-computed", initialStates: typed.initialStates });
    const statistics = {
      ...createResult("success"),
      generatedStates: typed.generatedStates,
      distinctStates: typed.distinctStates,
      queuedStates: 0,
      depth: typed.generatedStates === 0 ? 0 : Math.max(typed.depth, 1),
      initialStates: typed.initialStates,
    };
    if (liveness && options.livenessCheck !== "off") {
      options.onProgress?.({
        kind: "liveness-started",
        branches: liveness.branches,
        isComplete: true,
        states: typed.distinctStates,
        statistics,
      });
      options.onProgress?.({ kind: "liveness-finished" });
    }
    return {
      ...statistics,
      outDegrees: typed.outDegrees,
      fingerprintDistance: options.isFingerprintDistanceSkipped
        ? undefined
        : getFingerprintDistance(
            typed.distinctStates,
            typed.generatedStates,
            typed.forEachFingerprintRun,
          ),
    };
  }
  runtime.startInitialStates();
  if (spec.init !== undefined) {
    let overflowState: Value[] | undefined;
    const reportInitialFailure = (error: unknown, state: Value[]) => {
      const replay = captureOutput(() => runtime.replayInitialStates(replayInit));
      return createFailureResult(
        error,
        { trace: [], traceOutput: replay.lines },
        { initialState: state, ...replay.result, phase: "initial" },
      );
    };
    try {
      const onInitialState = (state: Value[]) => {
        reportInitialStatesProgress(runtime.initialStates, options.onProgress);
        if (overflowState) {
          runtime.generatedStates++;
          runtime.initialStates++;
          return false;
        }
        try {
          outcome = visitInitialState(state);
        } catch (error) {
          if (isStackOverflow(error)) overflowState = state;
          else outcome = reportInitialFailure(error, state);
        }
        return outcome !== undefined;
      };
      if (coverage) {
        try {
          replayInitialStates(
            coverageInits,
            environment,
            (state) => {
              if (onInitialState(state)) throw STOP_WALK;
            },
            rejectIncompleteInitialState,
          );
        } catch (error) {
          if (error !== STOP_WALK) throw error;
        }
      } else {
        forEachInitialState(spec.init, environment, onInitialState, spec.scope);
      }
      if (overflowState) {
        outcome = reportInitialFailure(createStackOverflowError(), overflowState);
      }
    } catch (error) {
      if (error instanceof TlaIllegalStateError) {
        const illegal = createFailureResult(
          error,
          { trace: [] },
          { initialState: error.state.map((value) => value ?? createModelValue("null")) },
        );
        return {
          ...illegal,
          failure: illegal.failure && { ...illegal.failure, isWithoutStatistics: true },
        };
      }
      const replay = captureOutput(() => runtime.replayInitialStates(replayInit));
      return createFailureResult(
        error,
        { trace: [], traceOutput: replay.lines },
        { ...replay.result, phase: "initial" },
      );
    }
  }
  if (outcome) return outcome;
  computedInitialStates = {
    generated: runtime.initialStates,
    distinct: runtime.distinctStates,
  };
  options.onProgress?.({
    kind: "initial-states-computed",
    initialStates: computedInitialStates,
  });

  const next = spec.next;
  if (next === undefined) {
    if (seen.count > 0) {
      throw new TlaEvaluationError(
        "No next state actions defined to generate successor states from.",
      );
    }
    return createSuccess();
  }
  const isNamingEdges = Boolean(
    options.dump?.dot && (options.dump.actionLabels || options.dump.colorize),
  );
  const forEachActionSuccessor = (
    state: Value[],
    onUnsatisfied:
      | ((successor: Value[], conjunct: Expression, action?: SpecAction) => void)
      | undefined,
    onSuccessor: (successor: Successor, action: SpecAction | undefined) => boolean,
  ) => {
    if (coverage) {
      let successorCount = 0;
      try {
        coverageActions.forEach((action) => {
          coveredNode = action.node;
          walkCoveredSuccessors(action, state, (successor) => {
            successorCount++;
            if (onSuccessor({ state: successor, action: action.name }, action)) throw STOP_WALK;
          });
        });
      } catch (error) {
        if (error !== STOP_WALK) throw error;
      }
      return successorCount;
    }
    if (!isNamingEdges) {
      return forEachSuccessor(
        next,
        state,
        environment,
        (successor) => onSuccessor(successor, undefined),
        spec.scope,
        undefined,
        onUnsatisfied,
      );
    }
    let isStopped = false;
    return nextActions.reduce((count, action) => {
      if (isStopped) return count;
      return (
        count +
        forEachSuccessor(
          action.expression,
          state,
          environment,
          (successor) => {
            isStopped = onSuccessor(successor, action);
            return isStopped;
          },
          action.scope,
          undefined,
          onUnsatisfied && ((successor, conjunct) => onUnsatisfied(successor, conjunct, action)),
        )
      );
    }, 0);
  };
  const getLiveCallStack = (index: number, liveSuccessors: LiveSuccessor[]) => {
    if (!liveness || is174) return undefined;
    const replay = getReplayFailure(environment, (tracing) =>
      liveness.addState(index, getState(index), liveSuccessors, tracing),
    );
    return replay?.callStack.length && !replay.fingerprint ? replay.callStack : undefined;
  };
  const addFailedLiveState = (index: number, liveSuccessors: LiveSuccessor[]) => {
    try {
      liveness?.addState(index, getState(index), liveSuccessors);
      return undefined;
    } catch (error) {
      if (!(error instanceof TlaEvaluationError)) throw error;
      return getLiveCallStack(index, liveSuccessors);
    }
  };
  const replayFailedNext = (index: number, liveSuccessors: LiveSuccessor[]) => {
    const live = captureOutput(() => addFailedLiveState(index, liveSuccessors));
    if (live.result) {
      const result: FailureOptions = { phase: "behavior-graph", callStack: live.result };
      return { result, lines: live.lines };
    }
    const replay = captureOutput(() => replayNext(index));
    return { result: replay.result, lines: [...live.lines, ...replay.lines] };
  };
  const reportNextStateOverflow = (index: number) =>
    createFailureResult(createStackOverflowError(), traces.getStateTrace(index), { phase: "next" });
  const reportNextStateFailure = (
    error: unknown,
    index: number,
    liveSuccessors: LiveSuccessor[],
  ) => {
    if (isStackOverflow(error)) return reportNextStateOverflow(index);
    const replay = replayFailedNext(index, liveSuccessors);
    return createFailureResult(
      withNextStateExitCode(error),
      appendTraceOutput(traces.getStateTrace(index), replay.lines),
      { phase: "next", ...replay.result },
    );
  };
  const expandState = (index: number, liveSuccessors: LiveSuccessor[]) => {
    const state = getState(index);
    const source = stateWriter && getGraphState(state, seen.getDepth(index));
    const writeUnsatisfied =
      source && options.dump?.constrained
        ? (successor: Value[], conjunct: Expression, action?: SpecAction) =>
            stateWriter?.writeConstrained(
              source,
              getGraphState(successor, seen.getDepth(index) + 1),
              action && getGraphAction(action),
              describeRange(conjunct),
            )
        : undefined;
    const successorCount = forEachActionSuccessor(state, writeUnsatisfied, (successor, action) => {
      try {
        outcome = visitSuccessor(index, successor, liveSuccessors, source, action);
      } catch (error) {
        if (isStackOverflow(error)) {
          outcome = reportNextStateOverflow(index);
          return true;
        }
        const replay = is174
          ? captureOutput(() => replayNext(index))
          : replayFailedNext(index, liveSuccessors);
        if (is174 && error instanceof TlaEvaluationError && !isHeadedError(error)) {
          outcome = createFailureResult(
            new TlaEvaluationError(`${getWrappedException(error)}${error.message}`, error),
            appendTraceOutput({ trace: [] }, replay.lines),
            { phase: "next", ...replay.result },
          );
          return true;
        }
        const isTraceFailed = isUnfingerprintable(successor.state);
        const traceFailure =
          isTraceFailed && error instanceof TlaEvaluationError
            ? getFailureMessage(
                isEvalException(error)
                  ? new TlaEvaluationError(`${UNEXPECTED_EVAL_EXCEPTION}${error.message}`)
                  : error,
                true,
              )
            : undefined;
        outcome = createFailureResult(
          error,
          appendTraceOutput(
            traceFailure ? traces.getStateTrace(index) : traces.getFailedTrace(index, successor),
            replay.lines,
          ),
          { phase: "next", ...replay.result, traceFailure },
        );
      }
      return outcome !== undefined;
    });
    current.isExpanding = false;
    const result = outcome;
    if (result?.violation && !is174) {
      const live = captureOutput(() => addFailedLiveState(index, liveSuccessors));
      return {
        ...result,
        violation: appendTraceOutput(result.violation, live.lines),
        ...(live.result && { liveCallStack: live.result }),
      };
    }
    if (result) return result;
    const addLiveState = () => {
      if (liveness && source && (is174 || options.dump?.stuttering)) {
        stateWriter?.writeStuttering(source);
      }
      const accepted = liveness?.addState(index, state, liveSuccessors);
      if (!liveness || !accepted) return undefined;
      const steps = traces.getPathSteps(accepted);
      const properties = getViolatedProperties(
        liveness,
        steps.map((step) => step.state),
        steps.length - 1,
      );
      return reportViolation("property-violation", {
        name: properties[0],
        properties,
        ...traces.toTrace(steps, accepted, { known: 0 }),
      });
    };
    if (successorCount === 0 && config.checkDeadlock) {
      const deadlock = reportViolation("deadlock", {
        name: "deadlock",
        ...traces.getStateTrace(index),
      });
      const live = captureOutput(() => {
        try {
          return { additional: addLiveState()?.violation };
        } catch (error) {
          if (!(error instanceof TlaEvaluationError)) throw error;
          return { isFailed: true, callStack: getLiveCallStack(index, liveSuccessors) };
        }
      });
      const reported = deadlock.violation
        ? { ...deadlock, violation: appendTraceOutput(deadlock.violation, live.lines) }
        : deadlock;
      const counted = withOutDegrees(
        reported,
        live.result.isFailed ? getFailureOutDegrees() : getPartialOutDegrees(),
      );
      if (live.result.additional)
        return { ...counted, additionalViolation: live.result.additional };
      return live.result.callStack ? { ...counted, liveCallStack: live.result.callStack } : counted;
    }
    try {
      return addLiveState();
    } catch (error) {
      if (is174 && error instanceof TlaEvaluationError) {
        const unexpected = new TlaEvaluationError(
          `${isEvalException(error) ? UNEXPECTED_EVAL_EXCEPTION : UNEXPECTED_EXCEPTION}${error.message}`,
        );
        return createFailureResult(
          setExitCode(unexpected, EXIT_STATUS.error),
          { trace: [] },
          { outDegrees: getFailureOutDegrees() },
        );
      }
      const replay = getReplayFailure(environment, (tracing) =>
        liveness?.addState(index, state, liveSuccessors, tracing),
      );
      return createFailureResult(error, traces.getStateTrace(index), {
        outDegrees: getFailureOutDegrees(),
        callStack: replay?.callStack.length ? replay.callStack : undefined,
        phase: "liveness",
      });
    }
  };
  const getStutteringWarning = () => {
    const name = config.specification;
    const book =
      "(compare Chapter 8, page 87ff of Specifying Systems at https://lamport.azurewebsites.net/tla/book.html).";
    if (name === undefined) {
      return {
        code: MESSAGE_CODE.noSpecificationButProperty,
        text: `The stuttering counterexample above may be caused by the absence of a behavior specification (SPECIFICATION). Only INIT and NEXT have been provided, so TLC permits infinite stuttering. To rule out such counterexamples, use SPECIFICATION Spec, with Spec asserting a suitable fairness constraint ${book}`,
      };
    }
    const range = getDefinition(name, environment)?.location.range;
    if (spec.temporals.length > 0 || !range) return undefined;
    return {
      code: MESSAGE_CODE.noFairnessButProperty,
      text: `The stuttering counterexample above may be caused by the absence of a fairness constraint in the behavior specification ${name} defined at ${formatRange(range)}. To rule out such counterexamples, conjoin a suitable fairness constraint to ${name} ${book}`,
    };
  };
  const reportLasso = (checker: LivenessChecker, lasso: Lasso) => {
    const path = [...lasso.prefix, ...lasso.cycle];
    const loopStart = lasso.prefix.length;
    const steps = traces.getPathSteps(path, loopStart);
    const trace = traces.toTrace(steps, path, { loopStart, known: 0 }).trace;
    const properties = getViolatedProperties(
      checker,
      steps.map((step) => step.state),
      loopStart,
    );
    const cycleStart = path[loopStart];
    const cycleEnd = path[path.length - 1];
    const lastState = steps[steps.length - 1].state;
    const loop =
      cycleStart === cycleEnd
        ? { result: undefined, lines: [] }
        : captureOutput(() =>
            traces.findStepAction(cycleEnd, lastState, getState(cycleStart), false),
          );
    const isStutteringWarned = cycleStart === cycleEnd && lasso.isCycleConstrained;
    return reportViolation(
      "liveness-violation",
      {
        name: properties[0],
        properties,
        trace,
        traceOutput: loop.lines,
        loopStart,
        loopLabel: loop.result && getActionLabel(loop.result, options.version),
        warning: isStutteringWarned ? getStutteringWarning() : undefined,
      },
      () =>
        cycleStart === cycleEnd
          ? getStepAction(trace[trace.length - 1])
          : loop.result
            ? toActionValue(loop.result)
            : createActionRecord(
                traces.findSuccessorWithFingerprint(lastState, cycleStart)?.action,
              ),
    );
  };
  const isPeriodic = liveness !== undefined && (options.livenessCheck ?? "default") === "default";
  const startedAt = performance.now();
  const periodic = {
    at: startedAt + PERIODIC_START,
    size: 1,
    time: 0,
    coverageCount: Math.trunc(options.coverage ?? 0),
  };
  const findPeriodicViolation = (expanded: number) => {
    const now = performance.now();
    const isGrown = (expanded - periodic.size) / periodic.size > LIVENESS_GROWTH;
    if (!isGrown || periodic.time > (now - startedAt) * LIVENESS_RATIO) return undefined;
    periodic.size = expanded;
    if (!liveness) return undefined;
    options.onProgress?.({
      kind: "liveness-started",
      branches: liveness.branches,
      isComplete: false,
      states: expanded,
      statistics: createResult("success"),
    });
    const lasso = liveness.findViolation();
    if (!lasso) options.onProgress?.({ kind: "liveness-finished" });
    periodic.time += performance.now() - now;
    return lasso;
  };
  const adoptTypedSearch = (searched: TypedResult) => {
    runtime.generatedStates = searched.generatedStates;
    runtime.distinctStates = searched.distinctStates;
    runtime.queueSize = searched.queuedStates;
    runtime.diameter = searched.depth;
    searchDepth = searched.depth;
    outDegrees.splice(0, outDegrees.length, ...searched.outDegrees);
    newStateCount = 0;
  };
  // Finds the stored initial state isState matches, with its index. The search keeps a state's
  // values only while it waits in the queue unless it retains them, so it replays Init otherwise.
  const findStoredInitialState = (isState: (state: Value[]) => boolean) => {
    const state = (retained ?? captureOutput(getSpecInitialStates).result).find(isState);
    if (!state) return undefined;
    const index = retained ? retained.indexOf(state) : findSeen(getFingerprintState(state));
    return index === -1 ? undefined : { index, state };
  };
  // The tier found a state without successors, or a successor that violates an invariant or an
  // action property. The interpreter rebuilds the tier's path to it and reports it, as it would after
  // its own search.
  const path = typed?.deadlock ?? typed?.violation;
  const initial = path && findStoredInitialState((state) => path.isPathState(0, state));
  if (typed && path && initial) {
    const steps: { successor: Successor; action: SpecAction | undefined }[] = [];
    const getLastState = () => steps.at(-1)?.successor.state ?? initial.state;
    for (let position = 1; position === steps.length + 1 && position < path.length; position++) {
      // The interpreter expands only the states it stores, which it has canonicalized.
      canonicalizeState(getLastState());
      forEachActionSuccessor(getLastState(), undefined, (successor, action) => {
        if (!path.isPathState(position, successor.state)) return false;
        steps.push({ successor, action });
        return true;
      });
    }
    const isRebuilt = steps.length === path.length - 1;
    const violating = typed.violation && isRebuilt ? steps.pop() : undefined;
    if (violating && isCanonicalizedFirst) canonicalizeState(violating.successor.state);
    const violatedInvariant =
      violating && search.findViolatedInvariant(violating.successor.state, getLastState());
    const violatedProperty =
      violating &&
      !violatedInvariant &&
      search.findViolatedActionProperty(getLastState(), violating.successor.state);
    const violated = violatedInvariant ?? violatedProperty;
    const isDeadlocked =
      typed.deadlock !== undefined &&
      isRebuilt &&
      forEachActionSuccessor(getLastState(), undefined, () => true) === 0;
    if (isDeadlocked || violated) {
      // Each step is visited from its parent as the search would expand it, so the parent's values
      // are the expanding state's rather than a replay of its path.
      const parent = { ...initial };
      for (const step of steps) {
        Object.assign(current, { index: parent.index, isExpanding: true });
        expanding = parent.state;
        visitSuccessor(parent.index, step.successor, [], undefined, step.action);
        Object.assign(parent, { index: seen.count - 1, state: step.successor.state });
      }
      adoptTypedSearch(typed);
      expanding = parent.state;
      Object.assign(current, { index: parent.index, candidate: undefined, isExpanding: true });
      if (violating && violated) {
        lookUpSuccessor(parent.state, violating.successor);
        current.candidate = violating.successor.state;
        current.candidateLevel = seen.getDepth(parent.index) + 1;
        return withOutDegrees(
          reportViolation(violatedInvariant ? "invariant-violation" : "liveness-violation", {
            name: violated.name,
            ...traces.getTrace(parent.index, violating.successor),
          }),
        );
      }
      const result = expandState(parent.index, []);
      if (result) return result;
    }
  }
  // The tier found a behavior that violates a temporal property after its whole search. The
  // interpreter stores the paths to the lasso's states and reports it, as it would after its own.
  const lassoSteps = new Map<number, { successor: Successor; action: SpecAction | undefined }>();
  for (const entry of typed?.lasso?.states ?? []) {
    const initial = entry.parent === -1 ? findStoredInitialState(entry.isState)?.state : undefined;
    if (initial) {
      lassoSteps.set(entry.index, {
        successor: { state: initial, action: undefined },
        action: undefined,
      });
    }
    const parent = lassoSteps.get(entry.parent)?.successor.state;
    if (!parent) continue;
    // The interpreter expands only the states it stores, which it has canonicalized.
    canonicalizeState(parent);
    forEachActionSuccessor(parent, undefined, (successor, action) => {
      if (!entry.isState(successor.state)) return false;
      lassoSteps.set(entry.index, { successor, action });
      return true;
    });
  }
  const lassoIndexes = new Map<number, number>();
  const typedLasso =
    typed?.lasso && lassoSteps.size === typed.lasso.states.length ? typed.lasso : undefined;
  typedLasso?.states.forEach((entry) => {
    const step = lassoSteps.get(entry.index);
    const parent = lassoIndexes.get(entry.parent);
    const parentState = lassoSteps.get(entry.parent)?.successor.state;
    if (step && parent !== undefined && parentState) {
      Object.assign(current, { index: parent, isExpanding: true });
      expanding = parentState;
      visitSuccessor(parent, step.successor, [], undefined, step.action);
    }
    lassoIndexes.set(
      entry.index,
      parent === undefined ? (findStoredInitialState(entry.isState)?.index ?? -1) : seen.count - 1,
    );
  });
  if (typed && typedLasso) adoptTypedSearch(typed);
  for (let index = typedLasso ? seen.count : 0; index < seen.count && !runtime.isStopped; index++) {
    expanding = queue.shift();
    current.index = index;
    current.candidate = undefined;
    runtime.queueSize = seen.count - index - 1;
    current.isExpanding = true;
    const liveSuccessors: LiveSuccessor[] = [];
    try {
      outcome = expandState(index, liveSuccessors);
    } catch (error) {
      outcome =
        error instanceof TlaIncompleteSuccessorError
          ? reportIncompleteSuccessor(index, error)
          : reportNextStateFailure(error, index, liveSuccessors);
    }
    if (outcome) return outcome;
    outDegrees[getOutDegreeBucket()]++;
    newStateCount = 0;
    if ((isPeriodic || options.onProgress) && performance.now() >= periodic.at) {
      periodic.at = performance.now() + PROGRESS_INTERVAL;
      const lasso = isPeriodic ? findPeriodicViolation(index + 1) : undefined;
      if (liveness && lasso) return reportLasso(liveness, lasso);
      const isCoverageDue = options.coverage !== undefined && periodic.coverageCount === 0;
      periodic.coverageCount = isCoverageDue
        ? Math.trunc(options.coverage ?? 0)
        : periodic.coverageCount - 1;
      const progress = createResult("success");
      options.onProgress?.({
        kind: "progress",
        statistics: {
          ...progress,
          coverage: isCoverageDue ? progress.coverage : undefined,
          seenBytes: seen.byteLength,
          spillBytes: queue.spillBytes,
          spillStates: queue.spillStates,
        },
      });
    }
  }

  current.index = -1;
  current.candidate = undefined;
  expanding = undefined;
  const finalLiveness = options.livenessCheck === "off" ? undefined : liveness;
  if (finalLiveness) {
    options.onProgress?.({
      kind: "liveness-started",
      branches: finalLiveness.branches,
      isComplete: true,
      states: runtime.distinctStates,
      statistics: createResult("success"),
    });
  }
  const getLassoIndexes = (indexes: number[]) =>
    indexes.map((index) => lassoIndexes.get(index) ?? -1);
  const lasso = typedLasso
    ? {
        prefix: getLassoIndexes(typedLasso.prefix),
        cycle: getLassoIndexes(typedLasso.cycle),
        isCycleConstrained: typedLasso.isCycleConstrained,
      }
    : finalLiveness?.findViolation();
  if (finalLiveness && lasso) return reportLasso(finalLiveness, lasso);

  if (finalLiveness) options.onProgress?.({ kind: "liveness-finished" });
  const missingPostcondition = specification.moduleOperators.find(
    (operator) => operator.kind === "postcondition" && operator.key === undefined,
  );
  if (missingPostcondition) {
    return createFailureResult(createMissingOperatorError(missingPostcondition), { trace: [] });
  }
  const postcondition = search.findPostconditionViolation();
  if (postcondition) {
    return createResult("postcondition-violation", {
      name: postcondition.name,
      message: postcondition.message,
      counts: postcondition.counts,
      trace: [],
      postcondition,
    });
  }
  return continuedStatus ? createResult(continuedStatus) : createSuccess();
};
