import { createStateDecoder, createStateEncoder, type EncodedStates } from "./transfer";
import type { Value } from "./values";

/**
 * The first-in, first-out queue of states a breadth-first search has found and not yet
 * expanded. Only the chunk being read and the chunk being written hold values; each full chunk
 * between them is stored encoded, in a file of its own under a new directory in the spill
 * directory where the runtime has a file system and in memory otherwise, and decoded when its
 * turn comes. A chunk whose values have no encoding stays as it is.
 */
export interface StateQueue {
  readonly size: number;
  /** The bytes of the chunk files the queue holds now. */
  readonly spillBytes: number;
  /** The states those chunk files hold. */
  readonly spillStates: number;
  push: (state: Value[]) => void;
  /** Returns the oldest state in the queue and removes it, or undefined when it is empty. */
  shift: () => Value[] | undefined;
  /** Deletes the queue's files. */
  close: () => void;
}

interface FileSystem {
  mkdtempSync: (prefix: string) => string;
  writeFileSync: (path: string, data: Uint8Array) => void;
  readFileSync: (path: string) => Uint8Array;
  rmSync: (path: string, options: { recursive: boolean; force: boolean }) => void;
}

// A chunk is its values, its encoding in memory, or the file holding its encoding.
type StoredChunk = Value[][] | EncodedStates | ChunkFile;

interface ChunkFile {
  path: string;
  bytes: number;
  states: number;
}

const CHUNK_SIZE = 1 << 14;

// getFileSystem returns the file system and the spill directory: spillDirectory when given,
// else the runtime's temporary directory, which follows TMPDIR.
const getFileSystem = (spillDirectory: string | undefined) => {
  const getModule = (globalThis as { process?: { getBuiltinModule?: (name: string) => unknown } })
    .process?.getBuiltinModule;
  const fs = getModule?.("node:fs") as FileSystem | undefined;
  const os = getModule?.("node:os") as { tmpdir: () => string } | undefined;
  return fs && os && { fs, directory: spillDirectory ?? os.tmpdir() };
};

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();

// A chunk file is three 32-bit words, the state count, word count and string bytes, then the
// words, then the strings as JSON.
const toBytes = (encoded: EncodedStates) => {
  const strings = textEncoder.encode(JSON.stringify(encoded.strings));
  const bytes = new Uint8Array(12 + encoded.words.length * 4 + strings.length);
  new Int32Array(bytes.buffer, 0, 3).set([encoded.count, encoded.words.length, strings.length]);
  bytes.set(new Uint8Array(encoded.words.buffer, 0, encoded.words.length * 4), 12);
  bytes.set(strings, 12 + encoded.words.length * 4);
  return bytes;
};

const fromBytes = (bytes: Uint8Array): EncodedStates => {
  const aligned = bytes.byteOffset % 4 === 0 ? bytes : bytes.slice();
  const [count, wordCount, stringBytes] = new Int32Array(aligned.buffer, aligned.byteOffset, 3);
  const words = new Int32Array(aligned.buffer, aligned.byteOffset + 12, wordCount);
  const strings = JSON.parse(
    textDecoder.decode(aligned.subarray(12 + wordCount * 4, 12 + wordCount * 4 + stringBytes)),
  ) as string[];
  return { count, words, strings };
};

/**
 * Creates a queue of states of width variables. canonicalize prepares a decoded state as the
 * search prepared it before queueing it. With isEncoded false every chunk keeps its values, for
 * a search that keeps every state anyway. spillDirectory overrides where the chunk files go.
 */
export const createStateQueue = (
  width: number,
  canonicalize: (state: Value[]) => void,
  isEncoded: boolean,
  spillDirectory?: string,
): StateQueue => {
  const decode = createStateDecoder(width);
  const files = isEncoded ? getFileSystem(spillDirectory) : undefined;
  let directory: string | undefined;
  let fileCount = 0;
  let spillBytes = 0;
  let spillStates = 0;
  const stored: StoredChunk[] = [];
  let storedStart = 0;
  let head: Value[][] = [];
  let headPosition = 0;
  let tail: Value[][] = [];
  let size = 0;

  const store = (chunk: Value[][]): StoredChunk => {
    if (!isEncoded) return chunk;
    const encoder = createStateEncoder();
    if (!chunk.every(encoder.encode)) return chunk;
    const encoded = encoder.finish();
    if (!files) return encoded;
    directory ??= files.fs.mkdtempSync(`${files.directory}/tla-ts-queue-`);
    const path = `${directory}/${fileCount++}`;
    const bytes = toBytes(encoded);
    files.fs.writeFileSync(path, bytes);
    spillBytes += bytes.length;
    spillStates += chunk.length;
    return { path, bytes: bytes.length, states: chunk.length };
  };
  const load = (chunk: StoredChunk): Value[][] => {
    if (Array.isArray(chunk)) return chunk;
    if (!("path" in chunk)) return decode(chunk).map(withCanonical);
    const encoded = fromBytes(files!.fs.readFileSync(chunk.path));
    files!.fs.rmSync(chunk.path, { recursive: false, force: true });
    spillBytes -= chunk.bytes;
    spillStates -= chunk.states;
    return decode(encoded).map(withCanonical);
  };
  const withCanonical = (state: Value[]) => {
    canonicalize(state);
    return state;
  };
  return {
    get size() {
      return size;
    },
    get spillBytes() {
      return spillBytes;
    },
    get spillStates() {
      return spillStates;
    },
    push: (state) => {
      tail.push(state);
      size++;
      if (tail.length < CHUNK_SIZE) return;
      stored.push(store(tail));
      tail = [];
    },
    shift: () => {
      if (headPosition === head.length) {
        if (storedStart < stored.length) {
          head = load(stored[storedStart]);
          stored[storedStart++] = [];
          if (storedStart === stored.length) {
            stored.length = 0;
            storedStart = 0;
          }
        } else {
          head = tail;
          tail = [];
        }
        headPosition = 0;
        if (head.length === 0) return undefined;
      }
      size--;
      const state = head[headPosition];
      head[headPosition++] = undefined as unknown as Value[];
      return state;
    },
    close: () => {
      if (directory) files?.fs.rmSync(directory, { recursive: true, force: true });
      directory = undefined;
      spillBytes = 0;
      spillStates = 0;
    },
  };
};
