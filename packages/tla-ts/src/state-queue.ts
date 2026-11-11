import { createStateDecoder, createStateEncoder, type EncodedStates } from "./transfer";
import type { Value } from "./values";

/**
 * The first-in, first-out queue of states a breadth-first search has found and not yet
 * expanded. Only the chunk being read and the chunk being written hold values; each full chunk
 * between them is stored compact, in a file of its own under a new directory in the spill
 * directory where the runtime has a file system and in memory otherwise, and expanded when its
 * turn comes. A chunk whose values have no encoding stays as it is.
 *
 * A compact chunk keeps, for each variable, the distinct values its states take, each encoded
 * once, and for each state the index of its value in that table. A search changes few variables
 * per step, so a variable takes few values across a chunk, and a state costs a byte or two per
 * variable rather than its full encoding.
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

// A chunk is its values, its compact form in memory, or the file holding its compact form.
type StoredChunk = Value[][] | CompactChunk | ChunkFile;

interface CompactChunk {
  count: number;
  columns: Column[];
}

// A column is one variable's distinct values in a chunk, encoded as one-variable states, and
// each state's index into them.
interface Column {
  table: EncodedStates;
  ids: Uint8Array | Uint16Array;
}

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

const decodeTable = createStateDecoder(1);

// compact returns chunk's compact form, or undefined when a value has no encoding. A value seen
// before in its column, as the same object or with the same encoding, reuses its index.
const compact = (chunk: Value[][], width: number): CompactChunk | undefined => {
  const columns: Column[] = [];
  for (let column = 0; column < width; column++) {
    const encoder = createStateEncoder();
    const byIdentity = new Map<Value, number>();
    const byWords = new Map<string, number>();
    const ids = new Uint16Array(chunk.length);
    let size = 0;
    for (let index = 0; index < chunk.length; index++) {
      const value = chunk[index][column];
      let id = byIdentity.get(value);
      if (id === undefined) {
        const start = encoder.length;
        if (!encoder.encode([value])) return undefined;
        const key = encoder.key(start);
        id = byWords.get(key);
        if (id === undefined) byWords.set(key, (id = size++));
        else encoder.drop(start);
        byIdentity.set(value, id);
      }
      ids[index] = id;
    }
    // A chunk holds at most CHUNK_SIZE states, so every index fits in 16 bits.
    columns.push({ table: encoder.finish(), ids: size <= 1 << 8 ? Uint8Array.from(ids) : ids });
  }
  return { count: chunk.length, columns };
};

// expand returns a compact chunk's states. It canonicalizes each table value once, so states
// share the values they had in common when queued.
const expand = (chunk: CompactChunk, canonicalize: (state: Value[]) => void): Value[][] => {
  // Columns run in variable order, so pushing each column's value builds every state in order.
  const states = Array.from({ length: chunk.count }, (): Value[] => []);
  for (const { table, ids } of chunk.columns) {
    const values = decodeTable(table).map((state) => state[0]);
    canonicalize(values);
    for (let index = 0; index < chunk.count; index++) states[index].push(values[ids[index]]);
  }
  return states;
};

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();

const align = (bytes: number) => (bytes + 3) & ~3;

// A chunk file is the state count and width, then for each column its value count, word count,
// string bytes and bytes per index, then each column's words, its strings as JSON and its
// indices, each section padded to four bytes.
const toBytes = (chunk: CompactChunk) => {
  const strings = chunk.columns.map(({ table }) =>
    textEncoder.encode(JSON.stringify(table.strings)),
  );
  const header = 8 + chunk.columns.length * 16;
  let total = header;
  chunk.columns.forEach(({ table, ids }, column) => {
    total += table.words.length * 4 + align(strings[column].length) + align(ids.byteLength);
  });
  const bytes = new Uint8Array(total);
  const head = new Int32Array(bytes.buffer, 0, header / 4);
  head[0] = chunk.count;
  head[1] = chunk.columns.length;
  let offset = header;
  chunk.columns.forEach(({ table, ids }, column) => {
    head.set(
      [table.count, table.words.length, strings[column].length, ids.BYTES_PER_ELEMENT],
      2 + column * 4,
    );
    bytes.set(
      new Uint8Array(table.words.buffer, table.words.byteOffset, table.words.byteLength),
      offset,
    );
    offset += table.words.byteLength;
    bytes.set(strings[column], offset);
    offset += align(strings[column].length);
    bytes.set(new Uint8Array(ids.buffer, ids.byteOffset, ids.byteLength), offset);
    offset += align(ids.byteLength);
  });
  return bytes;
};

const ID_ARRAYS = { 1: Uint8Array, 2: Uint16Array } as const;

const fromBytes = (bytes: Uint8Array): CompactChunk => {
  const aligned = bytes.byteOffset % 4 === 0 ? bytes : bytes.slice();
  // A file read returns a plain ArrayBuffer, never a shared one.
  const buffer = aligned.buffer as ArrayBuffer;
  const byteOffset = aligned.byteOffset;
  const [count, width] = new Int32Array(buffer, byteOffset, 2);
  const head = new Int32Array(buffer, byteOffset + 8, width * 4);
  let offset = 8 + width * 16;
  const columns = Array.from({ length: width }, (_, column): Column => {
    const [tableCount, wordCount, stringBytes, idBytes] = head.subarray(column * 4, column * 4 + 4);
    const words = new Int32Array(buffer, byteOffset + offset, wordCount);
    offset += wordCount * 4;
    const strings = JSON.parse(
      textDecoder.decode(aligned.subarray(offset, offset + stringBytes)),
    ) as string[];
    offset += align(stringBytes);
    const ids = new ID_ARRAYS[idBytes as 1 | 2](buffer, byteOffset + offset, count);
    offset += align(count * idBytes);
    return { table: { count: tableCount, words, strings }, ids };
  });
  return { count, columns };
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
    const encoded = compact(chunk, width);
    if (!encoded) return chunk;
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
    if (!("path" in chunk)) return expand(chunk, canonicalize);
    const encoded = fromBytes(files!.fs.readFileSync(chunk.path));
    files!.fs.rmSync(chunk.path, { recursive: false, force: true });
    spillBytes -= chunk.bytes;
    spillStates -= chunk.states;
    return expand(encoded, canonicalize);
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
