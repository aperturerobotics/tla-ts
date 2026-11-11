/**
 * The states a breadth-first search has seen, without their values. A state is its index, in the
 * order the search found it, and four typed columns: its 64-bit fingerprint, its parent's index
 * (-1 for an initial state), the action that reached it and its depth. An open-addressing index
 * over the fingerprints finds a state's index. The columns take 20 bytes a state and the index 8
 * to 16, and doubling can leave as much again unused; the search keeps a state's values only
 * while it waits to be expanded.
 *
 * Two states with one fingerprint are one state here, as in TLC: the search reports the chance
 * of that from the fingerprints it stored.
 */
export interface SeenStates {
  /** The number of states seen, which is also the index the next added state takes. */
  readonly count: number;
  /** Returns the index of the state with fingerprint words low and high, or -1. */
  find: (low: number, high: number) => number;
  /** Adds a state that find did not return and returns its index. */
  add: (
    low: number,
    high: number,
    parent: number,
    action: string | undefined,
    depth: number,
  ) => number;
  getParent: (index: number) => number;
  getAction: (index: number) => string | undefined;
  getDepth: (index: number) => number;
  /** Returns the fingerprint of every state seen, by index, as TLC orders its words. */
  getFingerprints: () => BigInt64Array;
}

const INITIAL_CAPACITY = 1 << 12;

const growInt32 = (column: Int32Array) => {
  const grown = new Int32Array(column.length * 2);
  grown.set(column);
  return grown;
};

export const createSeenStates = (): SeenStates => {
  // words holds each state's fingerprint as its low then its high word.
  let words = new Int32Array(INITIAL_CAPACITY * 2);
  let parents = new Int32Array(INITIAL_CAPACITY);
  let actions = new Int32Array(INITIAL_CAPACITY);
  let depths = new Int32Array(INITIAL_CAPACITY);
  // slots holds index + 1 of a state, 0 for an empty slot, and stays at most half full.
  let slots = new Int32Array(INITIAL_CAPACITY * 2);
  const actionNames: (string | undefined)[] = [undefined];
  const actionIds = new Map<string, number>();
  let count = 0;

  const getHome = (low: number, high: number, mask: number) =>
    Math.imul(low ^ Math.imul(high, 0x9e3779b1), 0x85ebca6b) & mask;
  const find = (low: number, high: number) => {
    const mask = slots.length - 1;
    for (let slot = getHome(low, high, mask); ; slot = (slot + 1) & mask) {
      const entry = slots[slot];
      if (entry === 0) return -1;
      if (words[2 * entry - 2] === low && words[2 * entry - 1] === high) return entry - 1;
    }
  };
  const place = (index: number) => {
    const mask = slots.length - 1;
    let slot = getHome(words[2 * index], words[2 * index + 1], mask);
    while (slots[slot] !== 0) slot = (slot + 1) & mask;
    slots[slot] = index + 1;
  };
  const getActionId = (action: string | undefined) => {
    if (action === undefined) return 0;
    const known = actionIds.get(action);
    if (known !== undefined) return known;
    actionIds.set(action, actionNames.length);
    actionNames.push(action);
    return actionNames.length - 1;
  };
  return {
    get count() {
      return count;
    },
    find,
    add: (low, high, parent, action, depth) => {
      if (count === parents.length) {
        words = growInt32(words);
        parents = growInt32(parents);
        actions = growInt32(actions);
        depths = growInt32(depths);
      }
      const index = count++;
      words[2 * index] = low;
      words[2 * index + 1] = high;
      parents[index] = parent;
      actions[index] = getActionId(action);
      depths[index] = depth;
      if (count * 2 > slots.length) {
        slots = new Int32Array(slots.length * 2);
        for (let entry = 0; entry < count; entry++) place(entry);
      } else {
        place(index);
      }
      return index;
    },
    getParent: (index) => parents[index],
    getAction: (index) => actionNames[actions[index]],
    getDepth: (index) => depths[index],
    getFingerprints: () => new BigInt64Array(words.buffer.slice(0, count * 8)),
  };
};
