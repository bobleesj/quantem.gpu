export type Codebooks = Array<{
    model: number;
    entries: Array<{
        symbol: number;
        bits: number;
        canonical_msb_code: number;
    }>;
}>;
export function encodingBook(books: Codebooks): Uint32Array {
    const data = new Uint32Array(10240);
    for (const [row, model] of [78, 79].entries()) {
        const rows = books.filter(b => b.model === model);
        if (rows.length !== 1)
            throw Error(`Expected one model ${model}.`);
        const symbols = new Map<number, {
            symbol: number;
            bits: number;
            canonical_msb_code: number;
        }>();
        const reverse = (v: number, n: number) => { let x = 0; for (let i = 0; i < n; i++)
            x = (x << 1) | ((v >>> i) & 1); return x; };
        for (const e of rows[0].entries) {
            if (!Number.isInteger(e.symbol) || e.symbol < 0 || e.symbol > 4095 || symbols.has(e.symbol) || !Number.isInteger(e.bits) || e.bits < 1 || e.bits > 10 || !Number.isInteger(e.canonical_msb_code) || e.canonical_msb_code < 0 || e.canonical_msb_code >= 2 ** e.bits)
                throw Error('Invalid or duplicate canonical Huffman entry.');
            symbols.set(e.symbol, e);
            const reversed = reverse(e.canonical_msb_code, e.bits);
            for (let suffix = 0; suffix < 2 ** (10 - e.bits); suffix++) {
                const i = 8192 + row * 1024 + reversed + (suffix << e.bits);
                if (data[i])
                    throw Error('Overlapping Huffman code.');
                data[i] = e.symbol | (e.bits << 12);
            }
        }
        const escape = symbols.get(4095);
        if (!escape)
            throw Error('Missing native pair escape.');
        for (let pair = 0; pair < 4096; pair++) {
            const direct = symbols.get(pair), e = direct ?? escape;
            data[row * 4096 + pair] = reverse(e.canonical_msb_code, e.bits) | (e.bits << 10) | (!direct || pair === 4095 ? 16384 : 0);
        }
        if (data.subarray(8192 + row * 1024, 8192 + (row + 1) * 1024).some(v => !v))
            throw Error('Incomplete Huffman code.');
    }
    return data;
}

/** Derive canonical books solely from the authenticated 81x1024 ANS table.
 * Heap ties use lexicographic symbol-list order, matching the independent Python
 * reference; canonical codes then sort by (length, native pair symbol). */
export function source112HuffmanBooks(decoding: Uint32Array): Codebooks {
  if (decoding.length !== 81 * 1024) throw new Error('Expected the complete authenticated Source112 decoding table.');
  return [78, 79].map(model => {
    const frequencies = new Map<number, number>();
    for (const word of decoding.subarray(model * 1024, (model + 1) * 1024)) {
      const symbol = word & 4095; frequencies.set(symbol, (frequencies.get(symbol) ?? 0) + 1);
    }
    const lengths = new Map([...frequencies.keys()].map(symbol => [symbol, 0]));
    const queue = [...frequencies].map(([symbol, weight]) => ({ weight, symbols: [symbol] }));
    const compare = (a: typeof queue[number], b: typeof queue[number]) => {
      if (a.weight !== b.weight) return a.weight - b.weight;
      for (let i = 0; i < Math.min(a.symbols.length, b.symbols.length); i++) {
        if (a.symbols[i] !== b.symbols[i]) return a.symbols[i] - b.symbols[i];
      }
      return a.symbols.length - b.symbols.length;
    };
    while (queue.length > 1) {
      queue.sort(compare); const left = queue.shift()!, right = queue.shift()!;
      const symbols = [...left.symbols, ...right.symbols];
      for (const symbol of symbols) lengths.set(symbol, lengths.get(symbol)! + 1);
      queue.push({ weight: left.weight + right.weight, symbols });
    }
    let code = 0, previous = 0;
    const entries = [...lengths].sort((a, b) => a[1] - b[1] || a[0] - b[0]).map(([symbol, bits]) => {
      code *= 2 ** (bits - previous); const entry = { symbol, bits, canonical_msb_code: code };
      code++; previous = bits; return entry;
    });
    return { model, entries };
  });
}
