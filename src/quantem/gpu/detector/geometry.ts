/** Row/column detector-mask rasterization shared by browser compute clients. */

export function diskMask(
  rows: number,
  columns: number,
  centerRow: number,
  centerColumn: number,
  radius: number,
): Uint32Array {
  const mask = new Uint32Array(rows * columns);
  const radiusSquared = radius * radius;
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const rowOffset = row - centerRow;
      const columnOffset = column - centerColumn;
      if (rowOffset * rowOffset + columnOffset * columnOffset <= radiusSquared) {
        mask[row * columns + column] = 1;
      }
    }
  }
  return mask;
}

export function annulusMask(
  rows: number,
  columns: number,
  centerRow: number,
  centerColumn: number,
  innerRadius: number,
  outerRadius: number,
): Uint32Array {
  const mask = new Uint32Array(rows * columns);
  const innerSquared = innerRadius * innerRadius;
  const outerSquared = outerRadius * outerRadius;
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const rowOffset = row - centerRow;
      const columnOffset = column - centerColumn;
      const distanceSquared = rowOffset * rowOffset + columnOffset * columnOffset;
      if (distanceSquared >= innerSquared && distanceSquared <= outerSquared) {
        mask[row * columns + column] = 1;
      }
    }
  }
  return mask;
}
