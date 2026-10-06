// jsfive has no bundled declarations. Its dynamic HDF5 objects are validated
// at the reader boundary; this declares only the constructor used there.
declare module "jsfive" {
  export class File {
    constructor(buffer: ArrayBuffer, filename?: string);
    get(path: string): any;
  }
}
