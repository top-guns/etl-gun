import { Writable, WritableOptions } from 'node:stream';
import { StringDecoder } from 'node:string_decoder';

export class StringWritable extends Writable {
    private data: string = '';

    constructor(options?: WritableOptions) {
        super(options);
    }

    _write(chunk, encoding, callback) {
        this.data += chunk;
        callback();
    }

    toString(): string {
        return this.data;
    }
}