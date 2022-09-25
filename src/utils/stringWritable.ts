import { Writable, WritableOptions } from 'node:stream';
import { StringDecoder } from 'node:string_decoder';

export class StringWritable extends Writable {
    private data: string = '';
    private decoder;

    constructor(options?: WritableOptions) {
        super(options);
        this.decoder = new StringDecoder(options ? options.defaultEncoding : undefined);
    }

    _write(chunk, encoding, callback) {
        if (encoding === 'buffer') chunk = this.decoder.write(chunk);
        this.data += chunk;
        callback();
    }

    _final(callback) {
        this.data += this.decoder.end();
        callback();
    }

    toString(): string {
        return this.data;
    }

    clear() {
        this.data = '';
    }
}