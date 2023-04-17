import * as rx from 'rxjs';
import * as etl from '../../lib/index.js';
import * as fs from 'fs';
import { StringWritable } from '../../utils/stringWritable.js';

describe('Operator log()', () => {
    test('log to StringWritable', async () => {
        const res = new StringWritable();

        const src$ = rx.of(100).pipe(
            etl.log('h', null, res)
        )

        await etl.run(src$);

        expect(res.toString()).toEqual("h 100\n");
    });
});