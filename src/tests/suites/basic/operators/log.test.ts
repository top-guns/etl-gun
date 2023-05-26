import { describe, test,  } from 'node:test';
import assert from 'node:assert';
import * as rx from 'rxjs';
import * as etl from '../../../../lib/index.js';
import { StringWritable } from '../../../../utils/stringWritable.js';

describe('Operator log()', () => {
    test('log to StringWritable', async () => {
        const res = new StringWritable();

        const src$ = rx.of(100).pipe(
            etl.log('h', null, res)
        )

        await etl.run(src$);

        assert.strictEqual(res.toString(), "h 100\n");
    });
});