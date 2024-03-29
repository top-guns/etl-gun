import { describe, test,  } from 'node:test';
import { should, expect, assert } from "chai";
import * as rx from 'rxjs';
import * as etl from '../../../../lib/index.js';
import { StringWritable } from '../../../../utils/stringWritable.js';

describe('Operator log()', () => {
    test('log to StringWritable', async () => {
        const res = new StringWritable();

        const src$ = rx.of(100).pipe(
            etl.operators.log('h', null, res)
        )

        await etl.operators.run(src$);

        const strVal = res.toString().replace(/\u001b[^m]*?m/g,"");

        expect(strVal).to.equal('h 100\n');
    });
});