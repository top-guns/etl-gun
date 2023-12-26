import { describe, test } from 'node:test';
import { should, expect, assert } from "chai";
import * as rx from 'rxjs';
import * as etl from '../../../../lib/index.js';
import { getError } from '../../../../utils/getError.js';
import { Memory } from '../../../../lib/endpoints/index.js'

describe('Operator addField()', () => {
    test('add field to objects', async () => {
        let res: {}[] = [];

        const mem = Memory.getEndpoint();
        const src = mem.getBuffer<{f1: number}>('bufer1', [{f1: 1}, {f1: 2}, {f1: 3}]);

        let stream$ = src.selectRx().pipe(
            etl.operators.addField("f2", v => v.f1 * 10),
            rx.tap(v => res.push(v))
        );

        await etl.operators.run(stream$);

        assert.deepStrictEqual(res, [{f1: 1, f2: 10}, {f1: 2, f2: 20}, {f1: 3, f2: 30}]);
    });

    test('try to add field to scalars', async () => {
        const mem = Memory.getEndpoint();
        const src = mem.getBuffer<number>('bufer1', [1, 2, 3]);

        let stream$ = src.selectRx().pipe(
            etl.operators.addField('f1', v => v * 10),
        );

        const error = await getError(async () => etl.operators.run(stream$));
        assert.strictEqual(error.constructor, Error);
    });
});