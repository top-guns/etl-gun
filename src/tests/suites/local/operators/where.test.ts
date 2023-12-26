import { describe, test } from 'node:test';
import { should, expect, assert } from "chai";
import { tap } from 'rxjs';
import * as etl from '../../../../lib/index.js';
import { Memory } from '../../../../lib/endpoints/index.js'

describe('Operator where()', () => {
    test('function-style criteria', async () => {
        let res;

        const mem = Memory.getEndpoint();
        const src = mem.getBuffer<number[]>('bufer1', [[0,1], [2,3], [4,5]]);
        let stream$ = src.selectRx().pipe(
            etl.operators.where(v => v[1] == 3),
            tap(v => res = v)
        );
        await etl.operators.run(stream$);

        assert.deepStrictEqual(res, [2,3]);
    });

    test('object-style criteria', async () => {
        let res;

        const mem = Memory.getEndpoint();
        const src = mem.getBuffer<{}>('bufer1', [{f1: 1, f2: 2}, {f1: 3, f2: 4}, {f1: 5, f2: 6}]);
        let stream$ = src.selectRx().pipe(
            etl.operators.where({f1: 3}),
            tap(v => res = v)
        );
        await etl.operators.run(stream$);

        assert.deepStrictEqual(res, {f1: 3, f2: 4});
    });
});