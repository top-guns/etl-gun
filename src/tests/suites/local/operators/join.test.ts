import { describe, test } from 'node:test';
import { should, expect, assert } from "chai";
import * as rx from 'rxjs';
import * as etl from '../../../../lib/index.js';
import { Memory } from '../../../../lib/endpoints/index.js'

describe('Operator join()', () => {
    test('join arrays', async () => {
        let res: number[][] = [];

        const mem = Memory.getEndpoint();
        const src1 = mem.getBuffer<number[]>('bufer1',[[1], [2]]);
        const src2 = mem.getBuffer<number[]>('bufer2',[[10], [11]]);

        let stream$ = src1.selectRx().pipe(
            etl.join(src2.selectRx()),
            rx.tap(v => res.push(v))
        );

        await etl.run(stream$);

        assert.deepStrictEqual(res, [[1, 10], [1, 11], [2, 10], [2, 11]]);
    });

    test('join objects', async () => {
        let res: {}[] = [];

        const mem = Memory.getEndpoint();
        const src1 = mem.getBuffer<{f1: number}>('bufer1', [{f1: 1}, {f1: 2}]);
        const src2 = mem.getBuffer<{f2: number}>('bufer2', [{f2: 10}, {f2: 11}]);

        let stream$ = src1.selectRx().pipe(
            etl.join(src2.selectRx()),
            rx.tap(v => res.push(v))
        );

        await etl.run(stream$);

        assert.deepStrictEqual(res, [{f1: 1, f2: 10}, {f1: 1, f2: 11}, {f1: 2, f2: 10}, {f1: 2, f2: 11}]);
    });

    test('join scalars', async () => {
        let res: number[][] = [];

        const mem = Memory.getEndpoint();
        const src1 = mem.getBuffer<number>('bufer1', [1, 2]);
        const src2 = mem.getBuffer<number>('bufer2', [10, 11]);

        let stream$ = src1.selectRx().pipe(
            etl.join(src2.selectRx()),
            rx.tap(v => res.push(v))
        );

        await etl.run(stream$);

        assert.deepStrictEqual(res, [[1, 10], [1, 11], [2, 10], [2, 11]]);
    });


    test('join array and object', async () => {
        let res: number[][] = [];

        const mem = Memory.getEndpoint();
        const src1 = mem.getBuffer<number[]>('bufer1', [[1], [2]]);
        const src2 = mem.getBuffer<{f1: number}>('bufer2', [{f1: 1}, {f1: 2}]);

        let stream$ = src1.selectRx().pipe(
            etl.join(src2.selectRx()),
            rx.tap(v => res.push(v))
        );

        await etl.run(stream$);

        assert.deepStrictEqual(res, [[1, 1], [1, 2], [2, 1], [2, 2]]);
    });

    test('join object and array', async () => {
        let res: number[][] = [];

        const mem = Memory.getEndpoint();
        const src1 = mem.getBuffer<{f1: number}>('bufer1', [{f1: 1}, {f1: 2}]);
        const src2 = mem.getBuffer<number[]>('bufer2', [[1], [2]]);

        let stream$ = src1.selectRx().pipe(
            etl.join(src2.selectRx()),
            rx.tap(v => res.push(v))
        );

        await etl.run(stream$);

        assert.deepStrictEqual(res, [[1, 1], [1, 2], [2, 1], [2, 2]]);
    });


    test('join array and scalar', async () => {
        let res: number[][] = [];

        const mem = Memory.getEndpoint();
        const src1 = mem.getBuffer<number[]>('bufer1', [[1], [2]]);
        const src2 = mem.getBuffer<number>('bufer2', [10, 20]);

        let stream$ = src1.selectRx().pipe(
            etl.join(src2.selectRx()),
            rx.tap(v => res.push(v))
        );

        await etl.run(stream$);

        assert.deepStrictEqual(res, [[1, 10], [1, 20], [2, 10], [2, 20]]);
    });

    test('join scalar and array', async () => {
        let res: number[][] = [];

        const mem = Memory.getEndpoint();
        const src1 = mem.getBuffer<number>('bufer1', [10, 20]);
        const src2 = mem.getBuffer<number[]>('bufer2', [[1], [2]]);

        let stream$ = src1.selectRx().pipe(
            etl.join(src2.selectRx()),
            rx.tap(v => res.push(v))
        );

        await etl.run(stream$);

        assert.deepStrictEqual(res, [[10, 1], [10, 2], [20, 1], [20, 2]]);
    });


    test('join object and scalar with field name specified', async () => {
        let res: { f1: number; f2: number; }[] = [];

        const mem = Memory.getEndpoint();
        const src1 = mem.getBuffer<{f1: number}>('bufer1', [{f1: 1}, {f1: 2}]);
        const src2 = mem.getBuffer<number>('bufer2', [10, 20]);

        let stream$ = src1.selectRx().pipe(
            etl.join(src2.selectRx(), 'f2'),
            rx.tap(v => res.push(v))
        );

        await etl.run(stream$);

        assert.deepStrictEqual(res, [{f1: 1, f2: 10}, {f1: 1, f2: 20}, {f1: 2, f2: 10}, {f1: 2, f2: 20}]);
    });

    test('join object and scalar without field name parameter', async () => {
        let res: number[][] = [];

        const mem = Memory.getEndpoint();
        const src1 = mem.getBuffer<{f1: number}>('bufer1', [{f1: 1}, {f1: 2}]);
        const src2 = mem.getBuffer<number>('bufer2', [10, 20]);

        let stream$ = src1.selectRx().pipe(
            etl.join(src2.selectRx()),
            rx.tap(v => res.push(v))
        );

        await etl.run(stream$);

        assert.deepStrictEqual(res, [[1, 10], [1, 20], [2, 10], [2, 20]]);
    });


    test('join scalar and object with field name specified', async () => {
        let res: { f1: number; f2: number; }[] = [];

        const mem = Memory.getEndpoint();
        const src1 = mem.getBuffer<number>('bufer1', [1, 2]);
        const src2 = mem.getBuffer<{f2: number}>('bufer2', [{f2: 10}, {f2: 20}]);

        let stream$ = src1.selectRx().pipe(
            etl.join(src2.selectRx(), 'f1'),
            rx.tap(v => res.push(v))
        );

        await etl.run(stream$);

        assert.deepStrictEqual(res, [{f1: 1, f2: 10}, {f1: 1, f2: 20}, {f1: 2, f2: 10}, {f1: 2, f2: 20}]);
    });

    test('join scalar and object without field name parameter', async () => {
        let res: number[][] = [];
        const mem = Memory.getEndpoint();
        const src1 = mem.getBuffer<number>('bufer1', [1, 2]);
        const src2 = mem.getBuffer<{f1: number}>('bufer2', [{f1: 10}, {f1: 20}]);

        let stream$ = src1.selectRx().pipe(
            etl.join(src2.selectRx()),
            rx.tap(v => res.push(v))
        );

        await etl.run(stream$);

        assert.deepStrictEqual(res, [[1, 10], [1, 20], [2, 10], [2, 20]]);
    });


    test('joinArrays operator', async () => {
        let res: number[][] = [];

        const mem = Memory.getEndpoint();
        const src1 = mem.getBuffer<number[]>('bufer1', [[1], [2]]);
        const src2 = mem.getBuffer<number[]>('bufer2', [[10], [11]]);

        let stream$ = src1.selectRx().pipe(
            etl.joinArrays(src2.selectRx()),
            rx.tap(v => res.push(v))
        );

        await etl.run(stream$);

        assert.deepStrictEqual(res, [[1, 10], [1, 11], [2, 10], [2, 11]]);
    });

    test('joinObjects operator', async () => {
        let res: {}[] = [];

        const mem = Memory.getEndpoint();
        const src1 = mem.getBuffer<{f1: number}>('bufer1', [{f1: 1}, {f1: 2}]);
        const src2 = mem.getBuffer<{f2: number}>('bufer2', [{f2: 10}, {f2: 11}]);

        let stream$ = src1.selectRx().pipe(
            etl.joinObjects(src2.selectRx()),
            rx.tap(v => res.push(v))
        );

        await etl.run(stream$);

        assert.deepStrictEqual(res, [{f1: 1, f2: 10}, {f1: 1, f2: 11}, {f1: 2, f2: 10}, {f1: 2, f2: 11}]);
    });
});