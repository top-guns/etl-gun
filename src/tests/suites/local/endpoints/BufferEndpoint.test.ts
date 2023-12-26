import { describe, test } from 'node:test';
import { should, expect, assert } from "chai";
import * as rx from 'rxjs';
import * as etl from '../../../../lib/index.js';
import { Memory } from '../../../../lib/endpoints/index.js'

describe('BufferEndpoint', () => {
    test('Constructor with parameters and buffer property', async () => {
        const mem = Memory.getEndpoint();
        const buf = mem.getBuffer<number>('bufer1', [1, 2, 3]);
        assert.deepStrictEqual(buf.buffer, [1, 2, 3]);
    });

    test('Constructor without parameters', async () => {
        const mem = Memory.getEndpoint();
        const buf = mem.getBuffer<number>('bufer1');
        assert.deepStrictEqual(buf.buffer, []);
    });
    
    test('sort method with parameter function which returns boolean', async () => {
        const mem = Memory.getEndpoint();
        const buf = mem.getBuffer<number>('bufer1', [1, 3, 2, 4]);
        buf.sort((v1, v2) => v1 > v2);
        assert.deepStrictEqual(buf.buffer, [1, 2, 3, 4]);
    });

    test('sort method with parameter function which returns number', async () => {
        const mem = Memory.getEndpoint();
        const buf = mem.getBuffer<number>('bufer1', [1, 3, 2, 4]);
        buf.sort((v1, v2) => v1 - v2);
        assert.deepStrictEqual(buf.buffer, [1, 2, 3, 4]);
    });

    test('push method', async () => {
        const mem = Memory.getEndpoint();
        const buf = mem.getBuffer<number>('bufer1');
        await buf.insert(1);
        await buf.insert(2);
        await buf.insert(3);
        assert.deepStrictEqual(buf.buffer, [1, 2, 3]);
    });

    test('clear method', async () => {
        const mem = Memory.getEndpoint();
        const buf = mem.getBuffer<number>('bufer1', [1, 2, 3]);
        await buf.delete();
        assert.deepStrictEqual(buf.buffer, []);
    });

    test('forEach method', async () => {
        const res: number[][] = [];
        const mem = Memory.getEndpoint();
        const buf = mem.getBuffer<number>('bufer1', [1, 2, 3]);
        buf.forEach((v, i) => res.push([v, i]));
        assert.deepStrictEqual(res, [[1, 0], [2, 1], [3, 2]]);
    });

    test('read method', async () => {
        const res: number[] = [];

        const mem = Memory.getEndpoint();
        const src = mem.getBuffer<number>('bufer1', [1, 2, 3]);

        let stream$ = src.selectRx().pipe(
            rx.tap(v => res.push(v))
        );
        await etl.operators.run(stream$);

        assert.deepStrictEqual(res, [1, 2, 3]);
    });

    test('select() method', async () => {
        const mem = Memory.getEndpoint();
        const src = mem.getBuffer<number>('bufer1', [1, 2, 3]);
        const res = await src.select();

        assert.deepStrictEqual(res, [1, 2, 3]);
    });

    test('selectGen() method', async () => {
        const res: number[] = [];

        const mem = Memory.getEndpoint();
        const src = mem.getBuffer<number>('bufer1', [1, 2, 3]);

        for await (const v of src.selectGen()) res.push(v);

        assert.deepStrictEqual(res, [1, 2, 3]);
    });

    test('selectIx() method', async () => {
        const res: number[] = [];

        const mem = Memory.getEndpoint();
        const src = mem.getBuffer<number>('bufer1', [1, 2, 3]);
        await src.selectIx().forEach(v => {
            res.push(v);
        });

        assert.deepStrictEqual(res, [1, 2, 3]);
    });

    test('selectRx() method', async () => {
        const res: number[] = [];

        const mem = Memory.getEndpoint();
        const src = mem.getBuffer<number>('bufer1', [1, 2, 3]);

        const stream$ = src.selectRx().pipe(
            rx.tap(v => res.push(v))
        );
        await etl.operators.run(stream$);

        assert.deepStrictEqual(res, [1, 2, 3]);
    });

    test('selectStream() method', async () => {
        const res: { done: false; value: number; }[] = [];

        const mem = Memory.getEndpoint();
        const src = mem.getBuffer<number>('bufer1', [1, 2, 3]);

        const reader = src.selectStream().getReader();
        let v;
        while(!(v = await reader.read()).done) res.push(v);

        assert.deepStrictEqual(res, [{done: false, value: 1}, {done: false, value: 2}, {done: false, value: 3}]);
    });
});