import * as rx from 'rxjs';
import * as etl from '../../lib/index.js';

describe('BufferEndpoint', () => {
    test('Constructor with parameters and buffer property', async () => {
        const mem = new etl.MemoryEndpoint();
        const buf = mem.getBuffer<number>('bufer1', [1, 2, 3]);
        expect(buf.buffer).toEqual([1, 2, 3]);
    });

    test('Constructor without parameters', async () => {
        const mem = new etl.MemoryEndpoint();
        const buf = mem.getBuffer<number>('bufer1');
        expect(buf.buffer).toEqual([]);
    });
    
    test('sort method with parameter function which returns boolean', async () => {
        const mem = new etl.MemoryEndpoint();
        const buf = mem.getBuffer<number>('bufer1', [1, 3, 2, 4]);
        buf.sort((v1, v2) => v1 > v2);
        expect(buf.buffer).toEqual([1, 2, 3, 4]);
    });

    test('sort method with parameter function which returns number', async () => {
        const mem = new etl.MemoryEndpoint();
        const buf = mem.getBuffer<number>('bufer1', [1, 3, 2, 4]);
        buf.sort((v1, v2) => v1 - v2);
        expect(buf.buffer).toEqual([1, 2, 3, 4]);
    });

    test('push method', async () => {
        const mem = new etl.MemoryEndpoint();
        const buf = mem.getBuffer<number>('bufer1');
        buf.push(1);
        buf.push(2);
        buf.push(3);
        expect(buf.buffer).toEqual([1, 2, 3]);
    });

    test('clear method', async () => {
        const mem = new etl.MemoryEndpoint();
        const buf = mem.getBuffer<number>('bufer1', [1, 2, 3]);
        buf.clear();
        expect(buf.buffer).toEqual([]);
    });

    test('forEach method', async () => {
        const res: number[][] = [];
        const mem = new etl.MemoryEndpoint();
        const buf = mem.getBuffer<number>('bufer1', [1, 2, 3]);
        buf.forEach((v, i) => res.push([v, i]));
        expect(res).toEqual([[1, 0], [2, 1], [3, 2]]);
    });

    test('read method', async () => {
        const res: number[] = [];

        const mem = new etl.MemoryEndpoint();
        const src = mem.getBuffer<number>('bufer1', [1, 2, 3]);
        let stream$ = src.list().pipe(
            rx.tap(v => res.push(v))
        );
        await etl.run(stream$);

        expect(res).toEqual([1, 2, 3]);
    });
});