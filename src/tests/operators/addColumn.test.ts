import * as rx from 'rxjs';
import * as etl from '../../lib/index.js';

describe('Operator addColumn()', () => {
    test('add column to arrays', async () => {
        let res: any[][] = [];

        const mem = new etl.MemoryEndpoint();
        const src = mem.getBuffer<number[]>('bufer1', [[1], [2], [3]]);

        let stream$ = src.list().pipe(
            etl.addColumn(v => v[0] * 10),
            rx.tap(v => res.push(v)),
        );

        await etl.run(stream$);

        expect(res).toEqual([[1, 10], [2, 20], [3, 30]]);
    });

    test('add column to scalars', async () => {
        let res: any[][] = [];

        const mem = new etl.MemoryEndpoint();
        const src = mem.getBuffer<number>('bufer1', [1, 2, 3]);

        let stream$ = src.list().pipe(
            etl.addColumn<number, number[]>(v => v * 10),
            rx.tap(v => res.push(v)),
        );

        await etl.run(stream$);

        expect(res).toEqual([[1, 10], [2, 20], [3, 30]]);
    });
});