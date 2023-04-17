import * as rx from 'rxjs';
import * as etl from '../../lib/index.js';
import { getError } from '../../utils/getError.js';
import { Endpoint as MemoryEndpoint } from '../../lib/endpoints/memory.js'

describe('Operator numerate()', () => {
    test('numerate arrays', async () => {
        let res: any[][] = [];

        const mem = new MemoryEndpoint();
        const src = mem.getBuffer<number[]>('bufer1', [[1], [2], [3]]);

        let stream$ = src.select().pipe(
            etl.numerate(10),
            rx.tap(v => res.push(v))
        );

        await etl.run(stream$);

        expect(res).toEqual([[1, 10], [2, 11], [3, 12]]);
    });

    test('numerate objects', async () => {
        let res: {}[] = [];

        const mem = new MemoryEndpoint();
        const src = mem.getBuffer<{}>('bufer1', [{f1: 1}, {f1: 2}, {f1: 3}]);

        let stream$ = src.select().pipe(
            etl.numerate(10, "index"),
            rx.tap(v => res.push(v))
        );

        await etl.run(stream$);

        expect(res).toEqual([{f1: 1, index: 10}, {f1: 2, index: 11}, {f1: 3, index: 12}]);
    });

    test('numerate scalars', async () => {
        let res: any[][] = [];

        const mem = new MemoryEndpoint();
        const src = mem.getBuffer<number>('bufer1', [1, 2, 3]);

        let stream$ = src.select().pipe(
            etl.numerate(10),
            rx.tap(v => res.push(v))
        );

        await etl.run(stream$);

        expect(res).toEqual([[1, 10], [2, 11], [3, 12]]);
    });


    test('numerate objects without field name specification', async () => {
        const mem = new MemoryEndpoint();
        const src = mem.getBuffer<{}>('bufer1', [{f1: 1}, {f1: 2}, {f1: 3}]);

        let stream$ = src.select().pipe(
            etl.numerate(10),
        );

        //expect(() => {etl.run(stream$)}).toThrow();
        const error = await getError(async () => etl.run(stream$));
        expect(error).toBeInstanceOf(Error);
    });

    test('numerate arrays with field name specification', async () => {
        const mem = new MemoryEndpoint();
        const src = mem.getBuffer<number[]>('bufer1', [[1], [2], [3]]);

        let stream$ = src.select().pipe(
            etl.numerate(10, "index"),
        );

        const error = await getError(async () => etl.run(stream$));
        expect(error).toBeInstanceOf(Error);
    });

    test('numerate scalars with field name specification', async () => {
        const mem = new MemoryEndpoint();
        const src = mem.getBuffer<number>('bufer1', [1, 2, 3]);

        let stream$ = src.select().pipe(
            rx.map(v => v as unknown as Record<string, any>),
            etl.numerate(10, "index"),
        );

        const error = await getError(async () => etl.run(stream$));
        expect(error).toBeInstanceOf(Error);
    });
});