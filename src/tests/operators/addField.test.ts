import * as rx from 'rxjs';
import * as etl from '../../lib';

describe('Operator addField()', () => {
    test('add column to arrays', async () => {
        let res: any[][] = [];

        const src = new etl.BufferEndpoint<number[]>([1], [2], [3]);

        let stream$ = src.read().pipe(
            etl.addField(v => v[0] * 10),
            rx.tap(v => res.push(v)),
        );

        await etl.run(stream$);

        expect(res).toEqual([[1, 10], [2, 20], [3, 30]]);
    });

    test('add field to objects', async () => {
        let res: {}[] = [];

        const src = new etl.BufferEndpoint<{f1: number}>({f1: 1}, {f1: 2}, {f1: 3});

        let stream$ = src.read().pipe(
            etl.addField("f2", v => v.f1 * 10),
            rx.tap(v => res.push(v))
        );

        await etl.run(stream$);

        expect(res).toEqual([{f1: 1, f2: 10}, {f1: 2, f2: 20}, {f1: 3, f2: 30}]);
    });

    test('add column to scalars', async () => {
        let res: any[][] = [];

        const src = new etl.BufferEndpoint<number>(1, 2, 3);

        let stream$ = src.read().pipe(
            etl.addField<number, number[]>(v => v * 10),
            rx.tap(v => res.push(v)),
        );

        await etl.run(stream$);

        expect(res).toEqual([[1, 10], [2, 20], [3, 30]]);
    });
});