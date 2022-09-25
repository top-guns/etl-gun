import * as rx from 'rxjs';
import * as etl from '../../lib';

describe('Operator join()', () => {
    test('join arrays', async () => {
        let res: number[][] = [];

        const src1 = new etl.BufferEndpoint<number[]>([1], [2]);
        const src2 = new etl.BufferEndpoint<number[]>([10], [11]);

        let stream$ = src1.read().pipe(
            etl.join(src2.read()),
            rx.tap(v => res.push(v))
        );

        await etl.run(stream$);

        expect(res).toEqual([[1, 10], [1, 11], [2, 10], [2, 11]]);
    });

    test('join objects', async () => {
        let res: {}[] = [];

        const src1 = new etl.BufferEndpoint<{f1: number}>({f1: 1}, {f1: 2});
        const src2 = new etl.BufferEndpoint<{f2: number}>({f2: 10}, {f2: 11});

        let stream$ = src1.read().pipe(
            etl.join(src2.read()),
            rx.tap(v => res.push(v))
        );

        await etl.run(stream$);

        expect(res).toEqual([{f1: 1, f2: 10}, {f1: 1, f2: 11}, {f1: 2, f2: 10}, {f1: 2, f2: 11}]);
    });

    test('join scalars', async () => {
        let res: number[] = [];

        const src1 = new etl.BufferEndpoint<number>(1, 2);
        const src2 = new etl.BufferEndpoint<number>(10, 11);

        let stream$ = src1.read().pipe(
            etl.join(src2.read()),
            rx.tap(v => res.push(v))
        );

        await etl.run(stream$);

        expect(res).toEqual([[1, 10], [1, 11], [2, 10], [2, 11]]);
    });
});