import * as rx from 'rxjs';
import * as etl from '../../lib';

describe('Operator numerate()', () => {
    test('numerate arrays', async () => {
        let res: any[][] = [];

        const src = new etl.BufferEndpoint<number[]>([1], [2], [3]);

        let stream$ = src.read().pipe(
            etl.numerate(10),
            rx.tap(v => res.push(v))
        );

        await etl.run(stream$);

        expect(res).toEqual([[1, 10], [2, 11], [3, 12]]);
    });

    test('numerate objects', async () => {
        let res: {}[] = [];

        const src = new etl.BufferEndpoint<{}>({f1: 1}, {f1: 2}, {f1: 3});

        let stream$ = src.read().pipe(
            etl.numerate(10, "index"),
            rx.tap(v => res.push(v))
        );

        await etl.run(stream$);

        expect(res).toEqual([{f1: 1, index: 10}, {f1: 2, index: 11}, {f1: 3, index: 12}]);
    });
});