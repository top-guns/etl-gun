import { tap } from 'rxjs';
import * as etl from '../../lib/index.js';
import { MemoryEndpoint } from '../../lib/endpoints/memory.js'

describe('Operator where()', () => {
    test('function-style criteria', async () => {
        let res;

        const mem = new MemoryEndpoint();
        const src = mem.getBuffer<number[]>('bufer1', [[0,1], [2,3], [4,5]]);
        let stream$ = src.select().pipe(
            etl.where(v => v[1] == 3),
            tap(v => res = v)
        );
        await etl.run(stream$);

        expect(res).toEqual([2,3]);
    });

    test('object-style criteria', async () => {
        let res;

        const mem = new MemoryEndpoint();
        const src = mem.getBuffer<{}>('bufer1', [{f1: 1, f2: 2}, {f1: 3, f2: 4}, {f1: 5, f2: 6}]);
        let stream$ = src.select().pipe(
            etl.where({f1: 3}),
            tap(v => res = v)
        );
        await etl.run(stream$);

        expect(res).toEqual({f1: 3, f2: 4});
    });
});