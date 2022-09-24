import { tap } from 'rxjs';
import * as etl from '../../lib';

describe('Operator where', () => {
    test('with functional criteria', async () => {
        let res;

        const src = new etl.BufferEndpoint<number[]>([0,1], [2,3], [4,5]);
        let stream$ = src.read().pipe(
            etl.where(v => v[1] == 3),
            tap(v => res = v)
        );
        await etl.run(stream$);

        expect(res).toEqual([2,3]);
    });
});