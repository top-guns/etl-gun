import * as rx from 'rxjs';
import * as etl from '../../lib';
import { getError } from '../../utils/getError';

describe('Operator addField()', () => {
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

    test('try to add field to scalars', async () => {
        const src = new etl.BufferEndpoint<number>(1, 2, 3);

        let stream$ = src.read().pipe(
            etl.addField('f1', v => v * 10),
        );

        const error = await getError(async () => etl.run(stream$));
        expect(error).toBeInstanceOf(Error);
    });
});