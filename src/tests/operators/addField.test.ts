import * as rx from 'rxjs';
import * as etl from '../../lib/index.js';
import { getError } from '../../utils/getError.js';
import { MemoryEndpoint } from '../../lib/endpoints/memory.js'

describe('Operator addField()', () => {
    test('add field to objects', async () => {
        let res: {}[] = [];

        const mem = new MemoryEndpoint();
        const src = mem.getBuffer<{f1: number}>('bufer1', [{f1: 1}, {f1: 2}, {f1: 3}]);

        let stream$ = src.select().pipe(
            etl.addField("f2", v => v.f1 * 10),
            rx.tap(v => res.push(v))
        );

        await etl.run(stream$);

        expect(res).toEqual([{f1: 1, f2: 10}, {f1: 2, f2: 20}, {f1: 3, f2: 30}]);
    });

    test('try to add field to scalars', async () => {
        const mem = new MemoryEndpoint();
        const src = mem.getBuffer<number>('bufer1', [1, 2, 3]);

        let stream$ = src.select().pipe(
            etl.addField('f1', v => v * 10),
        );

        const error = await getError(async () => etl.run(stream$));
        expect(error).toBeInstanceOf(Error);
    });
});