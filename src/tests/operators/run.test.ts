import * as rx from 'rxjs';
import * as etl from '../../lib/index.js';
import { Memory } from '../../lib/endpoints/index.js'

describe('Operator run()', () => {
    test('wait for two streams and check calls order', async () => {
        let res: number[] = [];

        const mem = Memory.getEndpoint();
        const src1 = mem.getBuffer<number>('bufer1', [1, 2]);
        const src2 = mem.getBuffer<number>('bufer2', [10, 11]);

        let stream1$ = src1.select().pipe(
            rx.delay(2000),
            rx.tap(v => res.push(v))
        );
        let stream2$ = src2.select().pipe(
            rx.delay(1500),
            rx.tap(v => res.push(v))
        );

        await etl.run(stream1$, stream2$);

        expect(res).toEqual([10, 11, 1, 2]);
    });
});