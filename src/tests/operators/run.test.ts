import * as rx from 'rxjs';
import * as etl from '../../lib';

describe('Operator run()', () => {
    test('wait for two streams and check calls order', async () => {
        let res: number[] = [];

        const src1 = new etl.BufferEndpoint<number>(1, 2);
        const src2 = new etl.BufferEndpoint<number>(10, 11);

        let stream1$ = src1.read().pipe(
            rx.delay(2000),
            rx.tap(v => res.push(v))
        );
        let stream2$ = src2.read().pipe(
            rx.delay(1500),
            rx.tap(v => res.push(v))
        );

        await etl.run(stream1$, stream2$);

        expect(res).toEqual([10, 11, 1, 2]);
    });
});