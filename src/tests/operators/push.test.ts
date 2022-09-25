import * as rx from 'rxjs';
import * as etl from '../../lib';

describe('Operator push()', () => {
    test('push to buffer endpoint', async () => {
        let res: number[] = [];

        const buf = new etl.BufferEndpoint<number>();

        const src$ = rx.of(1, 2, 3).pipe(
            etl.push(buf)
        )

        await rx.lastValueFrom(src$);

        let stream$ = buf.read().pipe(
            rx.tap(v => res.push(v))
        );

        await etl.run(stream$);

        expect(res).toEqual([1, 2, 3]);
    });
});