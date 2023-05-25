import * as rx from 'rxjs';
import * as etl from '../../lib/index.js';
import { Memory } from '../../lib/endpoints/index.js'

describe('Operator push()', () => {
    test('push to buffer endpoint', async () => {
        let res: number[] = [];

        const mem = Memory.getEndpoint();
        const buf = mem.getBuffer<number>('bufer1');

        const src$ = rx.of(1, 2, 3).pipe(
            etl.push(buf)
        )

        console.log(buf.buffer)

        await rx.lastValueFrom(src$);

        let stream$ = buf.select().pipe(
            rx.tap(v => res.push(v))
        );

        await etl.run(stream$);

        expect(res).toEqual([1, 2, 3]);
    });
});