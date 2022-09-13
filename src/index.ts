import { from, interval, map, of, tap, take } from 'rxjs';
import { BufferEndpoint, CsvEndpoint } from './lib/endpoints';
import { run } from './lib/utils/run';
import { log, push, numerate } from './lib/operators';


console.log("START");

async function f() {
    try {


        const source$ = of(1,6,2,9,0,7);
        const timer$ = interval(1000);
        const buf = new BufferEndpoint<number>();
        const scvRes = new CsvEndpoint("data/res.csv");
        const srcBuf = new BufferEndpoint({v: 1}, {v: 2}, {v: 3});

        let Src2Buf$ = timer$.pipe(
            map(v => ({v})),
            //map(v => ([v])),
            //take(3),
            numerate(5),
            log(),
            //map(v => parseInt(v[0]) * 2),
            map(v => v.index),
            //map(v => v[0]),
            push(buf)
        );

        await run(Src2Buf$);
        console.log("Src2Buf$ end");

        // scvRes.clear();

        // buf.sort((v1, v2) => v1 > v2);

        // let buf2log$ = buf.createReadStream()
        // .pipe(
        //     log("next: "),
        //     map(v => ["" + v]),
        //     push(scvRes)
        // );
        // run(buf2log$);


    }
    catch (err) {
        console.log(err);
    }
}
f();