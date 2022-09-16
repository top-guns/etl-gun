import { interval, map, take } from "rxjs";
import { BufferEndpoint, PostgresEndpoint, log, push } from './lib';


console.log("START");

async function f() {
    try {


        const timer$ = interval(1000);
        const buf = new BufferEndpoint<number>();
        const table = new PostgresEndpoint("users", "postgres://iiicrm:iiicrm@127.0.0.1:5432/iiicrm");

        let Src2Buf$ = table.find().pipe(
            take(5),
            //map(v => ({v})),
            //map(v => ([v])),
            //take(3),
            //numerate(5),
            log(),
            //map(v => parseInt(v[0]) * 2),
            map(v => v.id),
            //map(v => v[0]),
            push(buf)
        );

        //await run(Src2Buf$);
        await Src2Buf$.toPromise();
        console.log("Src2Buf$ end");


    }
    catch (err) {
        console.log(err);
    }
}
f();