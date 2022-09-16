import { interval, map, take } from "rxjs";
import * as etl from './lib';


console.log("START");

async function f() {
    try {


        const timer$ = interval(1000);
        const buf = new etl.BufferEndpoint<number>();
        const bufArrays = new etl.BufferEndpoint<any[]>([0,1], [2,3], [3,6]);
        const table = new etl.PostgresEndpoint("users", "postgres://iiicrm:iiicrm@127.0.0.1:5432/iiicrm");

        let tt$ = timer$.pipe(
            etl.numerate("index", "value", 10),
            //map(v => (v.)), 
            
            //etl.log(),
            etl.join(table.find().pipe(take(2))),
            //etl.join(bufArrays.find()),
            etl.log()
        )

        await etl.run(tt$);// .toPromise();
        console.log("END");


    }
    catch (err) {
        console.log(err);
    }
}
f();