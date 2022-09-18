import { interval, map, take, tap } from "rxjs";
import * as etl from './lib';


console.log("START");

async function f() {
    try {


        const timer$ = interval(1000);
        const buf = new etl.BufferEndpoint<number>();
        const bufArrays = new etl.BufferEndpoint<any[]>([0,1], [2,3], [3,6]);
        const table = new etl.PostgresEndpoint("users", "postgres://iiicrm:iiicrm@127.0.0.1:5432/iiicrm");
        const json = new etl.JsonEndpoint('data/test.json');
        const xml = new etl.XmlEndpoint('data/test.xml');

        // '$.store.book[*].author'
        // json.readByJsonPath('$.store.book[*].author')
        // xml.read('/store/book/author')
        let test$ = json.readByJsonPath('store.book[*]').pipe(
            // etl.numerate("index", "value", 10),
            //map(v => (v.)), 
            
            //etl.log(),
            //etl.join(table.read().pipe(take(2))),
            //etl.join(bufArrays.read()),

            //map(v => v.firstChild.nodeValue),
            etl.log(),
        )

        await etl.run(test$);// .toPromise();
        console.log("END");


    }
    catch (err) {
        console.log("ERROR: ", err);
    }
}
f();