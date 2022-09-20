import { interval, map, take, tap } from "rxjs";
import * as etl from './lib';


console.log("START");

async function f() {
    try {


        const timer$ = interval(1000);
        const buf = new etl.BufferEndpoint<number>();
        const bufArrays = new etl.BufferEndpoint<any[]>([0,1], [2,3], [3,6]);
        const table = new etl.PostgresEndpoint("users", "postgres://iiicrm:iiicrm@127.0.0.1:5432/iiicrm");
        const csv = new etl.CsvEndpoint('data/test.csv');
        const json = new etl.JsonEndpoint('data/test.json');
        const xml = new etl.XmlEndpoint('data/test.xml');
        //const telegram = new etl.TelegramEndpoint("");
        const fs = new etl.FilesystemEndpoint('src');

        // '$.store.book[*].author'
        // json.readByJsonPath('$.store.book[*].author')
        // xml.read('/store/book/author')
        //let test$ = xml.read('store', {searchReturns: 'foundedWithDescendants', addRelativePathAsAttribute: "path"})
        //let test$ = fs.read('**/*.ts', {objectsToSearch: 'all', includeRootDir: true})
        let test$ = csv.read()
        .pipe(
            // etl.numerate("index", "value", 10),
            //map(v => (v.)), 
            etl.addField(v => v[0] + "++++"),
            
            etl.log(),
            //etl.push(telegram)
            //etl.join(table.read().pipe(take(2))),
            //etl.join(bufArrays.read()),
            //tap(() => (0))

            //map(v => v.firstChild.nodeValue),
            //xml.logNode(),
        )

        // fs
        // .on("read.start", () => console.log("start event"))
        // .on("read.end", () => console.log("end event"))
        // .on("read.data", (data) => console.log("data event", data))
        // .on("read.up", () => console.log("up event"))
        // .on("read.down", () => console.log("down event"))
        // .on("read.error", (err) => console.log("error event: " + err))

        await etl.run(test$);// .toPromise();
        console.log("END");


    }
    catch (err) {
        console.log("ERROR: ", err);
    }
}
f();