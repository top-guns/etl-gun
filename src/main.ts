import { interval, map, take, tap, from, mergeMap } from "rxjs";
import * as dotenv from 'dotenv';

import * as etl from './lib';
import { GoogleTranslateHelper, GuiManager, NewProductAttributes } from "./lib";

dotenv.config()

console.log("START");

async function f() {
    try {

        GuiManager.startGui("Test ETL process", true, 20);

        // const timer$ = interval(1000);
        // const buf = new etl.BufferEndpoint<string>();
        // const bufArrays = new etl.BufferEndpoint<any[]>([[0,1], [2,3], [3,6]]);
        // const table = new etl.PostgresEndpoint("users", process.env.POSTGRES_CONNECTION_STRING!, {watch: v => `${v.name} [${v.id}]`});
        // const csv = new etl.CsvEndpoint('data/test.csv');
        // const json = new etl.JsonEndpoint('data/test.json');
        // const xml = new etl.XmlEndpoint('data/test.xml');
        //const telegram = new etl.TelegramEndpoint(process.env.TELEGRAM_BOT_TOKEN!);
        // const fs = new etl.FilesystemEndpoint('src');
        // const timer = new etl.IntervalEndpoint(500);
        const magento = new etl.MagentoProductsEndpoint('https://magento.test', process.env.MAGENTO_LOGIN!, process.env.MAGENTO_PASSWORD!, false);
        const csv = new etl.CsvEndpoint('data/products.csv');
        const header = new etl.Header('id', 'sku', 'name', 'price', 'visibility', 'type_id', 'status');
        const translator = new GoogleTranslateHelper(process.env.GOOGLE_CLOUD_API_KEY!, 'en', 'ru');

        // '$.store.book[*].author'
        // json.readByJsonPath('$.store.book[*].author')
        // xml.read('/store/book/author')
        //let test$ = xml.read('store', {searchReturns: 'foundedWithDescendants', addRelativePathAsAttribute: "path"})
        //let test$ = fs.read('**/*.ts', {objectsToSearch: 'all', includeRootDir: true})
        //let test$ = csv.read()


        let test$ = magento.read({})
        .pipe(
            // etl.numerate("index", "value", 10),
            //map(v => (v.)), 
            //etl.addField(v => v[0] + "++++"),
            
            //etl.log(),
            //map(v => (v.name)), 
            map(p => [
                p.id, 
                p.sku, 
                p.name, 
                p.price, 
                p.visibility, 
                p.type_id, 
                p.status, 
                etl.getChildByPropVal(p.custom_attributes, 'attribute_code', 'options_container')?.value,
                etl.getChildByPropVal(p.custom_attributes, 'attribute_code', 'url_key')?.value,
                etl.getChildByPropVal(p.custom_attributes, 'attribute_code', 'tax_class_id')?.value,
                // etl.getChildByPropVal(p.custom_attributes, 'attribute_code', 'category_ids')?.value,
                // etl.getChildByPropVal(p.custom_attributes, 'attribute_code', 'description')?.value,
                // etl.getChildByPropVal(p.custom_attributes, 'attribute_code', 'collection')?.value,
                // etl.getChildByPropVal(p.custom_attributes, 'attribute_code', 'erp_name')?.value,
                // etl.getChildByPropVal(p.custom_attributes, 'attribute_code', 'supplier')?.value
            ]),
            //mergeMap(p => translator.observable(p)),
            translator.operator(),
            etl.log()
            //etl.push(csv)
            //etl.join(table.read().pipe(take(2))),
            //etl.join(bufArrays.read()),
            //tap(() => (0))

            //map(v => v.firstChild.nodeValue),
            //xml.logNode(),
        );

        // fs
        // .on("read.start", () => console.log("start event"))
        // .on("read.end", () => console.log("end event"))
        // .on("read.data", (data) => console.log("data event", data))
        // .on("read.up", () => console.log("up event"))
        // .on("read.down", () => console.log("down event"))
        // .on("read.error", (err) => console.log("error event: " + err))

        await etl.run(test$);// .toPromise();

        const p: NewProductAttributes = {
            sku: 'test6',
            name: 'test product 6',
            price: 100,
            attribute_set_id: 4
        }
        //const res = await magento.push(p);
        //console.log(res);
        //console.log(res.id);


        if (GuiManager.isGuiStarted()) GuiManager.stopGui();
        console.log("END");
    }
    catch (err) {
        console.log("ERROR: ", err);
        GuiManager.quitApp();
    }
}
f();