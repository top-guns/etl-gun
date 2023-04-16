/*********************************************************************************************************************************** 
This file contains temporary code to test library while development process. 
It is not contains any usefull code, is not a part of library and is not an example of library using.
************************************************************************************************************************************/

import { interval, map, take, tap, from, mergeMap } from "rxjs";
import * as dotenv from 'dotenv';
import fetch, { RequestInit } from 'node-fetch';

import * as etl from './lib/index.js';
import { CsvEndpoint, DiscordHelper, GoogleTranslateHelper, GuiManager, Header, PostgresEndpoint, Product } from "./lib/index.js";

dotenv.config()

console.log("START");

async function f() {
    try {
        //GuiManager.startGui("Test ETL process", true, 20);

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

        const magento = new etl.MagentoEndpoint('https://magento.test', process.env.MAGENTO_LOGIN!, process.env.MAGENTO_PASSWORD!, false);
        const product = magento.getProducts();
        const csv = new etl.CsvEndpoint('data').getFile('products.csv');
        const header = new etl.Header('id', 'sku', 'name', 'price', 'visibility', 'type_id', 'status');
        const translate = new GoogleTranslateHelper(process.env.GOOGLE_CLOUD_API_KEY!, 'en', 'ru').operator;


        // const trello = new etl.TrelloEndpoint(process.env.TRELLO_API_KEY!, process.env.TRELLO_AUTH_TOKEN!);
        // const boards = trello.getUserBoards();
        // console.log(1);
        
        // const board = await boards.getByBrowserUrl(process.env.TRELLO_BOARD_URL!);
        // console.log(2);
        // const lists = trello.getBoardLists(board.id);
        // console.log(3);
        // const list = (await lists.get())[0];
        // console.log(4);
        // const cards = trello.getListCards(list.id);
        // const card = (await cards.get())[0];

        // const comments = trello.getCardComments(card.id);



        // //const cards = trello.getBoardCards(board.id);
        // console.log(5);

        // comments.push('test comment 1')

        // let trello$ = comments.list({}, ['id', 'data'])
        // .pipe(
        //     etl.log()
        // );

        // await etl.run(trello$);
        //console.log(obj);



        

        //const res = await getCredentialsByCode('3HFAXfsharzRRdenqHnjsDPKtaa07d');
        const discord = new DiscordHelper();
        const credential = await discord.loginViaBrowser();
        console.log(credential);



        //res = await res.json();
        //console.log(res)
        //fetch('%s/oauth2/token' % API_ENDPOINT, data=data, headers=headers)

        GuiManager.quitApp();

        // '$.store.book[*].author'
        // json.listByJsonPath('$.store.book[*].author')
        // xml.list('/store/book/author')
        //let test$ = xml.list('store', {searchReturns: 'foundedWithDescendants', addRelativePathAsAttribute: "path"})
        //let test$ = fs.list('**/*.ts', {objectsToSearch: 'all', includeRootDir: true})
        //let test$ = csv.list()


        let test$ = product.list({})
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
            translate(),
            etl.log()
            //etl.push(csv)
            //etl.join(table.list().pipe(take(2))),
            //etl.join(bufArrays.list()),
            //tap(() => (0))

            //map(v => v.firstChild.nodeValue),
            //xml.logNode(),
        );

        // fs
        // .on("list.start", () => console.log("start event"))
        // .on("list.end", () => console.log("end event"))
        // .on("list.data", (data) => console.log("data event", data))
        // .on("list.up", () => console.log("up event"))
        // .on("list.down", () => console.log("down event"))
        // .on("list.error", (err) => console.log("error event: " + err))

        await etl.run(test$);// .toPromise();

        const p: Partial<Product> = {
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