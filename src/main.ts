/*********************************************************************************************************************************** 
This file contains temporary code to test library while development process. 
It is not contains any usefull code, is not a part of library and is not an example of library using.
************************************************************************************************************************************/

import * as rx from "rxjs";
import * as dotenv from 'dotenv';
import fetch, { RequestInit } from 'node-fetch';
import * as etl from './lib/index.js';
import { GuiManager, Magento } from "./lib/index.js";
//import { DiscordHelper } from "./lib/index.js";

dotenv.config()

const START = new Date;
//GuiManager.startGui("Test ETL process", true, 20);
console.log("START", START);



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
// const discord = new DiscordHelper();
// const credential = await discord.loginViaBrowser();
// console.log(credential);



//res = await res.json();
//console.log(res)
//fetch('%s/oauth2/token' % API_ENDPOINT, data=data, headers=headers)



// '$.store.book[*].author'
// json.listByJsonPath('$.store.book[*].author')
// xml.list('/store/book/author')
//let test$ = xml.list('store', {searchReturns: 'foundedWithDescendants', addRelativePathAsAttribute: "path"})
//let test$ = fs.list('**/*.ts', {objectsToSearch: 'all', includeRootDir: true})
//let test$ = csv.list()


// const magento = new etl.MagentoEndpoint('https://magento.test', process.env.MAGENTO_LOGIN!, process.env.MAGENTO_PASSWORD!, false);
// const product = magento.getProducts();
// const csv = new etl.CsvEndpoint('data').getFile('products.csv');
// const header = new etl.Header('id', 'sku', 'name', 'price', 'visibility', 'type_id', 'status');
// const translate = new GoogleTranslateHelper(process.env.GOOGLE_CLOUD_API_KEY!, 'en', 'ru').operator;


// let test$ = product.select({})
// .pipe(
//     // etl.numerate("index", "value", 10),
//     //map(v => (v.)), 
//     //etl.addField(v => v[0] + "++++"),
    
//     //etl.log(),
//     //map(v => (v.name)), 
//     map(p => [
//         p.id, 
//         p.sku, 
//         p.name, 
//         p.price, 
//         p.visibility, 
//         p.type_id, 
//         p.status, 
//         etl.getChildByPropVal(p.custom_attributes, 'attribute_code', 'options_container')?.value,
//         etl.getChildByPropVal(p.custom_attributes, 'attribute_code', 'url_key')?.value,
//         etl.getChildByPropVal(p.custom_attributes, 'attribute_code', 'tax_class_id')?.value,
//         // etl.getChildByPropVal(p.custom_attributes, 'attribute_code', 'category_ids')?.value,
//         // etl.getChildByPropVal(p.custom_attributes, 'attribute_code', 'description')?.value,
//         // etl.getChildByPropVal(p.custom_attributes, 'attribute_code', 'collection')?.value,
//         // etl.getChildByPropVal(p.custom_attributes, 'attribute_code', 'erp_name')?.value,
//         // etl.getChildByPropVal(p.custom_attributes, 'attribute_code', 'supplier')?.value
//     ]),
//     //mergeMap(p => translator.observable(p)),
//     translate(),
//     etl.log()
//     //etl.push(csv)
//     //etl.join(table.list().pipe(take(2))),
//     //etl.join(bufArrays.list()),
//     //tap(() => (0))

//     //map(v => v.firstChild.nodeValue),
//     //xml.logNode(),
// );

// fs
// .on("list.start", () => console.log("start event"))
// .on("list.end", () => console.log("end event"))
// .on("list.data", (data) => console.log("data event", data))
// .on("list.up", () => console.log("up event"))
// .on("list.down", () => console.log("down event"))
// .on("list.error", (err) => console.log("error event: " + err))

const magentoEndpoint = new etl.Magento.Endpoint('https://magento.test', process.env.MAGENTO_LOGIN!, process.env.MAGENTO_PASSWORD!, false);
const magentoProducts = magentoEndpoint.getProducts();

const headerMagento = new etl.Header([
    'id', 
    'sku', 
    'name', 
    {'price': 'number', undefinedValue: ''},
    'visibility',
    'type_id',
    'status', 
    'attribute_set_id',
    'options_container',
    'url_key',
    'tax_class_id'
])

const headerPuma = new etl.Header([
    'image',
    'name',
    'vendor_code',
    {'price': 'number'},
    'currency',
    {'availability': 'boolean', trueValue: 'Да', falseValue: 'Нет'},
    'category',
    'url',
    'desctiption',
    'renk',
    'beden'
])

type DbProduct = {
    id?: number, 
    sku: string, 
    name: string, 
    price?: number, 
    visibility?: number, 
    type_id?: string, 
    status?: number, 
    attribute_set_id?: number,
    options_container?: string,
    url_key?: string,
    tax_class_id?: string,
    availability?: number,
    category?: string,
    desctiption?: string
}

const csv = new etl.Csv.Endpoint("./data");
const csvMagento = csv.getFile("magento.csv", headerMagento);
const csvPuma = csv.getFile("puma.csv", headerPuma, ';',);

const mysql = new etl.Mysql.Endpoint('mysql://test:test@localhost:7306/test');
const table = mysql.getTable<DbProduct>('test1');

const translator = new etl.GoogleTranslateHelper(process.env.GOOGLE_CLOUD_API_KEY!, 'en', 'ru');


function magentoProduct2ShortForm(p: Partial<etl.Magento.Product>) {
    return {
        id: p.id, 
        sku: p.sku, 
        name: p.name, 
        price: p.price, 
        visibility: p.visibility, 
        type_id: p.type_id, 
        status: p.status, 
        attribute_set_id: p.attribute_set_id,
        options_container: etl.getChildByPropVal(p.custom_attributes, 'attribute_code', 'options_container')?.value,
        url_key: etl.getChildByPropVal(p.custom_attributes, 'attribute_code', 'url_key')?.value,
        tax_class_id: etl.getChildByPropVal(p.custom_attributes, 'attribute_code', 'tax_class_id')?.value,
    }
}

function pumaProduct2Db(p: any): DbProduct {
    return {
        sku: p.vendor_code, 
        name: p.name, 
        price: p.price, 
        availability: p.availability ? 1 : 0, 
        category: p.category, 
        status: p.status, 
        desctiption: p.desctiption
    }
}

function db2Magento(p: DbProduct): Partial<Magento.Product> {
    return {
        sku: p.sku, 
        name: p.name, 
        price: p.price ?? 0, 
        status: p.status ?? 0,
        attribute_set_id: 4
    }
}

//console.log(translate)
//console.log(await translate.function('hello world', 'en', 'ru'));

let Magento_to_Csv$ = magentoProducts.select().pipe(
    rx.map(p => magentoProduct2ShortForm(p)),
    etl.log(),
    //rx.map(p => headerMagento.objToArr(p)),
    //etl.log(),
    //etl.push(csvMagento)
)

let MagentoCsv_to_MySql$ = csvMagento.select().pipe(
    rx.take(1),
    //etl.log(),
    rx.map(p => headerMagento.arrToObj(p)),
    //etl.where({id: 3}),
    etl.log(),
    //translate.operator(),
    //rx.tap(p => table.upsert(p))
)

let PumaCsv_to_MySql$ = csvPuma.select(true).pipe(
    //etl.log(),
    //translator.operator([1]),
    //rx.take(1),
    rx.map(p => headerPuma.arrToObj(p)),
    rx.map(pumaProduct2Db),
    //translator.operator([], ['name']),
    //etl.push(table),
    etl.expect<DbProduct>('price = 6800', {price: 6800, category: 'Мужчины'}),
    etl.log(),
)

let MySql_to_Magento$ = table.select().pipe(
    //etl.log(),
    rx.take(100),
    rx.map(db2Magento),
//    translator.operator([], ['name']),
//    etl.log(),
    etl.push(magentoProducts)
)

// await csv.delete();
//await etl.run(magento_to_Csv$);

await etl.run(PumaCsv_to_MySql$);

//mysql.releaseEndpoint();
//if (etl.GuiManager.isGuiStarted()) etl.GuiManager.stopGui();
console.log("END");
console.log('start', START);
console.log('end', new Date());
//etl.GuiManager.quitApp();


//mysql.releaseEndpoint();

// const p: Partial<Product> = {
//     sku: 'test6',
//     name: 'test product 6',
//     price: 100,
//     attribute_set_id: 4
// }
//const res = await magento.push(p);
//console.log(res);
//console.log(res.id);




