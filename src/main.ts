/*********************************************************************************************************************************** 
This file contains temporary code to test library while development process. 
It is not contains any usefull code, is not a part of library and is not an example of library using.
************************************************************************************************************************************/

import * as rx from "rxjs";
import fetch, { RequestInit } from 'node-fetch';
import { Rools, Rule } from 'rools';
import * as etl from './lib/index.js';
import { EtlRoolsResult, GuiManager, Magento } from "./lib/index.js";
import { CsvCellType } from "./lib/endpoints/csv.js";
import { sqlvalue } from "./lib/endpoints/databases/condition.js";
//import { DiscordHelper } from "./lib/index.js";


const START = new Date;
//GuiManager.startGui(true, 20);
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
const magentoStockItems = magentoEndpoint.getStockItems();

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

const csv = etl.Csv.getEndpoint("./data");
const csvMagento = csv.getFile("magento.csv", headerMagento);
const csvPuma = csv.getFile("puma.csv", headerPuma, ';');

const memory = etl.Memory.getEndpoint();
const queue = memory.getQueue<DbProduct>('queue');


const errorsEndpoint = etl.Errors.getEndpoint();
const errors = errorsEndpoint.getCollection('all');



const translator = new etl.GoogleTranslateHelper(process.env.GOOGLE_CLOUD_API_KEY!, 'en', 'ru');
const http = new etl.HttpClientHelper();


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

let ErrorProcessing$ = errors.select(false).pipe(
    etl.log('Error occurred: ')
)



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
    rx.take(3),
    rx.map(p => headerPuma.arrToObj(p)),

    //etl.where({price: 6800, category: 'Мужчины'}),
    //etl.log(),

    rx.map(pumaProduct2Db),
    //translator.operator([], ['name']),
    //etl.push(table),
    //etl.expect<DbProduct>('price = 1800', { price: 1800 }),
    //etl.push(queue),
    //rx.delay(5000),
    etl.log(),
)

// let MySql_to_Magento$ = table.select().pipe(
//     //etl.log(),
//     rx.take(100),
//     rx.map(db2Magento),
// //    translator.operator([], ['name']),
// //    etl.log(),
//     etl.push(magentoProducts)
// )

// await csv.delete();
//await etl.run(magento_to_Csv$);

//etl.run(ErrorProcessing$);

// etl.run(csvPuma.selectErrors().pipe(
//     etl.log()
// ));

// await etl.run(PumaCsv_to_MySql$);


const PrintMagentoProducts$ = magentoProducts.select().pipe(
    rx.take(10),
    etl.log()
)
const PrintStockItems$ = magentoStockItems.select().pipe(
    rx.take(10),
    etl.log()
)



type PumaCsvItem = {
    images: string[];
    name: string; 
    vendor_code: string; 
    price: number;
    currency: string;
    availability: boolean;
    category: string; 
    url: string;
    desctiption: string;
    renk: string;
    beden: string;
}

type PumaItem = {
    product: PumaCsvItem;
    image?: Blob;
}

function PumaArr_2_Obj(arr: CsvCellType[]): PumaCsvItem {
    const res = csvPuma.header.arrToObj(arr);
    const images: string[] = (res.image as string).split(';')
    const item: any = {...res, images};
    delete item.image;
    return item as PumaCsvItem;
}

const getImage = http.getBlobOperator<PumaItem>(v => v.product.images[0], 'image');

const PrintPuma$ = csvPuma.select(true).pipe(
    rx.take( 1 ),
    rx.map( PumaArr_2_Obj ),
    rx.map( p => ({product: p} as PumaItem) ),
    getImage,
    etl.log()
)


type PumaDataItem = {
    product: PumaCsvItem;
    imageContents?: Blob;
}

const PrintPuma1$ = csvPuma.select(true).pipe(
    rx.take( 1 ),
    rx.map( PumaArr_2_Obj ),
    rx.map<PumaCsvItem, PumaDataItem>( p => ({product: p}) ),

    http.getFileContentsOperator( v => v.product.images[0], 'imageContents' ),

    magentoProducts.uploadImageOperator<PumaDataItem>(v => ({
        product: v.product.vendor_code,
        imageContents: v.imageContents!,
        filename: v.product.vendor_code + '.' + etl.extractFileName(v.product.images[0]),
        label: v.product.vendor_code + ' product image',
        type: v.imageContents!.type
    })),

    etl.log()
)

// const imageBlob = await http.getFileContents('https://images.puma.com/image/upload/f_auto,q_auto,b_rgb:fafafa/global/024136/02/fnd/TUR/w/1000/h/1000/fmt/png');

// const magentoStage = new Magento.Endpoint(process.env.MAGENTO_STAGE!, process.env.MAGENTO_STAGE_LOGIN!, process.env.MAGENTO_STAGE_PASSWORD!);
// const magentoStageProducts = magentoStage.getProducts();

// const res = await magentoStageProducts.uploadImage(
//     '024136_02',
//     imageBlob,
//     '1111.png',
//     '1111 product image',
//     imageBlob.type
// )

// console.log(res)


//await etl.run(PrintMagentoProducts$);
//await etl.run(PrintPuma$);


// const buf = memory.getBuffer<number>('buf', [0,1,2,3,4,5]);
// const p = buf.select().pipe(
//     etl.move<{ n: number }>({to: 'n'}),
//     etl.copy<{ n: number, p: {k: number} }, any>('n', 'p.k'),
//     //etl.copy('n', 'nn'),
//     //etl.move('nn', 'kk'),
//     //etl.where({ n: etl.VALUE.in({'1': 1, '2': 2, '3': 3}) }),
//     //etl.where({ n: etl.VALUE.of([1,2,3]) }),
//     etl.where({ n: etl.value.or(etl.value.of([0,1]), etl.value["=="](5)) }),
//     //etl.expect('check', {n: etl.VALUE.of([1,2])}),
//     etl.log()
// )

// //etl.run(p, buf.selectErrors().pipe(etl.log()));


// const zendesk = new etl.Zendesk.Endpoint(process.env.ZENDESK_URL!, process.env.ZENDESK_USERNAME!, 
//     process.env.ZENDESK_TOKEN!);
// const tickets = zendesk.getTickets();
// const ticketFields = zendesk.getTicketFields();

// const PrintTickets$ = tickets.select().pipe(
//     rx.take(1),
//     //etl.move({to: 'ticket'}),
//     //etl.copy('ticket.status', 'status'),
//     //etl.copy('ticket.id', 'id'),
//     //etl.remove('ticket'),
//    //rx.distinct(),
//     etl.log()
// )

// //etl.run(PrintTickets$)

// const res = await ticketFields.get(360015269411)
// console.log(res);

// const res = await tickets.insert({
//     subject: "Callback to test1 (111-111-111)",
//     comment: "Callback request",
//     submitter_id: parseInt(process.env.ZENDESK_USER_ID!),
//     assignee_id: parseInt(process.env.ZENDESK_USER_ID!),
//     ticket_form_id: 11981095371156,
//     fields: [
//         {
//           id: 360027213031,
//           value: '11'
//         },
//         {
//           id: 14189942624660,
//           value: '22'
//         }
//     ]
// })

// console.log(res);

//const db = new etl.databases.Postgres.Endpoint(process.env.POSTGRES_CONNECTION_STRING!);
//const table = db.getTable('cities');
// const db = new etl.databases.MySql.Endpoint(process.env.MYSQL_CONNECTION_STRING!, undefined, 'mysql2');
// const table = db.getTable('test1');

// //const res = await table.update({name: 'test7'}, {id: sqlvalue.of([20])});
// //console.log(res);

// const PrintTable$ = table.select().pipe(
//     etl.log()
// )

// await etl.run(PrintTable$);
// db.releaseEndpoint();


// const ftp = new etl.filesystems.Ftp.Endpoint({host: process.env.FTP_HOST, user: process.env.FTP_USER, password: process.env.FTP_PASSWORD});
// const folder = ftp.getFolder('Vorne');
// const PrintFolder$ = folder.select().pipe(
//     etl.log(),
//     rx.tap(v => folder.insert('111', {isFolder: true, contents: "ttttttt"}))
// )
// await etl.run(PrintFolder$);




// const ruleSkipCheapProducts = new Rule({
//     name: 'skip products with price <= 1000',
//     when: (product: DbProduct) => product.price! <= 1000,
//     then: (product: DbProduct & EtlRoolsResult) => {
//         product.etl = {skip: true};
//     },
// });

// const ruleSetProductTaxClass = new Rule({
//     name: 'update product tax class',
//     when: (product: DbProduct) => product.price! > 1000,
//     then: (product: DbProduct & EtlRoolsResult) => {
//         product.tax_class_id = '10';
//     },
// });




// const rools = new Rools();
// await rools.register([ruleSkipCheapProducts, ruleSetProductTaxClass]);

// //const magento = new Magento.Endpoint('https://magento.test', process.env.MAGENTO_LOGIN!, process.env.MAGENTO_PASSWORD!, false);
// //const magento = new Magento.Endpoint(process.env.MAGENTO_STAGE!, process.env.MAGENTO_STAGE_LOGIN!, process.env.MAGENTO_STAGE_PASSWORD!);
// //const magentoStageProducts = magento.getProducts();

// const db = new etl.databases.MySql.Endpoint(process.env.MYSQL_CONNECTION_STRING!, undefined, 'mysql2');
// const table = db.getTable<DbProduct>('test1');

// const PrintStageCategories$ = table.select().pipe(
//     rx.take(10),
//     etl.rools(rools),
//     etl.log(),
// )

// await etl.run(PrintStageCategories$);


const ftp = new etl.filesystems.Ftp.Endpoint({host: process.env.FTP_HOST, user: process.env.FTP_USER_NAME, password: process.env.FTP_USER_PASS});
const folder = ftp.getFolder('ftp');



const res: any[] = [];

let stream$ = folder.select().pipe(
    etl.log(),
    rx.tap(v => res.push(v))
);
await etl.run(stream$);

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




