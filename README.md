# RxJs-ETL-Kit

<img src="https://github.com/igor-berezhnoy/rxjs-etl-kit/raw/main/static/ETL.png" alt="Logo" title="Logo" style="max-width: 100%;">

RxJs-ETL-Kit is a platform that employs RxJs observables, allowing developers to build stream-based ETL (Extract, Transform, Load) pipelines complete with buffering and bulk-insertions.

[![NPM Version][npm-image]][npm-url]
[![NPM Downloads][downloads-image]][downloads-url]
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/igor-berezhnoy/rxjs-etl-kit/actions/workflows/project-ci.yml/badge.svg?branch=main)](https://github.com/igor-berezhnoy/rxjs-etl-kit/actions?query=branch%3Amain+workflow%3A"Project%20CI")
[![Coverage Status](https://codecov.io/gh/igor-berezhnoy/rxjs-etl-kit/branch/main/graph/badge.svg)](https://codecov.io/gh/igor-berezhnoy/rxjs-etl-kit)

[//]: # (https://img.shields.io/codecov/c/github/igor-berezhnoy/rxjs-etl-kit/.svg   https://codecov.io/gh/igor-berezhnoy/rxjs-etl-kit)


[npm-image]: https://img.shields.io/npm/v/rxjs-etl-kit.svg
[npm-url]: https://npmjs.org/package/rxjs-etl-kit
[downloads-image]: https://img.shields.io/npm/dm/rxjs-etl-kit.svg
[downloads-url]: https://npmjs.org/package/rxjs-etl-kit

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Table of Contents
* [Why / when would I need this?](#why--when-would-i-need-this)
* [Installation](#installation)
* [Usage](#usage)
* [Concept](#concept)
* [Features](#features)
* [GUI](#gui)
* [Examples (how to)](#examples-how-to)
    * [Export rows from Postgres table to csv-file (postgresql -> .csv)](#export-rows-from-postgres-table-to-csv-file-postgresql---csv)
    * [Sort rows in csv-file by the first column (.csv -> .csv)](#sort-rows-in-csv-file-by-the-first-column-csv---csv)
    * [Create telegram bot with 'echo' functionality](#create-telegram-bot-with-echo-functionality)
* [API Reference](#api-reference)
    * [Core](#core)
        * [Collection](#collection)
    * [Endpoints and it's collections](#endpoints-and-its-collections)
        * [MemoryEndpoint](#memoryendpoint)
        * [FilesystemEndpoint](#filesystemendpoint)
        * [CsvEndpoint](#csvendpoint)
        * [JsonEndpoint](#jsonendpoint)
        * [XmlEndpoint](#xmlendpoint)
        * [PostgresEndpoint](#postgresendpoint)
        * [MagentoEndpoint](#magentoendpoint)
        * [TrelloEndpoint](#trelloendpoint)
        * [TelegramEndpoint](#telegramendpoint)
        * [IntervalEndpoint](#intervalendpoint)
    * [Operators](#operators)
        * [run](#run)
        * [log](#log)
        * [where](#where)
        * [push](#push)
        * [numerate](#numerate)
        * [addField](#addfield)
        * [addColumn](#addcolumn)
        * [join](#join)
    * [Misc](#misc)
        * [GoogleTranslateHelper](#googletranslatehelper)
        * [Header](#header)
        * [Utility functions](#utility-functions) 
- [License](#license)

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Why / when would I need this?

**RxJs-ETL-Kit** is a simple **ETL glue** represented as an extention to the **RxJs** library. 
Typically, you'd use **RxJs-ETL-Kit** to help with ETL processes. It can extract data from the one or more sources, transform it and load to one or more destinations in nedded order.

You can use javascript and typescript with it.

**RxJs-ETL-Kit** will **NOT** help you with "big data" - it executes on the one computer and is not supports clustering from the box.

Here's some ways to use it:

1. Read some data from database and export it to the .csv file and vice versa
2. Create file converters
3. Filter or sort content of some files
4. Run some queries in database
5. Create Telegram bots with [TelegramEndpoint](#telegramendpoint)

You can find many examples of using **RxJs-ETL-Kit** in the API Reference section of this file.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Installation

```
npm install rxjs-etl-kit
```
or
```
yarn add rxjs-etl-kit
```

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Usage

Require the RxJs-ETL-Kit library in the desired file to make it accessible.

Introductory example: postgresql -> .csv
```typescript
import { map } from "rxjs";
import { CsvEndpoint, GuiManager, Header, PostgresEndpoint, log, push, run } from "./lib";

// If you want to view GUI, uncomment the next line of code
// new GuiManager();

// Step 1: endpoint creation
const postgres = new PostgresEndpoint("postgres://user:password@127.0.0.1:5432/database");
const source = postgres.getTable('users');

const csvEndpoint = new CsvEndpoint('./dest-folder');
const dest = csvEndpoint.getFile('users.scv');

const header = new Header("id", "name", "login", "email");

// Step 2: transformation streams creation
const sourceToDest$ = source.list().pipe(
    log(),
    map(v => header.objToArr(v)),
    push(dest)
);

// Step 3: runing transformations (and wait until they finish, if necessary)
await run(sourceToDest$);
 ```

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Concept

**RxJs-ETL-Kit** contains several main concepts: 
* Endpoints - sources and destinations of data, which holds connection to the one system instance and other parameters of this system, and groups methods to get collections related this system
* Collections - data object types exists in the endpoint system
* Piplines (or streams) - routs of data transformation and delivery, based on **RxJs** streams

Using of this library consists of 3 steps:

1. Define your endpoints and collections for sources and destinations
2. Define data transformation pipelines using **pipe()** method of input streams of your source endpoints
3. Run transformation pipelines in order and wait for completion

ETL process:

* **Extract**: Data extraction from the source collection performs with **list()** method, which returns the **RxJs** stream
* **Transform**: Use any **RxJs** and **RxJs-ETL-Kit** operators inside **pipe()** method of the input stream to transform the input data. To complex data transformation you can use the **MemoryEndpoint** class, which can store data and which collections have **forEach()** and some other methods to manipulate with data in it
* **Load**: Loading of data to the destination endpoint performs with **push()** collection operator

Chaining:

Chaning of data transformation performs with **pipe()** method of the input data stream. 
Chaning of several streams performs by using **await** with **run()** procedure.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Features

* Simple way to use, consists of 3 steps: 1) Endpoints creation and collections getting 2) Piplines creation 3) Run piplines in order you want (and if you want to use GUI - then the zerro step is creating instance of **GuiManager** class)
* This library can be used to build simple console application or application with console GUI, which support many usefull functions to debug process (see [GUI](#gui))
* It written in typescript and you can use it in javascript and typescript applications
* Fully compatible with **RsJs** library, it's observables, operators etc.
* Create pipelines of data extraction, transformation and loading, and run this pipelines in nedded order
* Many kind of source and destination endpoints, for example PostgreSql, csv, json, xml
* Work with any type of data, including hierarchical data structures (json, xml) and support typescript types
* With endpoint events mechanism you can handle different stream events, for example stream start/end, errors and other (see [Endpoint](#endpoint))
* You can create Telegram bots with [TelegramEndpoint](#telegramendpoint) to control the ETL process for example
* You can translate some data to another language with [GoogleTranslateHelper](#googletranslatehelper)

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# GUI

<img src="https://github.com/igor-berezhnoy/rxjs-etl-kit/raw/main/static/GUI.jpg" alt="GUI" title="GUI" style="max-width: 100%">

* Simple way to use, you need only create instance of **GuiManager** class before any endpoint creation (at the begin of the program)
* You can pause the ETL-process and resume it with 'space' on keyboard
* With 'enter' you can execute ETL process step-by-step in pause mode
* With 'esc' you can quit the program
* GUI display full list of created endpoints, collections, their statuses and last values recived from (or pushed to) them
* Logs are displayed in footer part of console window
* You can select the log window with 'ctrl + l' and scroll it with up/down arrows

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Examples (how to)

### Export rows from Postgres table to csv-file (postgresql -> .csv)

```typescript
const { PostgresEndpoint, CsvEndpoint, Header, log, push, run } = require("rxjs-etl-kit");
const { map } = require("rxjs");

const postgres = new PostgresEndpoint("postgres://user:password@127.0.0.1:5432/database");
const source = postgres.getTable('users');

const csvEndpoint = new CsvEndpoint('./dest-folder');
const dest = csvEndpoint.getFile('users.scv');

const header = new Header("id", "name", "login", "email");

const sourceToDest$ = source.list().pipe(
    log(),
    map(v => header.objToArr(v)),
    push(dest)
);

await run(sourceToDest$);
 ```

 ### Sort rows in csv-file by the first column (.csv -> .csv)

```typescript
const etl = require('rxjs-etl-kit');

const csvEndpoint = etl.CsvEndpoint();
const csv = csvEndpoint.getFile('users.scv');
const memory = etl.BufferEndpoint();
const buffer = memory.getBuffer('buffer 1');

const scvToBuffer$ = csv.list().pipe(
    etl.push(buffer)
);
const bufferToCsv$ = buffer.list().pipe(
    etl.push(csv)
);

await etl.run(scvToBuffer$);

buffer.sort((row1, row2) => row1[0] > row2[0]);
csv.clear();

await etl.run(bufferToCsv$)
 ```

 ### Create telegram bot with translation functionality

 ```typescript
const etl = require('rxjs-etl-kit');

const telegram = new etl.TelegramEndpoint();
const bot = telegram.startBot('bot 1', process.env.TELEGRAM_BOT_TOKEN!);
const translator = new etl.GoogleTranslateHelper(process.env.GOOGLE_CLOUD_API_KEY!, 'en', 'ru');

const startTelegramBot$ = telegram.list().pipe(
    etl.log(),          // log user messages to the console
    translator.operator([], [message]), // translate 'message' field
    etl.push(telegram)  // echo input message back to the user
);

etl.run(startTelegramBot$);
```

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# API Reference

## Core

### Collection

Base class for all collections. Declares public interface of collection and implements event mechanism.

Methods:

```typescript
// Read elements from the collection and create data stream to process it
list(): Observable<any>;

// Add value to the collection (usually to the end of stream)
// value: what will be added to the collection
async push(value: any);

// Clear data of the collection
async clear();

// Add listener of specified event
// event: which event we want to listen, see below
// listener: callback function to handle events
on(event: CollectionEvent, listener: (...data: any[]) => void);
```

Types:

```typescript
export type CollectionEvent = 
    "list.start" |  // fires at the start of stream
    "list.end" |    // at the end of stream
    "list.data" |   // for every data value in the stream 
    "list.error" |  // on error
    "list.skip" |   // when the endpoint skip some data 
    "list.up" |     // when the endpoint go to the parent element while the tree data processing
    "list.down" |   // when the endpoint go to the child element while the tree data processing
    "push" |        // when data is pushed to the endpoint
    "clear";        // when the Endpoint.clear method is called
```

## Endpoints and it's collections

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### MemoryEndpoint

```typescript
// Creates new memory buffer. This is a generic method so you can specify type of data which will be stored in
// collectionName: identificator of the creating collection object
// values: initial data
// guiOptions: Some options how to display this endpoint
getBuffer<T>(collectionName: string, values: T[] = [], guiOptions: CollectionGuiOptions<T> = {}): BufferCollection;

// Release buffer data
// collectionName: identificator of the releasing collection object
releaseBuffer(collectionName: string);
```

### BufferCollection

Buffer to store values in memory and perform complex operations on it. Should be created with **getBuffer** method of **MemoryEndpoint**

Methods:

```typescript
// Create the observable object and send data from the buffer to it
list(): Observable<T>;

// Pushes the value to the buffer 
// value: what will be added to the buffer
async push(value: T);

// Clear endpoint data buffer
async clear();

// Sort buffer data
// compareFn: You can spacify the comparison function which returns number 
//            (for example () => v1 - v2, it is behaviour equals to Array.sort())
//            or which returns boolean (for example () => v1 > v2)
sort(compareFn: (v1: T, v2: T) => number | boolean);

// This function is equals to Array.forEach
forEach(callbackfn: (value: T, index: number, array: T[]) => void);
```

Example:

```typescript
const etl = require('rxjs-etl-kit');

const csvEndpoint = etl.CsvEndpoint();
const csv = csvEndpoint.getFile('users.scv');
const memory = etl.BufferEndpoint();
const buffer = memory.getBuffer('buffer 1');

const scvToBuffer$ = csv.list().pipe(
    etl.push(buffer);
)
const bufferToCsv$ = buffer.list().pipe(
    etl.push(csv)
)

await etl.run(scvToBuffer$);

buffer.sort((row1, row2) => row1[0] > row2[0]);
csv.clear();

etl.run(bufferToCsv$)
```

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### FilesystemEndpoint

Search for files and folders with standart unix shell wildcards [see glob documentation for details](https://www.npmjs.com/package/glob).

Methods:

```typescript
// rootFolder: full or relative path to the folder of intetest
constructor(rootFolder: string);

// Creates new FilesystemCollection
// folderName: subfolder of the root folder and identificator of the creating collection object
// guiOptions: Some options how to display this endpoint
getFolder(folderName: string = '.', guiOptions: CollectionGuiOptions<PathDetails> = {}): FilesystemCollection;

// Release FilesystemCollection
// folderName: identificator of the releasing collection object
releaseFolder(folderName: string);
```

### FilesystemCollection

Methods:

```typescript
// Create the observable object and send files and folders information to it
// mask: search path mask in glob format (see glob documentation)
//       for example:
//       *.js - all js files in root folder
//       **/*.png - all png files in root folder and subfolders
// options: Search options, see below
list(mask: string = '*', options?: ReadOptions): Observable<string[]>;

// Create folder or file
// pathDetails: Information about path, which returns from list() method
// filePath: File or folder path
// isFolder: Is it file or folder
// data: What will be added to the file, if it is a file, ignore for folders
async push(pathDetails: PathDetails, data?: string | NodeJS.ArrayBufferView | Iterable<string | NodeJS.ArrayBufferView> | AsyncIterable<string | NodeJS.ArrayBufferView> | internal.Stream);
async push(filePath: string, data?: string | NodeJS.ArrayBufferView | Iterable<string | NodeJS.ArrayBufferView> | AsyncIterable<string | NodeJS.ArrayBufferView> | internal.Stream, isFolder?: boolean);

// Clear the root folder by mask
// mask: Which files and folders we need to delete
// options: Search options, see below 
//          IMPORTANT! Be careful with option includeRootDir because if it is true, and the objectsToSearch is not 'filesOnly',
//          then the root folder will be deleted with all its content! Including folder itself.
async clear(mask: string = '*', options?: ReadOptions);
```

Types:

```typescript
type ReadOptions = {
    includeRootDir?: boolean;   // Is root folder itself will be included to search results
                                // false by default
    
    objectsToSearch?:           // Which object types will be included to the search results
        'filesOnly' |           // Only files
        'foldersOnly' |         // Only folders
        'all';                  // Both files and folders
                                // all is default option
}

type PathDetails = {
    isFolder: boolean 
    name: string;
    relativePath: string; // Empty for root folder
    fullPath: string;
    parentFolderRelativePath: string; // '..' for root folder
    parentFolderFullPath: string;
}

```

Example:

```typescript
const etl = require('rxjs-etl-kit');
const rx = require('rxjs');

const fs = new etl.FilesystemEndpoint('~');
const scripts = ep.getFolder('scripts');

const printAllJsFileNames$ = scripts.list('**/*.js').pipe(
    rx.map(v => v.name)
    etl.log()
);

etl.run(printAllJsFileNames$)
```

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### CsvEndpoint

Parses source csv file into individual records or write record to the end of destination csv file. Every record is csv-string and presented by array of values.

Methods:

```typescript
// Create collection object for the specified file
// filename: full or relative name of the csv file and identificator of the creating collection object
// delimiter: delimiter of values in one string of file data, equals to ',' by default
// guiOptions: Some options how to display this endpoint
getFile(filename: string, delimiter: string = ",", guiOptions: CollectionGuiOptions<string[]> = {}): CsvCollection;

// Release collection object
// filename: identificator of the releasing collection object
releaseFile(filename: string);
```

### CsvCollection

Methods:

```typescript
// Create the observable object and send file data to it string by string
// skipFirstLine: skip the first line in the file, useful for skip header
// skipEmptyLines: skip all empty lines in file
list(skipFirstLine: boolean = false, skipEmptyLines = false): Observable<string[]>;

// Add row to the end of file with specified value 
// value: what will be added to the file
async push(value: string[]);

// Clear the csv file
async clear();
```

Example:

```typescript
const etl = require('rxjs-etl-kit');

const csv = etl.CsvEndpoint('~');
const testFile = csv.getFile('test.csv')

const logTestFileRows$ = testFile.list().pipe(
    etl.log()
);

etl.run(logTestFileRows$)
```

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### JsonEndpoint

Read and write json file with buffering it in memory. You can get objects from json by path specifing in JSONPath format or in lodash simple path manner (see logash 'get' function documentation).

Methods:

```typescript
// Create collection object for the specified file
// filename: full or relative name of the json file and identificator of the creating collection object
// autosave: save json from memory to the file after every change
// autoload: load json from the file to memory before every get or search operation
// encoding: file encoding
// guiOptions: Some options how to display this endpoint
getFile(filename: string, autosave: boolean = true, autoload: boolean = false, encoding?: BufferEncoding, guiOptions: CollectionGuiOptions<number> = {}): JsonCollection;

// Release collection object
// filename: identificator of the releasing collection object
releaseFile(filename: string);
```

### JsonCollection

Methods:

```typescript
// Find and send to observable child objects by specified path
// path: search path in lodash simple path manner
// jsonPath: search path in JSONPath format
// options: see below
list(path: string, options?: ReadOptions): Observable<any>;
listByJsonPath(jsonPath: string | string[], options?: ReadOptions): Observable<any>;

// Find and return child object by specified path
// path: search path in lodash simple path manner
// jsonPath: search path in JSONPath format
get(path: string): any;
getByJsonPath(jsonPath: string): any;

// If fieldname is specified, the function find the object by path and add value as its field
// If fieldname is not specified, the function find the array by path and push value to it
// value: what will be added to the json
// path: where value will be added as child, specified in lodash simple path manner
// fieldname: name of the field to which the value will be added, 
//            and flag - is we add value to array or to object
async push(value: any, path?: string, fieldname?: string);

// Clear the json file and write an empty object to it
async clear();

// Reload the json to the memory from the file
load();

// Save the json from the memory to the file
save();
```

Types:

```typescript
type JsonReadOptions = {
    searchReturns?: 'foundedOnly'           // Default value, means that only search results objects will be sended to observable by the function
        | 'foundedImmediateChildrenOnly'    // Only the immidiate children of search results objects will be sended to observable 
        | 'foundedWithDescendants';         // Recursive send all objects from the object tree of every search result, including search result object itself

    addRelativePathAsField?: string;        // If specified, the relative path will be added to the sended objects as addRelativePathAsField field 
}
```

Example:

```typescript
const etl = require('rxjs-etl-kit');
const { tap } = require('rxjs');

const json = etl.JsonEndpoint('~');
const testFile = etl.getFile('test.json');

const printJsonBookNames$ = testFile.list('store.book').pipe(
    tap(book => console.log(book.name))
);

const printJsonAuthors$ = testFile.listByJsonPath('$.store.book[*].author', {searchReturns: 'foundedOnly', addRelativePathAsField: "path"}).pipe(
    etl.log()
);

await etl.run(printJsonAuthors$, printJsonBookNames$);
```

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### XmlEndpoint

<a name="xml" href="#xml">#</a> etl.<b>XmlEndpoint</b>(<i>filename, autosave?, autoload?, encoding?</i>)

Read and write XML document with buffering it in memory. You can get nodes from XML by path specifing in XPath format.

Methods:

```typescript
// Create collection object for the specified file
// filename: full or relative name of the xml file and identificator of the creating collection object
// autosave: save xml from memory to the file after every change
// autoload: load xml from the file to memory before every get or search operation
// encoding: file encoding
// guiOptions: Some options how to display this endpoint
getFile(filename: string, autosave: boolean = true, autoload: boolean = false, encoding?: BufferEncoding, guiOptions: CollectionGuiOptions<string[]> = {}): XmlCollection;

// Release collection object
// filename: identificator of the releasing collection object
releaseFile(filename: string);
```

### XmlCollection

Methods:

```typescript
// Find and send to observable child objects by specified xpath
// xpath: xpath to search
// options: see below
list(xpath: string = '', options: XmlReadOptions = {}): EtlObservable<Node>;

// Find and return child node by specified path
// xpath: search path
get(xpath: string = ''): XPath.SelectedValue

// If attribute is specified, the function find the object by xpath and add value as its attribute
// If attribute is not specified, the function find the node by xpath and push value as its child node
// value: what will be added to the xml
// xpath: where value will be added as child, specified in lodash simple path manner
// attribute: name of the attribute which value will be setted, 
//            and flag - is we add value as attribute or as node
async push(value: any, xpath: string = '', attribute: string = '');

// Clear the xml file and write an empty object to it
async clear();

// Reload the xml to the memory from the file
load();

// Save the xml from the memory to the file
save();
```

Types:

```typescript
export type XmlReadOptions = {
    searchReturns?: 'foundedOnly'           // Default value, means that only search results nodes will be sended to observable by the function
        | 'foundedImmediateChildrenOnly'    // Only the immediate children of search results nodes will be sended to observable 
        | 'foundedWithDescendants';         // Recursive send all nodes from the tree of every searched result, including searched result node itself

    addRelativePathAsAttribute?: string;    // If specified, the relative path will be added to the sended nodes as attribute, specified with this value 
}
```

Example

```typescript
const etl = require('rxjs-etl-kit');
const { map } = require('rxjs');

const xml = etl.XmlEndpoint('/tmp');
const testFile = xml.XmlCollection('test.xml');

const printXmlAuthors$ = testFile.list('/store/book/author').pipe(
    map(v => v.firstChild.nodeValue),
    etl.log()
);

await etl.run(printXmlAuthors$);
```

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### PostgresEndpoint

Represents PostgreSQL database. 

Methods:

```typescript
// Connection to the database can be performed using connection string or through the existing pool.
constructor(connectionString: string);
constructor(connectionPool: pg.Pool);

// Create collection object for the specified database table
// table: name of database table and identificator of the creating collection object
// guiOptions: Some options how to display this endpoint
getTable(table: string, guiOptions: CollectionGuiOptions<string[]> = {}): TableCollection;

// Release collection object
// table: identificator of the releasing collection object
releaseTable(table: string);
```

### TableCollection

Presents the table from the PostgreSQL database. 

Methods:

```typescript
// Create the observable object and send data from the database table to it
// where: you can filter incoming data by this parameter
//        it can be SQL where clause 
//        or object with fields as collumn names 
//        and its values as needed collumn values
list(where: string | {} = ''): Observable<T>;

// Insert value to the database table
// value: what will be added to the database
async push(value: T);

// Clear database table
// where: you can filter table rows to deleting by this parameter
//        it can be SQL where clause 
//        or object with fields as collumn names 
//        and its values as needed collumn values
async clear(where: string | {} = '');
```

Example:

```typescript
const etl = require('rxjs-etl-kit');

const pg = etl.PostgresEndpoint('postgres://user:password@127.0.0.1:5432/database');
const table = pg.getTable('users');

const logUsers$ = table.list().pipe(
    etl.log()
);

etl.run(logUsers$)
```

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### MagentoEndpoint

Presents Magento CMS objects.
Go to https://meetanshi.com/blog/create-update-product-using-rest-api-in-magento-2/ for details how to configure Magento integration to get access to it's API. 

Methods:

```typescript
// magentoUrl: Url of Magento
// login: admin login
// password: admin password
// rejectUnauthorized: You can set it to true to ignore ssl servificate problems while development.
constructor(magentoUrl: string, login: string, password: string, rejectUnauthorized: boolean = true);

// Create collection object for the Magento products
// guiOptions: Some options how to display this endpoint
getProducts(guiOptions: CollectionGuiOptions<Partial<Product>> = {}): ProductsCollection;

// Release products collection object
releaseProducts();
```

### ProductsCollection

Presents Magento CMS products. 

Methods:

```typescript
// Create the observable object and send product data from the Magento to it
// where: you can filter products by specifing object with fields as collumn names and it's values as fields values 
// fields: you can select which products fields will be returned (null means 'all fields') 
list(where: Partial<Product> = {}, fields: ProductFields[] = null): Observable<T>;

// Add new product to the Magento
// value: product fields values
async push(value: NewProductAttributes);
```

Example:

```typescript
const etl = require('rxjs-etl-kit');

const magento = etl.MagentoEndpoint('https://magento.test', process.env.MAGENTO_LOGIN!, process.env.MAGENTO_PASSWORD!);
const products = magento.getProducts();

const logProductsWithPrice100$ = products.list({price: 100}).pipe(
    etl.log()
);

etl.run(logProductsWithPrice100$)
```

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### TrelloEndpoint

Presents Trello task tracking system objects.
For details how to get API key and authorization token please read [Trello documentation](https://developer.atlassian.com/cloud/trello/guides/rest-api/api-introduction/).

Methods:

```typescript
// url: Trello web url
// apiKey: Trello API key
// authToken: Trello authorization token
// rejectUnauthorized: You can set it to true to ignore ssl servificate problems while development.
constructor(apiKey: string, authToken: string, url: string = "https://trello.com", rejectUnauthorized: boolean = true);

// Create collection object for the Trello user boards
// username: user, which boards we need to get, by default it is a Trello authorization token owner
// collectionName: identificator of the creating collection object
// guiOptions: Some options how to display this endpoint
getUserBoards(username: string = 'me', collectionName: string = 'Boards', guiOptions: CollectionGuiOptions<Partial<Board>> = {}): BoardsCollection;

// Create collection object for the Trello board lists
// boardId: board id
// collectionName: identificator of the creating collection object
// guiOptions: Some options how to display this endpoint
getBoardLists(boardId: string, collectionName: string = 'Lists', guiOptions: CollectionGuiOptions<Partial<List>> = {}): ListsCollection;

// Create collection object for the Trello list cards
// listId: list id
// collectionName: identificator of the creating collection object
// guiOptions: Some options how to display this endpoint
getListCards(listId: string, collectionName: string = 'Cards', guiOptions: CollectionGuiOptions<Partial<Card>> = {}): CardsCollection;

// Release collection data
// collectionName: identificator of the releasing collection object
releaseCollection(collectionName: string);
```

### BoardsCollection

Presents Trello boards accessible by user which was specified while collection creation. 

Methods:

```typescript
// Create the observable object and send boards data from the Trello to it
// where: you can filter boards by specifing object with fields as collumn names and it's values as fields values 
// fields: you can select which board fields will be returned (null means 'all fields') 
list(where: Partial<Board> = {}, fields: (keyof Board)[] = null): EtlObservable<Partial<Board>>;

// Add new board to the Trello
// value: board fields values
async push(value: Omit<Partial<Board>, 'id'>);

// Update board fields values by board id
// boardId: board id
// value: new board fields values as hash object
async update(boardId: string, value: Omit<Partial<Board>, 'id'>);

// Get all user boards
async get(): Promise<Board[]>;

// Get board by id
// boardId: board id
async get(boardId?: string): Promise<Board>;

// Get board by url from browser
async getByBrowserUrl(url: string): Promise<Board>;
```

### ListsCollection

Presents Trello lists on board which was specified while collection creation. 

Methods:

```typescript
// Create the observable object and send lists data from the Trello to it
// where: you can filter lists by specifing object with fields as collumn names and it's values as fields values 
// fields: you can select which list fields will be returned (null means 'all fields') 
list(where: Partial<List> = {}, fields: (keyof List)[] = null): EtlObservable<Partial<List>>;

// Add new list to the Trello
// value: list fields values
async push(value: Omit<Partial<List>, 'id'>);

// Update list fields values by list id
// listId: list id
// value: new list fields values as hash object
async update(listId: string, value: Omit<Partial<List>, 'id'>);

// Get all lists
async get(): Promise<List[]>;

// Get list by id
// listId: list id
async get(listId?: string): Promise<List>;

// Archive or unarchive a list
// listId: list id
async switchClosed(listId: string);

// Move list to another board
// listId: list id
// destBoardId: destination board id
async move(listId: string, destBoardId: string);

// Get list actions
// listId: list id
async getActions(listId: string);
```

### CardsCollection

Presents Trello cards in list which was specified while collection creation. 

Methods:

```typescript
// Create the observable object and send cards data from the Trello to it
// where: you can filter cards by specifing object with fields as collumn names and it's values as fields values 
// fields: you can select which card fields will be returned (null means 'all fields') 
list(where: Partial<Card> = {}, fields: (keyof Card)[] = null): EtlObservable<Partial<Card>>;

// Add new card to the Trello
// value: card fields values
async push(value: Omit<Partial<Card>, 'id'>);

// Update card fields values by card id
// listId: card id
// value: new list fields values as hash object
async update(cardId: string, value: Omit<Partial<Card>, 'id'>);

// Get all cards
async get(): Promise<Card[]>;

// Get card by id
// cardId: card id
async get(cardId?: string): Promise<Card>;

// Archive all cards in current list
async archiveListCards();

// Move all cards from the current list to another board and list
// destBoardId: destination board id
// destListId: destination list id
async moveListCards(destBoardId: string, destListId: string);
```

Example:

```typescript
import * as etl from 'rxjs-etl-kit';

const trello = etl.TrelloEndpoint(process.env.TRELLO_API_KEY!, process.env.TRELLO_AUTH_TOKEN!);

const boards = trello.getUserBoards();
const board = await boards.getByBrowserUrl('https://trello.com/b/C9zegsyz/board1');

const lists = trello.getBoardLists(board.id);
const list = (await lists.get())[0];

const cards = trello.getListCards(list.id);

const logCards$ = cards.list({}, ["id", "name"]).pipe(
    etl.log()
);

etl.run(logCards$)
```

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### TelegramEndpoint

With this endpoint you can create telegram bots and chats with users. It can listen for user messages and send the response massages. 
It also can set the user keyboard for the chat.

Methods:

```typescript
// Start bot and return collection object for the bot messages
// collectionName: identificator of the creating collection object
// token: Bot token
// keyboard: JSON keyboard description, see the node-telegram-bot-api for detailes
//           Keyboard example: [["Text for command 1", "Text for command 2"], ["Text for command 3"]]
// guiOptions: Some options how to display this endpoint
startBot(collectionName: string, token: string, keyboard?: any, guiOptions: CollectionGuiOptions<TelegramInputMessage> = {}): MessageCollection;

// Stop bot
// collectionName: identificator of the releasing collection object
releaseBot(collectionName: string);
```

### MessageCollection

Presents all chat bot messages.

Methods:

```typescript
// Start reciving of all users messages
list(): Observable<T>;

// Stop reciving of user messages
async stop();

// Pushes message to the chat
// value: Message in TelegramInputMessage type
// chatId: id of the destination chat, get it from input user messages
// message: Message to send
async push(value: TelegramInputMessage);
async push(chatId: string, message: string);

// Update keyboard structure to specified
// keyboard: JSON keyboard description, constructor for detailes
setKeyboard(keyboard: any)
```

Example:

```typescript
const etl = require('rxjs-etl-kit');

const telegram = new etl.TelegramEndpoint();
const bot = telegram.startBot('bot 1', '**********');

const startTelegramBot$ = bot.list().pipe(
    etl.log(),          // log user messages to the console
    etl.push(bot)  // echo input message back to the user
);

etl.run(startTelegramBot$);
```

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### IntervalEndpoint

This endpoint is analog of **RxJs** interval() operator, with GUI support. It emits simple counter, which increments every interval.

Methods:

```typescript
// Create new interval collection object
// collectionName: identificator of the creating collection object
// interval: Time interval in milliseconds between two emitted values 
// guiOptions: Some options how to display this endpoint
getSequence(collectionName: string, interval: number, guiOptions: CollectionGuiOptions<number> = {}): IntervalCollection;

// Stop interval
// collectionName: identificator of the releasing collection object
releaseSequence(collectionName: string);
```

### IntervalCollection

Methods:

```typescript
// Start interval generation, create observable and emit counter of intervals to it
list(): Observable<number>;

// Stop endpoint reading
async stop();

// Set value of interval counter
// value: new value of the interval counter
async push(value: number);

// Set interval counter to 0
async clear();
```

Example:

```typescript
const etl = require('rxjs-etl-kit');

const timer = new etl.IntervalEndpoint();
const seq = new etl.getSequence('every 500 ms', 500);

const startTimer$ = seq.list().pipe(
    etl.log()          // log counter
);

etl.run(startTimer$);
```

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Operators

Apart from operators from this library, you can use any operators of **RxJs** library.

### run

This function runs one or several streams and return promise to waiting when all streams are complites.

```typescript
const etl = require('rxjs-etl-kit');

let memory = etl.MemoryEndpoint();
let buffer = memory.getBuffer('test buffer', [1, 2, 3, 4, 5]);

let stream$ = buffer.list().pipe(
    etl.log()
);

etl.run(stream$);
```

### log

<a name="log" href="#log">#</a> etl.<b>log</b>([<i>options</i>])

Prints the value from the stream to the console.

Example

```typescript
const rx = require('rxjs');
const etl = require('rxjs-etl-kit');

let stream$ = rx.interval(1000).pipe(
    etl.log()
);

etl.run(stream$);
```

### where

<a name="where" href="#where">#</a> etl.<b>where</b>([<i>options</i>])

This operator is analog of **where** operation in SQL and is synonym of the **filter** operator from the **RxJS** library - but with improvements. It cat skip some values from the input stream by the specified condition. You can specify predicate function to determine filter conditions or you can specify map object as condition (like typeorm 'where' parameter in find() method).

Example

```typescript
const rx = require('rxjs');
const etl = require('rxjs-etl-kit');

let stream$ = rx.interval(1000).pipe(
    etl.where(v => v % 2 === 0),
    etl.log()
);
etl.run(stream$);
```

### push

<a name="push" href="#push">#</a> etl.<b>push</b>([<i>options</i>])

This operator call the **Endpoint.push** method to push value from stream to the specified endpoint.

Example

```typescript
const rx = require('rxjs');
const etl = require('rxjs-etl-kit');

let csv = etl.CsvEndpoint();
let dest = csv.getFile('test.csv');

let stream$ = rx.interval(1000).pipe(
    etl.push(dest)
);

etl.run(stream$);
```

### numerate

<a name="numerate" href="#numerate">#</a> etl.<b>numerate</b>([<i>options</i>])

This operator enumerate input values and add index field to value if it is object or index column if value is array. If the input stream values is objects, you should specify index field name as the second parameter of operator.

Example

```typescript
const etl = require('rxjs-etl-kit');

let csv = etl.CsvEndpoint();
let src = csv.getFile('test.csv');

let stream$ = src.list().pipe(
    etl.numerate(10), // 10 is the first value for numeration
    etl.log()
);

etl.run(stream$);
```

### addField

<a name="numerate" href="#numerate">#</a> etl.<b>addField</b>([<i>options</i>])

This operator applicable to the stream of objects. It calculate callback function and add result as new field to the input stream value.

Example

```typescript
const etl = require('rxjs-etl-kit');

const pg = etl.PostgresEndpoint('postgres://user:password@127.0.0.1:5432/database');
const table = pg.getTable('users');

const logUsers$ = table.list().pipe(
    etl.addField('NAME_IN_UPPERCASE', value => value.name.toUpperCase()),
    etl.log()
);

etl.run(logUsers$);
```

### addColumn

<a name="numerate" href="#numerate">#</a> etl.<b>addColumn</b>([<i>options</i>])

This operator applicable to the stream of arrays. It calculate callback function and add result as a new column to the input stream value.

Example

```typescript
const etl = require('rxjs-etl-kit');

let csv = etl.CsvEndpoint();
let src = csv.getFile('test.csv');

const stream$ = src.list().pipe(
    etl.addColumn(value => value[2].toUpperCase()), 
    etl.log()
);

etl.run(stream$);
```

### join

<a name="join" href="#join">#</a> etl.<b>join</b>([<i>options</i>])

This operator is analog of join operation in SQL. It takes the second input stream as the parameter, and gets all values from this second input stream for every value from the main input stream. Then it merges both values to one object (if values are objects) or to one array (if at least one of values are array), and put the result value to the main stream.

Example

```typescript
const etl = require('rxjs-etl-kit');

let csv = etl.CsvEndpoint();
let src = csv.getFile('test.csv');

let mem = etl.MemoryEndpoint();
let buffer = mem.getBuffer('buffer 1', [1, 2, 3, 4, 5]);

let stream$ = src.list().pipe(
    etl.join(buffer),
    etl.log()
);

etl.run(stream$);
```

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Misc

### GoogleTranslateHelper

This class help you to use Google translate service.

```typescript
const { CsvEndpoint, GoogleTranslateHelper, log, run } = require("rxjs-etl-kit");
const { mergeMap } = require("rxjs");

let csv = CsvEndpoint();
let src = csv.getFile('products.csv');

const translator = new GoogleTranslateHelper(process.env.GOOGLE_CLOUD_API_KEY!, 'en', 'ru');

let translateProducts$ = src.list().pipe(
    translator.operator(),
    log()
);
await run(translateProducts$);
 ```

### Header

This class can store array of column names and convert object to array or array to object representation.

```typescript
const { PostgresEndpoint, CsvEndpoint, Header, log, push, run } = require("rxjs-etl-kit");
const { map } = require("rxjs");

const pg = new PostgresEndpoint("postgres://user:password@127.0.0.1:5432/database");
const source = pg.getTable("users");

let csv = CsvEndpoint();
const dest = csv.getFile("users.csv");
const header = new Header("id", "name", "login", "email");

let sourceToDest$ = source.list().pipe(
    map(v => header.objToArr(v)),
    push(dest)
);
await run(sourceToDest$);
 ```

### Utility functions

This functions implements some useful things to manipulate data.

```typescript
// Join url parts (or path parts) to full url (or path) with delimeter
function pathJoin(parts: string[], sep: string = '/'): string;
// Get object part by json path
function getByJsonPath(obj: {}, jsonPath?: string): any;
// Get child element of array or object by element property value
function getChildByPropVal(obj: {}, propName: string, propVal?: any): any;
// Convert object to string
function dumpObject(obj: any, deep: number = 1): string;
 ```

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# License

This library is provided with [MIT](LICENSE) license. 

