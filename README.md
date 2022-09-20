# RxJs-ETL-Kit

RxJs-ETL-Kit is a platform that employs RxJs observables, allowing developers to build stream-based ETL (Extract, Transform, Load) pipelines complete with buffering and bulk-insertions.

[![NPM Version][npm-image]][npm-url]
[![NPM Downloads][downloads-image]][downloads-url]
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[npm-image]: https://img.shields.io/npm/v/rxjs-etl-kit.svg
[npm-url]: https://npmjs.org/package/rxjs-etl-kit
[downloads-image]: https://img.shields.io/npm/dm/rxjs-etl-kit.svg
[downloads-url]: https://npmjs.org/package/rxjs-etl-kit

---

### Table of Contents
* [Why / when would I need this?](#why--when-would-i-need-this)
* [Installation](#installation)
* [Usage](#usage)
* [Features](#features)
* [Concept](#concept)
* [API Reference](#api-reference)
    * [Core](#core)
        * [Endpoint](#endpoint)
    * [Endpoints](#endpoints)
        * [BufferEndpoint](#bufferendpoint)
        * [FilesystemEndpoint](#filesystemendpoint)
        * [CsvEndpoint](#csvendpoint)
        * [JsonEndpoint](#jsonendpoint)
        * [XmlEndpoint](#xmlendpoint)
        * [PostgresEndpoint](#postgresendpoint)
        * [TelegramEndpoint](#telegramendpoint)
    * [Operators](#operators)
        * [run](#run)
        * [log](#log)
        * [where](#where)
        * [push](#push)
        * [numerate](#numerate)
        * [addField](#addfield)
        * [join](#join)
    * [Misc](#misc)
        * [Header](#header)
- [License](#license)

---

# Why / when would I need this?

**RxJs-ETL-Kit** is a simple **ETL glue** represented as an extention to the **RxJs** library. 
Typically, you'd use **RxJs-ETL-Kit** to help with ETL processes. It can extract data from the one or more sources, transform it and load to one or more destinations in nedded order.

You can use javascript and typescript with it.

**RxJs-ETL-Kit** will **NOT** help you with "big data" - it executes on the one computer and is not supports clustering.

Here's some ways to use it:

1. Read some data from database and export it to the .csv file and vice versa
2. Create file converters
3. Filter or sort content of some files
4. Run some queries in database
5. Create Telegram bots with [TelegramEndpoint](#telegramendpoint)

You can find many examples of using **RxJs-ETL-Kit** in the API Reference section of this file.

---

# Installation

```
npm install rxjs-etl-kit
```
or
```
yarn add rxjs-etl-kit
```

---

# Usage

Require the RxJs-ETL-Kit library in the desired file to make it accessible.

Introductory example: postgresql -> .csv
```js
const { PostgresEndpoint, CsvEndpoint, Header, log, push, run } = require("rxjs-etl-kit");
const { map } = require("rxjs");

const source = new PostgresEndpoint("users", "postgres://user:password@127.0.0.1:5432/database");
const dest = new CsvEndpoint("users.csv");
const header = new Header(["id", "name", "login", "email"]);

const sourceToDest$ = source.read().pipe(
    log(),
    map(v => header.objToArr(v)),
    push(dest)
);

await run(sourceToDest$);
 ```

---

# Features

* Simple way to use, consists of 3 steps: 1) Endpoints creation 2) Piplines creation 3) Run piplines in needed order
* You can use javascript and typescript with **RxJs-ETL-Kit**, which writen in typescript itself
* Fully compatible with **RsJs** library, it's observables, operators etc.
* Extract data from the different source endpoints, for example PostgreSql, csv, json, xml
* Transform data with **RxJs** and **RxJs-ETL-Kit** operators and any custom js handlers via **map** operator for example
* Load data to the different destination endpoints, for example PostgreSql, csv, json, xml
* Create pipelines of data extraction, transformation and loading, and run this pipelines in needed order
* Working with any type of data, including hierarchical data structures (json, xml)
* With endpoint events mechanism you can handle different stream events, for example stream start/end, errors and other (see [Endpoint](#endpoint))
* You can create Telegram bots with [TelegramEndpoint](#telegramendpoint)

---

# Concept

**RxJs-ETL-Kit** contains several main concepts: 
* Endpoints - sources and destinations of data
* Piplines (or streams) - routs of data transformation and delivery, based on **RxJs** streams

Using of this library consists of 3 steps:

1. Define your endpoints for sources and destinations
2. Define data transformation pipelines using **pipe()** method of input streams of your source endpoints
3. Run transformation pipelines in needed order and wait for completion

ETL process:

* **Extract**: Data extraction from the source endpoint performs with **read()** endpoint method, which returns a **RxJs** stream
* **Transform**: Use any **RxJs** and **RxJs-ETL-Kit** operators inside **pipe()** method of the input stream to transform the input data. To complex data transformation you can use the **BufferEndpoint** class, which can store data and have **forEach()** and some other methods to manipulate with data in it
* **Load**: Loading of data to the destination endpoint performs with **push()** operator

Chaining:

Chaning of data transformation performs with **pipe()** method of the input data stream. 
Chaning of several streams performs by using **await** with **run()** procedure.

---

# API Reference

## Core

### Endpoint

Base class for all endpoints. Declares public interface of endpoint and implements event mechanism.

Methods:

```js
// Read data from the endpoint and create data stream to process it
read(): Observable<any>;

// Add value to the data of endpoint (usually to the end of stream)
// value: what will be added to the endpoint data
async push(value: any);

// Clear data of the endpoint
async clear();

// Add listener of specified event
// event: which event we want to listen, see below
// listener: callback function to handle events
on(event: EndpointEvent, listener: (...data: any[]) => void);
```

Types:

```js
export type EndpointEvent = 
    "read.start" |  // fires at the start of stream
    "read.end" |    // at the end of stream
    "read.data" |   // for every data value in the stream 
    "read.error" |  // on error
    "read.skip" |   // when the endpoint skip some data 
    "read.up" |     // when the endpoint go to the parent element while the tree data processing
    "read.down" |   // when the endpoint go to the child element while the tree data processing
    "push" |        // when data is pushed to the endpoint
    "clear";        // when the Endpoint.clear method is called
```

## Endpoints

### BufferEndpoint

Buffer to store values in memory and perform complex operations on it. 
This is a generic class and you can specify type of data which will be stored in the endpoint.

Constructor:

```js
// values: You can specify start data which will be placed to endpoint buffer
constructor(...values: T[]);
```

Methods:

```js
// Create the observable object and send data from the buffer to it
read(): Observable<T>;

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

```js
const etl = require('rxjs-etl-kit');

const csv = etl.CsvEndpoint('test.csv');
const buffer = etl.BufferEndpoint();

const scvToBuffer$ = csv.read().pipe(
    etl.push(buffer)
);
const bufferToCsv$ = buffer.read().pipe(
    etl.push(csv)
);

await etl.run(scvToBuffer$);

buffer.sort((row1, row2) => row1[0] > row2[0]);
csv.clear();

etl.run(bufferToCsv$)
```

### FilesystemEndpoint

Search for files and folders by standart unix shell wildcards [see glob documentation](https://www.npmjs.com/package/glob) for details.

Constructor:

```js
// rootFolderPath: full or relative path to the folder for search
constructor(rootFolderPath: string);
```

Methods:

```js
// Create the observable object and send files and folders information to it
// mask: search path mask in glob format (see glob documentation)
//       for example:
//       *.js - all js files in root folder
//       **/*.png - all png files in root folder and subfolders
// options: Search options, see below
read(mask: string = '*', options?: ReadOptions): Observable<string[]>;

// Create folder or file
// pathDetails: Information about path, which returns from read() method
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

```js
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

```js
const etl = require('rxjs-etl-kit');
const rx = require('rxjs');

const fs = new etl.FilesystemEndpoint('.');

const printAllJsFileNames$ = fs.read('**/*.js').pipe(
    rx.map(v => v.name)
    etl.log()
);

etl.run(printAllJsFileNames$)
```

### CsvEndpoint

Parses source csv file into individual records or write record to the end of destination csv file. Every record is csv-string and presented by array of values.

Constructor:

```js
// filename: full or relative name of the csv file
// delimiter: delimiter of values in one string of file data, equals to ',' by default
constructor(filename: string, delimiter?: string);
```

Methods:

```js
// Create the observable object and send file data to it string by string
// skipFirstLine: skip the first line in the file, useful for skip header
// skipEmptyLines: skip all empty lines in file
read(skipFirstLine: boolean = false, skipEmptyLines = false): Observable<string[]>;

// Add row to the end of file with specified value 
// value: what will be added to the file
async push(value: string[]);

// Clear the csv file
async clear();
```

Example:

```js
const etl = require('rxjs-etl-kit');

const csv = etl.CsvEndpoint('test.csv');

const logCsvRows$ = csv.read().pipe(
    etl.log()
);

etl.run(logCsvRows$)
```

### JsonEndpoint

Read and write json file with buffering it in memory. You can get objects from json by path specifing in JSONPath format or in lodash simple path manner (see logash 'get' function documentation).

Constructor:

```js
// filename: full or relative name of the json file
// autosave: save json from memory to the file after every change
// autoload: load json from the file to memory before every get or search operation
// encoding: file encoding
constructor(filename: string, autosave?: boolean, autoload?: boolean, encoding?: BufferEncoding);
```

Methods:

```js
// Find and send to observable child objects by specified path
// path: search path in lodash simple path manner
// jsonPath: search path in JSONPath format
// options: see below
read(path: string, options?: ReadOptions): Observable<any>;
readByJsonPath(jsonPath: string | string[], options?: ReadOptions): Observable<any>;

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

```js
type ReadOptions = {
    searchReturns?: 'foundedOnly'           // Default value, means that only search results objects will be sended to observable by the function
        | 'foundedImmediateChildrenOnly'    // Only the immidiate children of search results objects will be sended to observable 
        | 'foundedWithDescendants';         // Recursive send all objects from the object tree of every search result, including search result object itself

    addRelativePathAsField?: string;        // If specified, the relative path will be added to the sended objects as addRelativePathAsField field 
}
```

Example:

```js
const etl = require('rxjs-etl-kit');
const { tap } = require('rxjs');

const json = etl.JsonEndpoint('test.json');

const printJsonBookNames$ = json.read('store.book').pipe(
    tap(book => console.log(book.name))
);

const printJsonAuthors$ = json.readByJsonPath('$.store.book[*].author', {searchReturns: 'foundedOnly', addRelativePathAsField: "path"}).pipe(
    etl.log()
);

await etl.run(printJsonAuthors$, printJsonBookNames$);
```

### XmlEndpoint

<a name="xml" href="#xml">#</a> etl.<b>XmlEndpoint</b>(<i>filename, autosave?, autoload?, encoding?</i>)

Read and write XML document with buffering it in memory. You can get nodes from XML by path specifing in XPath format.

Example

```js
const etl = require('rxjs-etl-kit');
const { map } = require('rxjs');

const xml = etl.XmlEndpoint('test.xml');

const printXmlAuthors$ = xml.read('/store/book/author').pipe(
    map(v => v.firstChild.nodeValue),
    etl.log()
);

await etl.run(printXmlAuthors$);
```

### PostgresEndpoint

Presents the table from the PostgreSQL database. 
Connection to the database can be performed using connection string or through the existing pool.

Constructor:

```js
// table: Table name in database
// url: Connection string
// pool: You can specify the existing connection pool instead of new connection creation
constructor(table: string, url: string);
constructor(table: string, pool: any);
```

Methods:

```js
// Create the observable object and send data from the database table to it
// where: you can filter incoming data by this parameter
//        it can be SQL where clause 
//        or object with fields as collumn names 
//        and its values as needed collumn values
read(where: string | {} = ''): Observable<T>;

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

```js
const etl = require('rxjs-etl-kit');

const table = etl.PostgresEndpoint('users', 'postgres://user:password@127.0.0.1:5432/database');

const logUsers$ = table.read().pipe(
    etl.log()
);

etl.run(logUsers$)
```

### TelegramEndpoint

With this endpoint you can create telegram bots and chats with users. It can listen for user messages and send the response massages. It also can set the user keyboard for the chat.

Constructor:

```js
// token: Bot token
// keyboard: JSON keyboard description, see the node-telegram-bot-api for detailes
//           Keyboard example: [["Text for command 1", "Text for command 2"], ["Text for command 3"]]
constructor(token: string, keyboard?: any);
```

Methods:

```js
// Start bot, create observable and send all user messages to it
read(): Observable<T>;

// Stop bot
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

```js
const etl = require('rxjs-etl-kit');

const telegram = new etl.TelegramEndpoint('**********');

const startTelegramBot$ = telegram.read().pipe(
    etl.log(),          // log user messages to the console
    etl.push(telegram)  // echo input message back to the user
);

etl.run(startTelegramBot$);
```

## Operators

Apart from operators from this library, you can use any operators of **RxJs** library.

### run

This function runs one or several streams and return promise to waiting when all streams are complites.

```js
const etl = require('rxjs-etl-kit');

let buffer = etl.BufferEndpoint(1, 2, 3, 4, 5);

let stream$ = buffer.read().pipe(
    log()
);

etl.run(stream$)
```

### log

<a name="log" href="#log">#</a> etl.<b>log</b>([<i>options</i>])

Prints the value from the stream to the console.

Example

```js
const rx = require('rxjs');
const etl = require('rxjs-etl-kit');

let stream$ = rx.interval(1000).pipe(
    etl.log()
);

etl.run(stream$)
```

### where

<a name="where" href="#where">#</a> etl.<b>where</b>([<i>options</i>])

This operator is analog of **where** operation in SQL and is synonym of the **filter** operator from the **RxJS** library. It cat skip some values from the input stream by the specified condition.

Example

```js
const rx = require('rxjs');
const etl = require('rxjs-etl-kit');

let stream$ = rx.interval(1000).pipe(
    etl.where(v => v % 2 === 0),
    etl.log()
);
etl.run(stream$)
```

### push

<a name="push" href="#push">#</a> etl.<b>push</b>([<i>options</i>])

This operator call the **Endpoint.push** method to push value from stream to the specified endpoint.

Example

```js
const rx = require('rxjs');
const etl = require('rxjs-etl-kit');

let csv = etl.CsvEndpoint('test.csv');

let stream$ = rx.interval(1000).pipe(
    etl.push(csv)
);

etl.run(stream$)
```

### numerate

<a name="numerate" href="#numerate">#</a> etl.<b>numerate</b>([<i>options</i>])

This operator enumerate input values and add index field to value if it is object or index collumn if value is array.

Example

```js
const etl = require('rxjs-etl-kit');

let csv = etl.CsvEndpoint('test.csv');

let stream$ = csv.read().pipe(
    etl.numerate(),
    etl.log()
);

etl.run(stream$)
```

### addField

<a name="numerate" href="#numerate">#</a> etl.<b>addField</b>([<i>options</i>])

This operator calculate callback function from parameters and add result as new field to the input stream value (if it is object) or push result as new array item of input stream value (if it is array).

Example

```js
const etl = require('rxjs-etl-kit');

const table = etl.PostgresEndpoint('users', 'postgres://user:password@127.0.0.1:5432/database');

const logUsers$ = table.read().pipe(
    addField('NAME_IN_UPPERCASE', value => value.name.toUpperCase()),
    etl.log()
);

etl.run(logUsers$)
```

### join

<a name="join" href="#join">#</a> etl.<b>join</b>([<i>options</i>])

This operator is analog of join operation in SQL. It takes the second input stream as the parameter, and gets all values from this second input stream for every value from the main input stream. Then it merges both values to one object (if values are objects) or to one array (if at least one of values are array), and put the result value to the main stream.

Example

```js
const etl = require('rxjs-etl-kit');

let csv = etl.CsvEndpoint('test.csv');
let buffer = etl.BufferEndpoint(1, 2, 3, 4, 5);

let stream$ = csv.read().pipe(
    etl.join(buffer),
    log()
);

etl.run(stream$)
```

## Misc

### Header

This class can store array of column names and convert object to array or array to object representation..

```js
const { PostgresEndpoint, CsvEndpoint, Header, log, push, run } = require("rxjs-etl-kit");
const { map } = require("rxjs");

const source = new PostgresEndpoint("users", "postgres://user:password@127.0.0.1:5432/database");
const dest = new CsvEndpoint("users.csv");
const header = new Header(["id", "name", "login", "email"]);

let sourceToDest$ = source.read().pipe(
    map(v => header.objToArr(v)),
    push(dest)
);
await run(sourceToDest$);
 ```

# License

This library is provided with [MIT](LICENSE) license. 

