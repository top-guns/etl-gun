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
* [Why / when would I need this?](#whywhen-would-i-need-this)
* [Installation](#installation)
* [Usage](#usage)
* [Features](#features)
* [Concept](#concept)
    * [Extract](#extract)
    * [Transform](#transform)
    * [Load](#load)
    * [Chaining](#chaining)
* [API Reference](#api-reference)
    * [Endpoints](#databases)
        * [BufferEndpoint](#bufferendpoint)
        * [CsvEndpoint](#csvendpoint)
        * [JsonEndpoint](#jsonendpoint)
        * [XmlEndpoint](#xmlendpoint)
        * [PostgresEndpoint](#postgresendpoint)
    * [Operators](#operators)
        * [run](#run)
        * [log](#log)
        * [where](#where)
        * [push](#push)
        * [numerate](#numerate)
        * [join](#join)
    * [Misc](#misc)
        * [Header](#header)
- [License](#license)

---

# Why / when would I need this?

**RxJs-ETL-Kit** is a simple **ETL glue** represented as an extention to the **RxJs** library. 
Typically, you'd use **RxJs-ETL-Kit** to help with your existing ETL processes. It can extract data from the one or more sources, transform it and load to one or more destinations in nedded order.

You can use javascript and typescript to use it.

**RxJs-ETL-Kit** will **NOT** help you with "big data" - it executes on the one computer and is not supports clustering.

Here's somne ways to use it:

1. Read some data from database and export it to the .csv file
2. Run some queries in database
3. Do some processing the files (e.g. rename or even filter content of some files)

You can find many examples of using **RxJs-ETL-Kit** in the API section of this readme file.

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
const { PostgresEndpoint, CsvEndpoint, Header, log, push, run } = require('rxjs-etl-kit');
const { map } = require('rxjs');

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

* Extract data from the different source endpoints, for example PostgreSql, .csv-files
* Transform data with **RxJs** and **RxJs-ETL-Kit** operators
* Load data to the different destination endpoints, for example PostgreSql, .csv-files
* Create pipelines of data extraction, transformation and loading, and run this pipelines in needed order

---

# Concept

Using of this library should consists of 3 steps:

1. Define your endpoints for sources and destinations
2. Define data transformation pipelines using **pipe** operator of input streams of your source endpoints
3. Run transformation pipelines in needed order and wait for completion

## Extract

Data extraction from the source endpoint performs with **read** endpoint method, which returns a stream.

## Transform

Use any **RxJs** and **RxJs-ETL-Kit** operators inside **pipe** method of the input stream to transform the input data.

To complex data transformation you can use the **BufferEndpoint** class, which can store data and have **forEach** and some other methods to manipolate with data in it.

## Load

Loading of data to the destination endpoint performs with **push** operator.

## Chaining

Chaning of data transformation performs with **pipe** method of the input data stream.

Chaning of several streams performs by using **await** under the **run** procedure.

---

# API Reference

## Endpoints

### BufferEndpoint

<a name="buffer" href="#buffer">#</a> etl.<b>BufferEndpoint</b>([<i>options</i>])

Buffer to store values in memory and perform complex operations on it.

Example

```js
const etl = require('rxjs-etl-kit');

let csv = etl.CsvEndpoint('test.csv');
let buffer = etl.BufferEndpoint();

let scvToBuffer$ = csv.read().pipe(
    etl.push(buffer)
);
let bufferToCsv$ = buffer.read().pipe(
    etl.push(csv)
);

await etl.run(scvToBuffer$);
buffer.sort((row1, row2) => row1[0] > row2[0]);
csv.clear();

etl.run(bufferToCsv$)
```

### CsvEndpoint

<a name="csv" href="#csv">#</a> etl.<b>CsvEndpoint</b>([<i>options</i>])

Parses source csv file into individual records or write record to the end of destination csv file. Every record is csv-string and presented as array of values.

Example

```js
const etl = require('rxjs-etl-kit');

let csv = etl.CsvEndpoint('test.csv');

let logCsvRows$ = csv.read().pipe(
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
JsonEndpoint(filename: string, autosave?: boolean, autoload?: boolean, encoding?: BufferEncoding);
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

let json = etl.JsonEndpoint('test.json');

let printJsonBookNames$ = json.read('store.book').pipe(
    tap(book => console.log(book.name))
);

let printJsonAuthors$ = json.readByJsonPath('$.store.book[*].author', {searchReturns: 'foundedOnly', addRelativePathAsField: "path"}).pipe(
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

let xml = etl.XmlEndpoint('test.xml');

let printXmlAuthors$ = json.read('/store/book/author').pipe(
    map(v => v.firstChild.nodeValue),
    etl.log()
);

await etl.run(printXmlAuthors$);
```

### PostgresEndpoint

<a name="postgres" href="#postgres">#</a> etl.<b>PostgresEndpoint</b>([<i>options</i>])

Presents the table from the PostgreSQL database. 
Connection to the database can be performed using connection string or through existing pool.

Example

```js
const etl = require('rxjs-etl-kit');

let table = etl.PostgresEndpoint('users', "postgres://user:password@127.0.0.1:5432/database");

let logUsers$ = csv.read().pipe(
    etl.log()
);

etl.run(logUsers$)
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
const { PostgresEndpoint, CsvEndpoint, Header, log, push, run } = require('rxjs-etl-kit');
const { map } = require('rxjs');

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

