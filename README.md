# RxJs-ETL-Kit

<img src="https://github.com/igor-berezhnoy/rxjs-etl-kit/raw/main/static/ETL.png" alt="Logo" title="Logo" style="max-width: 100%;">

RxJs-ETL-Kit is a platform that employs RxJs observables, allowing developers to build stream-based ETL (Extract, Transform, Load) pipelines complete with buffering and bulk-insertions.

[![npm package](https://nodei.co/npm/rxjs-etl-kit.png?downloads=true&downloadRank=true&stars=true)](https://nodei.co/npm/rxjs-etl-kit/)

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
    * [Core classes](#core)
        * [BaseCollection](#basecollection) (base class for all collections)
        * [Errors](#errors) (queues to store and process errors which occurred in other endpoints)
    * [Endpoints](#endpoints-and-its-collections)
        * [Auxiliary]() (service endpoints)
            * [Memory](#memory)
            * [Interval](#interval)
        * [Filesystems]()
            * [Filesystem](#filesystem)
        * [File formates]()
            * [Csv](#csv)
            * [Json](#json)
            * [Xml](#xml)
        * [Databases]()
            * [PostgreSQL](#postgres)
            * [MySQL](#mysql)
        * [CMS]()
            * [Magento](#magento)
        * [Task tracking systems]()
            * [Trello](#trello)
            * [Zendesk](#zendesk)
        * [Messangers]()
            * [Telegram](#telegram)
    * [Operators](#operators)
        * [run](#run)
        * [log](#log)
        * [expect](#expect)
        * [where](#where)
        * [push](#push)
        * [move](#move)
        * [copy](#copy)
        * [numerate](#numerate)
        * [addField](#addfield)
        * [addColumn](#addcolumn)
        * [join](#join)
        * [mapAsync](#mapasync)
    * [Misc](#misc)
        * [GoogleTranslateHelper](#googletranslatehelper)
        * [HttpClientHelper](#httpclienthelper)
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
5. Create Telegram bots with [Telegram.Endpoint](#telegram)

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

Import the RxJs-ETL-Kit library in the desired file to make it accessible.

**Warning:** Since the version 2.0.4 this library is native [ESM](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Modules) and no longer provides a CommonJS export. If your project uses CommonJS, you will have to [convert to ESM](https://gist.github.com/sindresorhus/a39789f98801d908bbc7ff3ecc99d99c) or use the [dynamic `import()`](https://v8.dev/features/dynamic-import) function.

Introductory example of library using: postgresql -> .csv
```typescript
import { map } from "rxjs";
import { Csv, GuiManager, Header, Postgres, log, push, run } from "rxjs-etl-kit";

// If you want to view GUI, uncomment the next line of code
// new GuiManager();

// Step 1: endpoint creation
const postgres = new Postgres.Endpoint("postgres://user:password@127.0.0.1:5432/database");
const source = postgres.getTable('users');

const csv = new Csv.Endpoint('./dest-folder');
const dest = csv.getFile('users.scv');

const header = new Header("id", "name", "login", "email");

// Step 2: transformation streams creation
const sourceToDest$ = source.select().pipe(
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

* **Extract**: Data extraction from the source collection performs with **select()** method, which returns the **RxJs** stream
* **Transform**: Use any **RxJs** and **RxJs-ETL-Kit** operators inside **pipe()** method of the input stream to transform the input data. To complex data transformation you can use the **Memory.Endpoint** class, which can store data and which collections have **forEach()** and some other methods to manipulate with data in it
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
* You can create Telegram bots with [Telegram.Endpoint](#telegram) to control the ETL process for example
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
* You can select the log window with 'tab' and scroll it with up/down arrows

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Examples (how to)

### Export rows from Postgres table to csv-file (postgresql -> .csv)

```typescript
import { Postgres, Csv, Header, log, push, run } from "rxjs-etl-kit";
import { map } from "rxjs";

const postgres = new Postgres.Endpoint("postgres://user:password@127.0.0.1:5432/database");
const source = postgres.getTable('users');

const csv = new Csv.Endpoint('./dest-folder');
const dest = csv.getFile('users.scv');

const header = new Header("id", "name", "login", "email");

const sourceToDest$ = source.select().pipe(
    log(),
    map(v => header.objToArr(v)),
    push(dest)
);

await run(sourceToDest$);
 ```

 ### Sort rows in csv-file by the first column (.csv -> .csv)

```typescript
import * as etl from "rxjs-etl-kit";

const csvEndpoint = new etl.Csv.Endpoint();
const csv = csvEndpoint.getFile('users.scv');
const memory = new etl.Memory.Endpoint();
const buffer = memory.getBuffer('buffer 1');

const scvToBuffer$ = csv.select().pipe(
    etl.push(buffer)
);
const bufferToCsv$ = buffer.select().pipe(
    etl.push(csv)
);

await etl.run(scvToBuffer$);

buffer.sort((row1, row2) => row1[0] > row2[0]);
csv.delete();

await etl.run(bufferToCsv$)
 ```

 ### Create telegram bot with translation functionality

 ```typescript
import * as etl from "rxjs-etl-kit";

const telegram = new etl.Telegram.Endpoint();
const bot = telegram.startBot('bot 1', process.env.TELEGRAM_BOT_TOKEN!);
const translator = new etl.GoogleTranslateHelper(process.env.GOOGLE_CLOUD_API_KEY!, 'en', 'ru');

const startTelegramBot$ = bot.select().pipe(
    etl.log(),          // log user messages to the console
    translator.operator([], [message]), // translate 'message' field
    etl.push(bot)  // echo input message back to the user
);

etl.run(startTelegramBot$);
```

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# API Reference

## Core

### BaseCollection

Base class for all collections. Declares public interface of collection and implements event mechanism.

Methods:

```typescript
// Read elements from the collection and create data stream to process it
// where: condition of the element selection
select(where?: any): Observable<any>;

// Add value to the collection (usually to the end of stream)
// value: what will be added to the collection
async insert(value: any);

// Update collection elements
// where: condition of the element selection
// value: what will be added to the collection
async update(where: any, value: any);

// Clear data of the collection
// where: condition of the element selection
async delete(where?: any);

// Add listener of specified event
// event: which event we want to listen, see below
// listener: callback function to handle events
on(event: CollectionEvent, listener: (...data: any[]) => void);

// Readable/writable property wich contains errors collection instance for this collection
errors: Errors.ErrorsQueue;

// Calls select() method of errors collection
selectErrors(stopOnEmpty: boolean = false): BaseObservable<EtlError>;
```

Types:

```typescript
export type CollectionEvent = 
    "select.start" |  // fires at the start of stream
    "select.end" |    // at the end of stream
    "select.recive" | // for every data value in the stream 
    "select.error" |  // on error
    "select.skip" |   // when the collection skip some data 
    "select.up" |     // when the collection go to the parent element while the tree data processing
    "select.down" |   // when the collection go to the child element while the tree data processing
    "pipe.start" |    // when processing of any collection element was started
    "pipe.end" |      // when processing of one collection element was ended
    "insert" |        // when data is inserted to the collection
    "update" |        // when data is updated in the collection
    "delete";        // when data is deleted from the collection
```

## Endpoints and it's collections

------------------------------------------------------------------------------------------------------------------------------------------------------------------------


### Errors

Store and process etl errors. Every collection by default has errors property wich contains collection of errors from collection etl process. You can cancel default errors collection creation for any collection, and specify your own manualy created error collection.

#### Endpoint

```typescript
// Creates new errors collection
// collectionName: identificator of the creating collection object
// guiOptions: Some options how to display this endpoint
getCollection(collectionName: string, options: CollectionOptions<EtlError> = {}): ErrorsQueue;

// Release errors collection object
// collectionName: identificator of the releasing collection object
releaseCollection(collectionName: string);
```

#### ErrorsQueue

Queue in memory to store etl errors and process thea. Should be created with **getCollection** method of **Errors.Endpoint**

Methods:

```typescript
// Create the observable object and send errors data from the queue to it
// stopOnEmpty: is errors processing will be stopped when the queue is empty
select(stopOnEmpty: boolean = false): BaseObservable<EtlError>;

// Pushes the error to the queue 
// error: what will be added to the queue
async insert(error: EtlError);

// Clear queue
async delete();
```

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Memory

Create and manipulate with collections of objects in memory.

#### Endpoint

```typescript
// Creates new memory buffer. This is a generic method so you can specify type of data which will be stored in
// collectionName: identificator of the creating collection object
// values: initial data
// guiOptions: Some options how to display this endpoint
getBuffer<T>(collectionName: string, values: T[] = [], guiOptions: CollectionGuiOptions<T> = {}): BufferCollection;

// Release buffer data
// collectionName: identificator of the releasing collection object
releaseBuffer(collectionName: string);

getQueue<T>(collectionName: string, values: T[] = [], guiOptions: CollectionGuiOptions<T> = {}): QueueCollection;
releaseQueue(collectionName: string);
```

#### BufferCollection

Buffer to store values in memory and perform complex operations on it. Should be created with **getBuffer** method of **MemoryEndpoint**

Methods:

```typescript
// Create the observable object and send data from the buffer to it
select(): Observable<T>;

// Pushes the value to the buffer 
// value: what will be added to the buffer
async insert(value: T);

// Clear endpoint data buffer
async delete();

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
import * as etl from "rxjs-etl-kit";

const csvEndpoint = new etl.Csv.Endpoint();
const csv = csvEndpoint.getFile('users.scv');
const memory = new etl.Memory.Endpoint();
const buffer = memory.getBuffer('buffer 1');

const scvToBuffer$ = csv.select().pipe(
    etl.push(buffer);
)
const bufferToCsv$ = buffer.select().pipe(
    etl.push(csv)
)

await etl.run(scvToBuffer$);

buffer.sort((row1, row2) => row1[0] > row2[0]);
csv.delete();

etl.run(bufferToCsv$)
```

#### QueueCollection

Queue to store values in memory and perform ordered processing of it. Should be created with **getQueue** method of **MemoryEndpoint**

Methods:

```typescript
// Create the observable object wich send process queue elements one by one and remove processed element from queue
// dontStopOnEmpty - do we need stop queue processing (unsubscribe) when the queue will be empty
// interval - pause between elements processing, in milliseconds
select(dontStopOnEmpty: boolean = false, interval: number = 0): Observable<T>;

// Pushes the value to the queue 
// value: what will be added to the queue
async insert(value: T);

// Clear queue
async delete();
```

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Filesystem

Search for files and folders with standart unix shell wildcards [see glob documentation for details](https://www.npmjs.com/package/glob).

#### Endpoint

Methods:

```typescript
// rootFolder: full or relative path to the folder of intetest
constructor(rootFolder: string);

// Creates new FilesystemCollection
// folderName: subfolder of the root folder and identificator of the creating collection object
// guiOptions: Some options how to display this endpoint
getFolder(folderName: string = '.', guiOptions: CollectionGuiOptions<PathDetails> = {}): Collection;

// Release FilesystemCollection
// folderName: identificator of the releasing collection object
releaseFolder(folderName: string);
```

#### Collection

Methods:

```typescript
// Create the observable object and send files and folders information to it
// mask: search path mask in glob format (see glob documentation)
//       for example:
//       *.js - all js files in root folder
//       **/*.png - all png files in root folder and subfolders
// options: Search options, see below
select(mask: string = '*', options?: ReadOptions): Observable<string[]>;

// Create folder or file
// pathDetails: Information about path, which returns from select() method
// filePath: File or folder path
// isFolder: Is it file or folder
// data: What will be added to the file, if it is a file, ignore for folders
async insert(pathDetails: PathDetails, data?: string | NodeJS.ArrayBufferView | Iterable<string | NodeJS.ArrayBufferView> | AsyncIterable<string | NodeJS.ArrayBufferView> | internal.Stream);
async insert(filePath: string, data?: string | NodeJS.ArrayBufferView | Iterable<string | NodeJS.ArrayBufferView> | AsyncIterable<string | NodeJS.ArrayBufferView> | internal.Stream, isFolder?: boolean);

// Clear the root folder by mask
// mask: Which files and folders we need to delete
// options: Search options, see below 
//          IMPORTANT! Be careful with option includeRootDir because if it is true, and the objectsToSearch is not 'filesOnly',
//          then the root folder will be deleted with all its content! Including folder itself.
async delete(mask: string = '*', options?: ReadOptions);
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
import * as etl from "rxjs-etl-kit";
import * as rxjs from "rxjs";

const fs = new etl.Filesystem.Endpoint('~');
const scripts = ep.getFolder('scripts');

const printAllJsFileNames$ = scripts.select('**/*.js').pipe(
    rx.map(v => v.name)
    etl.log()
);

etl.run(printAllJsFileNames$)
```

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Csv

Parses source csv file into individual records or write record to the end of destination csv file. Every record is csv-string and presented by array of values.

#### Endpoint

Methods:

```typescript
// Create collection object for the specified file
// filename: full or relative name of the csv file and identificator of the creating collection object
// delimiter: delimiter of values in one string of file data, equals to ',' by default
// guiOptions: Some options how to display this endpoint
getFile(filename: string, delimiter: string = ",", guiOptions: CollectionGuiOptions<string[]> = {}): Collection;

// Release collection object
// filename: identificator of the releasing collection object
releaseFile(filename: string);
```

#### Collection

Methods:

```typescript
// Create the observable object and send file data to it string by string
// skipFirstLine: skip the first line in the file, useful for skip header
// skipEmptyLines: skip all empty lines in file
select(skipFirstLine: boolean = false, skipEmptyLines = false): Observable<string[]>;

// Add row to the end of file with specified value 
// value: what will be added to the file
async insert(value: string[]);

// Clear the csv file
async delete();
```

Example:

```typescript
import * as etl from "rxjs-etl-kit";

const csv = new etl.Csv.Endpoint('~');
const testFile = csv.getFile('test.csv')

const logTestFileRows$ = testFile.select().pipe(
    etl.log()
);

etl.run(logTestFileRows$)
```

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Json

Read and write json file with buffering it in memory. You can get objects from json by path specifing in JSONPath format or in lodash simple path manner (see logash 'get' function documentation).

#### Endpoint

Methods:

```typescript
// Create collection object for the specified file
// filename: full or relative name of the json file and identificator of the creating collection object
// autosave: save json from memory to the file after every change
// autoload: load json from the file to memory before every get or search operation
// encoding: file encoding
// guiOptions: Some options how to display this endpoint
getFile(filename: string, autosave: boolean = true, autoload: boolean = false, encoding?: BufferEncoding, guiOptions: CollectionGuiOptions<number> = {}): Collection;

// Release collection object
// filename: identificator of the releasing collection object
releaseFile(filename: string);
```

#### Collection

Methods:

```typescript
// Find and send to observable child objects by specified path
// path: search path in lodash simple path manner
// jsonPath: search path in JSONPath format
// options: see below
select(path: string, options?: ReadOptions): Observable<any>;
selectByJsonPath(jsonPath: string | string[], options?: ReadOptions): Observable<any>;

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
async insert(value: any, path?: string, fieldname?: string);

// Clear the json file and write an empty object to it
async delete();

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
import * as etl from "rxjs-etl-kit";
import { tap } from "rxjs";

const json = new etl.Json.Endpoint('~');
const testFile = etl.getFile('test.json');

const printJsonBookNames$ = testFile.select('store.book').pipe(
    tap(book => console.log(book.name))
);

const printJsonAuthors$ = testFile.selectByJsonPath('$.store.book[*].author', {searchReturns: 'foundedOnly', addRelativePathAsField: "path"}).pipe(
    etl.log()
);

await etl.run(printJsonAuthors$, printJsonBookNames$);
```

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Xml

<a name="xml" href="#xml">#</a> etl.<b>XmlEndpoint</b>(<i>filename, autosave?, autoload?, encoding?</i>)

Read and write XML document with buffering it in memory. You can get nodes from XML by path specifing in XPath format.

#### Endpoint

Methods:

```typescript
// Create collection object for the specified file
// filename: full or relative name of the xml file and identificator of the creating collection object
// autosave: save xml from memory to the file after every change
// autoload: load xml from the file to memory before every get or search operation
// encoding: file encoding
// guiOptions: Some options how to display this endpoint
getFile(filename: string, autosave: boolean = true, autoload: boolean = false, encoding?: BufferEncoding, guiOptions: CollectionGuiOptions<string[]> = {}): Collection;

// Release collection object
// filename: identificator of the releasing collection object
releaseFile(filename: string);
```

#### Collection

Methods:

```typescript
// Find and send to observable child objects by specified xpath
// xpath: xpath to search
// options: see below
select(xpath: string = '', options: XmlReadOptions = {}): EtlObservable<Node>;

// Find and return child node by specified path
// xpath: search path
get(xpath: string = ''): XPath.SelectedValue

// If attribute is specified, the function find the object by xpath and add value as its attribute
// If attribute is not specified, the function find the node by xpath and push value as its child node
// value: what will be added to the xml
// xpath: where value will be added as child, specified in lodash simple path manner
// attribute: name of the attribute which value will be setted, 
//            and flag - is we add value as attribute or as node
async insert(value: any, xpath: string = '', attribute: string = '');

// Clear the xml file and write an empty object to it
async delete();

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
import * as etl from "rxjs-etl-kit";
import { map } from "rxjs";

const xml = new etl.Xml.Endpoint('/tmp');
const testFile = xml.getFile('test.xml');

const printXmlAuthors$ = testFile.select('/store/book/author').pipe(
    map(v => v.firstChild.nodeValue),
    etl.log()
);

await etl.run(printXmlAuthors$);
```

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Postgres

Represents PostgreSQL database. Based on [node-postgres (aka 'pg')](https://github.com/brianc/node-postgres) module.

#### Endpoint

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

// Release all collection objects, endpoint object and release connections to database.
releaseEndpoint();
```

#### TableCollection

Presents the table from the PostgreSQL database. 

Methods:

```typescript
// Create the observable object and send data from the database table to it
// where: you can filter incoming data by this parameter
//        it can be SQL where clause 
//        or object with fields as collumn names 
//        and its values as needed collumn values
select(where: string | {} = ''): Observable<T>;

// Insert value to the database table
// value: what will be added to the database
async insert(value: T);

// Update all rows in database table which match to the specified condition
// where: you can filter table rows to deleting by this parameter
//        it can be SQL where clause 
//        or object with fields as collumn names 
//        and its values as needed collumn values
// value: what will be set as new value for updated rows
async update(where: string | {} = ''value: T);

// Delete rows from the database table by condition
// where: you can filter table rows to deleting by this parameter
//        it can be SQL where clause 
//        or object with fields as collumn names 
//        and its values as needed collumn values
async delete(where: string | {} = '');
```

Example:

```typescript
import * as etl from "rxjs-etl-kit";

const pg = new etl.Postgres.Endpoint('postgres://user:password@127.0.0.1:5432/database');
const table = pg.getTable('users');

const logUsers$ = table.select().pipe(
    etl.log()
);

etl.run(logUsers$)
```

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Mysql

Represents MySQL database. Based on [mysql2-async](https://github.com/txstate-etc/mysql2-async) module.

#### Endpoint

Methods:

```typescript
// Connection to the database can be performed using connection string or through the existing pool.
constructor(connectionString: string);
constructor(connectionPool: Db.Pool);

// Create collection object for the specified database table
// table: name of database table and identificator of the creating collection object
// guiOptions: Some options how to display this endpoint
getTable(table: string, guiOptions: CollectionGuiOptions<string[]> = {}): TableCollection;

// Release collection object
// table: identificator of the releasing collection object
releaseTable(table: string);

// Release all collection objects, endpoint object and release connections to database.
releaseEndpoint();
```

#### TableCollection

Presents the table from the MySQL database. 

Methods:

```typescript
// Create the observable object and send data from the database table to it
// where: you can filter incoming data by this parameter
//        it can be SQL where clause 
//        or object with fields as collumn names 
//        and its values as needed collumn values
select(where: string | {} = ''): Observable<T>;

// Insert value to the database table
// value: what will be added to the database
async insert(value: T);

// Update all rows in database table which match to the specified condition
// where: you can filter table rows to deleting by this parameter
//        it can be SQL where clause 
//        or object with fields as collumn names 
//        and its values as needed collumn values
// value: what will be set as new value for updated rows
async update(where: string | {} = ''value: T);

// Delete rows from the database table by condition
// where: you can filter table rows to deleting by this parameter
//        it can be SQL where clause 
//        or object with fields as collumn names 
//        and its values as needed collumn values
async delete(where: string | {} = '');
```

Example:

```typescript
import * as etl from "rxjs-etl-kit";

const mysql = new etl.Postgres.Endpoint('mysql://user:password@localhost:3306/database');
const table = mysql.getTable('users');

const logUsers$ = table.select().pipe(
    etl.log()
);

etl.run(logUsers$)
```

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Magento

Presents Magento CMS objects.
Go to https://meetanshi.com/blog/create-update-product-using-rest-api-in-magento-2/ for details how to configure Magento integration to get access to it's API. 

#### Endpoint

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

#### ProductsCollection

Presents Magento CMS products. 

Methods:

```typescript
// Create the observable object and send product data from the Magento to it
// where: you can filter products by specifing object with fields as collumn names and it's values as fields values 
// fields: you can select which products fields will be returned (null means 'all fields') 
select(where: Partial<Product> = {}, fields: (keyof Product)[] = null): BaseObservable<Partial<Product>> ;

// Add new product to the Magento
// value: product fields values
async insert(value: NewProductAttributes);

// Upload image to the magento and set it as image of specified product and returns total count of images for this product
// product: product sku
// imageContents: binary form of the image file
// filename: name of the file in with magento will store the image
// label: label of the product image
// type: mime type of the image
async uploadImage(product: {sku: string} | string, imageContents: Blob, filename: string, label: string, type: "image/png" | "image/jpeg" | string): Promise<number>;
// Operator to upload product image from the pipe
uploadImageOperator<T>(func: (value: T) => {product: {sku: string} | string, imageContents: Blob, filename: string, label: string, type: "image/png" | "image/jpeg" | string}): OperatorFunction<T, T>;

// Utility static function to get products as array
static async getProducts(endpoint: Endpoint, where: Partial<Product> = {}, fields: (keyof Product)[] = null): Promise<Partial<Product>[]>;
```

Example:

```typescript
import * as etl from "rxjs-etl-kit";

const magento = new etl.Magento.Endpoint('https://magento.test', process.env.MAGENTO_LOGIN!, process.env.MAGENTO_PASSWORD!);
const products = magento.getProducts();

const logProductsWithPrice100$ = products.select({price: 100}).pipe(
    etl.log()
);

etl.run(logProductsWithPrice100$)
```

#### StockCollection

Presents Magento CMS stock items. Stock items - is products on stock.

Methods:

```typescript
// Create the observable object and send stock items data from the Magento to it
// sku, product: you can filter stock items by product attributes
select(sku: string): BaseObservable<StockItem>;
select(product: Partial<Product>): BaseObservable<StockItem>;

// Get stock item for specified product
// sku, product: product, wich stock items we need to get
public async getStockItem(sku: string): Promise<StockItem>;
public async getStockItem(product: {sku: string}): Promise<StockItem>;

// Update product stock quantity 
public async updateStockQuantity(sku: string, quantity: number);
public async updateStockQuantity(product: {sku: string}, quantity: number);
```

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Trello

Presents Trello task tracking system objects.
For details how to get API key and authorization token please read [Trello documentation](https://developer.atlassian.com/cloud/trello/guides/rest-api/api-introduction/).

#### Endpoint

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

// Create collection object for the Trello card comments
// cardId: card id
// collectionName: identificator of the creating collection object
// guiOptions: Some options how to display this endpoint
getCardComments(cardId: string, collectionName: string = 'Comments', guiOptions: CollectionGuiOptions<Partial<Comment>> = {}): CommentsCollection;

// Release collection data
// collectionName: identificator of the releasing collection object
releaseCollection(collectionName: string);
```

#### BoardsCollection

Presents Trello boards accessible by user which was specified while collection creation. 

Methods:

```typescript
// Create the observable object and send boards data from the Trello to it
// where (does not working now!): you can filter boards by specifing object with fields as collumn names and it's values as fields values 
// fields (does not working now!): you can select which board fields will be returned (null means 'all fields') 
select(where: Partial<Board> = {}, fields: (keyof Board)[] = null): EtlObservable<Partial<Board>>;

// Add new board to the Trello
// value: board fields values
async insert(value: Omit<Partial<Board>, 'id'>);

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

#### ListsCollection

Presents Trello lists on board which was specified while collection creation. 

Methods:

```typescript
// Create the observable object and send lists data from the Trello to it
// where (does not working now!): you can filter lists by specifing object with fields as collumn names and it's values as fields values 
// fields (does not working now!): you can select which list fields will be returned (null means 'all fields') 
select(where: Partial<List> = {}, fields: (keyof List)[] = null): EtlObservable<Partial<List>>;

// Add new list to the Trello
// value: list fields values
async insert(value: Omit<Partial<List>, 'id'>);

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

#### CardsCollection

Presents Trello cards in list which was specified while collection creation. 

Methods:

```typescript
// Create the observable object and send cards data from the Trello to it
// where (does not working now!): you can filter cards by specifing object with fields as collumn names and it's values as fields values 
// fields (does not working now!): you can select which card fields will be returned (null means 'all fields') 
select(where: Partial<Card> = {}, fields: (keyof Card)[] = null): EtlObservable<Partial<Card>>;

// Add new card to the Trello
// value: card fields values
async insert(value: Omit<Partial<Card>, 'id'>);

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

#### CommentsCollection

Presents Trello card comments in card which was specified while collection creation. 

Methods:

```typescript
// Create the observable object and send comments from the Trello to it
// where (does not working now!): you can filter comments by specifing object with fields as collumn names and it's values as fields values 
// fields (does not working now!): you can select which comment fields will be returned (null means 'all fields') 
select(where: Partial<Comment> = {}, fields: (keyof Comment)[] = null): EtlObservable<Partial<Comment>>;

// Add new comment to the Trello card
// text: comment text
async insert(text: string);

// Update comment fields values by comment id
// commentId: comment id
// value: new comment fields values as hash object
async update(commentId: string, value: Omit<Partial<Comment>, 'id'>);

// Get all comments
async get(): Promise<Comment[]>;

// Get comment by id
// commentId: card id
async get(commentId?: string): Promise<Comment>;
```

Example:

```typescript
import * as rx from 'rxjs';
import * as etl from 'rxjs-etl-kit';

const trello = new etl.Trello.Endpoint(process.env.TRELLO_API_KEY!, process.env.TRELLO_AUTH_TOKEN!);

const boards = trello.getUserBoards();
const board = await boards.getByBrowserUrl('https://trello.com/b/C9zegsyz/board1');

const lists = trello.getBoardLists(board.id);
const list = (await lists.get())[0];

const cards = trello.getListCards(list.id);

const addCommentToAllCards$ = cards.select().pipe(
    rx.tap(card => {
        const comments = trello.getBoardLists(card.id, 'cards');
        comments.push('New comment');
        trello.releaseCollection('cards');
    })
);

etl.run(addCommentToAllCards$)
```

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Zendesk

Presents Zendesk task tracking system objects.
For details how to get API key and authorization token please read [Zendesk documentation](https://developer.zendesk.com/api-reference/).

#### Endpoint

Methods:

```typescript
// zendeskUrl: Zendesk web url
// username: user login
// token: Zendesk authorization token
// rejectUnauthorized: You can set it to true to ignore ssl servificate problems while development.
constructor(zendeskUrl: string, username: string, token: string, rejectUnauthorized: boolean = true);

// Create collection object for the Zendesk tickets
// collectionName: identificator of the creating collection object
// guiOptions: Some options how to display this endpoint
getTickets(collectionName: string = 'Tickets', options: CollectionOptions<Partial<Ticket>> = {}): TicketsCollection;

// Create collection object for the Zendesk tickets fields
// collectionName: identificator of the creating collection object
// guiOptions: Some options how to display this endpoint
getTicketFields(collectionName: string = 'TicketFields', options: CollectionOptions<Partial<Field>> = {}): TicketFieldsCollection;

// Release collection data
// collectionName: identificator of the releasing collection object
releaseCollection(collectionName: string);
```

#### TicketsCollection

Presents all Zendesk tickets. 

Methods:

```typescript
// Create the observable object and send tickets data from the Zendesk to it
// where: you can filter tickets by specifing object with fields as collumn names and it's values as fields values 
select(where: Partial<Ticket> = {}): BaseObservable<Partial<Ticket>>;

// Add new ticket to the Zendesk
// value: ticket fields values
async insert(value: Omit<Partial<Ticket>, 'id'>);

// Update ticket fields values by ticket id
// ticketId: ticket id
// value: new ticket fields values as hash object
async update(ticketId: number, value: Omit<Partial<Ticket>, 'id'>);

// Get all tickets
async get(): Promise<Ticket[]>;

// Get ticket by id
// ticketId: ticket id
async get(ticketId: number): Promise<Ticket>;
```

#### TicketFieldsCollection

Presents all Zendesk tickets fields. 

Methods:

```typescript
// Create the observable object and send fields description data from the Zendesk to it
select(): BaseObservable<Partial<Field>>;

// Add new field to the Zendesk
// value: field attributes values
async insert(value: Omit<Partial<Field>, 'id'>);

// Update field attributes by field id
// fieldId: field id
// value: new field attributes values as hash object
async update(fieldId: number, value: Omit<Partial<Field>, 'id'>);

// Get all fields
async get(): Promise<Field[]>;

// Get field by id
// fieldId: field id
async get(fieldId: number): Promise<Field>;
```

Example:

```typescript
import * as rx from 'rxjs';
import * as etl from 'rxjs-etl-kit';

const zendesk = new etl.Zendesk.Endpoint(process.env.ZENDESK_URL!, process.env.ZENDESK_USERNAME!, process.env.ZENDESK_TOKEN!);
const tickets = zendesk.getTickets();

const PrintAllClosedTickets$ = tickets.select().pipe(
    etl.where({status: 'closed'}),
    etl.log()
)

etl.run(PrintAllClosedTickets$);
```

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Telegram

With this endpoint you can create telegram bots and chats with users. It can listen for user messages and send the response massages. 
It also can set the user keyboard for the chat.

#### Endpoint

Methods:

```typescript
// Start bot and return collection object for the bot messages
// collectionName: identificator of the creating collection object
// token: Bot token
// keyboard: JSON keyboard description, see the node-telegram-bot-api for detailes
//           Keyboard example: [["Text for command 1", "Text for command 2"], ["Text for command 3"]]
// guiOptions: Some options how to display this endpoint
startBot(collectionName: string, token: string, keyboard?: any, guiOptions: CollectionGuiOptions<TelegramInputMessage> = {}): Collection;

// Stop bot
// collectionName: identificator of the releasing collection object
releaseBot(collectionName: string);
```

#### Collection

Presents all chat bot messages.

Methods:

```typescript
// Start reciving of all users messages
select(): Observable<T>;

// Stop reciving of user messages
async stop();

// Pushes message to the chat
// value: Message in TelegramInputMessage type
// chatId: id of the destination chat, get it from input user messages
// message: Message to send
async insert(value: TelegramInputMessage);
async insert(chatId: string, message: string);

// Update keyboard structure to specified
// keyboard: JSON keyboard description, constructor for detailes
setKeyboard(keyboard: any)
```

Example:

```typescript
import * as etl from "rxjs-etl-kit";

const telegram = new etl.Telegram.Endpoint();
const bot = telegram.startBot('bot 1', '**********');

const startTelegramBot$ = bot.select().pipe(
    etl.log(),          // log user messages to the console
    etl.push(bot)  // echo input message back to the user
);

etl.run(startTelegramBot$);
```

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Interval

This endpoint is analog of **RxJs** interval() operator, with GUI support. It emits simple counter, which increments every interval.

#### Endpoint

Methods:

```typescript
// Create new interval collection object
// collectionName: identificator of the creating collection object
// interval: Time interval in milliseconds between two emitted values 
// guiOptions: Some options how to display this endpoint
getSequence(collectionName: string, interval: number, guiOptions: CollectionGuiOptions<number> = {}): Collection;

// Stop interval
// collectionName: identificator of the releasing collection object
releaseSequence(collectionName: string);
```

#### Collection

Methods:

```typescript
// Start interval generation, create observable and emit counter of intervals to it
select(): Observable<number>;

// Stop endpoint reading
async stop();

// Set value of interval counter
// value: new value of the interval counter
async insert(value: number);

// Set interval counter to 0
async delete();
```

Example:

```typescript
import * as etl from "rxjs-etl-kit";

const timer = new etl.Interval.Endpoint();
const seq = new etl.getSequence('every 500 ms', 500);

const startTimer$ = seq.select().pipe(
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
import * as etl from "rxjs-etl-kit";

let memory = new etl.Memory.Endpoint();
let buffer = memory.getBuffer('test buffer', [1, 2, 3, 4, 5]);

let stream$ = buffer.select().pipe(
    etl.log()
);

etl.run(stream$);
```

### log

<a name="log" href="#log">#</a> etl.<b>log</b>([<i>options</i>])

Prints the value from the stream to the console.

Example:

```typescript
import * as etl from "rxjs-etl-kit";
import * as rx from "rxjs";

let stream$ = rx.interval(1000).pipe(
    etl.log()
);

etl.run(stream$);
```

### expect

This function checks condition end if false - throws an error to the error collection.

```typescript
import * as etl from "rxjs-etl-kit";

let memory = new etl.Memory.Endpoint();
let buffer = memory.getBuffer('test buffer', [{count: 1}, {count: 2}]);

const errorsEndpoint = etl.Errors.getEndpoint();
const errors = errorsEndpoint.getCollection('all');

let stream$ = buffer.select().pipe(
    etl.expect('count = 1', { count: 1 }, errors),
    etl.expect('count one of [1,2,3]', { count: etl.VALUE.of([1,2,3]) }),
    etl.expect('count not > 3', { count: etl.VALUE.not['>'](3) }),
    etl.expect('count check function', { count: v => v < 5 }),
    etl.log()
);

etl.run(stream$);
```

### where

Variants:

```typescript
function where<T>(condition: Condition<T>): OperatorFunction<T, T>;
```

This operator is analog of **where** operation in SQL and is synonym of the **filter** operator from the **RxJS** library - but with improvements. It cat skip some values from the input stream by the specified condition. You can specify predicate function to determine filter conditions or you can specify map object as condition (like typeorm 'where' parameter in find() method).

Example:

```typescript
import * as etl from "rxjs-etl-kit";
import * as rx from "rxjs";

let stream$ = rx.interval(1000).pipe(
    etl.where(v => v % 2 === 0),
    etl.where('count = 1', { count: 1 }),
    etl.where('count one of [1,2,3]', { count: etl.VALUE.of([1,2,3]) }),
    etl.where('count not > 3', { count: etl.VALUE.not['>'](3) }),
    etl.log()
);
etl.run(stream$);
```

### push

<a name="push" href="#push">#</a> etl.<b>push</b>([<i>options</i>])

This operator call the **Endpoint.push** method to push value from stream to the specified endpoint.

Variants:

```typescript
// Push value to collection with no wait for result
function push<S, T=S>(collection: BaseCollection<T>, options?: PushOptions<S, T> | null): OperatorFunction<S, S>;

// Push value to collection with waiting for result
function pushAndWait<S, T=S>(collection: BaseCollection<T>, options?: PushOptions<S, T> | null): OperatorFunction<S, S>;

// Push value to collection, wait for result, send result to log
function pushAndLog<S, T=S>(collection: BaseCollection<T>, options?: PushOptions<S, T> | null): OperatorFunction<S, S>;

// Push value to collection, get the result and put it as stream value or as property of stream value
function pushAndGet<S, T, R>(collection: BaseCollection<T>, options?: PushOptions<S, T> & {toProperty: string} | null): OperatorFunction<S, R>;
```

Example:

```typescript
import * as etl from "rxjs-etl-kit";
import * as rx from "rxjs";

let csv = new etl.Csv.Endpoint();
let dest = csv.getFile('test.csv');

let stream$ = rx.interval(1000).pipe(
    etl.push(dest)
);

etl.run(stream$);
```

### move

<a name="numerate" href="#numerate">#</a> etl.<b>toProperty</b>([<i>options</i>])

This operator moves to specified property the whole stream value or it's property. Lodash paths is supported.

Example:

```typescript
import * as etl from "rxjs-etl-kit";

const memory = etl.Memory.getEndpoint();
const buf = memory.getBuffer<number>('buf', [1,2,3,4,5]);

let stream$ = src.select().pipe(
    etl.move<{ nn: number }>({to: 'nn'}), // 1 -> { nn: 1 }
    etl.move<{ num: number }>({from: 'nn', to: 'num'}), // { nn: 1 } -> { num: 1 }
    etl.copy<{ num: number, kk: {pp: number} }>('nn', 'kk.pp'), // { nn: 1 } -> { nn: 1, kk: {pp: 1} }

    etl.log()
);

etl.run(stream$);
```

### copy

<a name="numerate" href="#numerate">#</a> etl.<b>toProperty</b>([<i>options</i>])

This operator copy the specified property of the stream value to the another property. Lodash paths is supported.

Example:

```typescript
import * as etl from "rxjs-etl-kit";

const memory = etl.Memory.getEndpoint();
const buf = memory.getBuffer<number>('buf', [1,2,3,4,5]);

let stream$ = src.select().pipe(
    etl.move<{ nn: number }>({to: 'nn'}), // 1 -> { nn: 1 }
    etl.move<{ num: number }>({from: 'nn', to: 'num'}), // { nn: 1 } -> { num: 1 }
    etl.copy<{ num: number, kk: {pp: number} }>('nn', 'kk.pp'), // { nn: 1 } -> { nn: 1, kk: {pp: 1} }

    etl.log()
);

etl.run(stream$);
```

### numerate

<a name="numerate" href="#numerate">#</a> etl.<b>numerate</b>([<i>options</i>])

This operator enumerate input values and add index field to value if it is object or index column if value is array. If the input stream values is objects, you should specify index field name as the second parameter of operator.

Example:

```typescript
import * as etl from "rxjs-etl-kit";

let csv = new etl.Csv.Endpoint();
let src = csv.getFile('test.csv');

let stream$ = src.select().pipe(
    etl.numerate(10), // 10 is the first value for numeration
    etl.log()
);

etl.run(stream$);
```

### addField

<a name="numerate" href="#numerate">#</a> etl.<b>addField</b>([<i>options</i>])

This operator applicable to the stream of objects. It calculate callback function and add result as new field to the input stream value.

Example:

```typescript
import * as etl from "rxjs-etl-kit";

const pg = new etl.Postgres.Endpoint('postgres://user:password@127.0.0.1:5432/database');
const table = pg.getTable('users');

const logUsers$ = table.select().pipe(
    etl.addField('NAME_IN_UPPERCASE', value => value.name.toUpperCase()),
    etl.log()
);

etl.run(logUsers$);
```

### addColumn

<a name="numerate" href="#numerate">#</a> etl.<b>addColumn</b>([<i>options</i>])

This operator applicable to the stream of arrays. It calculate callback function and add result as a new column to the input stream value.

Example:

```typescript
import * as etl from "rxjs-etl-kit";

let csv = new etl.Csv.Endpoint();
let src = csv.getFile('test.csv');

const stream$ = src.select().pipe(
    etl.addColumn(value => value[2].toUpperCase()), 
    etl.log()
);

etl.run(stream$);
```

### join

<a name="join" href="#join">#</a> etl.<b>join</b>([<i>options</i>])

This operator is analog of join operation in SQL. It takes the second input stream as the parameter, and gets all values from this second input stream for every value from the main input stream. Then it merges both values to one object (if values are objects) or to one array (if at least one of values are array), and put the result value to the main stream.

Example:

```typescript
import * as etl from "rxjs-etl-kit";

let csv = new etl.Csv.Endpoint();
let src = csv.getFile('test.csv');

let mem = new etl.Memory.Endpoint();
let buffer = mem.getBuffer('buffer 1', [1, 2, 3, 4, 5]);

let stream$ = src.select().pipe(
    etl.join(buffer),
    etl.log()
);

etl.run(stream$);
```

### mapAsync

<a name="mapAsync" href="#mapAsync">#</a> etl.<b>mapAsync</b>([<i>options</i>])

This operator is analog of rxjs map operator for async callback function. It call and wait for callback and then use it's result as new stream item.

Example:

```typescript
import * as etl from "rxjs-etl-kit";

let mem = new etl.Memory.Endpoint();
let buffer = mem.getBuffer('urls', ['1.json', '2.json', '3.json']);

const mySite = new HttpClientHelper('http://www.mysite.com/jsons');

let stream$ = src.select().pipe(
    mapAsync(async (url) => await mySite.getJson(url)),
    etl.log()
);

etl.run(stream$);
```

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Misc

### GoogleTranslateHelper

This class help you to use Google translate service.

```typescript
import { Csv, GoogleTranslateHelper, log, run } from "rxjs-etl-kit";

let csv = new Csv.Endpoint();
let src = csv.getFile('products.csv');

const translator = new GoogleTranslateHelper(process.env.GOOGLE_CLOUD_API_KEY!, 'en', 'ru');

let translateProducts$ = src.select().pipe(
    translator.operator(),
    log()
);
await run(translateProducts$);
```

 ### HttpClientHelper

This class help you to make requests to http and https resources and get data from it.

Methods:

```typescript
    // Create helper object
    // baseUrl: this string will be used as start part of urls in any helper methods
    // headers: will be added to headers in all requests maden with this helper instance
    constructor(baseUrl?: string, headers?: Record<string, string>);

    // GET request

    async get(url?: string, headers?: Record<string, string>): Promise<Response>;
    async getJson(url?: string, headers?: Record<string, string>): Promise<any>;
    async getText(url?: string, headers?: Record<string, string>): Promise<string>;
    async getBlob(url?: string, headers?: Record<string, string>): Promise<Blob>;
    async getFileContents(url?: string, headers?: Record<string, string>): Promise<Blob>;

    getJsonOperator<T, R = T>(): OperatorFunction<T, R>;
    getJsonOperator<T, R = T>(url: string, toProperty?: string, headers?: Record<string, string>): OperatorFunction<T, R>;
    getJsonOperator<T, R = T>(getUrl: (value: T) => string, toProperty?: string, headers?: Record<string, string>): OperatorFunction<T, R>;

    getTextOperator<T>(): OperatorFunction<T, string>;
    getTextOperator<T>(url: string, toProperty?: string, headers?: Record<string, string>): OperatorFunction<T, string>;
    getTextOperator<T>(getUrl: (value: T) => string, toProperty?: string, headers?: Record<string, string>): OperatorFunction<T, string>;

    getBlobOperator<T>(): OperatorFunction<T, Blob>;
    getBlobOperator<T, R = T>(url: string, toProperty?: string, headers?: Record<string, string>): OperatorFunction<T, R>;
    getBlobOperator<T, R = T>(getUrl: (value: T) => string, toProperty?: string, headers?: Record<string, string>): OperatorFunction<T, R>;

    getFileContentsOperator<T>(): OperatorFunction<T, Blob>;
    getFileContentsOperator<T, R = T>(url: string, toProperty?: string, headers?: Record<string, string>): OperatorFunction<T, R>;
    getFileContentsOperator<T, R = T>(getUrl: (value: T) => string, toProperty?: string, headers?: Record<string, string>): OperatorFunction<T, R>;

    // POST request

    async post(body: string, url?: string, headers?: Record<string, string>): Promise<Response>;
    async postJson(body: any, url?: string, headers?: Record<string, string>): Promise<any>;
    async postText(body: string, url?: string, headers?: Record<string, string>): Promise<string>;

    postJsonOperator<T, R = T>(bodyParam?: any | ((value: T) => any), urlParam?: string | ((value: T) => string), toProperty?: string, headers?: Record<string, string>): OperatorFunction<T, R>;

    // PUT request

    async put(body: string, url?: string, headers?: Record<string, string>): Promise<Response>;
    async putJson(body: any, url?: string, headers?: Record<string, string>): Promise<any>;
    async putText(body: string, url?: string, headers?: Record<string, string>): Promise<string>;

    putJsonOperator<T, R = T>(bodyParam?: any | ((value: T) => any), urlParam?: string | ((value: T) => string), toProperty?: string, headers?: Record<string, string>): OperatorFunction<T, R>;

    // Simple fetch method

    async fetch(url: string): Promise<Response>;
    async fetch(url: string, method: 'GET' | 'POST' | 'PUT' | 'DELETE', headers: Record<string, string>): Promise<Response>;
    async fetch(url: string, init: RequestInit): Promise<Response>;
```

Example:

```typescript
import { Csv, HttpClientHelper, run } from "rxjs-etl-kit";
import { map } from "rxjs";

let csv = new Csv.Endpoint();
let src = csv.getFile('products.csv');

const mySite = new HttpClientHelper('http://www.mysite.com');

let sendProductsToSite$ = src.select().pipe(
    map(p => mySite.post(p)),
);
await run(sendProductsToSite$);
 ```

### Header

This class can store array of column names and convert object to array or array to object representation.

```typescript
import { Postgres, Csv, Header, log, push, run } from "rxjs-etl-kit";
import { map } from "rxjs";

const pg = new Postgres.Endpoint("postgres://user:password@127.0.0.1:5432/database");
const source = pg.getTable("users");

let csv = new Csv.Endpoint();
const dest = csv.getFile("users.csv");
const header = new Header([{"id": "number"}, "name", {"login": "string", nullValue: "-null-"}, "email"]);

let sourceToDest$ = source.select().pipe(
    map(v => header.objToArr(v)),
    push(dest)
);
await run(sourceToDest$);
 ```

### Utility functions

This functions implements some useful things to manipulate data.

```typescript
// Stop thread for the specified in milliseconds delay.
async function wait(delay: number): Promise<void>;
// Join url parts (or path parts) to full url (or path) with delimeter
function pathJoin(parts: string[], sep: string = '/'): string;
// Extract 'html' from '/var/www/html'
function extractFileName(path: string): string;
// Extract '/var/www' from '/var/www/html'
function extractParentFolderPath(path: string): string
// Get object part by json path
function getByJsonPath(obj: {}, jsonPath?: string): any;
// Get child element of array or object by element property value
function getChildByPropVal(obj: {}, propName: string, propVal?: any): any;
// Convert object to string
function dumpObject(obj: any, deep: number = 1): string;
// Get child by it's property value
// For example getChildByPropVal([{prop1: 'val1'}, {prop1: 'val2'}], 'prop1', 'val1') -> returns {prop1: 'val1'}
function getChildByPropVal(obj: {} | [], propName: string, propVal?: any): any;
 ```

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# License

This library is provided with [MIT](LICENSE) license. 

