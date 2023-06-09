import * as ix from 'ix';
import * as fs from "fs";
import _ from 'lodash';
import { JSONPath } from 'jsonpath-plus';
import { BaseEndpoint} from "../core/endpoint.js";
import { pathJoin } from "../utils/index.js";
import { BaseObservable } from "../core/observable.js";
import { UpdatableCollection } from "../core/updatable_collection.js";
import { BaseCollection, CollectionOptions } from "../core/base_collection.js";
import { generator2Iterable, generator2Stream, promise2Generator, promise2Observable, selectOne_from_Promise, wrapGenerator, wrapObservable } from "../utils/flows.js";


export class Endpoint extends BaseEndpoint {
    protected rootFolder: string | null = null;
    constructor(rootFolder: string | null = null) {
        super();
        this.rootFolder = rootFolder;
    }

    getFile(filename: string, autosave: boolean = true, autoload: boolean = false, encoding?: BufferEncoding, options: CollectionOptions<number> = {}): LodashPathCollection {
        options.displayName ??= this.getName(filename);
        let path = filename;
        if (this.rootFolder) path = pathJoin([this.rootFolder, filename], '/');
        return this._addCollection(filename, new LodashPathCollection(this, filename, path, autosave, autoload, encoding, options));
    }

    getFileWithJsonPaths(filename: string, encoding?: BufferEncoding, options: CollectionOptions<number> = {}): JsonPathCollection {
        options.displayName ??= this.getName(filename);
        let path = filename;
        if (this.rootFolder) path = pathJoin([this.rootFolder, filename], '/');
        return this._addCollection(filename, new JsonPathCollection(this, filename, path, encoding, options));
    }

    releaseFile(filename: string) {
        this._removeCollection(filename);
    }

    protected getName(filename: string) {
        return filename.substring(filename.lastIndexOf('/') + 1);
    }

    get displayName(): string {
        return this.rootFolder ? `JSON (${this.rootFolder})` : `JSON (${this.instanceNo})`;
    }
}

export function getEndpoint(rootFolder: string | null = null): Endpoint {
    return new Endpoint(rootFolder);
}

export class LodashPathCollection extends UpdatableCollection<any> {
    protected static instanceNo = 0;

    protected filename: string;
    protected encoding: BufferEncoding | undefined;
    protected json: any;
    protected autosave: boolean;
    protected autoload: boolean;

    constructor(endpoint: Endpoint, collectionName: string, filename: string, autosave: boolean = true, autoload: boolean = false, encoding?: BufferEncoding, options: CollectionOptions<any> = {}) {
        LodashPathCollection.instanceNo++;
        super(endpoint, collectionName, options);
        this.filename = filename;
        this.encoding = encoding;
        this.autosave = autosave;
        this.autoload = autoload;
        this.load();
    }

    // Uses simple path syntax from lodash.get function
    // path example: 'store.book[5].author'
    // use path '' for the root object
    protected async _select(lodashPath: string = ''): Promise<any[]> {
        if (this.autoload) this.load();
        lodashPath = lodashPath.trim();
        let result: any = lodashPath ? _.get(this.json, lodashPath) : this.json;
        return Array.isArray(result) ? result : [result];
    }

    public async select(lodashPath: string = ''): Promise<any[]> {
        const values = await this._select(lodashPath);
        this.sendSelectEvent(values);
        return values;
    }

    public async* selectGen(lodashPath: string = ''): AsyncGenerator<any, void, void> {
        const values = this._select(lodashPath);
        const generator = wrapGenerator(promise2Generator(values), this);
        for await (const value of generator) yield value;
    }
    public selectIx(lodashPath: string = ''): ix.AsyncIterable<any> {
        const generator = this.selectGen(lodashPath);
        return generator2Iterable(generator);
    }

    public selectStream(lodashPath: string = ''): ReadableStream<any> {
        const generator = this.selectGen(lodashPath);
        return generator2Stream(generator);
    }

    public selectRx(lodashPath: string = ''): BaseObservable<any> {
        const values = this._select(lodashPath);
        return wrapObservable(promise2Observable(values), this);
    }

    public async selectOne(lodashPath: string = ''): Promise<any> {
        const values = this._select(lodashPath);
        const result = await selectOne_from_Promise(values);
        this.sendSelectOneEvent(result);
        return result;
    }
    

    // Pushes value to the array specified by simple path
    // or update property fieldname of object specified by simple path
    protected async _insert(value: any, lodashPath: string = '', fieldname: string = '') {
        const obj = await this._select(lodashPath);

        if (fieldname) obj[fieldname] = value;
        else obj.push(value);

        if (this.autosave) this.save();
    }

    public async update(value: any, lodashPath: string, fieldname: string): Promise<void> {
        this.sendUpdateEvent(value, { lodashPath, fieldname });
        const obj = await this._select(lodashPath);
        obj[fieldname] = value;
        if (this.autosave) this.save();
    }
    public async upsert(value: any, lodashPath: string, fieldname: string): Promise<boolean> {  
        const obj = await this._select(lodashPath);
        const exists = typeof obj[fieldname] !== 'undefined';
        if (!exists) this.sendInsertEvent(value, { lodashPath, fieldname });
        else this.sendUpdateEvent(value, { lodashPath, fieldname });
        obj[fieldname] = value;
        if (this.autosave) this.save();
        return exists;
    }

    public async delete(lodashPath: string = '', fieldname?: string): Promise<boolean> {
        this.sendDeleteEvent({ lodashPath, fieldname });
        let exists = false;
        if (!lodashPath && !fieldname) {
            exists = this.json && !!Object.keys(this.json).length;
            this.json = {};
        }
        else {
            const obj = await this._select(lodashPath);
            exists = typeof obj[fieldname!] !== 'undefined';
            delete obj[fieldname!];
        }
        if (this.autosave) this.save();
        return exists;
    }

    public load() {
        const text = fs.readFileSync(this.filename).toString(this.encoding);
        this.json = JSON.parse(text);
    }

    public save() {
        const text = JSON.stringify(this.json);
        fs.writeFile(this.filename, text, function(){});
    }
}

export class JsonPathCollection extends BaseCollection<any> {
    protected static instanceNo = 0;

    protected filename: string;
    protected encoding: BufferEncoding | undefined;
    protected json: any;

    constructor(endpoint: Endpoint, collectionName: string, filename: string, encoding?: BufferEncoding, options: CollectionOptions<any> = {}) {
        JsonPathCollection.instanceNo++;
        super(endpoint, collectionName, options);
        this.filename = filename;
        this.encoding = encoding;
        this.load();
    }

    // Uses complex JSONPath standart for path syntax
    // About path syntax read https://www.npmjs.com/package/jsonpath-plus
    // path example: '$.store.book[*].author'
    // use path '$' for the root object
    protected async _select(jsonPath: any = ''): Promise<any[]> {
        this.load();
        let result = JSONPath({ path: jsonPath, json: this.json, wrap: false });
        return Array.isArray(result) ? result : [result];
    }

    public select(jsonPath?: string): Promise<any[]>;
    public select(jsonPaths?: string[]): Promise<any[]>;
    public async select(jsonPath: any = ''): Promise<any[]> {
        const values = await this._select(jsonPath);
        this.sendSelectEvent(values);
        return values;
    }

    public selectGen(jsonPath?: string): AsyncGenerator<any, void, void>;
    public selectGen(jsonPaths?: string[]): AsyncGenerator<any, void, void>;    
    public async* selectGen(jsonPath: any = ''): AsyncGenerator<any, void, void> {
        const values = this._select(jsonPath);
        const generator = wrapGenerator(promise2Generator(values), this);
        for await (const value of generator) yield value;
    }
    public selectIx(jsonPath?: string): ix.AsyncIterable<any>;
    public selectIx(jsonPaths?: string[]): ix.AsyncIterable<any>;    
    public selectIx(jsonPath: any = ''): ix.AsyncIterable<any> {
        const generator = this.selectGen(jsonPath);
        return generator2Iterable(generator);
    }

    public selectStream(jsonPath?: string): ReadableStream<any>;
    public selectStream(jsonPaths?: string[]): ReadableStream<any>;    
    public selectStream(jsonPath: any = ''): ReadableStream<any> {
        const generator = this.selectGen(jsonPath);
        return generator2Stream(generator);
    }

    public selectRx(jsonPath?: string): BaseObservable<any>;
    public selectRx(jsonPaths?: string[]): BaseObservable<any>;    
    public selectRx(jsonPath: any = ''): BaseObservable<any> {
        const values = this._select(jsonPath);
        return wrapObservable(promise2Observable(values), this);
    }

    public selectOne(jsonPath?: string): Promise<any>;
    public selectOne(jsonPaths?: string[]): Promise<any>;
        public async selectOne(jsonPath: any = ''): Promise<any> {
        const values = this._select(jsonPath);
        const result = await selectOne_from_Promise(values);
        this.sendSelectOneEvent(result);
        return result;
    }

    protected load() {
        const text = fs.readFileSync(this.filename).toString(this.encoding);
        this.json = JSON.parse(text);
    }
}


