import * as fs from "fs";
import { Observable, Subscriber } from 'rxjs';
import _ from 'lodash';
import { JSONPath } from 'jsonpath-plus';
import { BaseEndpoint} from "../core/endpoint.js";
import { BaseCollection, CollectionGuiOptions } from "../core/collection.js";
import { pathJoin } from "../utils/index.js";

export type ReadOptions = {
    // foundedOnly is default
    searchReturns?: 'foundedOnly' | 'foundedImmediateChildrenOnly' | 'foundedWithDescendants';
    addRelativePathAsField?: string;
}

export class Endpoint extends BaseEndpoint {
    protected rootFolder: string = null;
    constructor(rootFolder: string = null) {
        super();
        this.rootFolder = rootFolder;
    }

    getFile(filename: string, autosave: boolean = true, autoload: boolean = false, encoding?: BufferEncoding, guiOptions: CollectionGuiOptions<number> = {}): Collection {
        guiOptions.displayName ??= this.getName(filename);
        let path = filename;
        if (this.rootFolder) path = pathJoin([this.rootFolder, filename], '/');
        return this._addCollection(filename, new Collection(this, path, autosave, autoload, encoding, guiOptions));
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

export class Collection extends BaseCollection<any> {
    protected static instanceNo = 0;

    get type(): string {
        return 'Json.Collection';
    }

    protected filename: string;
    protected encoding: BufferEncoding;
    protected json: any;
    protected autosave: boolean;
    protected autoload: boolean;

    constructor(endpoint: Endpoint, filename: string, autosave: boolean = true, autoload: boolean = false, encoding?: BufferEncoding, guiOptions: CollectionGuiOptions<any> = {}) {
        Collection.instanceNo++;
        super(endpoint, guiOptions);
        this.filename = filename;
        this.encoding = encoding;
        this.autosave = autosave;
        this.autoload = autoload;
        this.load();
    }

    // Uses simple path syntax from lodash.get function
    // path example: 'store.book[5].author'
    // use path '' for the root object
    public select(path: string = '', options: ReadOptions = {}): Observable<any> {
        const observable = new Observable<any>((subscriber) => {
            (async () => {
                try {
                    this.sendStartEvent();
                    path = path.trim();
                    let result: any = this.get(path);
                    if (result) {
                        if (options.searchReturns == 'foundedOnly' || !options.searchReturns) {
                            if (options.addRelativePathAsField) result[options.addRelativePathAsField] = ``;
                            await this.waitWhilePaused();
                            this.sendReciveEvent(result);
                            subscriber.next(result);
                        }
                        if (options.searchReturns == 'foundedWithDescendants') {
                            await this.sendElementWithChildren(result, subscriber, observable, options, '');
                        }
                        if (options.searchReturns == 'foundedImmediateChildrenOnly') {
                            if (Array.isArray(result)) {
                                for (let i = 0; i < result.length; i++) {
                                    const value = result[i];
                                    if (options.addRelativePathAsField) value[options.addRelativePathAsField] = `[${i}]`;
                                    if (subscriber.closed) break;
                                    await this.waitWhilePaused();
                                    this.sendReciveEvent(value);
                                    subscriber.next(value);
                                };
                            }
                            else if (typeof result === 'object') {
                                for (let key in result) {
                                    if (result.hasOwnProperty(key)) {
                                        if (options.addRelativePathAsField) result[key][options.addRelativePathAsField] = `${key}`;
                                        if (subscriber.closed) break;
                                        await this.waitWhilePaused();
                                        this.sendReciveEvent(result[key]);
                                        subscriber.next(result[key]);
                                    }
                                }
                            }
                        }
                    }
                    subscriber.complete();
                    this.sendEndEvent();
                }
                catch(err) {
                    this.sendErrorEvent(err);
                    subscriber.error(err);
                }
            })();
        });
        return observable;
    }

    // Uses complex JSONPath standart for path syntax
    // About path syntax read https://www.npmjs.com/package/jsonpath-plus
    // path example: '$.store.book[*].author'
    // use path '$' for the root object
    public selectByJsonPath(jsonPath?: string, options?: ReadOptions): Observable<any>;
    public selectByJsonPath(jsonPaths?: string[], options?: ReadOptions): Observable<any>;
    public selectByJsonPath(jsonPath: any = '', options: ReadOptions = {}): Observable<any> {
        const observable = new Observable<any>((subscriber) => {
            (async () => {
                try {
                    this.sendStartEvent();
                    let result: any = this.getByJsonPath(jsonPath);
                    if (options.searchReturns == 'foundedOnly' || !options.searchReturns) {
                        for (const value of result) {
                            if (options.addRelativePathAsField) value[options.addRelativePathAsField] = ``;
                            await this.waitWhilePaused();
                            this.sendReciveEvent(value);
                            subscriber.next(value);
                        };
                    }
                    if (options.searchReturns == 'foundedWithDescendants') {
                        for (const value of result) {
                            await this.sendElementWithChildren(value, subscriber, observable, options, ``);
                        };
                    }
                    if (options.searchReturns == 'foundedImmediateChildrenOnly') {
                        for (const value of result) {
                            if (Array.isArray(value)) {
                                for (let i = 0; i < value.length; i++) {
                                    const child = value[i];
                                    if (options.addRelativePathAsField) child[options.addRelativePathAsField] = `[${i}]`;
                                    if (subscriber.closed) break;
                                    await this.waitWhilePaused();
                                    this.sendReciveEvent(child);
                                    subscriber.next(child);
                                };
                            }
                            else if (typeof value === 'object') {
                                for (let key in value) {
                                    if (value.hasOwnProperty(key)) {
                                        if (options.addRelativePathAsField) value[key][options.addRelativePathAsField] = `${key}`;
                                        if (subscriber.closed) break;
                                        await this.waitWhilePaused();
                                        this.sendReciveEvent(value[key]);
                                        subscriber.next(value[key]);
                                    }
                                }
                            }
                        };
                    }
                    subscriber.complete();
                    this.sendEndEvent();
                }
                catch(err) {
                    this.sendErrorEvent(err);
                    subscriber.error(err);
                }
            })();
        });
        return observable;
    }

    protected async sendElementWithChildren(element: any, subscriber: Subscriber<any>, observable: Observable<any>, options: ReadOptions = {}, relativePath = '') {
        if (options.addRelativePathAsField) element[options.addRelativePathAsField] = relativePath;
        if (subscriber.closed) return;
        await this.waitWhilePaused();
        this.sendReciveEvent(element);
        subscriber.next(element);

        if (Array.isArray(element)) {
            if (element.length) this.sendDownEvent();
            for (let i = 0; i < element.length; i++) {
                const child = element[i];
                await this.sendElementWithChildren(child, subscriber, observable, options, `${relativePath}[${i}]`);
            };
            if (element.length) this.sendUpEvent();
        }
        else if (typeof element === 'object') {
            let sendedDown = false;
            for (let key in element) {
                if (element.hasOwnProperty(key)) {
                    if (Array.isArray(element[key]) || typeof element[key] === 'object') {
                        if (!sendedDown) {
                            this.sendDownEvent();
                            sendedDown = true;
                        }
                        await this.sendElementWithChildren(element[key], subscriber, observable, options, relativePath ? `${relativePath}.${key}` : `${key}`);
                    }
                }
            }
            if (sendedDown) this.sendUpEvent();
        }
    }

    // Uses simple path syntax from lodash.get function
    // path example: 'store.book[5].author'
    // use path '' for the root object
    public get(path: string = ''): any {
        if (this.autoload) this.load();
        path = path.trim();
        let result: any = path ? _.get(this.json, path) : this.json;
        return result;
    }

    // Uses complex JSONPath standart for path syntax
    // About path syntax read https://www.npmjs.com/package/jsonpath-plus
    // path example: '$.store.book[*].author'
    // use path '$' for the root object
    public getByJsonPath(jsonPath?: string): any;
    public getByJsonPath(jsonPaths?: string[]): any;
    public getByJsonPath(jsonPath: any = ''): any {
        if (this.autoload) this.load();
        let result: any = JSONPath({path: jsonPath, json: this.json, wrap: true});
        return result;
    }

    // Pushes value to the array specified by simple path
    // or update property fieldname of object specified by simple path
    public async insert(value: any, path: string = '', fieldname: string = '') {
        super.insert(value);

        const obj = this.get(path);

        if (fieldname) obj[fieldname] = value;
        else obj.push(value);

        if (this.autosave) this.save();
    }

    public async delete() {
        super.delete();
        this.json = {};
        if (this.autosave) this.save();
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


