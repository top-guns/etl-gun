import * as fs from "fs";
import { Observable, Subscriber } from 'rxjs';
import { get } from 'lodash';
import { JSONPath } from 'jsonpath-plus';
import { Endpoint } from "../core/endpoint";

export type ReadOptions = {
    // foundedOnly is default
    searchReturns?: 'foundedOnly' | 'foundedImmediateChildrenOnly' | 'foundedWithDescendants';
    addRelativePathAsField?: string;
}


export class JsonEndpoint extends Endpoint<any> {
    protected filename: string;
    protected encoding: BufferEncoding;
    protected json: any;
    protected autosave: boolean;
    protected autoload: boolean;

    constructor(filename: string, autosave: boolean = true, autoload: boolean = false, encoding?: BufferEncoding) {
        super();
        this.filename = filename;
        this.encoding = encoding;
        this.load();
        this.autosave = autosave;
        this.autoload = autoload;
    }

    // Uses simple path syntax from lodash.get function
    // path example: 'store.book[5].author'
    // use path '' for the root object
    public read(path: string = '', options: ReadOptions = {}): Observable<any> {
        return new Observable<any>((subscriber) => {
            try {
                if (this.autoload) this.load();

                path = path.trim();
                let result: any = this.get(path);
                if (result) {
                    if (options.searchReturns == 'foundedOnly' || !options.searchReturns) {
                        if (options.addRelativePathAsField) result[options.addRelativePathAsField] = ``;
                        subscriber.next(result);
                    }
                    if (options.searchReturns == 'foundedWithDescendants') {
                        this.sendElementWithChildren(result, subscriber, options, '');
                    }
                    if (options.searchReturns == 'foundedImmediateChildrenOnly') {
                        if (Array.isArray(result)) {
                            result.forEach((value, i) => {
                                if (options.addRelativePathAsField) value[options.addRelativePathAsField] = `[${i}]`;
                                subscriber.next(value);
                            });
                        }
                        else if (typeof result === 'object') {
                            for (let key in result) {
                                if (result.hasOwnProperty(key)) {
                                    if (options.addRelativePathAsField) result[key][options.addRelativePathAsField] = `${key}`;
                                    subscriber.next(result[key]);
                                }
                            }
                        }
                    }
                }
                subscriber.complete();
            }
            catch(err) {
                subscriber.error(err);
            }
        });
    }

    // Uses complex JSONPath standart for path syntax
    // About path syntax read https://www.npmjs.com/package/jsonpath-plus
    // path example: '$.store.book[*].author'
    // use path '$' for the root object
    public readByJsonPath(jsonPath?: string, options?: ReadOptions): Observable<any>;
    public readByJsonPath(jsonPaths?: string[], options?: ReadOptions): Observable<any>;
    public readByJsonPath(jsonPath: any = '', options: ReadOptions = {}): Observable<any> {
        return new Observable<any>((subscriber) => {
            try {
                if (this.autoload) this.load();

                let result: any = JSONPath({path: jsonPath, json: this.json, wrap: true});
                if (options.searchReturns == 'foundedOnly' || !options.searchReturns) {
                    result.forEach(value => {
                        if (options.addRelativePathAsField) value[options.addRelativePathAsField] = ``; 
                        subscriber.next(value);
                    });
                }
                if (options.searchReturns == 'foundedWithDescendants') {
                    result.forEach(value => {
                        this.sendElementWithChildren(value, subscriber, options, ``);
                    });
                }
                if (options.searchReturns == 'foundedImmediateChildrenOnly') {
                    result.forEach(value => {
                        if (Array.isArray(value)) {
                            value.forEach((child, i) => {
                                if (options.addRelativePathAsField) child[options.addRelativePathAsField] = `[${i}]`;
                                subscriber.next(child);
                            });
                        }
                        else if (typeof value === 'object') {
                            for (let key in value) {
                                if (value.hasOwnProperty(key)) {
                                    if (options.addRelativePathAsField) value[key][options.addRelativePathAsField] = `${key}`;
                                    subscriber.next(value[key]);
                                }
                            }
                        }
                    });
                }
                subscriber.complete();
            }
            catch(err) {
                subscriber.error(err);
            }
        });
    }

    protected sendElementWithChildren(element: any, subscriber: Subscriber<any>, options: ReadOptions = {}, relativePath = '') {
        if (options.addRelativePathAsField) element[options.addRelativePathAsField] = relativePath;
        subscriber.next(element);

        if (Array.isArray(element)) {
            element.forEach((child, i) => {
                this.sendElementWithChildren(child, subscriber, options, `${relativePath}[${i}]`);
            });
        }
        else if (typeof element === 'object') {
            for (let key in element) {
                if (element.hasOwnProperty(key)) {
                    if (Array.isArray(element[key]) || typeof element[key] === 'object') {
                        this.sendElementWithChildren(element[key], subscriber, options, relativePath ? `${relativePath}.${key}` : `${key}`);
                    }
                }
            }
        }
    }

    // Uses simple path syntax from lodash.get function
    // path example: 'store.book[5].author'
    // use path '' for the root object
    public get(path: string = ''): any {
        if (this.autoload) this.load();
        path = path.trim();
        let result: any = path ? get(this.json, path) : this.json;
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
        let result: any = JSONPath({path: jsonPath, json: this.json, wrap: false});
        return result;
    }

    // Pushes value to the array specified by simple path
    // or update property fieldname of object specified by simple path
    public async push(value: any, path: string = '', fieldname: string = '') {
        const obj = this.get(path);

        if (fieldname) obj[fieldname] = value;
        else obj.push(value);

        if (this.autosave) this.save();
    }

    public async clear() {
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


