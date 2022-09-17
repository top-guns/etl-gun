import * as fs from "fs";
import { Observable } from 'rxjs';
import { get } from 'lodash';
import { JSONPath } from 'jsonpath-plus';
import { Endpoint } from "../core/endpoint";

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
    public read(pathToArray: string = ''): Observable<any> {
        return new Observable<any>((subscriber) => {
            try {
                if (this.autoload) this.load();

                pathToArray = pathToArray.trim();
                let result: any = pathToArray ? get(this.json, pathToArray) : this.json;
                if (result) {
                    if (Array.isArray(result)) {
                        result.forEach(value => subscriber.next(result))
                    }
                    else if (typeof result == 'object') {
                        for (let key in result) {
                            if (result.hasOwnProperty(key)) {
                                subscriber.next(result[key]);
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
    public readByJsonPath(jsonPath?: string): Observable<any>;
    public readByJsonPath(jsonPaths?: string[]): Observable<any>;
    public readByJsonPath(jsonPath: any = ''): Observable<any> {
        return new Observable<any>((subscriber) => {
            try {
                if (this.autoload) this.load();

                let result: any = JSONPath({path: jsonPath, json: this.json, wrap: false});
                if (result) {
                    if (Array.isArray(result)) {
                        result.forEach(value => {
                            subscriber.next(value);
                        });
                    }
                    else if (typeof result == 'object') {
                        for (let key in result) {
                            if (result.hasOwnProperty(key)) {
                                subscriber.next(result[key]);
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
    public async push(value: any, path: string = '', fieldname: String = '') {
        const obj = get(this.json, path);

        // update property
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


