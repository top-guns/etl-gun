import * as ix from 'ix';
import * as fs from "fs";
import { parse, Options } from "csv-parse";
import { BaseEndpoint} from "../core/endpoint.js";
import { Header, pathJoin } from "../utils/index.js";
import { BaseObservable } from "../core/observable.js";
import { UpdatableCollection } from "../core/updatable_collection.js";
import { CollectionOptions } from "../core/base_collection.js";
import { observable2Generator, observable2Iterable, observable2Promise, observable2Stream, selectOne_from_Observable, wrapObservable, wrapPromise } from '../utils/flows.js';

export type Order = 'forward' | 'backward';

export class Endpoint extends BaseEndpoint {
    protected rootFolder: string | null = null;
    constructor(rootFolder: string | null = null) {
        super();
        this.rootFolder = rootFolder;
    }

    getFile(filename: string, header: Header | null = null, delimiter: string = ",", options: CollectionOptions<CsvCellType[]> = {}): Collection {
        options.displayName ??= this.getName(filename);
        let path = filename;
        if (this.rootFolder) path = pathJoin([this.rootFolder, filename], '/');
        return this._addCollection(filename, new Collection(this, filename, path, header, delimiter, options));
    }

    releaseFile(filename: string) {
        this._removeCollection(filename);
    }

    protected getName(filename: string) {
        return filename.substring(filename.lastIndexOf('/') + 1);
    }

    get displayName(): string {
        return this.rootFolder ? `CSV (${this.rootFolder})` : `CSV (${this.instanceNo})`;
    }
}

export function getEndpoint(rootFolder: string | null = null): Endpoint {
    return new Endpoint(rootFolder);
}

export type CsvCellType = string | boolean | number | undefined | null;

export type CsvWhere = { skipFirstLine: boolean, skipEmptyLines: boolean };

export class Collection extends UpdatableCollection<CsvCellType[]> {
    protected static instanceCount = 0;

    protected filename: string;
    protected delimiter: string;
    public header: Header | null;

    constructor(endpoint: Endpoint, collectionName: string, filename: string, header: Header | null = null, delimiter: string = ",", options: CollectionOptions<CsvCellType[]> = {}) {
        Collection.instanceCount++;
        super(endpoint, collectionName, options);
        this.filename = filename;
        this.delimiter = delimiter;
        this.header = header;
    }

    public select(skipFirstLine?: boolean, skipEmptyLines?: boolean, options: Options = {}): Promise<CsvCellType[][]> {
        const observable = this._selectRx(skipFirstLine, skipEmptyLines, 'forward', options);
        return wrapPromise(observable2Promise(observable), this);
    }
    public async* selectGen(skipFirstLine?: boolean, skipEmptyLines?: boolean, order: Order = 'forward', options: Options = {}): AsyncGenerator<CsvCellType[], void, void> {
        const observable = this.selectRx(skipFirstLine, skipEmptyLines, order, options);
        const generator = observable2Generator(observable);
        for await (const value of generator) yield value;
    }
    public selectIx(skipFirstLine?: boolean, skipEmptyLines?: boolean, order: Order = 'forward', options: Options = {}): ix.AsyncIterable<CsvCellType[]> {
        const observable = this.selectRx(skipFirstLine, skipEmptyLines, order, options);
        return observable2Iterable(observable);
    }

    public selectStream(skipFirstLine?: boolean, skipEmptyLines?: boolean, order: Order = 'forward', options: Options = {}): ReadableStream<CsvCellType[]> {
        const observable = this.selectRx(skipFirstLine, skipEmptyLines, order, options);
        return observable2Stream(observable);
    }
    public async selectOne(skipFirstLine?: boolean, skipEmptyLines?: boolean, options: Options = {}): Promise<CsvCellType[] | null> {
        const observable = this._selectRx(skipFirstLine, skipEmptyLines, 'forward', options);
        const value = await selectOne_from_Observable(observable);
        this.sendSelectOneEvent(value);
        return value;
    }

    public selectRx(skipFirstLine?: boolean, skipEmptyLines?: boolean, order: Order = 'forward', options: Options = {}): BaseObservable<CsvCellType[]> {
        const observable = this._selectRx(skipFirstLine, skipEmptyLines, order, options);
        return wrapObservable(observable, this);
    }

    /**
    * @param skipFirstLine skip the first line in file, useful for skip header
    * @param skipEmptyLines skip empty lines in file
    * @return Observable<string[]> 
    */
    protected _selectRx(skipFirstLine?: boolean, skipEmptyLines?: boolean, order: Order = 'forward', options: Options = {}): BaseObservable<CsvCellType[]> {
        const rows: any[] = [];
        const observable = new BaseObservable<CsvCellType[]>(this, (subscriber) => {
            try {
                const readStream = fs.createReadStream(this.filename)
                .pipe(parse({ delimiter: this.delimiter, from_line: skipFirstLine ? 2 : 1, ...options}))
                .on("data", (row: string[]) => {
                    // if (subscriber.closed) {
                    //     readStream.destroy();
                    //     return;
                    // }
                    // TODO process data in this listener instead of pushing it to the buffer
                    const r: CsvCellType[] = this.cropRowToHeader(row);
                    for (let i = 0; i < r.length; i++) r[i] = this.readedFieldValueToObj(i, r[i] as string);
                    rows.push(r);
                })
                .on("end", () => {
                    (async () => {
                        for (let i = 0; i < rows.length; i++) {
                            const row = rows[order == 'forward' ? i : rows.length - i - 1];
                            if (subscriber.closed) break;
                            await this.waitWhilePaused();
                            if (skipEmptyLines && (row.length == 0 || (row.length == 1 && row[0].trim() == ''))) {
                                return;
                            }
                            if (!subscriber.closed) subscriber.next(row);
                        }
                        if (!subscriber.closed) subscriber.complete();
                    })();
                })
                .on('error', (err) => {
                    if (!subscriber.closed) subscriber.error(err);
                }); 
            }
            catch(err) {
                if (!subscriber.closed) subscriber.error(err);
            }
        });
        return observable;
    }

    protected async _insert(value: CsvCellType[], nullValue: string = ''): Promise<void> {
        const strVal = this.getCsvStrFromArr(value) + "\n";
        // await fs.appendFile(this.filename, strVal, function (err) {
        //     if (err) throw err;
        // });
        fs.appendFileSync(this.filename, strVal);
    }

    public async update(value: any, where: any, ...params: any[]): Promise<void> {
        throw new Error("Method not implemented.");
    }
    public upsert(value: any, where?: any, ...params: any[]): Promise<boolean> {
        throw new Error("Method not implemented.");
    }


    public async delete(): Promise<boolean> {
        this.sendDeleteEvent();
        //const content = await fs.promises.readFile("myFile.txt");
        const stats = await fs.promises.stat(this.filename);
        const exists = stats && stats.size > 0;
        await fs.promises.writeFile(this.filename, '');
        return exists;
    }

    protected getCsvStrFromArr(vals: CsvCellType[]) {
        vals = this.cropRowToHeader(vals);
        let res = "";
        for (let i = 0; i < vals.length; i++) {
            if (res) res += ",";
            res += this.convertToCell(i, vals[i]);
        }
        return res;
    }

    protected convertToCell(i: number, val: CsvCellType): string {
        let strVal: string;
        if (!this.header) {
            if (val === null) return '"null"';
            if (typeof val === 'undefined') return '"undefined"';
            strVal = '' + val;
        }
        else strVal = this.header.valToStr(i, val);
        return '"' + strVal.replace(/"/g, '""') + '"';
    }

    protected readedFieldValueToObj(i: number, val: string): CsvCellType {
        if (!this.header) return val;
        return this.header.strToVal(i, val);
    }

    protected cropRowToHeader(row: CsvCellType[]): CsvCellType[] {
        if (!this.header) return row;
        return row.slice(0, this.header.getFieldsCount());
    }

}
