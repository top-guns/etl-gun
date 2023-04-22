import * as fs from "fs";
import { parse, Options } from "csv-parse";
import { BaseEndpoint} from "../core/endpoint.js";
import { BaseCollection, CollectionGuiOptions } from "../core/collection.js";
import { Header, pathJoin } from "../utils/index.js";
import { Observable } from "rxjs";

export class Endpoint extends BaseEndpoint {
    protected rootFolder: string = null;
    constructor(rootFolder: string = null) {
        super();
        this.rootFolder = rootFolder;
    }

    getFile(filename: string, header: Header = null, delimiter: string = ",", guiOptions: CollectionGuiOptions<string[]> = {}): Collection {
        guiOptions.displayName ??= this.getName(filename);
        let path = filename;
        if (this.rootFolder) path = pathJoin([this.rootFolder, filename], '/');
        return this._addCollection(filename, new Collection(this, path, header, delimiter, guiOptions));
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

export class Collection extends BaseCollection<string[]> {
    protected static instanceCount = 0;

    get type(): string {
        return 'Csv.Collection';
    }

    protected filename: string;
    protected delimiter: string;
    protected header: Header;

    constructor(endpoint: Endpoint, filename: string, header: Header = null, delimiter: string = ",", guiOptions: CollectionGuiOptions<string[]> = {}) {
        Collection.instanceCount++;
        super(endpoint, guiOptions);
        this.filename = filename;
        this.delimiter = delimiter;
        this.header = header;
    }

   /**
   * @param skipFirstLine skip the first line in file, useful for skip header
   * @param skipEmptyLines skip empty lines in file
   * @return Observable<string[]> 
   */
    public select(skipFirstLine: boolean = false, skipEmptyLines = false, options: Options = {}): Observable<string[]> {
        const rows = [];
        const observable = new Observable<string[]>((subscriber) => {
            try {
                this.sendStartEvent();
                const readStream = fs.createReadStream(this.filename)
                .pipe(parse({ delimiter: this.delimiter, from_line: skipFirstLine ? 2 : 1, ...options}))
                .on("data", (row: string[]) => {
                    // if (subscriber.closed) {
                    //     readStream.destroy();
                    //     return;
                    // }
                    // TODO process data in this listener instead of pushing it to the buffer
                    const r = this.cropRowToHeader(row);
                    for (let i = 0; i < r.length; i++) r[i] = this.readedFieldValueToObj(i, r[i]);
                    rows.push(r);
                })
                .on("end", () => {
                    (async () => {
                        for (const row of rows) {
                            if (subscriber.closed) break;
                            await this.waitWhilePaused();
                            if (skipEmptyLines && (row.length == 0 || (row.length == 1 && row[0].trim() == ''))) {
                                this.sendSkipEvent(row);
                                return;
                            }
                            this.sendReciveEvent(row);
                            subscriber.next(row);
                        }
                        subscriber.complete();
                        this.sendEndEvent();
                    })();
                })
                .on('error', (err) => {
                    this.sendErrorEvent(err);
                    subscriber.error(err);
                }); 
            }
            catch(err) {
                this.sendErrorEvent(err);
                subscriber.error(err);
            }
        });
        return observable;
    }

    public async insert(value: any[], nullValue: string = '') {
        await super.insert(value);
        const strVal = this.getCsvStrFromArr(value) + "\n";
        // await fs.appendFile(this.filename, strVal, function (err) {
        //     if (err) throw err;
        // });
        fs.appendFileSync(this.filename, strVal);
    }

    public async delete() {
        await super.delete();
        await fs.promises.writeFile(this.filename, '');
    }

    protected getCsvStrFromArr(vals: any[]) {
        vals = this.cropRowToHeader(vals);
        let res = "";
        for (let i = 0; i < vals.length; i++) {
            if (res) res += ",";
            res += this.convertToCell(i, vals[i]);
        }
        return res;
    }

    protected convertToCell(i: number, val: any): string {
        let strVal: string;
        if (!this.header) {
            if (val === null) return '"null"';
            if (typeof val === 'undefined') return '"undefined"';
            strVal = '' + val;
        }
        else strVal = this.header.valToStr(i, val);
        return '"' + strVal.replace(/"/g, '""') + '"';
    }

    protected readedFieldValueToObj(i: number, val: string): any {
        if (!this.header) return val;
        return this.header.strToVal(i, val);
    }

    protected cropRowToHeader(row: any[]): any[] {
        if (!this.header) return row;
        return row.slice(0, this.header.getFieldsCount());
    }

}
