import * as fs from "fs";
import { parse } from "csv-parse";
import { BaseEndpoint} from "../core/endpoint.js";
import { BaseCollection, CollectionGuiOptions } from "../core/collection.js";
import { EtlObservable } from "../core/observable.js";
import { pathJoin } from "../utils/index.js";

export class Endpoint extends BaseEndpoint {
    protected rootFolder: string = null;
    constructor(rootFolder: string = null) {
        super();
        this.rootFolder = rootFolder;
    }

    getFile(filename: string, delimiter: string = ",", guiOptions: CollectionGuiOptions<string[]> = {}): Collection {
        guiOptions.displayName ??= this.getName(filename);
        let path = filename;
        if (this.rootFolder) path = pathJoin([this.rootFolder, filename], '/');
        return this._addCollection(filename, new Collection(this, path, delimiter, guiOptions));
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
    protected filename: string;
    protected delimiter: string;

    constructor(endpoint: Endpoint, filename: string, delimiter: string = ",", guiOptions: CollectionGuiOptions<string[]> = {}) {
        Collection.instanceCount++;
        super(endpoint, guiOptions);
        this.filename = filename;
        this.delimiter = delimiter;
    }

   /**
   * @param skipFirstLine skip the first line in file, useful for skip header
   * @param skipEmptyLines skip empty lines in file
   * @return Observable<string[]> 
   */
    public select(skipFirstLine: boolean = false, skipEmptyLines = false): EtlObservable<string[]> {
        const rows = [];
        const observable = new EtlObservable<string[]>((subscriber) => {
            try {
                this.sendStartEvent();
                fs.createReadStream(this.filename)
                .pipe(parse({ delimiter: this.delimiter, from_line: skipFirstLine ? 2 : 1 }))
                .on("data", (row: string[]) => {
                    rows.push(row);
                })
                .on("end", () => {
                    (async () => {
                        for (const row of rows) {
                            await this.waitWhilePaused();
                            if (skipEmptyLines && (row.length == 0 || (row.length == 1 && row[0].trim() == ''))) {
                                this.sendSkipEvent(row);
                                return;
                            }
                            this.sendValueEvent(row);
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

    public async insert(value: any[]) {
        await super.insert(value);
        const strVal = this.getCsvStrFromStrArr(value) + "\n";
        // await fs.appendFile(this.filename, strVal, function (err) {
        //     if (err) throw err;
        // });
        fs.appendFileSync(this.filename, strVal);
    }

    public async delete() {
        await super.delete();
        await fs.promises.writeFile(this.filename, '');
    }

    protected getCsvStrFromStrArr(vals: string[]) {
        let res = "";
        for (let i = 0; i < vals.length; i++) {
            if (res) res += ",";

            let r = "" + vals[i];
            r = r.replace(/"/g, '""');
            r = '"' + r + '"';

            res += r;
        }
        return res;
    }

    protected arrayToScv(arr: string[][]) {
        let res = "";
        for (let i = 0; i < arr.length; i++) {
            if (res) res += "\n";
            res += this.getCsvStrFromStrArr(arr[i]);
        }
        return res;
    }

}
