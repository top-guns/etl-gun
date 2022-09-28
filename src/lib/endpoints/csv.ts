import * as fs from "fs";
import { Observable } from 'rxjs';
import { parse } from "csv-parse";
import { Endpoint, EndpointGuiOptions, EndpointImpl } from "../core/endpoint";
import { EtlObservable } from "../core/observable";

export class CsvEndpoint extends EndpointImpl<string[]> {
    protected static instanceNo = 0;
    protected filename: string;
    protected delimiter: string;

    constructor(filename: string, delimiter: string = ",", guiOptions: EndpointGuiOptions<string[]> = {}) {
        guiOptions.displayName = guiOptions.displayName ?? `CSV ${++CsvEndpoint.instanceNo}(${filename.substring(filename.lastIndexOf('/') + 1)})`;
        super(guiOptions);
        this.filename = filename;
        this.delimiter = delimiter;
    }

   /**
   * @param skipFirstLine skip the first line in file, useful for skip header
   * @param skipEmptyLines skip empty lines in file
   * @return Observable<string[]> 
   */
    public read(skipFirstLine: boolean = false, skipEmptyLines = false): EtlObservable<string[]> {
        const observable = new EtlObservable<string[]>((subscriber) => {
            try {
                this.sendStartEvent();
                fs.createReadStream(this.filename)
                .pipe(parse({ delimiter: this.delimiter, from_line: skipFirstLine ? 2 : 1 }))
                .on("data", (row: string[]) => {
                    if (skipEmptyLines && (row.length == 0 || (row.length == 1 && row[0].trim() == ''))) {
                        this.sendSkipEvent(row);
                        return;
                    }
                    this.sendDataEvent(row);
                    subscriber.next(row);
                })
                .on("end", () => {
                    subscriber.complete();
                    this.sendEndEvent();
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

    public async push(value: any[]) {
        await super.push(value);
        const strVal = this.getCsvStrFromStrArr(value) + "\n";
        // await fs.appendFile(this.filename, strVal, function (err) {
        //     if (err) throw err;
        // });
        fs.appendFileSync(this.filename, strVal);
    }

    public async clear() {
        await super.clear();
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


