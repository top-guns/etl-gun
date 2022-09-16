import * as fs from "fs";
import { Endpoint } from "../core/endpoint";
import { parse } from "csv-parse";
import { Observable } from 'rxjs';

export class CsvEndpoint extends Endpoint<string[]> {
    protected filename: string;
    protected delimiter: string;

    constructor(filename: string, delimiter: string = ",") {
        super();
        this.filename = filename;
        this.delimiter = delimiter;
    }

    public find(skipEmptyLines = false, skipFirstLine: boolean = false): Observable<string[]> {
        return new Observable<string[]>((subscriber) => {
            fs.createReadStream(this.filename)
            .pipe(parse({ delimiter: this.delimiter, from_line: skipFirstLine ? 2 : 1 }))
            .on("data", (row: string[]) => {
                if (skipEmptyLines && (row.length == 0 || (row.length == 1 && row[0].trim() == ''))) return;
                subscriber.next(row);
            })
            .on("end", () => {
                subscriber.complete();
            })
            .on('error', (err) => {
                subscriber.error(err);
            }); 

        });
    }

    public async push(value: string[]) {
        const strVal = this.getCsvStrFromStrArr(value) + "\n";
        await fs.appendFile(this.filename, strVal, function (err) {
            if (err) throw err;
        });
    }

    public async clear() {
        fs.writeFile(this.filename, '', function(){})
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


