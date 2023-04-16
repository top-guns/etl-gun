// Imports the Google Cloud client library
import { Translate } from '@google-cloud/translate/build/src/v2';
import * as rxjs from 'rxjs';

export class GoogleTranslateHelper {
    protected from: string;
    protected to: string;
    protected apiKey: string;

    protected _client: Translate = null;
    protected get client() {
        this._client ||= new Translate({key: this.apiKey});
        return this._client;
    }

    constructor(apiKey: string, from: string = null, to: string = null) {
        this.apiKey = apiKey;
        this.from = from;
        this.to = to;
    }

    operator(translateColumns?: number[], from?: string, to?: string);
    operator(translateKeyNames?: string[], translateKeyValues?: string[], from?: string, to?: string);
    operator(p1?: any, p2?: any, p3?: any, p4?: any) {
        return rxjs.mergeMap(p => this.observable(p, p1, p2, p3, p4)); 
    }

    observable(text: string, from?: string, to?: string): rxjs.Observable<string>;
    observable(arr: [], translateColumns?: number[], from?: string, to?: string): rxjs.Observable<string[]>;
    observable(obj: {}, translateKeyNames?: string[], translateKeyValues?: string[], from?: string, to?: string): rxjs.Observable<{}>;
    observable(value: any, p1?: any, p2?: any, p3?: any, p4?: any): rxjs.Observable<any> {
        return rxjs.from(this.function(value, p1, p2, p3, p4));
    }

    async function(text: string, from?: string, to?: string): Promise<string>;
    async function(arr: [], translateColumns?: number[], from?: string, to?: string): Promise<string[]>;
    async function(obj: {}, translateKeyNames?: string[], translateKeyValues?: string[], from?: string, to?: string): Promise<{}>;
    async function(value: any, p1?: any, p2?: any, p3?: any, p4?: any): Promise<any> {
        let from = p1 ?? this.from;
        let to = p2 ?? this.to;

        if (typeof value == 'string') {
            return this.translateStr(value, from, to);
        }

        if (typeof value.length != 'undefined') {
            const translateColumns: number[] = p1 ?? [];
            from = p2 ?? this.from;
            to = p3 ?? this.to;
            const res = [];
            for (let i = 0; i < value.length; i++) {
                let v = value[i];
                if (translateColumns.length && translateColumns.includes(i)) v = await this.translateStr(value[i], from, to);
                else if (!translateColumns.length && typeof value[i] == 'string') v = await this.translateStr(value[i], from, to);
                res.push(v);
            }
            return res;
        }

        if (typeof value == 'object') {
            const translateKeyNames = p1;
            const translateKeyValues = p2;
            from = p3 ?? this.from;
            to = p4 ?? this.to;
            const res = {};
            for (let key in value) {
                if (!value.hasOwnProperty) continue;

                let v = value[key];
                if (translateKeyValues && translateKeyValues.includes(key)) v = await this.translateStr(value[key], from, to);
                else if (!translateKeyValues && typeof value[key] == 'string') v = await this.translateStr(value[key], from, to);
                if (translateKeyNames && translateKeyNames.includes(key)) key = await this.translateStr(key, from, to);
                else if (!translateKeyNames) key = await this.translateStr(key, from, to);
                res[key] = v;
            }
            return res;
        }

        return this.translateStr('' + value, from, to);
    }

    protected async translateStr(value: string, from: string = this.from, to: string = this.to) {
        let [result] = await this.client.translate(value, {from, to});
        return result;
    }

    protected async translateArr(values: string[], from: string = this.from, to: string = this.to) {
        let [result] = await this.client.translate(values, {from, to});
        return result;
    }
}