// Imports the Google Cloud client library
import Translate from '@google-cloud/translate';
import * as rx from 'rxjs';
import { MonoTypeOperatorFunction, OperatorFunction } from 'rxjs';

export class GoogleTranslateHelper {
    protected from: string;
    protected to: string;
    protected apiKey: string;

    protected _client: Translate.v2.Translate = null;
    protected get client() {
        this._client ||= new Translate.v2.Translate({key: this.apiKey});
        return this._client;
    }

    constructor(apiKey: string, from: string = null, to: string = null) {
        this.apiKey = apiKey;
        this.from = from;
        this.to = to;
    }

    operator(text: string, from?: string, to?: string): MonoTypeOperatorFunction<string>;
    operator<T = []>(translateColumns?: number[], from?: string, to?: string): MonoTypeOperatorFunction<T>;
    operator<T = {}, R = T>(translateKeyNames?: string[], translateKeyValues?: string[], from?: string, to?: string): MonoTypeOperatorFunction<R>;
    operator(p1?: any, p2?: any, p3?: any, p4?: any) {
        return rx.mergeMap(p => this.observable(p, p1, p2, p3, p4)); 
    }

    observable(text: string, from?: string, to?: string): rx.Observable<string>;
    observable<T = []>(arr: T, translateColumns?: number[], from?: string, to?: string): rx.Observable<T>;
    observable<T = {}, R = T>(obj: T, translateKeyNames?: string[], translateKeyValues?: string[], from?: string, to?: string): rx.Observable<R>;
    observable(value: any, p1?: any, p2?: any, p3?: any, p4?: any): rx.Observable<any> {
        return rx.from(this.function(value, p1, p2, p3, p4));
    }

    async function(text: string, from?: string, to?: string): Promise<string>;
    async function<T = []>(arr: T, translateColumns?: number[], from?: string, to?: string): Promise<T>;
    async function<T = {}, R = T>(obj: T, translateKeyNames?: string[], translateKeyValues?: string[], from?: string, to?: string): Promise<R>;
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