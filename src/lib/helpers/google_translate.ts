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

    operator(text: string, from?: string, to?: string): rxjs.Observable<string>;
    operator(arr: [], from?: string, to?: string): rxjs.Observable<string[]>;
    operator(obj: {}, transleteKeys?: boolean, translateValues?: boolean, from?: string, to?: string): rxjs.Observable<{}>;
    operator(value: any, p2?: any, p3?: any, p4: string = this.from, p5: string = this.to): rxjs.Observable<any> {
        return rxjs.from(this.translate(value, p2, p3, p4, p5));
    }

    async translate(text: string, from?: string, to?: string): Promise<string>;
    async translate(arr: [], from?: string, to?: string): Promise<string[]>;
    async translate(obj: {}, transleteKeys?: boolean, translateValues?: boolean, from?: string, to?: string): Promise<{}>;
    async translate(value: any, p2?: any, p3?: any, p4: string = this.from, p5: string = this.to): Promise<any> {
        let from = typeof p2 == 'string' ? p2 : this.from;
        let to = typeof p3 == 'string' ? p3 : this.to;

        if (typeof value == 'string') {
            return this.translateStr(value, from, to);
        }

        if (typeof value.length != 'undefined') {
            const res = [];
            for (let i = 0; i < value.length; i++) {
                const v = (typeof value[i] != 'string') ? value[i]: await this.translateStr(value[i], from, to);
                res.push(v);
            }
            return res;
        }

        if (typeof value == 'object') {
            const transleteKeys = typeof p2 == 'boolean' ? p2 : false;
            const translateValues = typeof p3 == 'boolean' ? p3 : false;
            from = p4;
            to = p5;
            const res = {};
            for (let key in value) {
                if (!value.hasOwnProperty) continue;

                const v = !translateValues && (typeof value[key] != 'string') ? value[key]: await this.translateStr(value[key], from, to);
                if (transleteKeys) key = await this.translateStr(key, from, to);
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