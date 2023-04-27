import fetch, { RequestInit } from 'node-fetch';
import https from 'node:https';
import { pathJoin } from '../utils/index.js';
import * as rxjs from 'rxjs';
import { OperatorFunction } from 'rxjs';

export class HttpClientHelper {
    protected baseUrl: string;
    protected headers: Record<string, string>;
    protected rejectUnauthorized: boolean;
    protected agent: https.Agent;

    constructor(baseUrl?: string, headers?: Record<string, string>, rejectUnauthorized: boolean = true) {
        if (baseUrl && baseUrl.startsWith('http:') && !rejectUnauthorized) throw new Error('HttpClientHelper error: you can use rejectUnauthorized = false only for https urls')

        this.baseUrl = baseUrl;
        this.headers = headers;
        this.rejectUnauthorized = rejectUnauthorized;
        this.agent = rejectUnauthorized ? null : new https.Agent({
            rejectUnauthorized
        });
    }

    // GET

    async get(url?: string, headers?: Record<string, string>): Promise<Response> {
        const res = await this.fetch(url, 'GET', headers);
        return res;
    }
    async getJson(url?: string, headers?: Record<string, string>): Promise<any> {
        headers = {...headers, "Content-Type": "application/json"};
        const res = await (await this.get(url, headers)).json();
        return res;
    }
    async getText(url?: string, headers?: Record<string, string>): Promise<string> {
        const res = await (await this.get(url, headers)).text();
        return res;
    }
    async getBlob(url?: string, headers?: Record<string, string>): Promise<Blob> {
        const res = await (await this.get(url, headers)).blob();
        return res;
    }

    getJsonOperator<T, R = T>(): OperatorFunction<T, R>;
    getJsonOperator<T, R = T>(url: string, toProperty?: string, headers?: Record<string, string>): OperatorFunction<T, R>;
    getJsonOperator<T, R = T>(getUrl: (value: T) => string, toProperty?: string, headers?: Record<string, string>): OperatorFunction<T, R>;
    getJsonOperator<T, R = T>(urlParam?: string | ((value: T) => string), toProperty?: string, headers?: Record<string, string>): OperatorFunction<T, R> {
        return rxjs.mergeMap(val => rxjs.from(this.getOperatorResult<T, R>(val, toProperty, this.getJson(this.getOperatorUrl(val, urlParam), headers)))); 
    }

    getTextOperator<T>(): OperatorFunction<T, string>;
    getTextOperator<T, R = T>(url: string, toProperty?: string, headers?: Record<string, string>): OperatorFunction<T, R>;
    getTextOperator<T, R = T>(getUrl: (value: T) => string, toProperty?: string, headers?: Record<string, string>): OperatorFunction<T, R>;
    getTextOperator<T, R = T>(urlParam?: string | ((value: T) => string), toProperty?: string, headers?: Record<string, string>): OperatorFunction<T, R> {
        return rxjs.mergeMap(val => rxjs.from(this.getOperatorResult<T, R>(val, toProperty, this.getText(this.getOperatorUrl(val, urlParam), headers)))); 
    }

    getBlobOperator<T>(): OperatorFunction<T, Blob>;
    getBlobOperator<T, R = T>(url: string, toProperty?: string, headers?: Record<string, string>): OperatorFunction<T, R>;
    getBlobOperator<T, R = T>(getUrl: (value: T) => string, toProperty?: string, headers?: Record<string, string>): OperatorFunction<T, R>;
    getBlobOperator<T, R = T>(urlParam?: string | ((value: T) => string), toProperty?: string, headers?: Record<string, string>): OperatorFunction<T, R> {
        return rxjs.mergeMap(val => rxjs.from(this.getOperatorResult<T, R>(val, toProperty, this.getBlob(this.getOperatorUrl(val, urlParam), headers)))); 
    }

    // POST

    async post(body: string, url?: string, headers?: Record<string, string>): Promise<Response> {
        let init: RequestInit = {
            method: 'POST',
            headers: {...this.headers},
            body
        };
        
        const res = await this.fetch(url, init);
        return res;
    }
    async postJson(body: any, url?: string, headers?: Record<string, string>): Promise<any> {
        headers = {...headers, "Content-Type": "application/json"};
        const resp = await this.post(JSON.stringify(body), url, headers);
        const res = await resp.json();
        return res;
    }
    async postText(body: string, url?: string, headers?: Record<string, string>): Promise<string> {
        const resp = await this.post(body, url, headers);
        const res = await resp.text();
        return res;
    }

    postJsonOperator<T, R = T>(): OperatorFunction<T, R>;

    postJsonOperator<T, R = T>(body: any): OperatorFunction<T, R>;
    postJsonOperator<T, R = T>(body: any, url: string, toProperty?: string, headers?: Record<string, string>): OperatorFunction<T, R>;
    postJsonOperator<T, R = T>(body: any, getUrl: (value: T) => string, toProperty?: string, headers?: Record<string, string>): OperatorFunction<T, R>;

    postJsonOperator<T, R = T>(getBody: (value: T) => any): OperatorFunction<T, R>;
    postJsonOperator<T, R = T>(getBody: (value: T) => any, url: string, toProperty?: string, headers?: Record<string, string>): OperatorFunction<T, R>;
    postJsonOperator<T, R = T>(getBody: (value: T) => any, getUrl: (value: T) => string, toProperty?: string, headers?: Record<string, string>): OperatorFunction<T, R>;

    postJsonOperator<T, R = T>(bodyParam?: any | ((value: T) => any), urlParam?: string | ((value: T) => string), toProperty?: string, headers?: Record<string, string>): OperatorFunction<T, R> {
        return rxjs.mergeMap(val => rxjs.from(this.getOperatorResult<T, R>(val, toProperty, this.postJson(this.getOperatorBody(val, bodyParam), this.getOperatorUrl(val, urlParam), headers)))); 
    }

    // PUT

    async put(body: string, url?: string, headers?: Record<string, string>): Promise<Response> {
        let init: RequestInit = {
            method: 'PUT',
            headers: {...this.headers},
            body
        };
        
        const res = await this.fetch(url, init);
        return res;
    }
    async putJson(body: any, url?: string, headers?: Record<string, string>): Promise<any> {
        headers = {...headers, "Content-Type": "application/json"};
        const resp = await this.put(JSON.stringify(body), url, headers);
        const res = await resp.json();
        return res;
    }
    async putText(body: string, url?: string, headers?: Record<string, string>): Promise<string> {
        const resp = await this.put(body, url, headers);
        const res = await resp.text();
        return res;
    }

    putJsonOperator<T, R = T>(): OperatorFunction<T, R>;

    putJsonOperator<T, R = T>(body: any): OperatorFunction<T, R>;
    putJsonOperator<T, R = T>(body: any, url: string, toProperty?: string, headers?: Record<string, string>): OperatorFunction<T, R>;
    putJsonOperator<T, R = T>(body: any, getUrl: (value: T) => string, toProperty?: string, headers?: Record<string, string>): OperatorFunction<T, R>;

    putJsonOperator<T, R = T>(getBody: (value: T) => any): OperatorFunction<T, R>;
    putJsonOperator<T, R = T>(getBody: (value: T) => any, url: string, toProperty?: string, headers?: Record<string, string>): OperatorFunction<T, R>;
    putJsonOperator<T, R = T>(getBody: (value: T) => any, getUrl: (value: T) => string, toProperty?: string, headers?: Record<string, string>): OperatorFunction<T, R>;

    putJsonOperator<T, R = T>(bodyParam?: any | ((value: T) => any), urlParam?: string | ((value: T) => string), toProperty?: string, headers?: Record<string, string>): OperatorFunction<T, R> {
        return rxjs.mergeMap(val => rxjs.from(this.getOperatorResult<T, R>(val, toProperty, this.putJson(this.getOperatorBody(val, bodyParam), this.getOperatorUrl(val, urlParam), headers)))); 
    }

    // FETCH

    async fetch(url: string): Promise<Response>;
    async fetch(url: string, method: 'GET' | 'POST' | 'PUT' | 'DELETE', headers: Record<string, string>): Promise<Response>;
    async fetch(url: string, init: RequestInit): Promise<Response>;
    async fetch(url: string, p1?: any, p2?: Record<string, string>): Promise<Response> {
        let init: RequestInit = {
            headers: {...this.headers},
            agent: this.agent,
        };

        if (!p1) init.method = 'GET';
        else {
            if (typeof p1 !== 'string') init = p1;
            else {
                init.method = p1;
                init.headers = {...init.headers, ...p2}
            }
        }

        const res: Response = await fetch(this.getUrl(url), init);
        return res;
    }


    protected getUrl(relativeUrl: string) {
        if (this.baseUrl) relativeUrl = pathJoin([this.baseUrl, relativeUrl], '/');
        return relativeUrl;
    }

    protected getOperatorUrl<T>(v: T, urlParam?: string | ((value: T) => string)): string {
        if (!urlParam) return '';
        if (typeof urlParam === 'string') return urlParam;
        return urlParam(v);
    }

    protected getOperatorBody<T>(v: T, bodyParam?: string | ((value: T) => string)): any {
        if (!bodyParam) return v;
        if (typeof bodyParam === 'function') return bodyParam(v);
        return bodyParam;
    }

    protected async getOperatorResult<T, R>(val: T, toProperty: string, resPromise: Promise<any>): Promise<R> {
        const res = await resPromise;
        if (typeof toProperty === 'undefined') return res;
        if (!toProperty) return val as unknown as R; 
        val[toProperty] = res;
        return val as unknown as R;
    }
}