import _ from 'lodash';
import fetch, { RequestInit } from 'node-fetch';
import https from 'node:https';
import { BaseEndpoint} from "../../core/endpoint.js";
import { pathJoin } from '../../utils/index.js';


export type RestFetchOptions = {
    body?: any;
    headers?: {};
    params?: {};
}


export class RestEndpoint extends BaseEndpoint {
    protected apiUrl: string;
    protected agent: https.Agent | null;

    constructor(apiUrl: string, rejectUnauthorized: boolean = true) {
        super();
        this.apiUrl = apiUrl;
        this.agent = rejectUnauthorized ? null : new https.Agent({
            rejectUnauthorized
        });
    }

    async fetchJson<T = any>(url: string, method: 'GET' | 'PUT' | 'POST' | 'DELETE' = 'GET', options: RestFetchOptions = {}): Promise<T> {
        const init: RequestInit = {
            method,
            agent: this.agent,
            headers: {
                "Content-Type": "application/json",
                ...options.headers
            }
        }
        if (options.body) init.body = JSON.stringify(options.body);

        let fullUrl = this.makeUrl([url], [options.params!]);
        //console.log(fullUrl)
        const res = await fetch(fullUrl, init);
        //console.log(res)
        const jsonRes = await res.json();
        //console.log(jsonRes);
        //for (let key in jsonRes) console.log(key)
        return jsonRes as T;
    }

    get displayName(): string {
        return `HTTP REST (${this.apiUrl})`;
    }

    makeUrl(urlParts: string[] = [], params: ({}|string)[] = []): string {
        let url = '';
        if (urlParts && urlParts.length) {
            const subUrl = pathJoin(urlParts, '/');
            url = subUrl.startsWith('http') ? subUrl : pathJoin([this.apiUrl, subUrl], '/');
        }
        else url = this.apiUrl;

        let queryParams = '';
        for (let p of params) {
            if (queryParams) queryParams += '&';
            if (typeof p === 'string') queryParams += p;
            else queryParams += this.converHashToQueryParams(p);
        }

        if (queryParams) {
            if (url.includes('?')) url += '&';
            else url += '?';
            url += queryParams;
        }

        return url;
    }

    converHashToQueryParams(where: {} = {}): string {
        let queryParams = _.toQuery(where);
        // const queryParams = new URLSearchParams(where as {}).toString();
        return queryParams;
    }
}
