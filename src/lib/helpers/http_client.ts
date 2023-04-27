import { pathJoin } from '../utils/index.js';

export class HttpClientHelper {
    protected baseUrl: string;
    protected headers: Record<string, string>;

    constructor(baseUrl?: string, headers?: Record<string, string>) {
        this.baseUrl = baseUrl;
        this.headers = headers;
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

    // FETCH

    async fetch(url: string): Promise<Response>;
    async fetch(url: string, method: 'GET' | 'POST' | 'PUT' | 'DELETE', headers: Record<string, string>): Promise<Response>;
    async fetch(url: string, init: RequestInit): Promise<Response>;
    async fetch(url: string, p1?: any, p2?: Record<string, string>): Promise<Response> {
        let init: RequestInit = {
            headers: {...this.headers}
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
}