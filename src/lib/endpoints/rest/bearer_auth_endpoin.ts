import fetch, { RequestInit } from 'node-fetch';
import { RestEndpoint, RestFetchOptions} from "./endpoint.js";

const INFINITE = -1;

export class BearerAuthEndpoint extends RestEndpoint {
    protected login: string;
    protected password: string;
    protected tokenUrl: string;
    protected tokenLifetime: number;

    protected token: string | undefined;
    protected tokenTS: Date | null = null;

    constructor(apiUrl: string, login: string, password: string, tokenUrl: string, tokenLifetime: number = INFINITE, rejectUnauthorized: boolean = true) {
        super(apiUrl, rejectUnauthorized);
        this.login = login;
        this.password = password;
        this.tokenUrl = tokenUrl;
        this.tokenLifetime = tokenLifetime;
    }

    async fetchJson<T = any>(url: string, method: 'GET' | 'PUT' | 'POST' | 'DELETE' = 'GET', options: RestFetchOptions = {}): Promise<T> {
        await this.updateToken();
        
        return super.fetchJson(url, method, { 
            body: options.body,
            params: options.params,
            headers: { 
                "Authorization": 'Bearer ' + this.token,
                "Content-Type": "application/json",
                ...options.headers 
            }
        });
    }

    protected async updateToken() {
        if (this.tokenTS && (this.tokenLifetime === INFINITE || new Date().getTime() - this.tokenTS.getTime() < this.tokenLifetime)) return;
        
        let init: RequestInit = {
            method: "POST", 
            agent: this.agent,
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                username: this.login,
                password: this.password
            })
        }
        
        const url = this.makeUrl([this.tokenUrl]);
        const res = await fetch(url, init);

        if (!res.ok) throw new Error(await res.text());

        this.token = await res.json() as string;
        this.tokenTS = new Date();
    }
}
