import { RestEndpoint, RestFetchOptions} from "./endpoint.js";


export class BasicAuthEndpoint extends RestEndpoint {
    protected login: string;
    protected password: string;

    constructor(apiUrl: string, login: string, password: string, rejectUnauthorized: boolean = true) {
        super(apiUrl, rejectUnauthorized);
        this.login = login;
        this.password = password;
    }

    async fetchJson<T = any>(url: string, method: 'GET' | 'PUT' | 'POST' | 'DELETE' = 'GET', options: RestFetchOptions = {}): Promise<T> {
        return super.fetchJson(url, method, {
            body: options.body, 
            headers: { 
            'Authorization': 'Basic ' + Buffer.from(this.login + ":" + this.password).toString('base64'),
            ...options.headers 
        }});
    }
}
