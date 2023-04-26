import fetch, { RequestInit } from 'node-fetch';
//let {fetch, RequestInit} = await import('node-fetch');
import https from 'node:https';
import { BaseEndpoint} from "../../core/endpoint.js";
import { CollectionOptions } from "../../core/collection.js";
import { pathJoin } from '../../utils/index.js';
import { Product, ProductsCollection } from './products.js';


enum COLLECTIONS_NAMES {
    products = 'products',
    categories = 'categories'
}

export class Endpoint extends BaseEndpoint {
    protected _magentoUrl: string;
    protected login: string;
    protected password: string;
    protected token: string;
    protected rejectUnauthorized: boolean;
    protected agent: https.Agent;

    constructor(magentoUrl: string, login: string, password: string, rejectUnauthorized: boolean = true) {
        super();

        this._magentoUrl = magentoUrl;
        this.login = login;
        this.password = password;
        this.rejectUnauthorized = rejectUnauthorized;

        this.agent = rejectUnauthorized ? null : new https.Agent({
            rejectUnauthorized
        });
    }

    async updateToken() {
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
        
        this.token = await (await fetch(this.getUrl('/rest/V1/integration/admin/token'), init)).json() as string;
    }

    protected getUrl(...parts: string[]) {
        return pathJoin([this._magentoUrl, ...parts], '/');
    }

    get url(): string {
        return this._magentoUrl;
    }

    async fetch(relativeUrl: string) {
        let init: RequestInit = {
            agent: this.agent,
            headers: {
                "Authorization": 'Bearer ' + this.token,
                "Content-Type": "application/json"
            }
        }

        return await (await fetch(this.getUrl(relativeUrl), init)).json();
    }

    async push(relativeUrl: string, value: any) {
        let init: RequestInit = {
            method: "POST", 
            agent: this.agent,
            headers: {
                "Authorization": 'Bearer ' + this.token,
                "Content-Type": "application/json"
            },
            body: JSON.stringify(value)
        }
        return await (await fetch(this.getUrl(relativeUrl), init)).json();
    }

    getProducts(options: CollectionOptions<Partial<Product>> = {}): ProductsCollection {
        options.displayName ??= `products`;
        const collection = new ProductsCollection(this, COLLECTIONS_NAMES.products, options);
        this._addCollection(COLLECTIONS_NAMES.products, collection);
        return collection;
    }

    releaseProducts() {
        this._removeCollection(COLLECTIONS_NAMES.products);
    }

    get displayName(): string {
        return `Magento (${this._magentoUrl})`;
    }
}

export function getEndpoint(magentoUrl: string, login: string, password: string, rejectUnauthorized: boolean = true): Endpoint {
    return new Endpoint(magentoUrl, login, password, rejectUnauthorized);
}
