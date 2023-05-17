import fetch, { RequestInit } from 'node-fetch';
//let {fetch, RequestInit} = await import('node-fetch');
import https from 'node:https';
import { BaseEndpoint} from "../../core/endpoint.js";
import { CollectionOptions } from '../../core/readonly_collection.js';
import { pathJoin } from '../../utils/index.js';
import { Category, CategoryCollection } from './categories.js';
import { Product, ProductCollection } from './products.js';
import { StockItem, StockCollection } from './stock.js';


enum COLLECTIONS_NAMES {
    products = 'products',
    categories = 'categories',
    stock_items = 'stock_items'
}

const TOKEN_LIFE_TIME = 1 * 60 * 60 * 1000;

export class Endpoint extends BaseEndpoint {
    protected _magentoUrl: string;
    protected login: string;
    protected password: string;
    protected token: string;
    protected tokenTS: Date = null;
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
        if (this.tokenTS && new Date().getTime() - this.tokenTS.getTime() < TOKEN_LIFE_TIME) return;
        
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
        this.tokenTS = new Date();
    }

    protected getUrl(...parts: string[]) {
        return pathJoin([this._magentoUrl, ...parts], '/');
    }

    get url(): string {
        return this._magentoUrl;
    }

    async get<T = any>(relativeUrl: string): Promise<T> {
        await this.updateToken();
        let init: RequestInit = {
            agent: this.agent,
            headers: {
                "Authorization": 'Bearer ' + this.token,
                "Content-Type": "application/json"
            }
        }

        return (await (await fetch(this.getUrl(relativeUrl), init)).json()) as T;
    }

    async post<T = any>(relativeUrl: string, value: any): Promise<T> {
        await this.updateToken();
        let init: RequestInit = {
            method: "POST", 
            agent: this.agent,
            headers: {
                "Authorization": 'Bearer ' + this.token,
                "Content-Type": "application/json"
            },
            body: JSON.stringify(value)
        }
        return (await (await fetch(this.getUrl(relativeUrl), init)).json() as T);
    }

    async put<T = any>(relativeUrl: string, value: any): Promise<T> {
        await this.updateToken();
        let init: RequestInit = {
            method: "PUT", 
            agent: this.agent,
            headers: {
                "Authorization": 'Bearer ' + this.token,
                "Content-Type": "application/json"
            },
            body: JSON.stringify(value)
        }
        return (await (await fetch(this.getUrl(relativeUrl), init)).json()) as T;
    }

    getProducts(options: CollectionOptions<Partial<Product>> = {}): ProductCollection {
        options.displayName ??= `products`;
        const collection = new ProductCollection(this, COLLECTIONS_NAMES.products, options);
        this._addCollection(COLLECTIONS_NAMES.products, collection);
        return collection;
    }

    releaseProducts() {
        this._removeCollection(COLLECTIONS_NAMES.products);
    }

    getCategories(options: CollectionOptions<Partial<Category>> = {}): CategoryCollection {
        options.displayName ??= `categories`;
        const collection = new CategoryCollection(this, COLLECTIONS_NAMES.categories, options);
        this._addCollection(COLLECTIONS_NAMES.categories, collection);
        return collection;
    }

    releaseCategories() {
        this._removeCollection(COLLECTIONS_NAMES.categories);
    }

    getStockItems(options: CollectionOptions<StockItem> = {}): StockCollection {
        options.displayName ??= `stock items`;
        const collection = new StockCollection(this, COLLECTIONS_NAMES.stock_items, options);
        this._addCollection(COLLECTIONS_NAMES.stock_items, collection);
        return collection;
    }

    releaseStockItems() {
        this._removeCollection(COLLECTIONS_NAMES.stock_items);
    }

    get displayName(): string {
        return `Magento (${this._magentoUrl})`;
    }
}

export function getEndpoint(magentoUrl: string, login: string, password: string, rejectUnauthorized: boolean = true): Endpoint {
    return new Endpoint(magentoUrl, login, password, rejectUnauthorized);
}
