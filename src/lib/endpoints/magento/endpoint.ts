import fetch, { RequestInit } from 'node-fetch';
//let {fetch, RequestInit} = await import('node-fetch');
import https from 'node:https';
import { CollectionOptions } from '../../core/base_collection.js';
import { BaseEndpoint} from "../../core/endpoint.js";
import { pathJoin } from '../../utils/index.js';
import { RestEndpoint, RestFetchOptions } from '../rest/endpoint.js';
import { Category, CategoryCollection } from './categories.js';
import { Product, ProductCollection } from './products.js';
import { StockItem, StockCollection } from './stock.js';


enum COLLECTIONS_NAMES {
    products = 'products',
    categories = 'categories',
    stock_items = 'stock_items'
}

const TOKEN_LIFE_TIME = 1 * 60 * 60 * 1000;

export class Endpoint extends RestEndpoint {
    protected login: string;
    protected password: string;
    protected token: string | undefined;
    protected tokenTS: Date | null = null;


    constructor(magentoUrl: string, login: string, password: string, rejectUnauthorized: boolean = true) {
        let apiUrl = magentoUrl;
        if (!apiUrl.endsWith('/V1')) {
            if (!apiUrl.endsWith('/rest')) apiUrl = pathJoin([apiUrl, 'rest']);
            apiUrl = pathJoin([apiUrl, 'V1']);
        }

        super(apiUrl, rejectUnauthorized);
        this.login = login;
        this.password = password;
    }

    protected async updateToken() {
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
        
        const url = this.makeUrl([`integration/admin/token`]);
        const res = await fetch(url, init);
        this.token = await res.json() as string;
        this.tokenTS = new Date();
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
        return `Magento (${this.apiUrl})`;
    }
}

export function getEndpoint(magentoUrl: string, login: string, password: string, rejectUnauthorized: boolean = true): Endpoint {
    return new Endpoint(magentoUrl, login, password, rejectUnauthorized);
}
