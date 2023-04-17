import fetch, { RequestInit } from 'node-fetch';
//let {fetch, RequestInit} = await import('node-fetch');
import https from 'node:https';
import { BaseEndpoint} from "../core/endpoint.js";
import { BaseCollection, CollectionGuiOptions } from "../core/collection.js";
import { EtlObservable } from '../core/observable.js';
import { pathJoin } from '../utils/index.js';

export type CustomAttributeCodes = 'release_date' | 'options_container' | 'gift_message_available' | 'msrp_display_actual_price_type' | 'url_key' | 'required_options' | 'has_options' | 'tax_class_id' | 'category_ids' | 'description' | string;

export type Product = {
    id: number;
    sku: string;
    name: string;
    attribute_set_id: number;
    price: number;
    status: number;
    visibility: number;
    type_id: string;
    created_at: string;
    updated_at: string;
    extension_attributes: {
        website_ids: number[];
        category_links: {}[];
    }
    product_links: [];
    options: [];
    media_gallery_entries: [];
    tier_prices: [];
    custom_attributes: {attribute_code: CustomAttributeCodes, value: any}[];
}

enum COLLECTIONS_NAMES {
    products = 'products',
    categories = 'categories'
}

export class MagentoEndpoint extends BaseEndpoint {
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

    getProducts(guiOptions: CollectionGuiOptions<Partial<Product>> = {}): ProductsCollection {
        guiOptions.displayName ??= `Products`;
        const collection = new ProductsCollection(this, guiOptions);
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

export class ProductsCollection extends BaseCollection<Partial<Product>> {
    protected static instanceNo = 0;

    constructor(endpoint: MagentoEndpoint, guiOptions: CollectionGuiOptions<Partial<Product>> = {}) {
        ProductsCollection.instanceNo++;
        super(endpoint, guiOptions);
    }

    public select(where: Partial<Product> = {}, fields: (keyof Product)[] = null): EtlObservable<Partial<Product>> {
        const observable = new EtlObservable<Partial<Product>>((subscriber) => {
            (async () => {
                try {
                    await this.endpoint.updateToken();

                    let getParams = '';

                    let n = 0;
                    for (let key in where) {
                        if (!where.hasOwnProperty(key)) continue;
                        if (getParams) getParams += '&';
                        getParams += `searchCriteria[filterGroups][${n}][filters][0][field]=${key}&`;
                        getParams += `searchCriteria[filterGroups][${n}][filters][0][value]=${where[key]}`;
                        n++;
                    }
                    if (!getParams) getParams += 'searchCriteria';

                    if (fields) getParams += `fields=items[${fields.join(',')}]`;

                    const products = await this.endpoint.fetch('/rest/V1/products?' + getParams) as {items: Partial<Product>[]};

                    this.sendStartEvent();
                    for (const p of products.items) {
                        await this.waitWhilePaused();
                        this.sendValueEvent(p);
                        subscriber.next(p);
                    }
                    subscriber.complete();
                    this.sendEndEvent();
                }
                catch(err) {
                    this.sendErrorEvent(err);
                    subscriber.error(err);
                }
            })();
        });
        return observable;
    }

    public async insert(value: Omit<Partial<Product>, 'id'>) {
        super.insert(value as Partial<Product>);
        await this.endpoint.updateToken();
        return await this.endpoint.push('/rest/V1/products', {product: value}) as Partial<Product>;
    }

    get endpoint(): MagentoEndpoint {
        return super.endpoint as MagentoEndpoint;
    }
}
