import fetch, { RequestInit } from 'node-fetch';
//let {fetch, RequestInit} = await import('node-fetch');
import https from 'node:https';

import { EndpointGuiOptions, EndpointImpl } from '../core/endpoint';
import { EtlObservable } from '../core/observable';

export type ProductFields = 'id' | 'sku' | 'name' | 'price' | 'status' | 'visibility' | 'type_id' | 'created_at' | 'updated_at' | string;
export type CustomAttributeCodes = 'release_date' | 'options_container' | 'gift_message_available' | 'msrp_display_actual_price_type' | 'url_key' | 'required_options' | 'has_options' | 'tax_class_id' | 'category_ids' | 'description' | string;

export type Product = {
    id: number,
    sku: string,
    name: string,
    attribute_set_id: number,
    price: number,
    status: number,
    visibility: number,
    type_id: string,
    created_at: string,
    updated_at: string,
    extension_attributes: {
        website_ids: number[],
        category_links: {}[]
    },
    product_links: [],
    options: [],
    media_gallery_entries: [],
    tier_prices: [],
    custom_attributes: {attribute_code: CustomAttributeCodes, value: any}[]
}

export type NewProductAttributes = {
    sku: string,
    name: string,
    attribute_set_id: number,
    price?: number,
    status?: number,
    visibility?: number,
    type_id?: string,
    extension_attributes?: {
        website_ids: number[],
        category_links: {}[]
    },
    product_links?: [],
    options?: [],
    media_gallery_entries?: [],
    tier_prices?: [],
    custom_attributes?: {attribute_code: CustomAttributeCodes, value: any}[]
}


export class MagentoProductsEndpoint extends EndpointImpl<Partial<Product>> {
    protected static instanceNo = 0;
    protected magentoUrl: string;
    protected login: string;
    protected password: string;
    protected token: string;
    protected rejectUnauthorized: boolean;
    protected agent: https.Agent;

    constructor(magentoUrl: string, login: string, password: string, rejectUnauthorized: boolean = true, guiOptions: EndpointGuiOptions<Partial<Product>> = {}) {
        guiOptions.displayName = guiOptions.displayName ?? `Magento (${magentoUrl})`;
        MagentoProductsEndpoint.instanceNo++;
        super(guiOptions);
        this.magentoUrl = magentoUrl;
        this.login = login;
        this.password = password;
        this.rejectUnauthorized = rejectUnauthorized;

        this.agent = rejectUnauthorized ? null : new https.Agent({
            rejectUnauthorized
        });
    }

    public read(where: Partial<Product> = {}, fields: ProductFields[] = null): EtlObservable<Partial<Product>> {
        const observable = new EtlObservable<Partial<Product>>((subscriber) => {
            (async () => {
                try {
                    await this.updateToken();

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

                    let init: RequestInit = {
                        agent: this.agent,
                        headers: {
                            "Authorization": 'Bearer ' + this.token,
                            "Content-Type": "application/json"
                        }
                    }

                    const products: {items: Partial<Product>[]} = await (await fetch(this.getUrl('/rest/V1/products?' + getParams), init)).json() as {items: Partial<Product>[]};

                    this.sendStartEvent();
                    for (const p of products.items) {
                        await this.waitWhilePaused();
                        this.sendDataEvent(p);
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

    public async push(value: NewProductAttributes) {
        super.push(value as Partial<Product>);
        await this.updateToken();
        let init: RequestInit = {
            method: "POST", 
            agent: this.agent,
            headers: {
                "Authorization": 'Bearer ' + this.token,
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                product: value
            })
        }
        const res = await (await fetch(this.getUrl('/rest/V1/products'), init)).json();
        return res as Partial<Product>;
    }

    protected async updateToken() {
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

    protected pathJoin(parts: string[], sep: string = '/') {
        return parts
          .map(part => {
            const part2 = part.endsWith(sep) ? part.substring(0, part.length - 1) : part;
            return part2.startsWith(sep) ? part2.substr(1) : part2;
          })
          .join(sep);
    }

    protected getUrl(...parts: string[]) {
        return this.pathJoin([this.magentoUrl, ...parts], '/');
    }

}
