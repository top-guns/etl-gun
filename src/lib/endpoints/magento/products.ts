import { BaseCollection, CollectionOptions } from "../../core/collection.js";
import { BaseObservable } from '../../core/observable.js';
import { Endpoint } from './endpoint.js';

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

export class ProductsCollection extends BaseCollection<Partial<Product>> {
    protected static instanceNo = 0;

    constructor(endpoint: Endpoint, collectionName: string, options: CollectionOptions<Partial<Product>> = {}) {
        ProductsCollection.instanceNo++;
        super(endpoint, collectionName, options);
    }

    public select(where: Partial<Product> = {}, fields: (keyof Product)[] = null): BaseObservable<Partial<Product>> {
        const observable = new BaseObservable<Partial<Product>>(this, (subscriber) => {
            (async () => {
                try {
                    const products = await ProductsCollection.getProducts(this.endpoint, where, fields);

                    this.sendStartEvent();
                    for (const p of products) {
                        if (subscriber.closed) break;
                        await this.waitWhilePaused();
                        this.sendReciveEvent(p);
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
        return await this.endpoint.push('/rest/V1/products', {product: value}) as Partial<Product>;
    }

    public static async getProducts(endpoint: Endpoint, where: Partial<Product> = {}, fields: (keyof Product)[] = null) {
        let getParams = '';
        if (!where) where = {};

        let n = 0;
        for (let key in where) {
            if (!where.hasOwnProperty(key)) continue;
            if (getParams) getParams += '&';
            getParams += `searchCriteria[filterGroups][${n}][filters][0][field]=${key}&`;
            getParams += `searchCriteria[filterGroups][${n}][filters][0][value]=${where[key]}`;
            n++;
        }
        if (!getParams) getParams += 'searchCriteria';

        if (fields) getParams += `&fields=items[${fields.join(',')}]`;

        const products = await endpoint.get('/rest/V1/products?' + getParams) as {items: Partial<Product>[]};
        return products.items;
    }

    public async updateStockQuantity(product: Partial<Product>, quantity: number);
    public async updateStockQuantity(sku: string, quantity: number);
    public async updateStockQuantity(product: Partial<Product> | string, quantity: number) {
        if (typeof product !== 'string') product = product.sku;
        return await this.endpoint.put(`/rest/V1/products/${product}/stockItems/1`, { stockItem: { qty: quantity, is_in_stock: quantity > 0 } }) as Partial<Product>;
    }

    get endpoint(): Endpoint {
        return super.endpoint as Endpoint;
    }
}
