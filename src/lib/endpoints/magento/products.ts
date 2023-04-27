import * as rx from 'rxjs';
import { OperatorFunction } from 'rxjs';
import { BaseCollection, CollectionOptions } from "../../core/collection.js";
import { BaseObservable } from '../../core/observable.js';
import { mapAsync } from '../../index.js';
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
        return await this.endpoint.post('/rest/V1/products', {product: value}) as Partial<Product>;
    }

    public static async getProducts(endpoint: Endpoint, where: Partial<Product> = {}, fields: (keyof Product)[] = null): Promise<Partial<Product>[]> {
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

    // "product": {
    //     "attribute_set_id": 4,
    //     "type_id": "simple",
    //     "sku": "B201-SKU",
    //     "name": "B201",
    //     "price": 25,
    //     "status": 1,
    //     "custom_attributes": {
    //       "description": "Heavy Duty Brake Cables",
    //       "meta_description": "Some describing text",
    //       "image" : "/w/i/sample_1.jpg",
    //       "small_image": "/w/i/sample_2.jpg",
    //       "thumbnail": "/w/i/sample_3.jpg"
    //     }
    // }

    async uploadImage(product: {sku: string} | string, imageContents: Blob, filename: string, label: string, type: "image/png" | "image/jpeg" | string): Promise<number> {
        //const imageBase64 = URL.createObjectURL(imageContents);
        //const imageBase64 = Buffer.from('username:password', 'utf8').toString('base64') 
        const buf = await imageContents.arrayBuffer();
        const imageBase64 = Buffer.from(buf).toString('base64');

        const body = {
            entry: {
                media_type: "image",
                label,                  // "I am an image!"
                position: 1,
                //disabled: false,
                //file: filename,
                types: [
                    "image",
                    //"small_image",
                    //"thumbnail"
                ],
                content: {
                    base64_encoded_data: imageBase64,
                    type,
                    name: filename      // "choose_any_name.png"
                }
            }
        }
        if (typeof product !== 'string') product = product.sku;
        return await this.endpoint.post(`/rest/V1/products/${product}/media`, body);
    }

    uploadImageOperator<T>(func: (value: T) => {product: {sku: string} | string, imageContents: Blob, filename: string, label: string, type: "image/png" | "image/jpeg" | string}): OperatorFunction<T, T> {
        const f = async (v: T) => {
            const params = await func(v);
            await this.uploadImage(params.product, params.imageContents, params.filename, params.label, params.type);
            return v;
        }
        return mapAsync( p => f(p)); 
    }

    get endpoint(): Endpoint {
        return super.endpoint as Endpoint;
    }
}
