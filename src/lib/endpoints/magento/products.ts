import { OperatorFunction } from 'rxjs';
import { CollectionOptions } from '../../core/base_collection.js';
import { mapAsync } from '../../index.js';
import { Endpoint } from './endpoint.js';
import { MagentoCollection } from './magento_collection.js';

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

export class ProductCollection extends MagentoCollection<Product> {
    protected static instanceNo = 0;

    constructor(endpoint: Endpoint, collectionName: string, options: CollectionOptions<Product> = {}) {
        ProductCollection.instanceNo++;
        super(endpoint, collectionName, 'product', 'products', options);
    }

    protected async _selectOne(sku: string): Promise<Product> {
        return super._selectOne(sku);
    }

    public async selectOne(sku: string): Promise<Product> {
        return super.selectOne(sku);
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
                disabled: false,
                //file: filename,
                types: [
                    "image",
                    "small_image",
                    "thumbnail"
                ],
                content: {
                    base64_encoded_data: imageBase64,
                    type,
                    name: filename      // "choose_any_name.png"
                }
            }
        }
        if (typeof product !== 'string') product = product.sku;
        return await this.endpoint.fetchJson(`${this.resourceNameS}/${product}/media`, 'POST', { body });
    }

    uploadImageOperator<T>(func: (value: T) => {product: {sku: string} | string, imageContents: Blob, filename: string, label: string, type: "image/png" | "image/jpeg" | string}): OperatorFunction<T, T> {
        const f = async (v: T) => {
            const params = await func(v);
            await this.uploadImage(params.product, params.imageContents, params.filename, params.label, params.type);
            return v;
        }
        return mapAsync( p => f(p)); 
    }

    static async _findWithoutEvent<TT = any>(endpoint: Endpoint, where?: Partial<TT>): Promise<TT[]> {
        const searchParams = ProductCollection.convertWhereToQueryParams(where);
        let url = endpoint.makeUrl(['products'], [searchParams])
        
        const res = await endpoint.fetchJson(url);
        return res ? (typeof res.items === 'undefined' ? res : res.items) : null;
    }
}
