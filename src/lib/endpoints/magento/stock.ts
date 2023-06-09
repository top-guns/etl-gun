import * as ix from 'ix';
import { CollectionOptions } from '../../core/base_collection.js';
import { BaseObservable } from '../../core/observable.js';
import { Endpoint } from './endpoint.js';
import { Product, ProductCollection } from "./products.js";
import { ResourceCollection } from '../rest/resource_collection.js';

export type StockItem = {
    "item_id": number;
    "product_id": number;
    "stock_id": number;
    "qty": number;
    "is_in_stock": boolean;
    "is_qty_decimal": boolean;
    "show_default_notification_message": boolean;
    "use_config_min_qty": boolean;
    "min_qty": number;
    "use_config_min_sale_qty": number;
    "min_sale_qty": number;
    "use_config_max_sale_qty": boolean;
    "max_sale_qty": number;
    "use_config_backorders": boolean;
    "backorders": number;
    "use_config_notify_stock_qty": boolean;
    "notify_stock_qty": number;
    "use_config_qty_increments": boolean;
    "qty_increments": number;
    "use_config_enable_qty_inc": boolean;
    "enable_qty_increments": boolean;
    "use_config_manage_stock": boolean;
    "manage_stock": boolean;
    "low_stock_date": Date | null,
    "is_decimal_divided": boolean;
    "stock_status_changed_auto": number;
}


export class StockCollection extends ResourceCollection<StockItem> {
    protected static instanceNo = 0;

    constructor(endpoint: Endpoint, collectionName: string, options: CollectionOptions<StockItem> = {}) {
        StockCollection.instanceNo++;
        super(endpoint, collectionName, 'stockItem', 'stockItems', options);
    }

    protected async _select(product?: string | Partial<Product>): Promise<StockItem[]> {
        const sku: string = typeof product === 'string' ? product : product?.sku!;
        const products = await ProductCollection._findWithoutEvent(this.endpoint, { sku }); //, ['sku']
        const result: StockItem[] = [];
        for (const p of products) {
            const res: any = await this.endpoint.fetchJson(this.getResourceUrl(p.sku));
            result.push(res);
        }
        return result;
    }

    public select(sku: string): Promise<StockItem[]>;
    public select(product: Partial<Product>): Promise<StockItem[]>;
    public async select(param?: string | Partial<Product>): Promise<StockItem[]> {
        return super.select(param);
    }

    public selectGen(sku: string): AsyncGenerator<StockItem, void, void>;
    public selectGen(product: Partial<Product>): AsyncGenerator<StockItem, void, void>;
    public async* selectGen(params?: any): AsyncGenerator<StockItem, void, void> {
        for await (const item of super.selectGen(params)) yield item;
    }

    public selectRx(sku: string): BaseObservable<StockItem>;
    public selectRx(product: Partial<Product>): BaseObservable<StockItem>;
    public selectRx(params?: any): BaseObservable<StockItem> {
        return super.selectRx(params);
    }

    public selectIx(sku: string): ix.AsyncIterable<StockItem>;
    public selectIx(product: Partial<Product>): ix.AsyncIterable<StockItem>;
    public selectIx(params?: any): ix.AsyncIterable<StockItem> {
        return super.selectIx(params);
    }

    public selectStream(sku: string): ReadableStream<StockItem>;
    public selectStream(product: Partial<Product>): ReadableStream<StockItem>;
    public selectStream(params?: any): ReadableStream<StockItem> {
        return super.selectStream(params);
    }

    public async updateStockQuantity(product: {sku: string}, quantity: number);
    public async updateStockQuantity(sku: string, quantity: number);
    public async updateStockQuantity(product: {sku: string} | string, quantity: number) {
        if (typeof product !== 'string') product = product.sku;
        return await this.endpoint.fetchJson(`products/${product}/${this.resourceNameS}/1`, 'PUT', {body: { stockItem: { qty: quantity, is_in_stock: quantity > 0 } }}) as Partial<Product>;
    }

    get endpoint(): Endpoint {
        return super.endpoint as Endpoint;
    }
}
