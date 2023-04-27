import { BaseCollection, CollectionOptions } from "../../core/collection.js";
import { BaseObservable } from '../../core/observable.js';
import { Endpoint } from './endpoint.js';
import { Product, ProductsCollection } from "./products.js";

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

export class StockItemsCollection extends BaseCollection<StockItem> {
    protected static instanceNo = 0;

    constructor(endpoint: Endpoint, collectionName: string, options: CollectionOptions<StockItem> = {}) {
        StockItemsCollection.instanceNo++;
        super(endpoint, collectionName, options);
    }

    public select(): BaseObservable<StockItem>;
    public select(sku: string): BaseObservable<StockItem>;
    public select(product: Partial<Product>): BaseObservable<StockItem>;
    public select(param?: Partial<Product> | string | any): BaseObservable<StockItem> {
        const observable = new BaseObservable<StockItem>(this, (subscriber) => {
            (async () => {
                try {
                    if (param && typeof param !== 'string') param = {sku: param};
                    
                    const products = await ProductsCollection.getProducts(this.endpoint, param, ['sku']);

                    this.sendStartEvent();
                    for (const p of products) {
                        if (subscriber.closed) break;
                        await this.waitWhilePaused();
                        const result: any = await this.endpoint.get('/rest/V1/stockItems/' + p.sku);
                        this.sendReciveEvent(result);
                        subscriber.next(result);
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

    public async getProductStockItem(product: {sku: string}): Promise<StockItem>;
    public async getProductStockItem(sku: string): Promise<StockItem>;
    public async getProductStockItem(product: {sku: string} | string): Promise<StockItem> {
        if (typeof product !== 'string') product = product.sku;
        return await this.endpoint.get(`/rest/V1/stockItems/${product}`) as StockItem;
    }

    public async updateProductStockQuantity(product: {sku: string}, quantity: number);
    public async updateProductStockQuantity(sku: string, quantity: number);
    public async updateProductStockQuantity(product: {sku: string} | string, quantity: number) {
        if (typeof product !== 'string') product = product.sku;
        return await this.endpoint.put(`/rest/V1/products/${product}/stockItems/1`, { stockItem: { qty: quantity, is_in_stock: quantity > 0 } }) as Partial<Product>;
    }

    get endpoint(): Endpoint {
        return super.endpoint as Endpoint;
    }
}
