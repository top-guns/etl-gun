import { CollectionOptions } from '../../core/base_collection.js';
import { BaseObservable } from '../../core/observable.js';
import { Endpoint } from './endpoint.js';
import { MagentoCollection } from './magento_collection.js';
import { Product, ProductCollection } from "./products.js";

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

export class StockCollection extends MagentoCollection<Partial<StockItem>> {
    protected static instanceNo = 0;

    constructor(endpoint: Endpoint, collectionName: string, options: CollectionOptions<Partial<StockItem>> = {}) {
        StockCollection.instanceNo++;
        super(endpoint, collectionName, 'stockItem', 'stockItems', options);
    }

    public select(): BaseObservable<Partial<StockItem>>;
    public select(sku: string): BaseObservable<Partial<StockItem>>;
    public select(product: Partial<Product>): BaseObservable<Partial<StockItem>>;
    public select(param?: Partial<Product> | string | any): BaseObservable<Partial<StockItem>> {
        const observable = new BaseObservable<Partial<StockItem>>(this, (subscriber) => {
            (async () => {
                try {
                    if (param && typeof param !== 'string') param = {sku: param};
                    
                    const products = await ProductCollection._findWithoutEvent(this.endpoint, param); //, ['sku']

                    this.sendStartEvent();
                    for (const p of products) {
                        if (subscriber.closed) break;
                        await this.waitWhilePaused();
                        const result: any = await this.endpoint.fetchJson(this.getResourceUrl(p.sku));
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

    public async getStockItem(product: {sku: string}): Promise<StockItem>;
    public async getStockItem(sku: string): Promise<StockItem>;
    public async getStockItem(product: {sku: string} | string): Promise<StockItem> {
        if (typeof product !== 'string') product = product.sku;
        return await this.endpoint.fetchJson(this.getResourceUrl(product)) as StockItem;
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
