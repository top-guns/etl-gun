import { CollectionOptions } from '../../core/base_collection.js';
import { pathJoin } from '../../utils/index.js';
import { Category, CategoryCollection } from './categories.js';
import { Product, ProductCollection } from './products.js';
import { StockItem, StockCollection } from './stock.js';
import { BearerAuthEndpoint } from '../rest/bearer_auth_endpoin.js';


enum COLLECTIONS_NAMES {
    products = 'products',
    categories = 'categories',
    stock_items = 'stock_items'
}

// Generally, the admin token in Magento 2 is valid for four hours. 
// It can be changed from Stores > Settings > Configuration > Services > OAuth > Access Token Expiration > Admin Token Lifetime (hours).
const TOKEN_LIFE_TIME = 4 * 60 * 60 * 1000;

const GET_TOKEN_URL = `integration/admin/token`;

export class Endpoint extends BearerAuthEndpoint {

    constructor(magentoUrl: string, login: string, password: string, rejectUnauthorized: boolean = true, tokenLifetime: number = TOKEN_LIFE_TIME) {
        let apiUrl = magentoUrl;
        if (!apiUrl.endsWith('/V1')) {
            if (!apiUrl.endsWith('/rest')) apiUrl = pathJoin([apiUrl, 'rest']);
            apiUrl = pathJoin([apiUrl, 'V1']);
        }

        super(apiUrl, login, password, GET_TOKEN_URL, tokenLifetime, rejectUnauthorized);
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
