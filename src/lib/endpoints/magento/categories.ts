import { CollectionOptions } from '../../core/base_collection.js';
import { Endpoint } from './endpoint.js';
import { MagentoCollection } from './magento_collection.js';

export type Category = {
    id: number;
    parent_id: number;
    name: string;
    is_active: boolean;
    position: number;
    level: number;
    product_count: number;
    children_data: any[];
}

export class CategoryCollection extends MagentoCollection<Category> {
    protected static instanceNo = 0;

    constructor(endpoint: Endpoint, collectionName: string, options: CollectionOptions<Category> = {}) {
        CategoryCollection.instanceNo++;
        super(endpoint, collectionName, 'category', 'categories', options);
    }
}
