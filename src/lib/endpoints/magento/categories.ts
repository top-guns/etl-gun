import { BaseObservable } from '../../core/observable.js';
import { CollectionOptions } from '../../core/readonly_collection.js';
import { UpdatableCollection } from '../../core/updatable_collection.js';
import { Endpoint } from './endpoint.js';

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

export class CategoryCollection extends UpdatableCollection<Partial<Category>> {
    protected static instanceNo = 0;

    constructor(endpoint: Endpoint, collectionName: string, options: CollectionOptions<Partial<Category>> = {}) {
        CategoryCollection.instanceNo++;
        super(endpoint, collectionName, options);
    }

    public select(where: Partial<Category> = {}, fields: (keyof Category)[] = null): BaseObservable<Partial<Category>> {
        const observable = new BaseObservable<Partial<Category>>(this, (subscriber) => {
            (async () => {
                try {
                    const categories = await CategoryCollection.getCategories(this.endpoint, where, fields);                    

                    this.sendStartEvent();
                    for (const cat of categories) {
                        if (subscriber.closed) break;
                        await this.waitWhilePaused();
                        this.sendReciveEvent(cat);
                        subscriber.next(cat);
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

    public async insert(value: Omit<Partial<Category>, 'id'>) {
        await super.insert(value as Partial<Category>);
        return await this.endpoint.post('/rest/V1/categories', {category: value}) as Partial<Category>;
    }

    public static async getCategories(endpoint: Endpoint, where: Partial<Category> = {}, fields: (keyof Category)[] = null): Promise<Partial<Category>[]> {
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

        const categories = await endpoint.get('/rest/V1/categories?' + getParams);
        if (categories.items) return categories.items;
        return [categories];
    }

    get endpoint(): Endpoint {
        return super.endpoint as Endpoint;
    }
}
