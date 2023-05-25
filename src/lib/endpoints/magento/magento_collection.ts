import { CollectionOptions } from '../../core/base_collection.js';
import { BaseObservable } from '../../core/observable.js';
import { RestResourceCollection } from '../rest/resource_collection.js';
import { Endpoint } from './endpoint.js';


export class MagentoCollection<T> extends RestResourceCollection<T> {
    protected static instanceNo = 0;

    constructor(endpoint: Endpoint, collectionName: string, resourceName: string, resourceNameS: string, options: CollectionOptions<T> = {}) {
        MagentoCollection.instanceNo++;
        super(endpoint, collectionName, resourceName, resourceNameS, options);
    }

    async find(where?: Partial<T>, fields?: (keyof T)[]): Promise<T[]> {
        return await super.find({ ...where, fields });
    }

    public select(where: Partial<T> = {}, fields?: (keyof T)[]): BaseObservable<T>{
        return super.select({ ...where, fields });
    }

    async _findWithoutEvent(where?: Partial<T>): Promise<T[]> {
        let url = this.getSearchUrl();
        const searchParams = MagentoCollection.convertWhereToQueryParams(where);
        url = this.endpoint.makeUrl([url], [searchParams])
        
        const res = await this.endpoint.fetchJson(url);
        return res ? (typeof res[this.resourceNameS] === 'undefined' ? res : res[this.resourceNameS]) : null;
    }

    protected static convertWhereToQueryParams(where: any = {}): string {
        let getParams = '';
        let n = 0;
        for (let key in where) {
            if (!where.hasOwnProperty(key)) continue;
            if (getParams) getParams += '&';
            getParams += `searchCriteria[filterGroups][${n}][filters][0][field]=${key}&`;
            getParams += `searchCriteria[filterGroups][${n}][filters][0][value]=${where[key]}`;
            n++;
        }
        if (!getParams) getParams = 'searchCriteria';
        return getParams;
    }


    get endpoint(): Endpoint {
        return super.endpoint as Endpoint;
    }
}
