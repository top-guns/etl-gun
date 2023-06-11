import * as ix from 'ix';
import { CollectionOptions } from '../../core/base_collection.js';
import { BaseObservable } from '../../core/observable.js';
import { generator2Iterable, observable2Stream } from '../../utils/flows.js';
import { ResourceCollection } from '../rest/resource_collection.js';
import { Endpoint } from './endpoint.js';



export class MagentoCollection<T> extends ResourceCollection<T> {
    protected static instanceNo = 0;

    constructor(endpoint: Endpoint, collectionName: string, resourceName: string, resourceNameS: string, options: CollectionOptions<T> = {}) {
        MagentoCollection.instanceNo++;
        super(endpoint, collectionName, resourceName, resourceNameS, options);
    }

    protected async _select(where?: any, fields?: (keyof T)[]): Promise<T[]> {
        if (fields) where['fields'] = fields;

        let url = this.getSearchUrl();
        const searchParams = MagentoCollection.convertWhereToQueryParams(where);
        url = this.endpoint.makeUrl([url], [searchParams])
        
        const res = await this.endpoint.fetchJson(url);
        const result = res ? res['items'] : null;
        return result;
    }

    public select(where?: Partial<T>, fields?: (keyof T)[]): Promise<T[]> {
        if (fields) where['fields'] = fields;
        return super.select(where);
    }

    public async* selectGen(where?: Partial<T>, fields?: (keyof T)[]): AsyncGenerator<T, void, void> {
        if (fields) where['fields'] = fields;
        for await (const item of super.selectGen(where)) yield item;
    }

    public selectRx(where?: Partial<T>, fields?: (keyof T)[]): BaseObservable<T>{
        if (fields) where['fields'] = fields;
        return super.selectRx(where);
    }

    public selectIx(where?: Partial<T>, fields?: (keyof T)[]): ix.AsyncIterable<T> {
        if (fields) where['fields'] = fields;
        return super.selectIx(where);
    }

    public selectStream(where?: Partial<T>, fields?: (keyof T)[]): ReadableStream<T> {
        if (fields) where['fields'] = fields;
        return super.selectStream(where);
    }

    protected async _selectOne(sku: string): Promise<T> {
        return super._selectOne(sku);
    }

    public async selectOne(sku: string): Promise<T> {
        return super.selectOne(sku);
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
