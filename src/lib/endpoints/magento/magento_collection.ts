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

    protected async _select(where?: any): Promise<T[]> {
        let url = this.getSearchUrl();
        const searchParams = MagentoCollection.convertWhereToQueryParams(where);
        url = this.endpoint.makeUrl([url], [searchParams])
        
        const res = await this.endpoint.fetchJson(url);
        const result = res ? (typeof res[this.resourceNameS] === 'undefined' ? res : res[this.resourceNameS]) : null;
        return result;
    }

    public select(where?: Partial<T>, fields?: (keyof T)[]): Promise<T[]> {
        return super.select({...where, fields});
    }

    public async* selectGen(where?: Partial<T>, fields?: (keyof T)[]): AsyncGenerator<T, void, void> {
        for await (const item of super.selectGen(where)) yield item;
    }

    public selectRx(where?: Partial<T>, fields?: (keyof T)[]): BaseObservable<T>{
        return super.selectRx({...where, fields});
    }

    public selectIx(where?: Partial<T>, fields?: (keyof T)[]): ix.AsyncIterable<T> {
        return super.selectIx({...where, fields});
    }

    public selectStream(where?: Partial<T>, fields?: (keyof T)[]): ReadableStream<T> {
        return super.selectStream({...where, fields});
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
