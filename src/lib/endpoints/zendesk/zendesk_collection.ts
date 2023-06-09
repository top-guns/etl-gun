import * as ix from 'ix';
import { CollectionOptions } from "../../core/base_collection.js";
import { BaseObservable } from "../../core/observable.js";
import { ResourceCollection } from "../rest/resource_collection.js";
import { Endpoint } from './endpoint.js';


export class ZendeskCollection<T> extends ResourceCollection<T> {
    protected static instanceNo = 0;

    constructor(endpoint: Endpoint, collectionName: string, resourceName: string, resourceNameS: string, options: CollectionOptions<T> = {}) {
        ZendeskCollection.instanceNo++;
        super(endpoint, collectionName, resourceName, resourceNameS, options);
    }

    protected async _select(where?: Partial<T>): Promise<T[]> {
        let url = `${this.resourceNameS}`;

        if (where) {
            let queryParams = `type%3A${this.resourceName}`;
            for (let key in where) {
                if (!where.hasOwnProperty(key)) continue;
                queryParams += '+' + key + '%3A' + where[key];
            }
            url = `search.json?query=${queryParams}`;
        }
        
        const res = await this.endpoint.fetchJson(url);
        const result = res ? (typeof res[this.resourceNameS] === 'undefined' ? res : res[this.resourceNameS]) : null;
        return result;
    }

    public async select(where?: Partial<T>): Promise<T[]> {
        return super.select(where);
    }

    public async* selectGen(where?: Partial<T>): AsyncGenerator<T, void, void> {
        for await (const item of super.selectGen(where)) yield item;
    }

    public selectRx(where?: Partial<T>): BaseObservable<T>{
        return super.selectRx(where);
    }

    public selectIx(where?: Partial<T>): ix.AsyncIterable<T> {
        return super.selectIx(where);
    }

    public selectStream(where?: Partial<T>): ReadableStream<T> {
        return super.selectStream(where);
    }

    get endpoint(): Endpoint {
        return super.endpoint as Endpoint;
    }
}
