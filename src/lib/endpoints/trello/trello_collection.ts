import * as ix from 'ix';
import { CollectionOptions } from "../../core/base_collection.js";
import { BaseObservable } from "../../core/observable.js";
import { generator2Iterable, observable2Stream } from "../../utils/flows.js";
import { ResourceCollection } from "../rest/resource_collection.js";
import { Endpoint } from './endpoint.js';


export class TrelloCollection<T> extends ResourceCollection<T> {
    protected static instanceNo = 0;
    
    constructor(endpoint: Endpoint, collectionName: string, resourceName: string, resourceNameS: string, options: CollectionOptions<T> = {}) {
        TrelloCollection.instanceNo++;
        super(endpoint, collectionName, resourceName, resourceNameS, options);
    }

    // protected getSearchUrl(): string {
    //     return `${this.resourceNameS}?filter=${this.resourceName}`;
    // }

    async select(where?: Partial<T>, fields?: (keyof T)[]): Promise<T[]> {
        return await super.select({...where, fields});
    }

    public async* selectGen(where?: Partial<T>, fields?: (keyof T)[]): AsyncGenerator<T, void, void> {
        const generator = super.selectGen({...where, fields});
        for await (const item of generator) yield item;
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

    get endpoint(): Endpoint {
        return super.endpoint as Endpoint;
    }
}
