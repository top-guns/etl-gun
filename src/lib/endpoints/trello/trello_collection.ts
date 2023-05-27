import { CollectionOptions } from "../../core/base_collection.js";
import { BaseObservable } from "../../core/observable.js";
import { RestResourceCollection } from "../rest/resource_collection.js";
import { Endpoint } from './endpoint.js';


export class TrelloCollection<T> extends RestResourceCollection<T> {
    protected static instanceNo = 0;
    
    constructor(endpoint: Endpoint, collectionName: string, resourceName: string, resourceNameS: string, options: CollectionOptions<Partial<T>> = {}) {
        TrelloCollection.instanceNo++;
        super(endpoint, collectionName, resourceName, resourceNameS, options);
    }

    // protected getSearchUrl(): string {
    //     return `${this.resourceNameS}?filter=${this.resourceName}`;
    // }

    async find(where?: Partial<T>, fields?: (keyof T)[]): Promise<T[]> {
        return await super.find({ ...where, fields });
    }

    public select(where: Partial<T> = {}, fields?: (keyof T)[]): BaseObservable<T>{
        return super.select({ ...where, fields });
    }

    get endpoint(): Endpoint {
        return super.endpoint as Endpoint;
    }
}
