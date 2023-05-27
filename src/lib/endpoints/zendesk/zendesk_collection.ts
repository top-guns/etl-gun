import { CollectionOptions } from "../../core/base_collection.js";
import { RestResourceCollection } from "../rest/resource_collection.js";
import { Endpoint } from './endpoint.js';


export class ZendeskCollection<T> extends RestResourceCollection<T> {
    protected static instanceNo = 0;
    protected resourceNameS: string;
    protected resourceName: string;

    constructor(endpoint: Endpoint, collectionName: string, resourceName: string, resourceNameS: string, options: CollectionOptions<T> = {}) {
        ZendeskCollection.instanceNo++;
        super(endpoint, collectionName, resourceName, resourceNameS, options);
    }

    async _findWithoutEvent(where?: Partial<T>): Promise<T[]> {
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
        return res ? (typeof res[this.resourceNameS] === 'undefined' ? res : res[this.resourceNameS]) : null;
    }

    get endpoint(): Endpoint {
        return super.endpoint as Endpoint;
    }
}
