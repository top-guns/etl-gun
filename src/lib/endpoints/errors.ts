import { BaseCollection, CollectionOptions } from "../core/collection.js";
import { BaseEndpoint } from "../core/endpoint.js";
import { QueueCollection } from "./memory/queue.js";


export type EtlError = {
    name: string;
    message: string;
}

export type EtlErrorData<T = any> = EtlError & {
    data: T;
}


export class Endpoint extends BaseEndpoint {
    static _instance: Endpoint;
    static get instance(): Endpoint {
        return Endpoint._instance ||= new Endpoint();
    }

    getCollection(collectionName: string, options: CollectionOptions<EtlError> = {}): ErrorsQueue {
        options.displayName ??= collectionName;
        const collection = new QueueCollection<EtlError>(this, collectionName, options);
        return this._addCollection(collectionName, collection);
    }
    releaseCollection(collectionName: string) {
        const collection: QueueCollection<any> = this.collections[collectionName] as QueueCollection<any>;
        collection.stop();
        this._removeCollection(collectionName);
    }

    releaseEndpoint() {
        for (let key in this.collections) this.collections[key].stop();
        super.releaseEndpoint();
    }

    get displayName(): string {
        return `Errors`;
    }
}

export class ErrorsQueue extends QueueCollection<EtlError> {
    
}