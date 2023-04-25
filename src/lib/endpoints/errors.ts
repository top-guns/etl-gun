import { CollectionOptions } from "../core/collection.js";
import { BaseEndpoint } from "../core/endpoint.js";
import { BaseObservable } from "../core/observable.js";
import { QueueCollection } from "./memory/queue.js";


export type EtlError = {
    name: string;
    message: string;
    //error: any
}

export type EtlErrorData<T = any> = EtlError & {
    data: T;
}


export class Endpoint extends BaseEndpoint {
    protected static _instance: Endpoint;
    static get instance(): Endpoint {
        return Endpoint._instance ||= new Endpoint();
    }

    protected constructor() {
        super(false, true);
    }

    getCollection(collectionName: string, options: CollectionOptions<EtlError> = {}): ErrorsQueue {
        options.displayName ??= collectionName;
        const collection = new ErrorsQueue(this, collectionName, options);
        return this._addCollection(collectionName, collection);
    }
    releaseCollection(collectionName: string) {
        const collection: ErrorsQueue = this.collections[collectionName] as ErrorsQueue;
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

export function getEndpoint(): Endpoint {
    return Endpoint.instance;
}

export class ErrorsQueue extends QueueCollection<EtlError> {
    public select(stopOnEmpty: boolean = false): BaseObservable<EtlError> {
        return super.select(stopOnEmpty, 0);
    }
}
