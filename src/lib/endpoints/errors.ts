import { BaseCollection, CollectionGuiOptions } from "../core/collection.js";
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
    getCollection(collectionName: string, guiOptions: CollectionGuiOptions<EtlError> = {}): QueueCollection<EtlError> {
        guiOptions.displayName ??= collectionName;
        const collection = new QueueCollection<EtlError>(this, guiOptions);
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
