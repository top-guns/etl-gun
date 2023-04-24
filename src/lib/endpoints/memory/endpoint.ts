import { BaseEndpoint} from "../../core/endpoint.js";
import { CollectionOptions } from "../../core/collection.js";
import { BufferCollection } from "./buffer.js";
import { QueueCollection } from "./queue.js";

export class Endpoint extends BaseEndpoint {
    getBuffer<T>(collectionName: string, values: T[] = [], options: CollectionOptions<T> = {}): BufferCollection<T> {
        options.displayName ??= collectionName;
        return this._addCollection(collectionName, new BufferCollection<T>(this, collectionName, values, options));
    }
    releaseBuffer(collectionName: string) {
        this._removeCollection(collectionName);
    }

    getQueue<T>(collectionName: string, options: CollectionOptions<T> = {}): QueueCollection<T> {
        options.displayName ??= collectionName;
        return this._addCollection(collectionName, new QueueCollection<T>(this, collectionName, options));
    }
    releaseQueue(collectionName: string) {
        const collection: QueueCollection<any> = this.collections[collectionName] as QueueCollection<any>;
        collection.stop();
        this._removeCollection(collectionName);
    }

    releaseEndpoint() {
        for (let key in this.collections) {
            if (!this.collections.hasOwnProperty(key)) continue;
            if (this.collections[key].constructor.name == QueueCollection.name) this.collections[key].stop();
        }
        super.releaseEndpoint();
    }

    get displayName(): string {
        return `Memory`;
    }    
}
