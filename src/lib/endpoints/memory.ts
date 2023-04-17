import { from, Observable } from "rxjs";
import { BaseEndpoint} from "../core/endpoint.js";
import { BaseCollection, CollectionGuiOptions } from "../core/collection.js";
import { EtlObservable } from "../core/observable.js";

export class Endpoint extends BaseEndpoint {
    getBuffer<T>(collectionName: string, values: T[] = [], guiOptions: CollectionGuiOptions<T> = {}): Collection {
        guiOptions.displayName ??= collectionName;
        return this._addCollection(collectionName, new Collection(this, values, guiOptions));
    }
    releaseBuffer(collectionName: string) {
        this._removeCollection(collectionName);
    }

    get displayName(): string {
        return `Memory buffers`;
    }
}

export class Collection<T = any> extends BaseCollection<T> {
    protected static instanceCount = 0;

    protected _buffer: T[];
    get buffer() {
        return this._buffer;
    }

    constructor(endpoint: Endpoint, values: T[] = [], guiOptions: CollectionGuiOptions<T> = {}) {
        Collection.instanceCount++;
        super(endpoint, guiOptions);
        this._buffer = [...values];
    }

    public select(): EtlObservable<T> {
        const observable = new EtlObservable<T>((subscriber) => {
            (async () => {
                try {
                    this.sendStartEvent();
                    for(const value of this._buffer) {
                        await this.waitWhilePaused();
                        this.sendValueEvent(value);
                        subscriber.next(value);
                    };
                    subscriber.complete();
                    this.sendEndEvent();
                }
                catch(err) {
                    this.sendErrorEvent(err);
                    subscriber.error(err);
                }
            })();
        });
        return observable;
    }

    public async insert(value: T) {
        super.insert(value);
        this._buffer.push(value); 
    }

    public async delete() {
        super.delete();
        this._buffer = [];
    }

    public sort(compareFn: ((v1: T, v2: T) => number | boolean) | undefined = undefined): void {
        if (compareFn === undefined) {
            this._buffer.sort();
            return;
        }

        this._buffer.sort((v1: T, v2: T) => {
            let res = compareFn(v1, v2);
            if (typeof res === "boolean") res = res? 1 : -1;
            return res;
        })
    }

    public forEach(callbackfn: (value: T, index: number, array: T[]) => void) {
        this._buffer.forEach(callbackfn);
    }
}
