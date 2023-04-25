import { from, Observable } from "rxjs";
import { BaseEndpoint } from "src/lib/core/endpoint.js";
import { BaseObservable } from "src/lib/core/observable.js";
import { BaseCollection, CollectionOptions } from "../../core/collection.js";
import { Endpoint } from "./endpoint.js";


export class BufferCollection<T = any> extends BaseCollection<T> {
    protected static instanceCount = 0;

    protected _buffer: T[];
    get buffer() {
        return this._buffer;
    }

    constructor(endpoint: BaseEndpoint, collectionName: string, values: T[] = [], options: CollectionOptions<T> = {}) {
        BufferCollection.instanceCount++;
        super(endpoint, collectionName, options);
        this._buffer = [...values];
    }

    public select(deleteProcessedElements?: true): BaseObservable<T>;
    public select(deleteProcessedElements?: false, filter?: (value: T, index: number) => T): BaseObservable<T>;
    public select(deleteProcessedElements: boolean = false, filter?: (value: T, index: number) => T): BaseObservable<T> {
        const observable = new BaseObservable<T>(this, (subscriber) => {
            (async () => {
                try {
                    this.sendStartEvent();
                    if (deleteProcessedElements) {
                        let value;
                        while (value = this._buffer.shift()) {
                            if (subscriber.closed) break;
                            await this.waitWhilePaused();
                            this.sendReciveEvent(value);
                            subscriber.next(value);
                        }
                    }
                    else {
                        const elements = filter ? this._buffer.filter(filter) : this._buffer;
                        for(const value of elements) {
                            if (subscriber.closed) break;
                            await this.waitWhilePaused();
                            this.sendReciveEvent(value);
                            subscriber.next(value);
                        };
                    }
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
