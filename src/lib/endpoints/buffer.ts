import { from, Observable } from "rxjs";
import { Endpoint} from "../core/endpoint";
import { Collection, CollectionGuiOptions, CollectionImpl } from "../core/collection";
import { EtlObservable } from "../core/observable";

export class MemoryEndpoint extends Endpoint {
    getBuffer<T>(name: string, values: T[] = [], guiOptions: CollectionGuiOptions<T> = {}): BufferCollection {
        guiOptions.displayName ??= name;
        return this._addCollection(name, new BufferCollection(this, values, guiOptions));
    }
    releaseBuffer(name: string) {
        this._removeCollection(name);
    }
}

export class BufferCollection<T = any> extends CollectionImpl<T> {
    protected static instanceCount = 0;

    protected _buffer: T[];
    get buffer() {
        return this._buffer;
    }

    constructor(endpoint: MemoryEndpoint, values: T[] = [], guiOptions: CollectionGuiOptions<T> = {}) {
        BufferCollection.instanceCount++;
        guiOptions.displayName ??= `Buffer ${BufferCollection.instanceCount}`;
        super(endpoint, guiOptions);
        this._buffer = [...values];
    }

    public list(): EtlObservable<T> {
        const observable = new EtlObservable<T>((subscriber) => {
            (async () => {
                try {
                    this.sendStartEvent();
                    for(const value of this._buffer) {
                        await this.waitWhilePaused();
                        this.sendDataEvent(value);
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

    public async push(value: T) {
        super.push(value);
        this._buffer.push(value); 
    }

    public async clear() {
        super.clear();
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
