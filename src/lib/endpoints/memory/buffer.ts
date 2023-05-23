import _ from 'lodash';
import { Condition, isMatch } from "../../core/condition.js";
import { BaseEndpoint } from "../../core/endpoint.js";
import { BaseObservable } from "../../core/observable.js";
import { CollectionOptions } from "../../core/readonly_collection.js";
import { UpdatableCollection } from "../../core/updatable_collection.js";


export class BufferCollection<T = any> extends UpdatableCollection<T> {
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

    public select(where?: (value: T, index: number) => T): BaseObservable<T> {
        const observable = new BaseObservable<T>(this, (subscriber) => {
            (async () => {
                try {
                    this.sendStartEvent();
                    
                    const elements = where ? this._buffer.filter(where) : this._buffer;
                    for(const value of elements) {
                        if (subscriber.closed) break;
                        await this.waitWhilePaused();
                        this.sendReciveEvent(value);
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

    public async get(condition?: Condition<T>): Promise<T[]> {
        if (!condition) {
            this.sendGetEvent(undefined, this._buffer);
            return this._buffer;
        }

        const elements = this._buffer.filter(v => isMatch(v, condition))
        return elements;
    }

    public async insert(value: T) {
        this.sendInsertEvent(value);
        this._buffer.push(value); 
    }

    public async update(value: Partial<T>, condition: Condition<T>): Promise<any> {
        this.sendUpdateEvent(value, condition);
        this._buffer = this._buffer.map(v => {
            if (isMatch(v, condition)) {
                for (let key in value) {
                    if (!value.hasOwnProperty(key)) continue;
                    _.set(v, key, value);
                }
            }
            return v;
        })
    }

    public async upsert(value: T, condition: Condition<T>): Promise<boolean> {
        let exists = false;
        this._buffer = this._buffer.map(v => {
            if (isMatch(v, condition)) {
                exists = true;
                this.sendUpdateEvent(value, condition);
                for (let key in value) {
                    if (!value.hasOwnProperty(key)) continue;
                    _.set(v, key, value);
                }
            }
            return v;
        })
        if (!exists) this.insert(value);
        return exists;
    }

    public async delete(condition?: Condition<T>): Promise<boolean> {
        this.sendDeleteEvent(condition);
        let exists = false;

        if (!condition) {
            exists = this._buffer.length > 0;
            if (exists) this._buffer = [];
            return exists;
        }
        
        this._buffer = this._buffer.filter(v => {
            if (isMatch(v, condition)) {
                exists = true;
                return false;
            }
            return true;
        })

        return exists;
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
