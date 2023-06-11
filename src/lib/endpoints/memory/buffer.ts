import * as ix from 'ix';
import _ from 'lodash';
import { CollectionOptions } from '../../core/base_collection.js';
import { Condition, isMatch } from "../../core/condition.js";
import { BaseEndpoint } from "../../core/endpoint.js";
import { BaseObservable } from "../../core/observable.js";
import { UpdatableCollection } from "../../core/updatable_collection.js";
import { generator2Iterable, observable2Stream, promise2Generator, promise2Iterable, promise2Observable, promise2Stream, selectOne_from_Promise, wrapGenerator, wrapIterable, wrapObservable, wrapStream } from '../../utils/flows.js';


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

    protected async _select(where?: Condition<T>): Promise<T[]> {
        if (!where) return this._buffer;
        return this._buffer.filter(v => isMatch(v, where));
    }

    public async select(where?: Condition<T>): Promise<T[]> {
        let list: T[] = await this._select(where);
        this.sendSelectEvent(list);
        return list;
    }

    public async* selectGen(where?: Condition<T>): AsyncGenerator<T, void, void> {
        const generator = wrapGenerator(promise2Generator(this._select(where)), this);
        for await (let item of generator) yield item;
    }

    public selectRx(where?: Condition<T>): BaseObservable<T> {
        return wrapObservable(promise2Observable(this._select(where)), this);
    }

    public selectIx(where?: Condition<T>): ix.AsyncIterable<T> {
        return wrapIterable(promise2Iterable(this._select(where)), this);
    }

    public selectStream(where?: Condition<T>): ReadableStream<T> {
        return wrapStream(promise2Stream(this._select(where)), this);
    }

    public async selectOne(where: Condition<T>): Promise<T | null> {
        const res = await selectOne_from_Promise(this._select(where));
        this.sendSelectOneEvent(res ?? null);
        return res ?? null;
    }

    protected async _insert(value: T) {
        this._buffer.push(value); 
    }

    public async insertBatch(values: T[]) {
        this.sendInsertBatchEvent(values);
        this._buffer.push(...values); 
    }

    public async update(value: Partial<T>, where?: Condition<T>): Promise<any> {
        this.sendUpdateEvent(value, where);
        this._buffer = this._buffer.map(v => {
            if (!where || isMatch(v, where)) {
                for (let key in value) {
                    if (!value.hasOwnProperty(key)) continue;
                    _.set(v as any, key, value);
                }
            }
            return v;
        })
    }

    public async upsert(value: T, where?: Condition<T>): Promise<boolean> {
        let exists = false;
        this._buffer = this._buffer.map(v => {
            if (where && isMatch(v, where)) {
                exists = true;
                this.sendUpdateEvent(value, where);
                for (let key in value) {
                    if (!(value as any).hasOwnProperty(key)) continue;
                    _.set(v as any, key, value);
                }
            }
            return v;
        })
        if (!exists) this.insert(value);
        return exists;
    }

    public async delete(where?: Condition<T>): Promise<boolean> {
        this.sendDeleteEvent(where);
        let exists = false;

        if (!where) {
            exists = this._buffer.length > 0;
            if (exists) this._buffer = [];
            return exists;
        }
        
        this._buffer = this._buffer.filter(v => {
            if (isMatch(v, where)) {
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
