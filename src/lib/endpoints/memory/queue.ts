import * as ix from 'ix';
import Signal from 'signal-promise';
import { CollectionOptions } from '../../core/base_collection.js';
import { BaseEndpoint } from "../../core/endpoint.js";
import { BaseObservable } from "../../core/observable.js";
import { BaseQueueCollection } from '../../core/queue_collection.js';
import { generator2Iterable, observable2Stream } from '../../utils/flows.js';


export type QueueSelectOptions = { 
    stopOnEmpty?: boolean;
    interval?: number;
}

export class QueueCollection<T = any> extends BaseQueueCollection<T> {
    protected static instanceCount = 0;

    protected _queue: T[];
    protected get queue(): T[] {
        return this._queue;
    }

    protected timestamp: Date | null = null;
    protected activateSignal: Signal;
    protected started: boolean;

    constructor(endpoint: BaseEndpoint, collectionName: string, options: CollectionOptions<T> = {}) {
        QueueCollection.instanceCount++;
        super(endpoint, collectionName, options);
        this._queue = [];
        this.activateSignal = new Signal();
        this.started = false;
    }

    protected async _select(): Promise<T[]> {
        const result = [...this._queue];
        this._queue = [];
        return result;
    }

    public async select(): Promise<T[]> {
        let list: T[] = await this._select();
        this.sendSelectEvent(list);
        return list;
    }

    public async* selectGen(options: QueueSelectOptions = {}): AsyncGenerator<T, void, void> {
        this.sendStartEvent();

        while(this.started && !(options.stopOnEmpty && !this._queue.length)) {
            if (!this._queue.length) await this.activateSignal.wait();
            await this.waitWhilePaused();

            const curTimestamp = new Date();
            const delay = this.timestamp ? (options.interval ?? 0) - (curTimestamp.getTime() - this.timestamp.getTime()) : 0;
            if (options.interval && delay > 0) await this.wait(delay);

            if (!this.started) break;
            const value = this._queue.shift();
            this.sendReciveEvent(value!);
            yield value!;
            
            this.timestamp = new Date();
        };

        this.started = false;
        this.sendEndEvent();
    }

    public selectRx(options: QueueSelectOptions = {}): BaseObservable<T> {
        this.timestamp = null;
        this.started = true;

        const observable = new BaseObservable<T>(this, (subscriber) => {
            (async () => {
                try {
                    this.sendStartEvent();
                    while(this.started && !(options.stopOnEmpty && !this._queue.length)) {
                        if (!this._queue.length) await this.activateSignal.wait();

                        if (subscriber.closed) break;
                        await this.waitWhilePaused();

                        const curTimestamp = new Date();
                        const delay = this.timestamp ? (options.interval ?? 0) - (curTimestamp.getTime() - this.timestamp.getTime()) : 0;
                        if (options.interval && delay > 0) await this.wait(delay);

                        if (!this.started) break;
                        const value = this._queue.shift();
                        this.sendReciveEvent(value!);
                        if (!subscriber.closed) subscriber.next(value);
                        
                        this.timestamp = new Date();
                    };

                    this.started = false;
                    if (!subscriber.closed) subscriber.complete();
                    this.sendEndEvent();
                }
                catch(err) {
                    this.sendErrorEvent(err);
                    if (!subscriber.closed) subscriber.error(err);
                }
            })();
        });
        return observable;
    }

    public selectIx(options: QueueSelectOptions = {}): ix.AsyncIterable<T> {
        return generator2Iterable(this.selectGen(options));
    }

    public selectStream(options: QueueSelectOptions = {}): ReadableStream<T> {
        return observable2Stream(this.selectRx(options));
    }

    public async selectOne(returnIfEmpty: boolean = false): Promise<T | null> {
        if (!this._queue.length) {
            if (returnIfEmpty) return null;
            await this.activateSignal.wait();
        }
        const value = this._queue.shift();
        this.sendSelectOneEvent(value!);
        return value!;
    }


    protected async _insert(value: T): Promise<void> {
        this.queue.push(value);
        this.activateSignal.notify();
    }

    public async insertBatch(values: T[]): Promise<void> {
        this.sendInsertBatchEvent(values);
        this.queue.push(...values);
        this.activateSignal.notify();
    }
    
    public async delete(): Promise<boolean> {
        this.sendDeleteEvent();
        let exists = this.queue.length > 0;
        this._queue = [];
        return exists;
    }
    

    public stop() {
        this.started = false;
        this._queue = [];
        this.activateSignal.notify();
    }

    protected async wait(delay: number): Promise<void> {
        this.sendSleepEvent();
        await new Promise((r) => setTimeout(r, delay));
        this.sendStartEvent();
    }
}
