import Signal from 'signal-promise';
import { BaseEndpoint } from "../../core/endpoint.js";
import { BaseObservable } from "../../core/observable.js";
import { BaseQueueCollection } from '../../core/queue_collection.js';
import { BaseCollection, CollectionOptions } from "../../core/readonly_collection.js";
import { Endpoint } from "./endpoint.js";


export class QueueCollection<T = any> extends BaseQueueCollection<T> {
    protected static instanceCount = 0;


    protected _queue: T[];
    protected get queue(): T[] {
        return this._queue;
    }

    protected timestamp: Date;
    protected activateSignal: Signal;
    protected started: boolean;

    constructor(endpoint: BaseEndpoint, collectionName: string, options: CollectionOptions<T> = {}) {
        QueueCollection.instanceCount++;
        super(endpoint, collectionName, options);
        this._queue = [];
        this.activateSignal = new Signal();
        this.started = false;
    }

    public select(dontStopOnEmpty: boolean = false, interval: number = 0): BaseObservable<T> {
        this.timestamp = null;
        this.started = true;

        const observable = new BaseObservable<T>(this, (subscriber) => {
            (async () => {
                try {
                    this.sendStartEvent();
                    while(this.started && !(!dontStopOnEmpty && !this._queue.length)) {
                        if (!this._queue.length) await this.activateSignal.wait();

                        if (subscriber.closed) break;
                        await this.waitWhilePaused();

                        const curTimestamp = new Date();
                        const delay = this.timestamp ? interval - (curTimestamp.getTime() - this.timestamp.getTime()) : 0;
                        if (interval > 0 && delay > 0) await this.wait(delay);

                        if (!this.started) break;
                        const value = this._queue.shift();
                        this.sendReciveEvent(value);
                        subscriber.next(value);
                        
                        this.timestamp = new Date();
                    };

                    this.started = false;
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

    public async get(): Promise<T> {
        if (!this._queue.length) await this.activateSignal.wait();
        const value = this._queue.shift();
        this.sendGetEvent(value);
        return value;
    }


    public async insert(value: T): Promise<void> {
        this.sendInsertEvent(value);
        this.queue.push(value);
        this.activateSignal.notify();
    }
    
    public async delete(): Promise<boolean> {
        this.sendDeleteEvent();
        let exists = this.queue.length > 0;
        this.queue.length = 0;
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
