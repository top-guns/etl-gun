import { from, Observable } from "rxjs";
import Signal from 'signal-promise';
import { BaseEndpoint } from "src/lib/core/endpoint.js";
import { BaseCollection, CollectionGuiOptions } from "../../core/collection.js";
import { Endpoint } from "./endpoint.js";


export class QueueCollection<T = any> extends BaseCollection<T> {
    protected static instanceCount = 0;


    protected _queue: T[];
    protected get queue(): T[] {
        return this._queue;
    }

    protected timestamp: Date;
    protected activateSignal: Signal;
    protected started: boolean;

    constructor(endpoint: BaseEndpoint, guiOptions: CollectionGuiOptions<T> = {}) {
        QueueCollection.instanceCount++;
        super(endpoint, guiOptions);
        this._queue = [];
        this.activateSignal = new Signal();
        this.started = false;
    }

    public select(stopOnEmpty: boolean = true, interval: number = 0): Observable<T> {
        this.timestamp = null;
        this.started = true;

        const observable = new Observable<T>((subscriber) => {
            (async () => {
                try {
                    this.sendStartEvent();
                    while(this.started && !(stopOnEmpty && !this._queue.length)) {
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

    public async insert(value: T) {
        super.insert(value);
        this.queue.push(value);
        this.activateSignal.notify();
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
