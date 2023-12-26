import * as ix from 'ix';
import { Subscriber } from "rxjs";
import { BaseEndpoint} from "../core/endpoint.js";
import { BaseObservable } from "../core/observable.js";
import { BaseCollection, CollectionOptions } from "../core/base_collection.js";
import { generator2Iterable, observable2Stream } from "../utils/flows.js";
import { wait } from "../utils/index.js";

export class Endpoint extends BaseEndpoint {
    protected static _instance: Endpoint;
    static get instance(): Endpoint {
        return Endpoint._instance ||= new Endpoint();
    }

    protected constructor() {
        super();
    }

    getSequence(collectionName: string, interval: number, options: CollectionOptions<number> = {}): Collection {
        options.displayName ??= `${collectionName} (${interval}ms)`;
        return this._addCollection(collectionName, new Collection(this, collectionName, interval, options));
    }

    releaseSequence(collectionName: string) {
        this.collections[collectionName].stop();
        this._removeCollection(collectionName);
    }

    get displayName(): string {
        return `Intervals`;
    }
}

export function getEndpoint(): Endpoint {
    return Endpoint.instance;
}

export class Collection extends BaseCollection<number> {
    protected static instanceNo = 0;

    protected interval: number;
    protected intervalId: NodeJS.Timeout | null = null;
    protected counter: number = 0;
    protected subscriber: Subscriber<number> | null = null;
    protected stoped: Boolean = true;
    
    constructor(endpoint: Endpoint, collectionName: string, interval: number, options: CollectionOptions<number> = {}) {
        Collection.instanceNo++;
        super(endpoint, collectionName, options);
        this.interval = interval;
    }

    public selectRx(): BaseObservable<number> {
        const observable = new BaseObservable<number>(this, (subscriber) => {
            try {
                this.subscriber = subscriber;
                this.sendStartEvent();
                this.intervalId = setInterval(this.onTick.bind(this, false), this.interval);
            }
            catch(err) {
                this.sendErrorEvent(err);
                if (!subscriber.closed) subscriber.error(err);
            }
        });
        return observable;
    }

    public selectStream(): ReadableStream<number> {
        const observable = this.selectRx();
        return observable2Stream(observable);
    }

    public async* selectGen(): AsyncGenerator<number, void, void> {
        while (!this.stoped) {
            await wait(this.interval);
            yield this.counter;
            this.onTick(false);
        }
    }
    public selectIx(): ix.AsyncIterable<number> {
        const generator = this.selectGen();
        return generator2Iterable(generator);
    }

    public async select(): Promise<number[]> {
        try {
            const result = [await this._selectOne()];
            this.sendSelectEvent(result);
            return result;
        }
        catch(err) {
            this.sendErrorEvent(err);
            throw err;
        }
    }

    public async selectOne(): Promise<number> {
        try {
            const value = await this._selectOne();
            this.sendSelectOneEvent(value);
            return value;
        }
        catch(err) {
            this.sendErrorEvent(err);
            throw err;
        }
    }

    protected async _selectOne(): Promise<number> {
        await wait(this.interval);
        const value = this.counter;
        this.onTick(true);
        return value;
    }

    protected onTick(disableEvents: boolean) {
        if (this.isPaused) return;
        try {
            if (!disableEvents) this.sendReciveEvent(this.counter);
            if (this.subscriber && !this.subscriber.closed) this.subscriber.next(this.counter);
            this.counter++;
        }
        catch(err) {
            if (!disableEvents) this.sendErrorEvent(err);
            if (this.subscriber && !this.subscriber.closed) this.subscriber.error(err);
        }
    }

    public async stop() {
        try {
            this.stoped = true;
            if (this.intervalId) clearInterval(this.intervalId);
            if (!this.subscriber?.closed) this.subscriber?.complete();
            this.sendEndEvent();
            this.subscriber = null;
            this.intervalId = null;
        }
        catch(err) {
            this.sendErrorEvent(err);
            if (this.subscriber && !this.subscriber.closed) this.subscriber.error(err);
        }
    }
}
