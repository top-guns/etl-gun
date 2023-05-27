import { Subscriber } from "rxjs";
import { BaseEndpoint} from "../core/endpoint.js";
import { BaseObservable } from "../core/observable.js";
import { BaseCollection_G } from "../core/base_collection_g.js";
import { CollectionOptions } from "../core/base_collection.js";

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

export class Collection extends BaseCollection_G<number> {
    protected static instanceNo = 0;

    protected interval: number;
    protected intervalId: NodeJS.Timer;
    protected counter: number = 0;
    protected subscriber: Subscriber<number>;
    
    constructor(endpoint: Endpoint, collectionName: string, interval: number, options: CollectionOptions<number> = {}) {
        Collection.instanceNo++;
        super(endpoint, collectionName, options);
        this.interval = interval;
    }

    public select(): BaseObservable<number> {
        const observable = new BaseObservable<number>(this, (subscriber) => {
            try {
                this.subscriber = subscriber;
                this.sendStartEvent();
                this.intervalId = setInterval(this.onTick.bind(this), this.interval);
            }
            catch(err) {
                this.sendErrorEvent(err);
                subscriber.error(err);
            }
        });
        return observable;
    }

    public async get(): Promise<number> {
        const value = this.counter;
        this.sendGetEvent(value);
        return value;
    }

    protected onTick() {
        if (this.isPaused) return;
        try {
            this.sendReciveEvent(this.counter);
            this.subscriber.next(this.counter);
            this.counter++;
        }
        catch(err) {
            this.sendErrorEvent(err);
            this.subscriber.error(err);
        }
    }

    public async stop() {
        try {
            clearInterval(this.intervalId);
            this.subscriber.complete();
            this.sendEndEvent();
            this.subscriber = null;
        }
        catch(err) {
            this.sendErrorEvent(err);
            this.subscriber.error(err);
        }
    }
}
