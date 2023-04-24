import { Observable, Subscriber } from "rxjs";
import { GuiManager } from "../core/gui.js";
import { BaseEndpoint} from "../core/endpoint.js";
import { BaseCollection, CollectionOptions } from "../core/collection.js";

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
    protected intervalId: NodeJS.Timer;
    protected counter: number = 0;
    protected subscriber: Subscriber<number>;
    
    constructor(endpoint: Endpoint, collectionName: string, interval: number, options: CollectionOptions<number> = {}) {
        Collection.instanceNo++;
        super(endpoint, collectionName, options);
        this.interval = interval;
    }

    public select(): Observable<number> {
        const observable = new Observable<number>((subscriber) => {
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

    public async insert(value: number) {
        super.insert(value);
        this.counter = value;
    }

    public async delete() {
        super.delete();
        this.counter = 0;
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
