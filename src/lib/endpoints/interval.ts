import { Observable, Subscriber } from "rxjs";
import { GuiManager } from "../core/gui.js";
import { BaseEndpoint} from "../core/endpoint.js";
import { BaseCollection, CollectionGuiOptions } from "../core/collection.js";

export class Endpoint extends BaseEndpoint {
    getSequence(collectionName: string, interval: number, guiOptions: CollectionGuiOptions<number> = {}): Collection {
        guiOptions.displayName ??= `${collectionName} (${interval}ms)`;
        return this._addCollection(collectionName, new Collection(this, interval, guiOptions));
    }

    releaseSequence(collectionName: string) {
        this.collections[collectionName].stop();
        this._removeCollection(collectionName);
    }

    get displayName(): string {
        return `Intervals`;
    }
}

export class Collection extends BaseCollection<number> {
    protected static instanceNo = 0;

    protected interval: number;
    protected intervalId: NodeJS.Timer;
    protected counter: number = 0;
    protected subscriber: Subscriber<number>;
    
    constructor(endpoint: Endpoint, interval: number, guiOptions: CollectionGuiOptions<number> = {}) {
        Collection.instanceNo++;
        super(endpoint, guiOptions);
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
