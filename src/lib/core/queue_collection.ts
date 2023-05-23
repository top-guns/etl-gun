import { BaseEndpoint } from "./endpoint.js";
import { BaseObservable } from "./observable.js";
import { BaseCollectionEvent, BaseCollection, CollectionOptions, CollectionEventListener } from "./readonly_collection.js";


export type QueueCollectionEvent = BaseCollectionEvent |
    "insert" |
    "delete";

export abstract class BaseQueueCollection<T> extends BaseCollection<T> {
    protected get listeners(): Record<QueueCollectionEvent, CollectionEventListener[]> {
        return this._listeners as Record<QueueCollectionEvent, CollectionEventListener[]>;
    }

    constructor(endpoint: BaseEndpoint, collectionName: string, options: CollectionOptions<T> = {}) {
        super(endpoint, collectionName, options);

        this.listeners.insert = [];
        this.listeners.delete = [];
    }

    public abstract select(): BaseObservable<T>;
    public abstract get(): Promise<T>;
    public abstract insert(value: T): Promise<void>;
    public abstract delete(): Promise<boolean>;

    public on(event: QueueCollectionEvent, listener: CollectionEventListener): BaseQueueCollection<T> {
        if (!this.listeners[event]) this.listeners[event] = [];
        this.listeners[event].push(listener); 
        return this;
    }

    public sendEvent(event: QueueCollectionEvent, ...data: any[]) {
        if (!this.listeners[event]) this.listeners[event] = [];
        this.listeners[event].forEach(listener => listener(...data));
    }

    public sendGetEvent(value: T) {
        this.sendEvent("get", { value });
    }

    public sendInsertEvent(value: T) {
        this.sendEvent("insert", { value });
    }

    public sendDeleteEvent() {
        this.sendEvent("delete", { });
    }
}
  