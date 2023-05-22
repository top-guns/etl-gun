import { BaseEndpoint } from "./endpoint.js";
import { BaseCollectionEvent, BaseCollection, CollectionOptions, CollectionEventListener } from "./readonly_collection.js";


export type UpdatableCollectionEvent = BaseCollectionEvent |
    "insert" |
    "update" |
    "upsert" |
    "delete";

export abstract class UpdatableCollection<T> extends BaseCollection<T> {
    protected get listeners(): Record<UpdatableCollectionEvent, CollectionEventListener[]> {
        return this._listeners as Record<UpdatableCollectionEvent, CollectionEventListener[]>;
    }

    constructor(endpoint: BaseEndpoint, collectionName: string, options: CollectionOptions<T> = {}) {
        super(endpoint, collectionName, options);

        this.listeners.insert = [];
        this.listeners.update = [];
        this.listeners.upsert = [];
        this.listeners.delete = [];
    }

    public abstract insert(value: T | any, ...params: any[]): Promise<any>;
    public abstract update(value: T | any, where: any, ...params: any[]): Promise<any>;
    public abstract upsert(value: T | any, where?: any, ...params: any[]): Promise<boolean>;
    public abstract delete(where?: any, ...params: any[]): Promise<boolean>;

    public on(event: UpdatableCollectionEvent, listener: EventListener): UpdatableCollection<T> {
        if (!this.listeners[event]) this.listeners[event] = [];
        this.listeners[event].push(listener); 
        return this;
    }

    protected sendEvent(event: UpdatableCollectionEvent, ...data: any[]) {
        if (!this.listeners[event]) this.listeners[event] = [];
        this.listeners[event].forEach(listener => listener(...data));
    }

    protected sendInsertEvent(value: T | any, ...params: any[]) {
        this.sendEvent("insert", { value, params });
    }

    protected sendUpdateEvent(value: T | any, where: any, ...params: any[]) {
        this.sendEvent("update", { value, where, params });
    }

    protected sendUpsertEvent(value: T | any, where?: any, ...params: any[]) {
        this.sendEvent("upsert", { value, where, params });
    }

    protected sendDeleteEvent(where?: any, ...params: any[]) {
        this.sendEvent("delete", { where, params });
    }
}
  