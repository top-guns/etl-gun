import { BaseEndpoint } from "./endpoint.js";
import { BaseCollectionEvent, BaseCollection, CollectionOptions, CollectionEventListener } from "./readonly_collection.js";


export type UpdatableCollectionEvent = BaseCollectionEvent |
    "insert" |
    "update" |
    "delete";

export abstract class UpdatableCollection<T> extends BaseCollection<T> {
    protected get listeners(): Record<UpdatableCollectionEvent, CollectionEventListener[]> {
        return this._listeners as Record<UpdatableCollectionEvent, CollectionEventListener[]>;
    }

    constructor(endpoint: BaseEndpoint, collectionName: string, options: CollectionOptions<T> = {}) {
        super(endpoint, collectionName, options);

        this.listeners.insert = [];
        this.listeners.update = [];
        this.listeners.delete = [];
    }

    public abstract insert(value: T | any, ...params: any[]): Promise<any>;
    public abstract update(value: T | any, where: any, ...params: any[]): Promise<any>;
    public abstract upsert(value: T | any, where?: any, ...params: any[]): Promise<boolean>;
    public abstract delete(where?: any, ...params: any[]): Promise<boolean>;

    public on(event: UpdatableCollectionEvent, listener: CollectionEventListener): UpdatableCollection<T> {
        if (!this.listeners[event]) this.listeners[event] = [];
        this.listeners[event].push(listener); 
        return this;
    }

    public sendEvent(event: UpdatableCollectionEvent, ...data: any[]) {
        if (!this.listeners[event]) this.listeners[event] = [];
        this.listeners[event].forEach(listener => listener(...data));
    }

    public sendInsertEvent(value: T | any, ...params: any[]) {
        this.sendEvent("insert", { value, params });
    }

    public sendUpdateEvent(value: T | any, where: any, ...params: any[]) {
        this.sendEvent("update", { value, where, params });
    }

    public sendDeleteEvent(where?: any, ...params: any[]) {
        this.sendEvent("delete", { where, params });
    }
}
  