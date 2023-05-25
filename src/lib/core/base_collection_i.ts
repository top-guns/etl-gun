import { BaseCollection, BaseCollectionEvent, CollectionEventListener, CollectionOptions } from "./base_collection.js";
import { BaseEndpoint } from "./endpoint.js";


export type BaseCollection_I_Event = BaseCollectionEvent |
    "insert";

export abstract class BaseCollection_I<T> extends BaseCollection<T> {
    protected get listeners(): Record<BaseCollection_I_Event, CollectionEventListener[]> {
        return this._listeners as Record<BaseCollection_I_Event, CollectionEventListener[]>;
    }

    constructor(endpoint: BaseEndpoint, collectionName: string, options: CollectionOptions<T> = {}) {
        super(endpoint, collectionName, options);

        this.listeners.insert = [];
    }

    public abstract insert(value: T | any, ...params: any[]): Promise<void>;

    public on(event: BaseCollection_I_Event, listener: CollectionEventListener): BaseCollection_I<T> {
        if (!this.listeners[event]) this.listeners[event] = [];
        this.listeners[event].push(listener); 
        return this;
    }

    public sendEvent(event: BaseCollection_I_Event, ...data: any[]) {
        if (!this.listeners[event]) this.listeners[event] = [];
        this.listeners[event].forEach(listener => listener(...data));
    }

    public sendInsertEvent(value: T | any, ...params: any[]) {
        this.sendEvent("insert", { value, params });
    }
}
  