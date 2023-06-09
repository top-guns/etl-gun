import { CollectionEventListener, CollectionOptions } from "./base_collection.js";
import { BaseCollection_ID, BaseCollection_ID_Event } from "./base_collection_id.js";
import { BaseEndpoint } from "./endpoint.js";


export type BaseCollection_IDU_Event = BaseCollection_ID_Event |
    "update";

export abstract class BaseCollection_IDU<T> extends BaseCollection_ID<T> {
    protected get listeners(): Record<BaseCollection_IDU_Event, CollectionEventListener[]> {
        return this._listeners as Record<BaseCollection_IDU_Event, CollectionEventListener[]>;
    }

    constructor(endpoint: BaseEndpoint, collectionName: string, options: CollectionOptions<T> = {}) {
        super(endpoint, collectionName, options);
        this.listeners.update = [];
    }

    public abstract update(value: T | any, where: any, ...params: any[]): Promise<void>;
    public abstract upsert(value: T | any, where?: any, ...params: any[]): Promise<boolean>;

    public on(event: BaseCollection_IDU_Event, listener: CollectionEventListener): BaseCollection_IDU<T> {
        if (!this.listeners[event]) this.listeners[event] = [];
        this.listeners[event].push(listener); 
        return this;
    }

    public sendEvent(event: BaseCollection_IDU_Event, ...data: any[]) {
        if (!this.listeners[event]) this.listeners[event] = [];
        this.listeners[event].forEach(listener => listener(...data));
    }

    public sendUpdateEvent(value: T | any, where: any, ...params: any[]) {
        this.sendEvent("update", { value, where, params });
    }
}
  