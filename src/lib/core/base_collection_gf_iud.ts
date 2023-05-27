import { CollectionEventListener, CollectionOptions } from "./base_collection.js";
import { BaseCollection_GF, BaseCollection_GF_Event } from "./base_collection_gf.js";
import { BaseEndpoint } from "./endpoint.js";


export type BaseCollection_GF_IUD_Event = BaseCollection_GF_Event |
    "insert" |
    "update" |
    "delete";

export abstract class BaseCollection_GF_IUD<T> extends BaseCollection_GF<T> {
    protected get listeners(): Record<BaseCollection_GF_IUD_Event, CollectionEventListener[]> {
        return this._listeners as Record<BaseCollection_GF_IUD_Event, CollectionEventListener[]>;
    }

    constructor(endpoint: BaseEndpoint, collectionName: string, options: CollectionOptions<T> = {}) {
        super(endpoint, collectionName, options);

        this.listeners.insert = [];
        this.listeners.update = [];
        this.listeners.delete = [];
    }

    public abstract insert(value: T | any, ...params: any[]): Promise<void>;
    public abstract update(value: T | any, where: any, ...params: any[]): Promise<void>;
    public abstract upsert(value: T | any, where?: any, ...params: any[]): Promise<boolean>;
    public abstract delete(where?: any, ...params: any[]): Promise<boolean>;

    public on(event: BaseCollection_GF_IUD_Event, listener: CollectionEventListener): BaseCollection_GF_IUD<T> {
        if (!this.listeners[event]) this.listeners[event] = [];
        this.listeners[event].push(listener); 
        return this;
    }

    public sendEvent(event: BaseCollection_GF_IUD_Event, ...data: any[]) {
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
  