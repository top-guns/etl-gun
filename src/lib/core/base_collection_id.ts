import { CollectionEventListener, CollectionOptions } from "./base_collection.js";
import { BaseCollection_I, BaseCollection_I_Event } from "./base_collection_i.js";
import { BaseEndpoint } from "./endpoint.js";


export type BaseCollection_ID_Event = BaseCollection_I_Event |
    "delete";

export abstract class BaseCollection_ID<T> extends BaseCollection_I<T> {
    protected get listeners(): Record<BaseCollection_ID_Event, CollectionEventListener[]> {
        return this._listeners as Record<BaseCollection_ID_Event, CollectionEventListener[]>;
    }

    constructor(endpoint: BaseEndpoint, collectionName: string, options: CollectionOptions<T> = {}) {
        super(endpoint, collectionName, options);
        this.listeners.delete = [];
    }

    public abstract delete(where?: any, ...params: any[]): Promise<boolean>;

    public on(event: BaseCollection_ID_Event, listener: CollectionEventListener): BaseCollection_ID<T> {
        if (!this.listeners[event]) this.listeners[event] = [];
        this.listeners[event].push(listener); 
        return this;
    }

    public sendEvent(event: BaseCollection_ID_Event, ...data: any[]) {
        if (!this.listeners[event]) this.listeners[event] = [];
        this.listeners[event].forEach(listener => listener(...data));
    }

    public sendDeleteEvent(where?: any, ...params: any[]) {
        this.sendEvent("delete", { where, params });
    }
}
