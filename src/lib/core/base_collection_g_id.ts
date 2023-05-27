import { CollectionEventListener, CollectionOptions } from "./base_collection.js";
import { BaseCollection_I, BaseCollection_I_Event } from "./base_collection_i.js";
import { BaseEndpoint } from "./endpoint.js";


export type BaseCollection_G_ID_Event = BaseCollection_I_Event |
    "get" |
    "delete";

export abstract class BaseCollection_G_ID<T> extends BaseCollection_I<T> {
    protected get listeners(): Record<BaseCollection_G_ID_Event, CollectionEventListener[]> {
        return this._listeners as Record<BaseCollection_G_ID_Event, CollectionEventListener[]>;
    }

    constructor(endpoint: BaseEndpoint, collectionName: string, options: CollectionOptions<T> = {}) {
        super(endpoint, collectionName, options);

        this.listeners.get = [];
        this.listeners.delete = [];
    }

    public abstract get(): Promise<T | any>;
    public abstract delete(): Promise<boolean>;

    public on(event: BaseCollection_G_ID_Event, listener: CollectionEventListener): BaseCollection_G_ID<T> {
        if (!this.listeners[event]) this.listeners[event] = [];
        this.listeners[event].push(listener); 
        return this;
    }

    public sendEvent(event: BaseCollection_G_ID_Event, ...data: any[]) {
        if (!this.listeners[event]) this.listeners[event] = [];
        this.listeners[event].forEach(listener => listener(...data));
    }

    public sendGetEvent(value: T) {
        this.sendEvent("get", { value });
    }

    public sendDeleteEvent() {
        this.sendEvent("delete", { });
    }
}
  