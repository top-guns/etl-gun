import { BaseEndpoint } from "./endpoint.js";
import { BaseCollection, BaseCollectionEvent, CollectionEventListener, CollectionOptions } from "./base_collection.js";


export type BaseCollection_G_Event = BaseCollectionEvent |
    "get";

export abstract class BaseCollection_G<T> extends BaseCollection<T> {
    constructor(endpoint: BaseEndpoint, collectionName: string, options: CollectionOptions<T> = {}) {
        super(endpoint, collectionName, options);

        this.listeners.get = [];
    }


    public abstract get(where: any, ...params: any[]): Promise<T | any>;


    public on(event: BaseCollection_G_Event, listener: CollectionEventListener): BaseCollection_G<T> {
        if (!this.listeners[event]) this.listeners[event] = [];
        this.listeners[event].push(listener); 
        return this;
    }

    public sendEvent(event: BaseCollection_G_Event, ...data: any[]) {
        if (!this.listeners[event]) this.listeners[event] = [];
        this.listeners[event].forEach(listener => listener(...data));
    }

    public sendGetEvent(value: T, ...params: any[]) {
        this.sendEvent("get", { value, params });
    }
    
}