import { BaseEndpoint } from "./endpoint.js";
import { BaseCollection, BaseCollectionEvent, CollectionEventListener, CollectionOptions } from "./base_collection.js";


export type BaseCollection_GF_Event = BaseCollectionEvent |
    "get" |
    "find";

export abstract class BaseCollection_GF<T> extends BaseCollection<T> {
    constructor(endpoint: BaseEndpoint, collectionName: string, options: CollectionOptions<T> = {}) {
        super(endpoint, collectionName, options);

        this.listeners.get = [];
        this.listeners.find = [];
    }


    public abstract get(where: any, ...params: any[]): Promise<T | any>;
    public abstract find(where?: any, ...params: any[]): Promise<T[]>;
  

    public on(event: BaseCollection_GF_Event, listener: CollectionEventListener): BaseCollection_GF<T> {
        if (!this.listeners[event]) this.listeners[event] = [];
        this.listeners[event].push(listener); 
        return this;
    }

    public sendEvent(event: BaseCollection_GF_Event, ...data: any[]) {
        if (!this.listeners[event]) this.listeners[event] = [];
        this.listeners[event].forEach(listener => listener(...data));
    }

    public sendListEvent(values: T[], where?: any, ...params: any[]) {
        this.sendEvent("find", { where, values, params });
    }

    public sendGetEvent(value: T, where: any, ...params: any[]) {
        this.sendEvent("get", { where, value, params });
    }
    
}