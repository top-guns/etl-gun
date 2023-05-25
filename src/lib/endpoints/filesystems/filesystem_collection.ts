import { Readable } from "stream";
import { CollectionOptions } from "../../core/base_collection.js";
import { BaseEndpoint } from "../../core/endpoint.js";
import { BaseObservable } from "../../core/observable.js";
import { UpdatableCollection } from "../../core/updatable_collection.js";

// export type FilesystemCollectionEvent = UpdatableCollectionEvent |
//     "append" |
//     "copy" |
//     "move";

export type AlreadyExistsAction = 'update' | 'append' | 'skip' | 'error';

export abstract class FilesystemCollection<T> extends UpdatableCollection<T> {
    // protected get listeners(): Record<UpdatableCollectionEvent, CollectionEventListener[]> {
    //     return this._listeners as Record<UpdatableCollectionEvent, CollectionEventListener[]>;
    // }

    constructor(endpoint: BaseEndpoint, collectionName: string, options: CollectionOptions<T> = {}) {
        super(endpoint, collectionName, options);

        // this.listeners.append = [];
        // this.listeners.copy = [];
        // this.listeners.move = [];
    }


    public abstract select(folderPath?: string, ...params: any[]): BaseObservable<T>;
    public abstract list(folderPath?: string): Promise<T[]>;
    public abstract find(folderPath?: string, ...params: any[]): Promise<T[]>;
    public abstract get(filePath: string): Promise<string>;

    public abstract insert(path: string, fileContents?: string | Readable): Promise<void>;
    public abstract update(filePath: string, fileContents: string | Readable): Promise<void>;
    public abstract upsert(filePath: string, fileContents: string | Readable): Promise<boolean>;
    public abstract delete(path: string): Promise<boolean>;

    public abstract append(filePath: string, fileContents: string | Readable): Promise<void>;
    public abstract clear(path: string): Promise<void>;
    public abstract copy(srcPath: string, dstPath: string): Promise<void>; // , ifAlreadyExists?: AlreadyExistsAction
    public abstract move(srcPath: string, dstPath: string): Promise<void>; // , ifAlreadyExists?: AlreadyExistsAction

    public abstract isExists(path: string): Promise<boolean>;
    public abstract isFolder(path: string): Promise<boolean>;
    public abstract getInfo(path: string): Promise<any>;


    // public on(event: UpdatableCollectionEvent, listener: EventListener): UpdatableCollection<T> {
    //     if (!this.listeners[event]) this.listeners[event] = [];
    //     this.listeners[event].push(listener); 
    //     return this;
    // }

    // protected sendEvent(event: FilesystemCollectionEvent, ...data: any[]) {
    //     if (!this.listeners[event]) this.listeners[event] = [];
    //     this.listeners[event].forEach(listener => listener(...data));
    // }

    
    public sendGetEvent(value: T | string, path: string, operation: string = 'get', params: {} = {}) {
        this.sendEvent("get", { where: path, value, params: { operation, ...params } } );
    }

    public sendListEvent(values: T[], path: string) {
        this.sendEvent("find", { where: path, values } );
    }

    public sendFindEvent(values: T[], path: string, params?: {}) {
        this.sendEvent("find", { values, where: { path, ...params } } );
    }


    public sendInsertEvent(path: string, fileContents?: string | NodeJS.ArrayBufferView | Iterable<string | NodeJS.ArrayBufferView> | AsyncIterable<string | NodeJS.ArrayBufferView> | Readable, operation: string = 'insert', params?: {}) {
        super.sendInsertEvent(
            { 
                path, 
                isFolder: typeof fileContents === 'undefined', 
                fileContents: typeof fileContents === undefined ? undefined : typeof fileContents === 'string' ? fileContents : '[readable stream]' 
            }, 
            { ...params, operation });
    }

    public sendUpdateEvent(filePath: string, fileContents: string | NodeJS.ArrayBufferView | Iterable<string | NodeJS.ArrayBufferView> | AsyncIterable<string | NodeJS.ArrayBufferView> | Readable, operation: string = 'update', params?: {}) {
        super.sendUpdateEvent({ filePath, fileContents: typeof fileContents === 'string' ? fileContents : '[readable stream]' }, filePath, { ...params, operation });
    }

    public sendDeleteEvent(path: string) {
        super.sendDeleteEvent(path, { operation: 'delete' });
    }


    public sendAppendEvent(filePath: string, fileContents: string | NodeJS.ArrayBufferView | Iterable<string | NodeJS.ArrayBufferView> | AsyncIterable<string | NodeJS.ArrayBufferView> | Readable) {
        super.sendUpdateEvent({ filePath, fileContents: typeof fileContents === 'string' ? fileContents : '[readable stream]' }, filePath, { operation: 'append' });
    }

    public sendCopyEvent(srcPath: string, dstPath: string) {
        super.sendUpdateEvent({ srcPath, dstPath }, dstPath, { operation: 'copy' });
    }

    public sendMoveEvent(srcPath: string, dstPath: string) {
        super.sendUpdateEvent({ srcPath, dstPath }, dstPath, { operation: 'move' });
    }
}
  