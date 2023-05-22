import { Readable } from "stream";
import { BaseEndpoint } from "../../core/endpoint.js";
import { BaseObservable } from "../../core/observable.js";
import { CollectionOptions } from "../../core/readonly_collection.js";
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


    public abstract select(folderPath?: string): BaseObservable<T>;
    public abstract get(filePath: string): Promise<any>;

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

    protected sendInsertEvent(path: string, fileContents?: string | Readable) {
        super.sendInsertEvent({ 
            path, 
            isFolder: typeof fileContents === 'undefined', 
            fileContents: typeof fileContents === undefined ? undefined : typeof fileContents === 'string' ? fileContents : '[readable stream]' 
        }, { operation: typeof fileContents === 'undefined' ? 'create folder' : 'create file' });
    }

    protected sendUpdateEvent(filePath: string, fileContents: string | Readable, operation: string = 'update', params?: {}) {
        super.sendUpdateEvent({ filePath, fileContents: typeof fileContents === 'string' ? fileContents : '[readable stream]' }, filePath, { ...params, operation });
    }

    protected sendUpsertEvent(path: string, fileContents?: string | Readable, operation: string = 'upsert', params?: {}) {
        super.sendUpsertEvent({ 
            path, 
            isFolder: typeof fileContents === 'undefined', 
            fileContents: typeof fileContents === undefined ? undefined : typeof fileContents === 'string' ? fileContents : '[readable stream]'  
        }, path, { ...params, operation });
    }

    protected sendDeleteEvent(path: string) {
        super.sendDeleteEvent(path, { operation: 'delete' });
    }


    protected sendAppendEvent(filePath: string, fileContents: string | Readable) {
        super.sendUpsertEvent({ filePath, fileContents: typeof fileContents === 'string' ? fileContents : '[readable stream]' }, filePath, { operation: 'append' });
    }

    protected sendCopyEvent(srcPath: string, dstPath: string) {
        super.sendUpsertEvent({ srcPath, dstPath }, dstPath, { operation: 'copy' });
    }

    protected sendMoveEvent(srcPath: string, dstPath: string) {
        super.sendUpsertEvent({ srcPath, dstPath }, dstPath, { operation: 'move' });
    }


    protected sendGetEvent(path: string, value: any, operation: string = 'get', params?: {}) {
        this.sendEvent("get", { where: path, value, params: { ...params, operation } } );
    }
}
  