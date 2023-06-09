import * as ix from 'ix';
import { Readable } from "stream";
import { CollectionOptions } from "../../core/base_collection.js";
import { BaseEndpoint } from "../../core/endpoint.js";
import { BaseObservable } from "../../core/observable.js";
import { UpdatableCollection } from "../../core/updatable_collection.js";
import { generator2Iterable, observable2Stream, promise2Generator, promise2Observable, selectOne_from_Promise, wrapGenerator, wrapObservable } from '../../utils/flows.js';

// export type FilesystemCollectionEvent = UpdatableCollectionEvent |
//     "append" |
//     "copy" |
//     "move";

export type AlreadyExistsAction = 'update' | 'append' | 'skip' | 'error';

export enum FilesystemItemType {
    Unknown = 0,
    File = 1,
    Directory = 2,
    SymbolicLink = 3
}

export type FilesystemItem = {
    name: string;
    path: string;
    fullPath: string;
    size?: number;
    type: FilesystemItemType;
    modifiedAt?: Date;
    fileContents?: string;
}

export type FilesystemCollectionOptions = CollectionOptions<FilesystemItem>;

export abstract class FilesystemCollection extends UpdatableCollection<FilesystemItem> {
    // protected get listeners(): Record<UpdatableCollectionEvent, CollectionEventListener[]> {
    //     return this._listeners as Record<UpdatableCollectionEvent, CollectionEventListener[]>;
    // }

    constructor(endpoint: BaseEndpoint, collectionName: string, options: FilesystemCollectionOptions = {}) {
        super(endpoint, collectionName, options);

        // this.listeners.append = [];
        // this.listeners.copy = [];
        // this.listeners.move = [];
    }

    
    protected abstract _select(mask?: string, ...params: any[]): Promise<FilesystemItem[]>;
    public abstract read(filePath: string): Promise<string>;

    public async select(mask?: string, ...params: any[]): Promise<FilesystemItem[]> {
        const values = await this._select(mask, ...params);
        this.sendSelectEvent(values);
        return values;
    }

    public async* selectGen(mask: string = '', ...params: any[]): AsyncGenerator<FilesystemItem, void, void> {
        const values = this._select(mask, ...params);
        const generator = wrapGenerator(promise2Generator(values), this);
        for await (const value of generator) yield value;
    }

    public selectRx(mask: string = '', ...params: any[]): BaseObservable<FilesystemItem> {
        const values = this._select(mask, ...params);
        return wrapObservable(promise2Observable(values), this);
    }

    public selectIx(mask?: string, ...params: any[]): ix.AsyncIterable<FilesystemItem> {
        return generator2Iterable(this.selectGen(mask, ...params));
    }

    public selectStream(mask?: string, ...params: any[]): ReadableStream<FilesystemItem> {
        return observable2Stream(this.selectRx(mask, ...params));
    }

    public async selectOne(path: string = ''): Promise<FilesystemItem | null> {
        const values = this._select(path);
        const value = await selectOne_from_Promise(values);
        this.sendSelectOneEvent(value);
        return value;
    }


    public async list(folderPath: string = '', ...params: any[]): Promise<FilesystemItem[]> {
        return await this.select(folderPath, ...params);
    }
    

    public async insert(folderPath: string): Promise<void>;
    public async insert(filePath: string, fileContents: string): Promise<void>;
    public async insert(filePath: string, sourceStream: Readable): Promise<void>;
    public async insert(item: FilesystemItem): Promise<void>;
    public async insert(p1: string | FilesystemItem, fileContents?: string | Readable): Promise<void> {
        if (typeof p1 === 'string') {
            this.sendInsertEvent(p1, fileContents);
            await this._insert(p1, fileContents);
            return;
        }
        this.sendInsertEvent(p1.path, p1.fileContents);
        await this._insert(p1.path, p1.fileContents);
    }

    public async insertBatch(items: FilesystemItem[]): Promise<void> {
        this.sendInsertBatchEvent(items);
        for (const item of items) this._insert(item.path, item.fileContents);
    }

    protected abstract _insert(path: string, fileContents?: string | Readable): Promise<void>;

    public abstract update(filePath: string, fileContents: string | Readable): Promise<void>;
    public abstract upsert(filePath: string, fileContents: string | Readable): Promise<boolean>;
    public abstract delete(path: string): Promise<boolean>;

    public abstract append(filePath: string, fileContents: string | Readable): Promise<void>;
    public abstract clear(path: string): Promise<void>;
    public abstract copy(srcPath: string, dstPath: string): Promise<void>; // , ifAlreadyExists?: AlreadyExistsAction
    public abstract move(srcPath: string, dstPath: string): Promise<void>; // , ifAlreadyExists?: AlreadyExistsAction

    public abstract isExists(path: string): Promise<boolean>;
    public abstract isFolder(path: string): Promise<boolean | undefined>;
    public abstract getInfo(path: string): Promise<any | undefined>;


    // public on(event: UpdatableCollectionEvent, listener: EventListener): UpdatableCollection<T> {
    //     if (!this.listeners[event]) this.listeners[event] = [];
    //     this.listeners[event].push(listener); 
    //     return this;
    // }

    // protected sendEvent(event: FilesystemCollectionEvent, ...data: any[]) {
    //     if (!this.listeners[event]) this.listeners[event] = [];
    //     this.listeners[event].forEach(listener => listener(...data));
    // }

    
    public sendGetEvent(value: FilesystemItem | string, path: string, operation: string = 'get', params: {} = {}) {
        this.sendEvent("get", { where: path, value, params: { operation, ...params } } );
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

    public sendInsertBatchEvent(items: FilesystemItem[], operation: string = 'insertBatch', params?: {}) {
        super.sendInsertBatchEvent(items, { ...params, operation });
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
  