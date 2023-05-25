import * as ftp from "basic-ftp";
import { AccessOptions, Client, FileInfo } from "basic-ftp";
import { Readable } from "stream";
import { WritableStreamBuffer } from 'stream-buffers';
import { BaseEndpoint} from "../../core/endpoint.js";
import { extractParentFolderPath } from "../../utils/index.js";
import { BaseObservable } from "../../core/observable.js";
import { RemoteFilesystemCollection } from "./remote_filesystem_collection.js";
import { join } from "path";
import { CollectionOptions } from "../../core/base_collection.js";


export class Endpoint extends BaseEndpoint {
    protected _client: Client = null;
    protected verbose: boolean;
    protected options: AccessOptions;

    async getConnection(): Promise<Client> {
        if (!this._client) {
            this._client = new ftp.Client();
            this._client.ftp.verbose = this.verbose;
        }

        if (this._client.closed) await this._client.access(this.options);
        
        return this._client;
    }

    
    constructor(options: AccessOptions, verbose: boolean = false) {
        super();
        this.verbose = verbose;
        this.options = options;
    }

    getFolder(folderPath: string = '', options: CollectionOptions<FileInfo> = {}): Collection {
        options.displayName ??= folderPath;
        return this._addCollection(folderPath, new Collection(this, folderPath, folderPath, options));
    }

    releaseFolder(folderPath: string) {
        this._removeCollection(folderPath);
    }

    get displayName(): string {
        return this.options.host ? `FTP (${this.options.host})` : `FTP (${this.instanceNo})`;
    }
}


export function getEndpoint(options: AccessOptions, verbose: boolean = false): Endpoint {
    return new Endpoint(options, verbose);
}


export class Collection extends RemoteFilesystemCollection<FileInfo> {
    protected static instanceCount = 0;

    protected rootPath: string;

    constructor(endpoint: Endpoint, collectionName: string, rootPath: string = '', options: CollectionOptions<FileInfo> = {}) {
        Collection.instanceCount++;
        super(endpoint, collectionName, options);
        this.rootPath = rootPath.trim();
        if (this.rootPath.endsWith('/')) this.rootPath.substring(0, this.rootPath.lastIndexOf('/'));
    }

    public select(folderPath: string = ''): BaseObservable<FileInfo> {
        const observable = new BaseObservable<any>(this, (subscriber) => {
            this.sendStartEvent();
            try {
                (async () => {
                    try {
                        const connection = await this.endpoint.getConnection();
                        const list: FileInfo[] = await this._list(folderPath);
                        
                        for (let fileInfo of list) {
                            if (subscriber.closed) break;
                            await this.waitWhilePaused();
                            this.sendReciveEvent(fileInfo);
                            subscriber.next(fileInfo);
                        }

                        if (!subscriber.closed) {
                            subscriber.complete();
                            this.sendEndEvent();
                        }
                    }
                    catch(err) {
                        this.sendErrorEvent(err);
                        subscriber.error(err);
                    }
                })();
            }
            catch(err) {
                this.sendErrorEvent(err);
                subscriber.error(err);
            }
        });
        return observable;
    }

    // Get

    public async get(filePath: string): Promise<string> {
        const connection = await this.endpoint.getConnection();
        if (await this.isFolder(filePath)) throw new Error("Error: method 'get' of filesystem collections can be used only with file path");

        const writableStreamBuffer = new WritableStreamBuffer();
        await connection.downloadTo(writableStreamBuffer, this.getFullPath(filePath));
        const res = writableStreamBuffer.getContentsAsString() as string;

        this.sendGetEvent(res, filePath);
        return res;
    }

    public async list(folderPath: string = ''): Promise<FileInfo[]> {
        const res = await this._list(folderPath);
        this.sendListEvent(res, folderPath);
        return res;
    }

    public async _list(folderPath: string = ''): Promise<FileInfo[]> {
        const connection = await this.endpoint.getConnection();
        let res: FileInfo[] = await connection.list(this.getFullPath(folderPath)) ?? [];
        return res;
    }

    public async find(folderPath: string = ''): Promise<FileInfo[]> {
        const connection = await this.endpoint.getConnection();
        let res: FileInfo[] = await connection.list(this.getFullPath(folderPath)) ?? [];
        this.sendFindEvent(res, folderPath);
        return res;
    }

    // Insert

    public async insert(folderPath: string): Promise<void>;
    public async insert(filePath: string, fileContents: string): Promise<void>;
    public async insert(filePath: string, sourceStream: Readable): Promise<void>;
    public async insert(remotePath: string, fileContents?: string | Readable): Promise<void> {
        this.sendInsertEvent(remotePath, fileContents);
        const path = this.getFullPath(remotePath);
        if (await this.isExists(remotePath)) throw new Error(`Path ${path} already exists`);
        const connection = await this.endpoint.getConnection();

        if (typeof fileContents === 'undefined') {
            await this.ensureDir(remotePath);
            return;
        }

        await this.ensureParentDir(remotePath);

        if (typeof fileContents === 'string') fileContents = Readable.from(fileContents);
        await connection.uploadFrom(fileContents, path);
    }

    // Update

    public async update(filePath: string, fileContents: string): Promise<void>;
    public async update(filePath: string, sourceStream: Readable): Promise<void>;
    public async update(remoteFilePath: string, fileContents: string | Readable): Promise<void> {
        this.sendUpdateEvent(remoteFilePath, fileContents);
        const path = this.getFullPath(remoteFilePath);
        const connection = await this.endpoint.getConnection();

        if (!await this.isExists(remoteFilePath)) throw new Error(`File ${path} does not exists`);

        if (typeof fileContents === 'string') fileContents = Readable.from(fileContents);
        await connection.uploadFrom(fileContents, path);
    }

    // Upsert

    public async upsert(folderPath: string): Promise<boolean>;
    public async upsert(filePath: string, fileContents: string): Promise<boolean>;
    public async upsert(filePath: string, sourceStream: Readable): Promise<boolean>;
    public async upsert(remotePath: string, fileContents?: string | Readable): Promise<boolean> {
        const path = this.getFullPath(remotePath);
        const connection = await this.endpoint.getConnection();
        const exists = await this.isExists(remotePath);

        if (exists) this.sendUpdateEvent(remotePath, fileContents);
        else this.sendInsertEvent(remotePath, fileContents);

        if (typeof fileContents === 'undefined') return await this.ensureDir(remotePath);

        await this.ensureParentDir(remotePath);

        if (exists) await this.update(remotePath, fileContents as any);
        else await this.insert(remotePath, fileContents as any);
        return exists;
    }

    // Delete

    public async delete(path: string) {
        this.sendDeleteEvent(path);
        const connection = await this.endpoint.getConnection();
        if (!await this.isExists(path)) return false;

        if (await this.isFolder(path)) await connection.removeDir(this.getFullPath(path));
        else await connection.remove(this.getFullPath(path));

        return true;
    }

    // Append & clear

    public async append(filePath: string, fileContents: string): Promise<void>;
    public async append(filePath: string, sourceStream: Readable): Promise<void>;
    public async append(remoteFilePath: string, fileContents: string | Readable): Promise<void> {
        this.sendUpdateEvent(remoteFilePath, fileContents);
        const path = this.getFullPath(remoteFilePath);
        const connection = await this.endpoint.getConnection();
        
        if (!await this.isExists(remoteFilePath)) throw new Error(`File ${remoteFilePath} does not exists`);

        if (typeof fileContents === 'string') fileContents = Readable.from(fileContents);
        await connection.appendFrom(fileContents, path);
    }

    public async clear(path: string): Promise<void> {
        this.sendUpdateEvent(path, '');
        const connection = await this.endpoint.getConnection();
        
        if (!await this.isExists(path)) throw new Error(`Path ${path} does not exists`);

        if (await this.isFolder(path)) {
            const curpath = await connection.pwd();
            await connection.cd(this.getFullPath(path));
            await connection.clearWorkingDir();
            await connection.cd(curpath);
            return;
        }

        await connection.uploadFrom(Readable.from(''), path);
    }

    //

    public async copy(remoteSrcPath: string, remoteDstPath: string): Promise<void> {
        throw new Error(`Method copy for the ftp collection is not implemented`);
    }
    public async move(remoteSrcPath: string, remoteDstPath: string): Promise<void> {
        throw new Error(`Method copy for the ftp collection is not implemented`);
    }
  
    public async download(remotePath: string, localPath: string): Promise<void> {
        const connection = await this.endpoint.getConnection();
        const path = this.getFullPath(remotePath);
        this.sendDownloadEvent(remotePath, localPath);
        await connection.downloadTo(localPath, path);
    }
    public async upload(localPath: string, remotePath: string): Promise<void> {
        const connection = await this.endpoint.getConnection();
        const path = this.getFullPath(remotePath);
        this.sendUploadEvent(localPath, remotePath);
        await connection.uploadFrom(localPath, path);
    }

    // Get info

    public async getInfo(path: string) {
        const connection = await this.endpoint.getConnection();
        const list: FileInfo[] = await connection.list(this.getFullPath(path));
        if (!list || !list.length) return null;
        return list[0];
    }

    public async isFolder(path: string): Promise<boolean> {
        const info = await this.getInfo(path);
        return info.isDirectory;
    }

    public async isExists(path: string) {
        const info = await this.getInfo(path);
        return !!info;
    }


    protected getFullPath(path: string): string {
        return join(this.rootPath, path);
    }

    protected async ensureDir(folderPath: string): Promise<boolean> {
        const connection = await this.endpoint.getConnection();
        const exists = await this.isExists(folderPath);
        await connection.ensureDir(this.getFullPath(folderPath));
        return exists;
    }

    protected async ensureParentDir(path: string): Promise<boolean> {
        const connection = await this.endpoint.getConnection();
        const parentPath = extractParentFolderPath(this.getFullPath(path));
        
        const info = await this.getInfo(parentPath);
        await connection.ensureDir(parentPath);
        return !!info;
    }


    get endpoint(): Endpoint {
        return super.endpoint as Endpoint;
    }
}


