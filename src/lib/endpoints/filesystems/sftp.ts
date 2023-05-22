import { default as SFTPClient, ConnectOptions, FileInfo, FileStats } from "ssh2-sftp-client";
import { Readable, Writable } from "stream";
import { WritableStreamBuffer } from 'stream-buffers';
import { BaseEndpoint} from "../../core/endpoint.js";
import { extractParentFolderPath } from "../../utils/index.js";
import { BaseObservable } from "../../core/observable.js";
import { CollectionOptions } from "../../core/readonly_collection.js";
import { UpdatableCollection } from "../../core/updatable_collection.js";
import { RemoteFilesystemCollection } from "./remote_filesystem_collection.js";
import { join } from "path";
import { AlreadyExistsAction } from "./filesystem_collection.js";


export class Endpoint extends BaseEndpoint {
    protected _client: SFTPClient = null;
    protected options: ConnectOptions;

    async getConnection(): Promise<SFTPClient> {
        if (!this._client) {
            this._client = new SFTPClient();
            await this._client.connect(this.options);
        }
        
        return this._client;
    }

    
    constructor(options: ConnectOptions) {
        super();
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
        return this.options.host ? `SFTP (${this.options.host})` : `SFTP (${this.instanceNo})`;
    }
}


export function getEndpoint(options: ConnectOptions): Endpoint {
    return new Endpoint(options);
}


export class Collection extends RemoteFilesystemCollection<FileInfo> {
    protected static instanceCount = 0;

    protected rootPath: string;

    constructor(endpoint: Endpoint, collectionName: string, rootPath: string, options: CollectionOptions<FileInfo> = {}) {
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
                        const list: FileInfo[] = await connection.list(this.getFullPath(folderPath));
                        
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

    public async get(remotePath: string): Promise<undefined | string | FileInfo[]> {
        const connection = await this.endpoint.getConnection();
        const path = this.getFullPath(remotePath);

        if (!await this.isExists(remotePath)) {
            this.sendGetEvent(remotePath, undefined);
            return undefined;
        }

        if (await this.isFolder(remotePath)) {
            const res = await connection.list(path);
            this.sendGetEvent(remotePath, res);
            return res;
        }

        const writableStreamBuffer = new WritableStreamBuffer();
        await connection.get(path, writableStreamBuffer);
        const res: string = writableStreamBuffer.getContentsAsString() as string;

        this.sendGetEvent(remotePath, res);
        return res;
    }

    // Insert

    public async insert(remoteFolderPath: string): Promise<void>;
    public async insert(remoteFilePath: string, fileContents: string): Promise<void>;
    public async insert(remoteFilePath: string, sourceStream: Readable): Promise<void>;
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
        await connection.put(fileContents, path);
    }

    // Update

    public async update(remoteFilePath: string, fileContents: string): Promise<void>;
    public async update(remoteFilePath: string, sourceStream: Readable): Promise<void>;
    public async update(remoteFilePath: string, fileContents: string | Readable): Promise<void> {
        this.sendUpdateEvent(remoteFilePath, fileContents);
        const path = this.getFullPath(remoteFilePath);
        const connection = await this.endpoint.getConnection();

        if (!await this.isExists(remoteFilePath)) throw new Error(`File ${path} does not exists`);

        if (typeof fileContents === 'string') fileContents = Readable.from(fileContents);
        await connection.put(fileContents, path);
    }

    // Upsert

    public async upsert(remoteFolderPath: string): Promise<boolean>;
    public async upsert(remoteFilePath: string, fileContents: string): Promise<boolean>;
    public async upsert(remoteFilePath: string, sourceStream: Readable): Promise<boolean>;
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

    public async delete(remotePath: string): Promise<boolean> {
        this.sendDeleteEvent(remotePath);
        return this._delete(remotePath);
    }

    protected async _delete(remotePath: string): Promise<boolean> {
        const path = this.getFullPath(remotePath);
        const connection = await this.endpoint.getConnection();
        if (!await this.isExists(remotePath)) return false;

        if (await this.isFolder(remotePath)) await connection.rmdir(path, true);
        else await connection.delete(path);

        return true;
    }

    // Append & clear

    public async append(remoteFilePath: string, fileContents: string): Promise<void>;
    public async append(remoteFilePath: string, sourceStream: Readable): Promise<void>;
    public async append(remoteFilePath: string, fileContents: string | Readable): Promise<void> {
        this.sendUpdateEvent(remoteFilePath, fileContents);
        const path = this.getFullPath(remoteFilePath);
        const connection = await this.endpoint.getConnection();
        
        if (!await this.isExists(remoteFilePath)) throw new Error(`File ${remoteFilePath} does not exists`);

        if (typeof fileContents === 'string') fileContents = Readable.from(fileContents);
        await connection.append(fileContents, path);
    }

    public async clear(remotePath: string): Promise<void> {
        this.sendUpdateEvent(remotePath, '');
        const path = this.getFullPath(remotePath);
        const connection = await this.endpoint.getConnection();
        
        if (!await this.isExists(remotePath)) throw new Error(`Path ${remotePath} does not exists`);

        if (await this.isFolder(remotePath)) {
            await connection.rmdir(path, true)
            await connection.mkdir(path);
            return;
        }

        await connection.put(Readable.from(''), path);
    }

    //

    public async copy(remoteSrcPath: string, remoteDstPath: string): Promise<void> {
        const connection = await this.endpoint.getConnection();
        this.sendCopyEvent(remoteSrcPath, remoteDstPath);

        await connection.rcopy(this.getFullPath(remoteSrcPath), this.getFullPath(remoteDstPath));
    }
    public async move(remoteSrcPath: string, remoteDstPath: string): Promise<void> {
        const connection = await this.endpoint.getConnection();
        this.sendMoveEvent(remoteSrcPath, remoteDstPath);

        await connection.rcopy(this.getFullPath(remoteSrcPath), this.getFullPath(remoteDstPath));
        await this._delete(remoteSrcPath);
    }
  
    public async download(remotePath: string, localPath: string): Promise<void> {
        const connection = await this.endpoint.getConnection();
        const path = this.getFullPath(remotePath);
        this.sendDownloadEvent(remotePath, localPath);

        if (await this.isFolder(remotePath)) {
            await connection.downloadDir(path, localPath);
        }
        else {
            await connection.fastGet(path, localPath);
        }
    }
    public async upload(localPath: string, remotePath: string): Promise<void> {
        const connection = await this.endpoint.getConnection();
        const path = this.getFullPath(remotePath);
        this.sendUploadEvent(localPath, remotePath);
        
        if (await this.isFolder(remotePath)) {
            await connection.uploadDir(localPath, path);
        }
        else {
            await connection.fastPut(localPath, path);
        }
    }

    // Get info

    public async getInfo(remotePath: string) {
        const connection = await this.endpoint.getConnection();
        const list: FileInfo[] = await connection.list(this.getFullPath(remotePath));
        if (!list || !list.length) return null;
        return list[0];
    }

    public async getInfoExt(remotePath: string) {
        const connection = await this.endpoint.getConnection();
        const stats: FileStats = await connection.stat(this.getFullPath(remotePath));
        if (!stats) return null;
        return stats;
    }

    public async isFolder(remotePath: string): Promise<boolean> {
        const info = await this.getInfoExt(remotePath);
        return info.isDirectory;
    }

    public async isExists(remotePath: string): Promise<boolean> {
        const connection = await this.endpoint.getConnection();
        return !!(await connection.exists(this.getFullPath(remotePath)));
    }


    protected getFullPath(path: string): string {
        return join(this.rootPath, path);
    }

    protected async ensureDir(remotePath: string): Promise<boolean> {
        const connection = await this.endpoint.getConnection();
        const exists = await this.isExists(remotePath);
        if (!exists) await connection.mkdir(this.getFullPath(remotePath));
        return exists;
    }

    protected async ensureParentDir(path: string): Promise<boolean> {
        const connection = await this.endpoint.getConnection();
        const parentPath = extractParentFolderPath(this.getFullPath(path));
        const exists = !!(await connection.exists(parentPath));
        if (!exists) await connection.mkdir(parentPath);
        return exists;
    }
    

    get endpoint(): Endpoint {
        return super.endpoint as Endpoint;
    }
}


