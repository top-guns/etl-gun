import * as ix from 'ix';
import { default as SFTPClient, ConnectOptions, FileInfo, FileStats } from "ssh2-sftp-client";
import { Readable } from "stream";
import { WritableStreamBuffer } from 'stream-buffers';
import { BaseEndpoint} from "../../core/endpoint.js";
import { extractParentFolderPath } from "../../utils/index.js";
import { BaseObservable } from "../../core/observable.js";
import { RemoteFilesystemCollection } from "./remote_filesystem_collection.js";
import { join } from "path";
import { FilesystemCollectionOptions, FilesystemItem, FilesystemItemType } from './filesystem_collection.js';


export class Endpoint extends BaseEndpoint {
    protected _client: SFTPClient | null = null;
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

    getFolder(folderPath: string = '', options: FilesystemCollectionOptions = {}): Collection {
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


export class Collection extends RemoteFilesystemCollection {
    protected static instanceCount = 0;

    protected rootPath: string;

    constructor(endpoint: Endpoint, collectionName: string, rootPath: string, options: FilesystemCollectionOptions = {}) {
        Collection.instanceCount++;
        super(endpoint, collectionName, options);
        this.rootPath = rootPath.trim();
        if (this.rootPath.endsWith('/')) this.rootPath.substring(0, this.rootPath.lastIndexOf('/'));
    }

    protected makeFilesystemItem(info: FileInfo, folderPath: string, contents?: string): FilesystemItem {
        const path = join(folderPath, info.name);
        return {
            name: info.name,
            path,
            fullPath: this.getFullPath(path),
            size: info.size, 
            type: info.type === 'd' ? FilesystemItemType.Directory : info.type === '-' ? FilesystemItemType.File : info.type === 'l' ? FilesystemItemType.SymbolicLink : FilesystemItemType.Unknown,
            modifiedAt: new Date(info.modifyTime),
            fileContents: contents
        }
    }

    protected async _select(folderPath: string = ''): Promise<FilesystemItem[]> {
        const connection = await this.endpoint.getConnection();
        const list: FileInfo[] = await connection.list(this.getFullPath(folderPath)) ?? [];
        const result = list.map(info => this.makeFilesystemItem(info, folderPath));
        return result;
    }

    public async select(folderPath?: string): Promise<FilesystemItem[]> {
        return super.select(folderPath);
    }

    public async* selectGen(folderPath: string = ''): AsyncGenerator<FilesystemItem, void, void> {
        const generator = super.selectGen(folderPath);
        for await (const value of generator) yield value;
    }

    public selectRx(folderPath: string = ''): BaseObservable<FilesystemItem> {
        return super.selectRx(folderPath);
    }

    public selectIx(folderPath?: string): ix.AsyncIterable<FilesystemItem> {
        return super.selectIx(folderPath);
    }

    public selectStream(folderPath?: string): ReadableStream<FilesystemItem> {
        return super.selectStream(folderPath);
    }

    // Get

    public async read(filePath: string): Promise<string> {
        const connection = await this.endpoint.getConnection();
        if (await this.isFolder(filePath)) throw new Error("Error: method 'get' of filesystem collections can be used only with file path");

        const writableStreamBuffer = new WritableStreamBuffer();
        await connection.get(this.getFullPath(filePath), writableStreamBuffer);
        const res = writableStreamBuffer.getContentsAsString() as string;

        this.sendGetEvent(res, filePath);
        return res;
    }

    // Insert

    protected async _insert(remotePath: string, fileContents?: string | Readable): Promise<void> {
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

    public async update(filePath: string, fileContents: string): Promise<void>;
    public async update(filePath: string, sourceStream: Readable): Promise<void>;
    public async update(remoteFilePath: string, fileContents: string | Readable): Promise<void> {
        this.sendUpdateEvent(remoteFilePath, fileContents);
        const path = this.getFullPath(remoteFilePath);
        const connection = await this.endpoint.getConnection();

        if (!await this.isExists(remoteFilePath)) throw new Error(`File ${path} does not exists`);

        if (typeof fileContents === 'string') fileContents = Readable.from(fileContents);
        await connection.put(fileContents, path);
    }

    // Upsert

    public async upsert(folderPath: string): Promise<boolean>;
    public async upsert(filePath: string, fileContents: string): Promise<boolean>;
    public async upsert(filePath: string, sourceStream: Readable): Promise<boolean>;
    public async upsert(remotePath: string, fileContents?: string | Readable): Promise<boolean> {
        const path = this.getFullPath(remotePath);
        const connection = await this.endpoint.getConnection();
        const exists = await this.isExists(remotePath);

        if (exists) this.sendUpdateEvent(remotePath, fileContents!);
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

    protected async _delete(path: string): Promise<boolean> {
        const connection = await this.endpoint.getConnection();
        if (!await this.isExists(path)) return false;

        if (await this.isFolder(path)) await connection.rmdir(this.getFullPath(path), true);
        else await connection.delete(this.getFullPath(path));

        return true;
    }

    // Append & clear

    public async append(filePath: string, fileContents: string): Promise<void>;
    public async append(filePath: string, sourceStream: Readable): Promise<void>;
    public async append(filePath: string, fileContents: string | Readable): Promise<void> {
        this.sendUpdateEvent(filePath, fileContents);
        const path = this.getFullPath(filePath);
        const connection = await this.endpoint.getConnection();
        
        if (!await this.isExists(filePath)) throw new Error(`File ${filePath} does not exists`);

        if (typeof fileContents === 'string') fileContents = Readable.from(fileContents);
        await connection.append(fileContents, path);
    }

    public async clear(path: string): Promise<void> {
        this.sendUpdateEvent(path, '');
        
        const connection = await this.endpoint.getConnection();
        
        if (!await this.isExists(path)) throw new Error(`Path ${path} does not exists`);

        if (await this.isFolder(path)) {
            const fullPath = this.getFullPath(path);
            await connection.rmdir(fullPath, true)
            await connection.mkdir(fullPath);
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

    public async getInfo(path: string) {
        const connection = await this.endpoint.getConnection();
        const list: FileInfo[] = await connection.list(this.getFullPath(path));
        if (!list || !list.length) return null;
        return list[0];
    }

    public async getInfoExt(path: string) {
        const connection = await this.endpoint.getConnection();
        const stats: FileStats = await connection.stat(this.getFullPath(path));
        if (!stats) return null;
        return stats;
    }

    public async isFolder(path: string): Promise<boolean | undefined> {
        const info = await this.getInfoExt(path);
        return info?.isDirectory;
    }

    public async isExists(path: string): Promise<boolean> {
        const connection = await this.endpoint.getConnection();
        return !!(await connection.exists(this.getFullPath(path)));
    }


    protected getFullPath(path: string): string {
        return join(this.rootPath, path);
    }

    protected async ensureDir(folderPath: string): Promise<boolean> {
        const connection = await this.endpoint.getConnection();
        const exists = await this.isExists(folderPath);
        if (!exists) await connection.mkdir(this.getFullPath(folderPath));
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


