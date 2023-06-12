import * as ix from 'ix';
import * as ftp from "comlog-ftp";
import { Readable } from "stream";
import { WritableStreamBuffer } from 'stream-buffers';
import { BaseEndpoint} from "../../core/endpoint.js";
import { extractFileName, extractParentFolderPath } from "../../utils/index.js";
import { BaseObservable } from "../../core/observable.js";
import { RemoteFilesystemCollection } from "./remote_filesystem_collection.js";
import { join } from "path";
import { FilesystemCollectionOptions, FilesystemItem, FilesystemItemType } from './filesystem_collection.js';
import { inspect } from 'util';


export type FtpConnectionOptions = {
    host: string;
    port?: number;
    user: string;
    password: string;
}

type FtpFileInfo = {
    name: string;
    type: number;
    time: number;
    size: string;
    owner: string;
    group: string;
    userPermissions: { read: boolean, write: boolean, exec: boolean },
    groupPermissions: { read: boolean, write: boolean, exec: boolean },
    otherPermissions: { read: boolean, write: boolean, exec: boolean }
}

export class Endpoint extends BaseEndpoint {
    protected _client: ftp.Client | null = null;
    protected verbose: boolean;
    protected options: FtpConnectionOptions;

    async getConnection(): Promise<ftp.Client> {
        if (!this._client) {
            this._client = new ftp.Client();
            //this._client.ftp.verbose = this.verbose;

            await this._client.connectAsync(this.options.port ?? 21, this.options.host);
            await this._client.login(this.options.user , this.options.password);
            await this._client.pasv();
        }

        //if (this._client.closed) await this._client.access(this.options);

        return this._client;
    }

    
    constructor(options: FtpConnectionOptions, verbose: boolean = false) {
        super();
        this.verbose = verbose;
        this.options = options;
    }

    getFolder(folderPath: string = '', options: FilesystemCollectionOptions = {}): Collection {
        options.displayName ??= folderPath;
        return this._addCollection(folderPath, new Collection(this, folderPath, folderPath, options));
    }

    releaseFolder(folderPath: string) {
        this._removeCollection(folderPath);
    }

    async releaseEndpoint(): Promise<void> {
        if (this._client) await this._client.quit();
    }

    get displayName(): string {
        return this.options.host ? `FTP (${this.options.host})` : `FTP (${this.instanceNo})`;
    }
}


export function getEndpoint(options: FtpConnectionOptions, verbose: boolean = false): Endpoint {
    return new Endpoint(options, verbose);
}


export class Collection extends RemoteFilesystemCollection {
    protected static instanceCount = 0;

    protected rootPath: string;

    constructor(endpoint: Endpoint, collectionName: string, rootPath: string = '', options: FilesystemCollectionOptions = {}) {
        Collection.instanceCount++;
        super(endpoint, collectionName, options);
        this.rootPath = rootPath.trim();
        if (this.rootPath.endsWith('/')) this.rootPath.substring(0, this.rootPath.lastIndexOf('/'));
    }

    protected makeFilesystemItem(info: FtpFileInfo, path: string, contents?: string): FilesystemItem {
        return {
            name: extractFileName(path),
            path,
            fullPath: this.getFullPath(path),
            size: parseInt(info.size),
            type: info.type === 1 ? FilesystemItemType.Directory : info.type === 0 ? FilesystemItemType.File : FilesystemItemType.Unknown,
            modifiedAt: new Date(info.time),
            fileContents: contents
        }
    }

    // protected async getFilesystemItem(path: string, contents?: string): Promise<FilesystemItem> {
    //     if (!await this.isExists(path)) return undefined;
    //     const fullPath = this.getFullPath(path);
    //     const connection = await this.endpoint.getConnection();
    //     return {
    //         name: extractFileName(path),
    //         path,
    //         fullPath,
    //         size: await connection.size(fullPath),
    //         type: await this.isFolder(path) ? FilesystemItemType.Directory : FilesystemItemType.File,
    //         modifiedAt: await connection.lastMod(path),
    //         fileContents: contents
    //     }
    // }

    protected async _select(folderPath: string = ''): Promise<FilesystemItem[]> {
        const connection = await this.endpoint.getConnection();
        const list: any[] = await connection.list(this.getFullPath(folderPath)) ?? [];
        const result = list.map(info => this.makeFilesystemItem(info, join(folderPath, info.name)));
        return result;
    }

    public async selectOne(path: string = ''): Promise<FilesystemItem | undefined> {
        const value = await this.getInfo(path);
        this.sendSelectOneEvent(value);
        return value;
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
        await connection.downloadTo(writableStreamBuffer, this.getFullPath(filePath));
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

        if (exists) this.sendUpdateEvent(remotePath, fileContents!);
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

    public async getInfo(path: string): Promise<FilesystemItem | undefined> {
        const fullPath = this.getFullPath(path);
        const connection = await this.endpoint.getConnection();

        const res: FtpFileInfo[] = await connection.list(fullPath);
        if (!res) return undefined;

        if (res.length === 1 && !res[0].name.localeCompare(fullPath)) return this.makeFilesystemItem(res[0], path);

        if (res.length) return {
            name: extractFileName(path),
            path,
            fullPath: fullPath,
            type: FilesystemItemType.Directory
        }

        // Test the folder existance
        try {
            const curpath = await connection.pwd();
            const resp: { code: number, message: string }[] = await connection.cwd(fullPath);
            await connection.cwd(curpath);
        }
        catch (err) {
            // Can't change directory => folder is not exists
            if (err.code == 550) return undefined;
            throw err;
        }
        
        return {
            name: extractFileName(path),
            path,
            fullPath: this.getFullPath(path),
            type: FilesystemItemType.Directory
        }
    }

    public async isFolder(path: string): Promise<boolean | undefined> {
        const info = await this.getInfo(path);
        if (!info) return undefined;
        return info.type === FilesystemItemType.Directory;
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


