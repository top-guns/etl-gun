import * as ix from 'ix';
import * as fs from "fs";
import * as webdav from "webdav";
import { Readable } from "stream";
import { BaseEndpoint} from "../../core/endpoint.js";
import { extractParentFolderPath, pathJoin } from "../../utils/index.js";
import { BaseObservable } from "../../core/observable.js";
import { RemoteFilesystemCollection } from "./remote_filesystem_collection.js";
import { join } from "path";
import { FilesystemCollectionOptions, FilesystemItem, FilesystemItemType } from './filesystem_collection.js';


export class Endpoint extends BaseEndpoint {
    protected _client: webdav.WebDAVClient = null;
    protected url: string;
    protected options: webdav.WebDAVClientOptions;

    async getConnection(): Promise<webdav.WebDAVClient> {
        if (!this._client) {
            this._client = webdav.createClient(this.url, this.options);
        }
        return this._client;
    }

    
    constructor(url: string, options: webdav.WebDAVClientOptions) {
        super();
        this.url = url;
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
        return `WebDAV (${this.url})`;
    }
}


export function getEndpoint(url: string, options: webdav.WebDAVClientOptions): Endpoint {
    return new Endpoint(url, options);
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

    protected makeFilesystemItem(info: webdav.FileStat, folderPath: string, contents?: string): FilesystemItem {
        const path = join(folderPath, info.filename);
        return {
            name: info.filename,
            path,
            fullPath: this.getFullPath(path),
            size: info.size, 
            type: info.type === 'directory' ? FilesystemItemType.Directory : info.type === 'file' ? FilesystemItemType.File : FilesystemItemType.Unknown,
            modifiedAt: new Date(info.lastmod),
            fileContents: contents
        }
    }

    protected async _select(folderPath: string = '', mask: string = '*', recursive: boolean = false): Promise<FilesystemItem[]> {
        const connection = await this.endpoint.getConnection();
        const list: webdav.FileStat[] = await connection.getDirectoryContents(this.getFullPath(folderPath), { deep: recursive, glob: mask }) as webdav.FileStat[];
        const result = list.map(info => this.makeFilesystemItem(info, folderPath));
        return result;
    }

    public async select(folderPath?: string, mask: string = '*', recursive: boolean = false): Promise<FilesystemItem[]> {
        return super.select(folderPath, mask, recursive);
    }

    public async* selectGen(folderPath: string = '', mask: string = '*', recursive: boolean = false): AsyncGenerator<FilesystemItem, void, void> {
        const generator = super.selectGen(folderPath, mask, recursive);
        for await (const value of generator) yield value;
    }

    public selectRx(folderPath: string = '', mask: string = '*', recursive: boolean = false): BaseObservable<FilesystemItem> {
        return super.selectRx(folderPath, mask, recursive);
    }

    public selectIx(folderPath?: string, mask: string = '*', recursive: boolean = false): ix.AsyncIterable<FilesystemItem> {
        return super.selectIx(folderPath, mask, recursive);
    }

    public selectStream(folderPath?: string, mask: string = '*', recursive: boolean = false): ReadableStream<FilesystemItem> {
        return super.selectStream(folderPath, mask, recursive);
    }

    // Get

    public async read(filePath: string): Promise<undefined | string> {
        const connection = await this.endpoint.getConnection();
        if (await this.isFolder(filePath)) throw new Error("Error: method 'get' of filesystem collections can be used only with file path");

        const res: string = await connection.getFileContents(this.getFullPath(filePath), { format: 'text' }) as string;
        this.sendGetEvent(res, filePath);
        return res;
    }

    // Insert

    protected async _insert(path: string, fileContents?: string | Readable): Promise<void> {
        if (await this.isExists(path)) throw new Error(`Path ${path} already exists`);

        if (typeof fileContents === 'undefined') {
            await this.ensureDir(path);
            return;
        }

        await this.ensureParentDir(path);
        const connection = await this.endpoint.getConnection();
        await connection.putFileContents(this.getFullPath(path), fileContents, { overwrite: false });
    }

    // Update

    public async update(filePath: string, fileContents: string): Promise<void>;
    public async update(filePath: string, sourceStream: Readable): Promise<void>;
    public async update(filePath: string, fileContents: string | Readable): Promise<void> {
        this.sendUpdateEvent(filePath, fileContents);
        const connection = await this.endpoint.getConnection();

        if (!await this.isExists(filePath)) throw new Error(`File ${filePath} does not exists`);

        if (typeof fileContents === 'string') fileContents = Readable.from(fileContents);
        await connection.putFileContents(this.getFullPath(filePath), fileContents, { overwrite: true });
    }

    // Upsert

    public async upsert(folderPath: string): Promise<boolean>;
    public async upsert(filePath: string, fileContents: string): Promise<boolean>;
    public async upsert(filePath: string, sourceStream: Readable): Promise<boolean>;
    public async upsert(path: string, fileContents?: string | Readable): Promise<boolean> {
        const exists = await this.isExists(path);

        if (exists) this.sendUpdateEvent(path, fileContents);
        else this.sendInsertEvent(path, fileContents);

        if (typeof fileContents === 'undefined') return await this.ensureDir(path);

        await this.ensureParentDir(path);

        if (exists) await this.update(path, fileContents as any);
        else await this.insert(path, fileContents as any);
        return exists;
    }

    // Delete

    public async delete(path: string) {
        this.sendDeleteEvent(path);
        const connection = await this.endpoint.getConnection();
        if (!await this.isExists(path)) return false;

        await connection.deleteFile(this.getFullPath(path));

        return true;
    }

    // Append & clear

    public async append(filePath: string, fileContents: string): Promise<void>;
    public async append(filePath: string, sourceStream: Readable): Promise<void>;
    public async append(filePath: string, fileContents: string | Readable): Promise<void> {
        throw new Error('Method append is not implemented');
        // this.sendUpdateEvent(remoteFilePath, fileContents);
        // const path = this.getFullPath(remoteFilePath);
        // const connection = await this.endpoint.getConnection();
        
        // if (!await this.isExists(remoteFilePath)) throw new Error(`File ${remoteFilePath} does not exists`);
        // if (await this.isFolder) throw new Error(`You cannot use append method for folders`);

        // if (typeof fileContents === 'string') fileContents = Readable.from(fileContents);
        // connection.putFileContents(path, fileContents, { overwrite: false });
    }

    public async clear(path: string): Promise<void> {
        this.sendUpdateEvent(path, '');
        const connection = await this.endpoint.getConnection();
        
        if (!await this.isExists(path)) throw new Error(`Path ${path} does not exists`);

        if (await this.isFolder(path)) {
            await this.delete(path);
            await this.ensureDir(path);
            return;
        }

        await connection.putFileContents(this.getFullPath(path), '');
    }

    //

    public async copy(remoteSrcPath: string, remoteDstPath: string): Promise<void> {
        //if (await this.isFolder) throw new Error(`Method copy for the webdav collection is not supports folders, but only files`);
        const connection = await this.endpoint.getConnection();
        await connection.copyFile(remoteSrcPath, remoteDstPath);
    }
    public async move(remoteSrcPath: string, remoteDstPath: string): Promise<void> {
        //if (await this.isFolder) throw new Error(`Method copy for the webdav collection is not supports folders, but only files`);
        const connection = await this.endpoint.getConnection();
        await connection.moveFile(remoteSrcPath, remoteDstPath);
    }
  
    public async download(remoteFilePath: string, localPath: string): Promise<void> {
        const connection = await this.endpoint.getConnection();
        this.sendDownloadEvent(remoteFilePath, localPath);

        let filePath = localPath;
        //let folderPath = localPath;
        if (filePath.endsWith('/')) {
            filePath = pathJoin([filePath, this.extractFilename(remoteFilePath)], '/');
            let exists = fs.existsSync(localPath);
            if (!exists) await fs.promises.mkdir(localPath, { recursive: true });
        }
        else {
            let stat = fs.statSync(filePath);
            if (stat && stat.isDirectory) filePath = pathJoin([filePath, this.extractFilename(remoteFilePath)], '/');
            else {
                const urlParts = filePath.split("/");
                urlParts.pop();
                const folderPath = urlParts.join('/');
                let exists = fs.existsSync(folderPath);
                if (!exists) await fs.promises.mkdir(folderPath, { recursive: true });
            }
        }
        
        return new Promise((resolve, reject) => {
            const file = fs.createWriteStream(filePath, { flags: "wx" });

            connection.createReadStream(this.getFullPath(remoteFilePath))
            .pipe(
                file
            )
            .on("error", err => {
                file.close();
                fs.unlink(filePath, () => {}); // Delete temp file
                reject(err.message);
            });

            file.on("finish", () => {
                resolve();
            });
        });
    }
    public async upload(localFilePath: string, remotePath: string): Promise<void> {
        const connection = await this.endpoint.getConnection();
        this.sendUploadEvent(localFilePath, remotePath);

        let url = this.getFullPath(remotePath);
        let exists = await this.isExists(remotePath);
        //let folderPath = localPath;
        if (url.endsWith('/') || (exists && await this.isFolder(remotePath))) {
            url = pathJoin([url, this.extractFilename(localFilePath)], '/');
            if (!exists) await connection.createDirectory(this.getFullPath(remotePath), { recursive: true });
        }
        else {
            const urlParts = url.split("/");
            urlParts.pop();
            const folderPath = urlParts.join('/');
            if (!await this.isExists(folderPath)) await connection.createDirectory(this.getFullPath(folderPath), { recursive: true });
        }
        
        return new Promise((resolve, reject) => {
            const link = connection.createWriteStream(url);

            fs.createReadStream(this.getFullPath(localFilePath))
            .pipe(
                link
            )
            .on("error", err => {
                reject(err.message);
            });

            link.on("finish", () => {
                resolve();
            });
        });
    }

    // Get info

    public async getInfo(path: string): Promise<webdav.FileStat> {
        const connection = await this.endpoint.getConnection();
        const res: any = await connection.stat(this.getFullPath(path));
        const stat: webdav.FileStat = ((res.data && res.headers) ? res.data : res) as webdav.FileStat;
        if (!stat) return null;
        return stat;
    }

    public async isFolder(path: string): Promise<boolean> {
        const info = await this.getInfo(path);
        return info.type == 'directory';
    }

    public async isExists(path: string) {
        const connection = await this.endpoint.getConnection();
        return await connection.exists(path);
    }


    protected getFullPath(path: string): string {
        return join(this.rootPath, path);
    }

    protected async ensureDir(folderPath: string): Promise<boolean> {
        const connection = await this.endpoint.getConnection();
        const exists = await this.isExists(folderPath);
        if (!exists) await connection.createDirectory(this.getFullPath(folderPath), { recursive: true });
        return exists;
    }

    protected async ensureParentDir(path: string): Promise<boolean> {
        const connection = await this.endpoint.getConnection();
        const parentPath = extractParentFolderPath(this.getFullPath(path));
        
        const exists = await connection.exists(parentPath);
        if (!exists) await connection.createDirectory(parentPath, { recursive: true });
        return exists;
    }


    protected extractFilename(urlOrPath: string): string {
        const urlParts = urlOrPath.split("/");
        return urlParts[urlParts.length - 1] ?? urlOrPath;
    }


    get endpoint(): Endpoint {
        return super.endpoint as Endpoint;
    }
}


