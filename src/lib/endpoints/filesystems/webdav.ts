import * as fs from "fs";
import * as webdav from "webdav";
import { Readable } from "stream";
import { BaseEndpoint} from "../../core/endpoint.js";
import { extractParentFolderPath, pathJoin } from "../../utils/index.js";
import { BaseObservable } from "../../core/observable.js";
import { RemoteFilesystemCollection } from "./remote_filesystem_collection.js";
import { join } from "path";
import { CollectionOptions } from "../../core/base_collection.js";


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

    getFolder(folderPath: string = '', options: CollectionOptions<webdav.FileStat> = {}): Collection {
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


export class Collection extends RemoteFilesystemCollection<webdav.FileStat> {
    protected static instanceCount = 0;

    protected rootPath: string;

    constructor(endpoint: Endpoint, collectionName: string, rootPath: string = '', options: CollectionOptions<webdav.FileStat> = {}) {
        Collection.instanceCount++;
        super(endpoint, collectionName, options);
        this.rootPath = rootPath.trim();
        if (this.rootPath.endsWith('/')) this.rootPath.substring(0, this.rootPath.lastIndexOf('/'));
    }

    public select(folderPath: string = ''): BaseObservable<webdav.FileStat> {
        const observable = new BaseObservable<any>(this, (subscriber) => {
            this.sendStartEvent();
            try {
                (async () => {
                    try {
                        const list: webdav.FileStat[] = await this._listWithoutEvent(folderPath);
                        
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

    public async get(filePath: string): Promise<undefined | string> {
        const connection = await this.endpoint.getConnection();
        if (await this.isFolder(filePath)) throw new Error("Error: method 'get' of filesystem collections can be used only with file path");

        const res: string = await connection.getFileContents(this.getFullPath(filePath), { format: 'text' }) as string;
        this.sendGetEvent(res, filePath);
        return res;
    }

    public async list(folderPath: string = ''): Promise<webdav.FileStat[]> {
        const res: webdav.FileStat[] = await this._listWithoutEvent(folderPath);
        this.sendListEvent(res, folderPath);
        return res;
    }

    public async find(folderPath: string = '', mask: string = '*', recursive: boolean = false): Promise<webdav.FileStat[]> {
        const connection = await this.endpoint.getConnection();
        const result: webdav.FileStat[] = await connection.getDirectoryContents(this.getFullPath(folderPath), { deep: recursive, glob: mask }) as webdav.FileStat[];
        this.sendFindEvent(result, folderPath, { mask, recursive });
        return result;
    }

    public async _listWithoutEvent(remotePath: string = ''): Promise<webdav.FileStat[]> {
        const connection = await this.endpoint.getConnection();
        const result: webdav.FileStat[] = await connection.getDirectoryContents(this.getFullPath(remotePath)) as webdav.FileStat[];
        return result;
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


