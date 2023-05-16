import * as ftp from "basic-ftp";
import { AccessOptions, Client, FileInfo } from "basic-ftp";
import { Readable } from "stream";
import { BaseEndpoint} from "../../core/endpoint.js";
import { extractParentFolderPath } from "../../utils/index.js";
import { BaseObservable } from "../../core/observable.js";
import { CollectionOptions } from "../../core/readonly_collection.js";
import { UpdatableCollection } from "../../core/updatable_collection.js";


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

    getFolder(folderPath: string = '.', options: CollectionOptions<FileInfo> = {}): Collection {
        options.displayName ??= folderPath;

        let path = folderPath == '.' ? '' : folderPath;
        return this._addCollection(folderPath, new Collection(this, folderPath, path, options));
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


export class Collection extends UpdatableCollection<FileInfo> {
    protected static instanceCount = 0;

    protected folderPath: string;

    constructor(endpoint: Endpoint, collectionName: string, folderPath: string, options: CollectionOptions<FileInfo> = {}) {
        Collection.instanceCount++;
        super(endpoint, collectionName, options);
        this.folderPath = folderPath.trim();
        if (this.folderPath.endsWith('/')) this.folderPath.substring(0, this.folderPath.lastIndexOf('/'));
    }

    public select(): BaseObservable<FileInfo> {
        const observable = new BaseObservable<any>(this, (subscriber) => {
            this.sendStartEvent();
            try {
                (async () => {
                    try {
                        const connection = await this.endpoint.getConnection();
                        const list: FileInfo[] = await connection.list(this.folderPath);
                        
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


    public async insertFolder(remoteFolderPath: string) {
        await super.insert(remoteFolderPath);
        const connection = await this.endpoint.getConnection();
        await connection.ensureDir(remoteFolderPath);
    }

    public async insertFile(remoteFilePath: string, localFilePath: string);
    public async insertFile(remoteFilePath: string, sourceStream: Readable);
    public async insertFile(remoteFilePath: string, source: string | Readable) {
        await super.insert(remoteFilePath);
        const connection = await this.endpoint.getConnection();

        const parentPath = extractParentFolderPath(remoteFilePath);
        if (parentPath) await connection.ensureDir(parentPath);

        await connection.uploadFrom(source, remoteFilePath);
    }

    public async insertFileWithContents(remoteFilePath: string, fileContents: string) {
        let stream = Readable.from(fileContents);
        await this.insertFile(remoteFilePath, stream);
    }

    public async insert(remotePath: string, contents: { isFolder: boolean, localFilePath?: string, sourceStream?: Readable, contents?: string }) {
        if (contents.isFolder) return await this.insertFolder(remotePath);

        let source = contents.localFilePath ?? contents.sourceStream;
        if (contents.contents) source = Readable.from(contents.contents);

        return await this.insertFile(remotePath, source as any);
    }


    public async deleteFolder(remoteFolderPath: string) {
        await super.delete(remoteFolderPath);
        const connection = await this.endpoint.getConnection();
        await connection.removeDir(remoteFolderPath);
    }

    public async deleteEmptyFolder(remoteFolderPath: string) {
        await super.delete(remoteFolderPath);
        const connection = await this.endpoint.getConnection();
        await connection.removeEmptyDir(remoteFolderPath);
    }

    public async deleteFile(remoteFilePath: string) {
        await super.delete(remoteFilePath);
        const connection = await this.endpoint.getConnection();
        await connection.remove(remoteFilePath);
    }

    public async delete(remotePath: string) {
        const fileInfo = await this.getPathInfo(remotePath);
        if (!fileInfo) {
            await super.delete(remotePath);
            return;
        }

        if (fileInfo.isDirectory) return await this.deleteFolder(remotePath);
        return await this.deleteFile(remotePath);
    }

    public async getPathInfo(remotePath: string) {
        const connection = await this.endpoint.getConnection();
        const list: FileInfo[] = await connection.list(remotePath);
        if (!list || !list.length) return null;
        return list[0];
    }

    get endpoint(): Endpoint {
        return super.endpoint as Endpoint;
    }
}


