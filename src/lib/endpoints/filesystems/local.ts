import * as fs from "fs";
import { glob, Glob } from 'glob';
import PATH from 'path';
import internal from "stream";
import { BaseEndpoint} from "../../core/endpoint.js";
import { extractFileName, extractParentFolderPath, pathJoin } from "../../utils/index.js";
import { BaseObservable } from "../../core/observable.js";
import { FilesystemCollection } from "./filesystem_collection.js";
import { CollectionOptions } from "../../core/base_collection.js";


export type PathDetails = {
    isFolder: boolean
    name: string;
    relativePath: string; // Empty for root folder
    fullPath: string;
    parentFolderRelativePath: string; // '..' for root folder
    parentFolderFullPath: string;
    content?: Buffer;
}

export type ReadOptions = {
    // false by default
    includeRootDir?: boolean;
    // all is default
    objectsToSearch?: 'filesOnly' | 'foldersOnly' | 'all';
    // false by default
    withContent?: boolean;
}

export class Endpoint extends BaseEndpoint {
    protected rootFolder: string = null;

    constructor(rootFolder: string) {
        super();
        this.rootFolder = rootFolder.trim();
        if (this.rootFolder.endsWith('/')) this.rootFolder.substring(0, this.rootFolder.lastIndexOf('/'));
    }

    getFolder(folderName: string = '.', options: CollectionOptions<PathDetails> = {}): Collection {
        options.displayName ??= this.getName(folderName);

        let path = folderName == '.' ? '' : folderName;
        if (this.rootFolder) path = pathJoin([this.rootFolder, path], '/');
        return this._addCollection(folderName, new Collection(this, folderName, path, options));
    }

    releaseFolder(folderName: string) {
        this._removeCollection(folderName);
    }

    protected getName(folderPath: string) {
        return folderPath.substring(folderPath.lastIndexOf('/') + 1);
    }

    get displayName(): string {
        return this.rootFolder ? `Local filesystem (${this.rootFolder})` : `Local filesystem (${this.instanceNo})`;
    }
}

export function getEndpoint(rootFolder: string = null): Endpoint {
    return new Endpoint(rootFolder);
}

export class Collection extends FilesystemCollection<PathDetails> {
    protected static instanceCount = 0;

    protected rootPath: string;

    constructor(endpoint: Endpoint, collectionName: string, rootPath: string, options: CollectionOptions<PathDetails> = {}) {
        Collection.instanceCount++;
        super(endpoint, collectionName, options);
        this.rootPath = rootPath.trim();
        if (this.rootPath.endsWith('/')) this.rootPath.substring(0, this.rootPath.lastIndexOf('/'));
    }

    // Uses simple path syntax from lodash.get function
    // path example: 'store.book[5].author'
    // use path '' for the root object
    public select(mask: string = '*', options: ReadOptions = {}): BaseObservable<PathDetails> {
        const observable = new BaseObservable<any>(this, (subscriber) => {
            this.sendStartEvent();
            try {
                (async () => {
                    try {
                        if (options.includeRootDir && (options.objectsToSearch == 'all' || options.objectsToSearch == 'foldersOnly' || !options.objectsToSearch)) {
                            let res = this.getRootFolderDetails();
                            this.sendReciveEvent(res);
                            subscriber.next(res);
                        }
                
                        const matches = new Glob(mask, { cwd: this.rootPath });

                        for await (const path of matches) {
                            const res: PathDetails = await this.getInfo(path, options);
                            if ( (res.isFolder && (options.objectsToSearch == 'all' || options.objectsToSearch == 'foldersOnly' || !options.objectsToSearch))
                                || (!res.isFolder && (options.objectsToSearch == 'filesOnly' || options.objectsToSearch == 'all' || !options.objectsToSearch)) )
                            {
                                if (subscriber.closed) break;
                                await this.waitWhilePaused();
                                this.sendReciveEvent(res);
                                subscriber.next(res);
                            }
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

    public async get(filePath: string): Promise<string> {
        if (await this.isFolder(filePath)) throw new Error("Error: method 'get' of filesystem collections can be used only with file path");

        let resStr = '';
        if (await this.isExists(filePath)) {
            const res = await fs.promises.readFile(this.getFullPath(filePath));
            resStr = res.toString();
        }
        
        this.sendGetEvent(resStr, filePath);
        return resStr;
    }

    public async list(folderPath: string = ''): Promise<PathDetails[]> {
        const res = await this._list(folderPath);
        this.sendListEvent(res, folderPath);
        return res;
    }

    public async _list(folderPath: string = ''): Promise<PathDetails[]> {
        const res: PathDetails[] = [];
        const names = await fs.promises.readdir(this.getFullPath(folderPath));
        for (const path of names) res.push(await this.getInfo(path));
        return res;
    }


    public async find(folderPath: string = '', params: {mask?: string, options?: ReadOptions} = { mask: '*', options: {} }): Promise<PathDetails[]> {
        const res: PathDetails[] = [];

        if (params?.options?.includeRootDir && (params?.options?.objectsToSearch === 'all' || params?.options?.objectsToSearch === 'foldersOnly' || !params?.options?.objectsToSearch)) {
            res.push(this.getRootFolderDetails());
        }

        const matches = await glob(params?.mask ?? '*', { cwd: this.getFullPath(folderPath) });

        for (const path of matches) {
            const cur: PathDetails = await this.getInfo(path, params?.options);
            if ( (cur.isFolder && (params?.options?.objectsToSearch == 'all' || params?.options?.objectsToSearch == 'foldersOnly' || !params?.options?.objectsToSearch))
                || (!cur.isFolder && (params?.options?.objectsToSearch == 'filesOnly' || params?.options?.objectsToSearch == 'all' || !params?.options?.objectsToSearch)) )
            {
                res.push(cur);
            }
        }

        this.sendFindEvent(res, folderPath, params);
        return res;
    }

    public async insert(folderPathDetails: PathDetails): Promise<void>;
    public async insert(pathDetails: PathDetails, fileContents: string): Promise<void>;
    public async insert(pathDetails: PathDetails, sourceStream: NodeJS.ArrayBufferView | Iterable<string | NodeJS.ArrayBufferView> | AsyncIterable<string | NodeJS.ArrayBufferView> | internal.Readable): Promise<void>;
    public async insert(folderPath: string): Promise<void>;
    public async insert(filePath: string, fileContents: string): Promise<void>;
    public async insert(filePath: string, sourceStream: NodeJS.ArrayBufferView | Iterable<string | NodeJS.ArrayBufferView> | AsyncIterable<string | NodeJS.ArrayBufferView> | internal.Readable): Promise<void>;
    public async insert(pathDetails: string | PathDetails, data?: string | string | NodeJS.ArrayBufferView | Iterable<string | NodeJS.ArrayBufferView> | AsyncIterable<string | NodeJS.ArrayBufferView> | internal.Readable) {
        let path = (typeof pathDetails === 'string') ? pathDetails : pathDetails.fullPath;
        const fullPath = this.getFullPath(path);
        this.sendInsertEvent(path, data);
        
        if (await this.isExists(path)) throw new Error(`Path ${path} already exists`);

        if (typeof data === 'undefined') {
            await this.ensureDir(path);
            return;
        }

        await this.ensureParentDir(path);
        await fs.promises.writeFile(fullPath, data);
    }

    public async update(pathDetails: PathDetails, fileContents: string): Promise<void>;
    public async update(pathDetails: PathDetails, sourceStream: NodeJS.ArrayBufferView | Iterable<string | NodeJS.ArrayBufferView> | AsyncIterable<string | NodeJS.ArrayBufferView> | internal.Readable): Promise<void>;
    public async update(filePath: string, fileContents: string): Promise<void>;
    public async update(filePath: string, sourceStream: NodeJS.ArrayBufferView | Iterable<string | NodeJS.ArrayBufferView> | AsyncIterable<string | NodeJS.ArrayBufferView> | internal.Readable): Promise<void>;
    public async update(pathDetails: string | PathDetails, data: string | string | NodeJS.ArrayBufferView | Iterable<string | NodeJS.ArrayBufferView> | AsyncIterable<string | NodeJS.ArrayBufferView> | internal.Readable): Promise<void> {
        let filePath = (typeof pathDetails === 'string') ? pathDetails : pathDetails.fullPath;
        this.sendUpdateEvent(filePath, data);
        if (!await this.isExists(filePath)) throw new Error(`File ${filePath} does not exists`);

        await fs.promises.writeFile(this.getFullPath(filePath), data);
    }

    public async upsert(folderPathDetails: PathDetails): Promise<boolean>;
    public async upsert(pathDetails: PathDetails, fileContents: string): Promise<boolean>;
    public async upsert(pathDetails: PathDetails, sourceStream: NodeJS.ArrayBufferView | Iterable<string | NodeJS.ArrayBufferView> | AsyncIterable<string | NodeJS.ArrayBufferView> | internal.Readable): Promise<boolean>;
    public async upsert(folderPath: string): Promise<boolean>;
    public async upsert(filePath: string, fileContents: string): Promise<boolean>;
    public async upsert(filePath: string, sourceStream: NodeJS.ArrayBufferView | Iterable<string | NodeJS.ArrayBufferView> | AsyncIterable<string | NodeJS.ArrayBufferView> | internal.Readable): Promise<boolean>;
    public async upsert(pathDetails: string | PathDetails, data?: string | string | NodeJS.ArrayBufferView | Iterable<string | NodeJS.ArrayBufferView> | AsyncIterable<string | NodeJS.ArrayBufferView> | internal.Readable): Promise<boolean> {
        let path = (typeof pathDetails === 'string') ? pathDetails : pathDetails.fullPath;
        const exists = await this.isExists(path);

        if (exists) this.sendUpdateEvent(path, data);
        else this.sendInsertEvent(path, data);

        if (typeof data === 'undefined') return await this.ensureDir(path);

        await this.ensureParentDir(path);

        if (exists) await this.update(path, data);
        else await this.insert(path, data);
        return exists;
    }

    public async delete(mask: string = '*', options: ReadOptions = {}): Promise<boolean> {
        this.sendDeleteEvent(mask);

        if (options.includeRootDir && (options.objectsToSearch == 'all' || options.objectsToSearch == 'foldersOnly' || !options.objectsToSearch)) {
            let res = this.isExists(this.rootPath);
            fs.rmSync(this.rootPath, { recursive: true, force: true });
            return res;
        }

        const matches = glob.sync(mask, {cwd: this.rootPath});
        let res = false;
        for (let i = 0; i < matches.length; i++) {
            const matchPath = PATH.join(this.rootPath, matches[i]);
            const isFolder = (await fs.promises.lstat(matchPath)).isDirectory();

            if ( isFolder && (options.objectsToSearch == 'all' || options.objectsToSearch == 'foldersOnly' || !options.objectsToSearch) ) {
                fs.rmdirSync(matchPath, { recursive: true });
                res = true;
            }

            if ( !isFolder && (options.objectsToSearch == 'filesOnly' || options.objectsToSearch == 'all' || !options.objectsToSearch) ) {
                fs.rmSync(matchPath, { force: true });
                res = true;
            }
            
        };
        return res;
    }

    
    public async append(path: string, data: string | internal.Readable): Promise<void>;
    public async append(pathDetails: PathDetails, data: string | internal.Readable): Promise<void>;
    public async append(pathDetails: string | PathDetails, data: string | internal.Readable): Promise<void> {
        let filePath = (typeof pathDetails === 'string') ? pathDetails : pathDetails.fullPath;
        this.sendUpdateEvent(filePath, data);
        
        if (!await this.isExists(filePath)) throw new Error(`File ${filePath} does not exists`);

        if (typeof data !== 'string') data = await this.streamToString(data);
        await fs.promises.appendFile(this.getFullPath(filePath), data);
    }


    public async clear(path: string): Promise<void>;
    public async clear(pathDetails: PathDetails): Promise<void>;
    public async clear(pathDetails: string | PathDetails): Promise<void> {
        let filePath = (typeof pathDetails === 'string') ? pathDetails : pathDetails.fullPath;
        this.sendUpdateEvent(filePath, '');
        const path = this.getFullPath(filePath);
        
        if (!await this.isExists(filePath)) throw new Error(`Path ${filePath} does not exists`);

        if (await this.isFolder(filePath)) {
            await fs.promises.rmdir(path, {recursive: true});
            await fs.promises.mkdir(path);
            return;
        }

        await fs.promises.writeFile(path, '');
    }

    public async copy(srcPath: string, dstPath: string): Promise<void>;
    public async copy(srcPathDetails: PathDetails, dstPath: string): Promise<void>;
    public async copy(srcPathDetails: string | PathDetails, dstPath: string): Promise<void> {
        let srcPath = (typeof srcPathDetails === 'string') ? srcPathDetails : srcPathDetails.fullPath;
        this.sendCopyEvent(srcPath, dstPath);
        const srcPathFull = this.getFullPath(srcPath);
        const dstPathFull = this.getFullPath(dstPath);
        if (await this.isFolder(srcPath)) {
            await fs.promises.cp(srcPathFull, dstPathFull);
            return;
        }
        await fs.promises.copyFile(srcPathFull, dstPathFull);
    }

    public async move(srcPath: string, dstPath: string): Promise<void>;
    public async move(srcPathDetails: PathDetails, dstPath: string): Promise<void>;
    public async move(srcPathDetails: string | PathDetails, dstPath: string): Promise<void> {
        let srcPath = (typeof srcPathDetails === 'string') ? srcPathDetails : srcPathDetails.fullPath;
        this.sendCopyEvent(srcPath, dstPath);
        const srcPathFull = this.getFullPath(srcPath);
        const dstPathFull = this.getFullPath(dstPath);
        await fs.promises.rename(srcPathFull, dstPathFull);
    }


    protected isRootCur(): boolean {
        const r = PATH.join(this.rootPath.trim()).trim();
        return this.rootPath == '.' || this.rootPath == '';
    }

    protected getRootFolderDetails(): PathDetails {
        const fullPath = PATH.resolve(this.rootPath);
        return {
            isFolder: true,
            name: extractFileName(fullPath),
            relativePath: '',
            fullPath,
            parentFolderRelativePath: '..',
            parentFolderFullPath: PATH.resolve(this.rootPath + '/..')
        }
    }

    protected removeLeadingRoot(path: string): string {
        if (path.startsWith(this.rootPath)) path = path.substring(this.rootPath.length);
        if (path.startsWith('./')) path = path.substring(2);
        if (path.startsWith('/')) path = path.substring(1);
        return path;
    }


    public async getInfo(relativePath: string, options?: ReadOptions) {
        const rootFolderPath = this.rootPath ? this.rootPath : '.';
        const fullPath = PATH.resolve(rootFolderPath + '/' + relativePath);

        const res = {
            isFolder: (await fs.promises.lstat(fullPath)).isDirectory(),
            name: extractFileName(relativePath),
            relativePath,
            fullPath,
            parentFolderRelativePath: extractParentFolderPath(relativePath),
            parentFolderFullPath: extractParentFolderPath(fullPath)
        } as PathDetails;
        if (options && options.withContent) {
            res.content = fs.readFileSync(fullPath);
        }
        return res;
    }


    public async getInfoExt(path: string): Promise<fs.Stats> {
        return await fs.promises.stat(this.getFullPath(path));
    } 

    public async isExists(path: string): Promise<boolean> {
        return fs.existsSync(this.getFullPath(path));
    }
    public async isFolder(path: string): Promise<boolean> {
        return (await this.getInfo(path)).isFolder;
    }


    protected getFullPath(path: string): string {
        return PATH.join(this.rootPath, path);
    }
     
    protected async ensureDir(path: string): Promise<boolean> {
        const exists = await this.isExists(path);
        if (!exists) await fs.promises.mkdir(this.getFullPath(path), {recursive: true})
        return exists;
    }

    protected async ensureParentDir(path: string): Promise<boolean> {
        const parentPath = extractParentFolderPath(this.getFullPath(path));
        
        const exists = fs.existsSync(parentPath);
        if (!exists) await fs.promises.mkdir(parentPath, {recursive: true})
        return exists;
    }


    protected streamToString(stream): Promise<string> {
        stream.setEncoding('utf-8'); // do this instead of directly converting the string
        const chunks = [];
        return new Promise((resolve, reject) => {
            stream.on('data', (chunk) => chunks.push(chunk));
            stream.on('error', (err) => reject(err));
            stream.on('end', () => resolve(chunks.join("")));
        })
    }
    
}


