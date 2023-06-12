import * as ix from 'ix';
import * as fs from "fs";
import { glob, Glob } from 'glob';
import PATH, { join } from 'path';
import internal from "stream";
import { BaseEndpoint} from "../../core/endpoint.js";
import { extractFileName, extractParentFolderPath, pathJoin } from "../../utils/index.js";
import { BaseObservable } from "../../core/observable.js";
import { FilesystemCollection, FilesystemCollectionOptions, FilesystemItem, FilesystemItemType } from "./filesystem_collection.js";
import { generator2Iterable, generator2Observable, generator2Promise, observable2Stream, selectOne_from_Generator, wrapGenerator, wrapObservable, wrapPromise } from "../../utils/flows.js";


export class Endpoint extends BaseEndpoint {
    protected rootFolder: string = '';

    constructor(rootFolder: string) {
        super();
        this.rootFolder = rootFolder.trim();
        if (this.rootFolder.endsWith('/')) this.rootFolder.substring(0, this.rootFolder.lastIndexOf('/'));
    }

    getFolder(folderName: string = '.', options: FilesystemCollectionOptions = {}): Collection {
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

export function getEndpoint(rootFolder: string = ''): Endpoint {
    return new Endpoint(rootFolder);
}

export class Collection extends FilesystemCollection {
    protected static instanceCount = 0;

    protected rootPath: string;

    constructor(endpoint: Endpoint, collectionName: string, rootPath: string, options: FilesystemCollectionOptions = {}) {
        Collection.instanceCount++;
        super(endpoint, collectionName, options);
        this.rootPath = rootPath.trim();
        if (this.rootPath.endsWith('/')) this.rootPath.substring(0, this.rootPath.lastIndexOf('/'));
    }

    protected makeFilesystemItem(info: fs.Stats, fileName: string, folderPath: string, contents?: string): FilesystemItem {
        const path = join(folderPath, fileName);
        return {
            name: fileName,
            path,
            fullPath: this.getFullPath(path),
            size: info.size,
            type: info.isDirectory() ? FilesystemItemType.Directory : info.isFile() ? FilesystemItemType.File : info.isSymbolicLink() ? FilesystemItemType.SymbolicLink : FilesystemItemType.Unknown,
            modifiedAt: info.mtime,
            fileContents: contents
        }
    }

    protected async* _selectGen(mask: string = '*'): AsyncGenerator<FilesystemItem, void, void> {
        const matches = new Glob(mask, { cwd: this.rootPath });
        for await (const path of matches) {
            const fullPath = PATH.resolve(this.rootPath, path);
            const info = await await fs.promises.lstat(fullPath);
            const value = this.makeFilesystemItem(info, extractFileName(path), extractParentFolderPath(path));
            yield value;
        }
    }

    protected async _select(mask: string = '*'): Promise<FilesystemItem[]> {
        const generator = this._selectGen(mask);
        const values = await wrapPromise(generator2Promise(generator), this);
        return values;
    }

    public async select(mask: string = '*'): Promise<FilesystemItem[]> {
        const generator = this._selectGen(mask);
        const values = await wrapPromise(generator2Promise(generator), this);
        return values;
    }

    public async* selectGen(mask: string = '*'): AsyncGenerator<FilesystemItem, void, void> {
        const generator = wrapGenerator(this._selectGen(mask), this);
        for await (const value of generator) yield value;
    }

    // Uses simple path syntax from lodash.get function
    // path example: 'store.book[5].author'
    // use path '' for the root object
    public selectRx(mask: string = '*'): BaseObservable<FilesystemItem> {
        const generator = this._selectGen(mask);
        return wrapObservable(generator2Observable(generator), this);
    }

    public selectIx(mask?: string): ix.AsyncIterable<FilesystemItem> {
        return generator2Iterable(this.selectGen(mask));
    }

    public selectStream(mask?: string): ReadableStream<FilesystemItem> {
        return observable2Stream(this.selectRx(mask));
    }


    public async read(filePath: string): Promise<string> {
        if (await this.isFolder(filePath)) throw new Error("Error: method 'get' of filesystem collections can be used only with file path");

        let resStr = '';
        if (await this.isExists(filePath)) {
            const res = await fs.promises.readFile(this.getFullPath(filePath));
            resStr = res.toString();
        }
        
        this.sendGetEvent(resStr, filePath);
        return resStr;
    }

    protected async _insert(pathDetails: string | FilesystemItem, data?: string | string | NodeJS.ArrayBufferView | Iterable<string | NodeJS.ArrayBufferView> | AsyncIterable<string | NodeJS.ArrayBufferView> | internal.Readable) {
        let path = (typeof pathDetails === 'string') ? pathDetails : pathDetails.path;
        const fullPath = this.getFullPath(path);
        
        if (await this.isExists(path)) throw new Error(`Path ${path} already exists`);

        if (typeof data === 'undefined') {
            await this.ensureDir(path);
            return;
        }

        await this.ensureParentDir(path);
        await fs.promises.writeFile(fullPath, data);
    }

    public async insertExt(folderPathDetails: FilesystemItem): Promise<void>;
    public async insertExt(pathDetails: FilesystemItem, fileContents: string): Promise<void>;
    public async insertExt(pathDetails: FilesystemItem, sourceStream: NodeJS.ArrayBufferView | Iterable<string | NodeJS.ArrayBufferView> | AsyncIterable<string | NodeJS.ArrayBufferView> | internal.Readable): Promise<void>;
    public async insertExt(folderPath: string): Promise<void>;
    public async insertExt(filePath: string, fileContents: string): Promise<void>;
    public async insertExt(filePath: string, sourceStream: NodeJS.ArrayBufferView | Iterable<string | NodeJS.ArrayBufferView> | AsyncIterable<string | NodeJS.ArrayBufferView> | internal.Readable): Promise<void>;
    public async insertExt(pathDetails: string | FilesystemItem, data?: string | string | NodeJS.ArrayBufferView | Iterable<string | NodeJS.ArrayBufferView> | AsyncIterable<string | NodeJS.ArrayBufferView> | internal.Readable) {
        let path = (typeof pathDetails === 'string') ? pathDetails : pathDetails.path;
        const fullPath = this.getFullPath(path);
        this.sendInsertEvent(path, data);
        return await this._insert(pathDetails, data);
    }

    public async update(pathDetails: FilesystemItem, fileContents: string): Promise<void>;
    public async update(pathDetails: FilesystemItem, sourceStream: NodeJS.ArrayBufferView | Iterable<string | NodeJS.ArrayBufferView> | AsyncIterable<string | NodeJS.ArrayBufferView> | internal.Readable): Promise<void>;
    public async update(filePath: string, fileContents: string): Promise<void>;
    public async update(filePath: string, sourceStream: NodeJS.ArrayBufferView | Iterable<string | NodeJS.ArrayBufferView> | AsyncIterable<string | NodeJS.ArrayBufferView> | internal.Readable): Promise<void>;
    public async update(pathDetails: string | FilesystemItem, data: string | string | NodeJS.ArrayBufferView | Iterable<string | NodeJS.ArrayBufferView> | AsyncIterable<string | NodeJS.ArrayBufferView> | internal.Readable): Promise<void> {
        let filePath = (typeof pathDetails === 'string') ? pathDetails : pathDetails.path;
        this.sendUpdateEvent(filePath, data);
        if (!await this.isExists(filePath)) throw new Error(`File ${filePath} does not exists`);

        await fs.promises.writeFile(this.getFullPath(filePath), data);
    }

    public async upsert(folderPathDetails: FilesystemItem): Promise<boolean>;
    public async upsert(pathDetails: FilesystemItem, fileContents: string): Promise<boolean>;
    public async upsert(pathDetails: FilesystemItem, sourceStream: NodeJS.ArrayBufferView | Iterable<string | NodeJS.ArrayBufferView> | AsyncIterable<string | NodeJS.ArrayBufferView> | internal.Readable): Promise<boolean>;
    public async upsert(folderPath: string): Promise<boolean>;
    public async upsert(filePath: string, fileContents: string): Promise<boolean>;
    public async upsert(filePath: string, sourceStream: NodeJS.ArrayBufferView | Iterable<string | NodeJS.ArrayBufferView> | AsyncIterable<string | NodeJS.ArrayBufferView> | internal.Readable): Promise<boolean>;
    public async upsert(pathDetails: string | FilesystemItem, data?: string | string | NodeJS.ArrayBufferView | Iterable<string | NodeJS.ArrayBufferView> | AsyncIterable<string | NodeJS.ArrayBufferView> | internal.Readable): Promise<boolean> {
        let path = (typeof pathDetails === 'string') ? pathDetails : pathDetails.path;
        const exists = await this.isExists(path);

        if (exists) this.sendUpdateEvent(path, data!);
        else this.sendInsertEvent(path, data);

        if (typeof data === 'undefined') return await this.ensureDir(path);

        await this.ensureParentDir(path);

        if (exists) await this.update(path, data);
        else await this._insert(path, data);
        return exists;
    }

    public async delete(mask: string = '*'): Promise<boolean> {
        this.sendDeleteEvent(mask);

        if (['.', ''].includes(mask)) {
            let res = this.isExists(this.rootPath);
            fs.rmSync(this.rootPath, { recursive: true, force: true });
            return res;
        }

        const matches = glob.sync(mask, {cwd: this.rootPath});
        let res = false;
        for (let i = 0; i < matches.length; i++) {
            const matchPath = PATH.join(this.rootPath, matches[i]);
            await fs.promises.rm(matchPath, { recursive: true, force: true });
            res = true;
        };
        return res;
    }

    
    public async append(path: string, data: string | internal.Readable): Promise<void>;
    public async append(pathDetails: FilesystemItem, data: string | internal.Readable): Promise<void>;
    public async append(pathDetails: string | FilesystemItem, data: string | internal.Readable): Promise<void> {
        let filePath = (typeof pathDetails === 'string') ? pathDetails : pathDetails.path;
        this.sendUpdateEvent(filePath, data);
        
        if (!await this.isExists(filePath)) throw new Error(`File ${filePath} does not exists`);

        if (typeof data !== 'string') data = await this.streamToString(data);
        await fs.promises.appendFile(this.getFullPath(filePath), data);
    }


    public async clear(path: string): Promise<void>;
    public async clear(pathDetails: FilesystemItem): Promise<void>;
    public async clear(pathDetails: string | FilesystemItem): Promise<void> {
        let filePath = (typeof pathDetails === 'string') ? pathDetails : pathDetails.path;
        this.sendUpdateEvent(filePath, '');
        const path = this.getFullPath(filePath);
        
        if (!await this.isExists(filePath)) throw new Error(`Path ${filePath} does not exists`);

        if (await this.isFolder(filePath)) {
            await fs.promises.rm(path, {recursive: true});
            await fs.promises.mkdir(path);
            return;
        }

        await fs.promises.writeFile(path, '');
    }

    public async copy(srcPath: string, dstPath: string): Promise<void>;
    public async copy(srcPathDetails: FilesystemItem, dstPath: string): Promise<void>;
    public async copy(srcPathDetails: string | FilesystemItem, dstPath: string): Promise<void> {
        let srcPath = (typeof srcPathDetails === 'string') ? srcPathDetails : srcPathDetails.path;
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
    public async move(srcPathDetails: FilesystemItem, dstPath: string): Promise<void>;
    public async move(srcPathDetails: string | FilesystemItem, dstPath: string): Promise<void> {
        let srcPath = (typeof srcPathDetails === 'string') ? srcPathDetails : srcPathDetails.path;
        this.sendCopyEvent(srcPath, dstPath);
        const srcPathFull = this.getFullPath(srcPath);
        const dstPathFull = this.getFullPath(dstPath);
        await fs.promises.rename(srcPathFull, dstPathFull);
    }


    protected isRootCur(): boolean {
        const r = PATH.join(this.rootPath.trim()).trim();
        return this.rootPath == '.' || this.rootPath == '';
    }

    protected getRootFolderDetails(): FilesystemItem {
        const fullPath = PATH.resolve(this.rootPath);
        return {
            name: extractFileName(fullPath),
            path: fullPath,
            fullPath: fullPath,
            size: undefined,
            type: FilesystemItemType.Directory,
            modifiedAt: undefined,
            fileContents: undefined
        }
    }

    protected removeLeadingRoot(path: string): string {
        if (path.startsWith(this.rootPath)) path = path.substring(this.rootPath.length);
        if (path.startsWith('./')) path = path.substring(2);
        if (path.startsWith('/')) path = path.substring(1);
        return path;
    }

    public async getInfo(path: string): Promise<fs.Stats | undefined> {
        return await fs.promises.stat(this.getFullPath(path));
    } 

    public async isExists(path: string): Promise<boolean> {
        return fs.existsSync(this.getFullPath(path));
    }
    public async isFolder(path: string): Promise<boolean | undefined> {
        const info = await this.getInfo(path);
        return info?.isDirectory();
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
        const chunks: string[] = [];
        return new Promise((resolve, reject) => {
            stream.on('data', (chunk) => chunks.push(chunk));
            stream.on('error', (err) => reject(err));
            stream.on('end', () => resolve(chunks.join("")));
        })
    }
    
}


