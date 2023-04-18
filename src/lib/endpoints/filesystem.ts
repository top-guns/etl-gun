import * as fs from "fs";
import glob from "glob";
import path from 'path';
import { Observable, Subscriber } from 'rxjs';
import internal from "stream";
import { BaseEndpoint} from "../core/endpoint.js";
import { BaseCollection, CollectionGuiOptions } from "../core/collection.js";
import { EtlObservable } from "../core/observable.js";
import { extractFileName, extractParentFolderPath, pathJoin } from "../utils/index.js";


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

    getFolder(folderName: string = '.', guiOptions: CollectionGuiOptions<PathDetails> = {}): Collection {
        guiOptions.displayName ??= this.getName(folderName);

        let path = folderName == '.' ? '' : folderName;
        if (this.rootFolder) path = pathJoin([this.rootFolder, path], '/');
        return this._addCollection(folderName, new Collection(this, path, guiOptions));
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

export class Collection extends BaseCollection<PathDetails> {
    protected static instanceCount = 0;
    protected folderPath: string;

    constructor(endpoint: Endpoint, folderPath: string, guiOptions: CollectionGuiOptions<PathDetails> = {}) {
        Collection.instanceCount++;
        super(endpoint, guiOptions);
        this.folderPath = folderPath.trim();
        if (this.folderPath.endsWith('/')) this.folderPath.substring(0, this.folderPath.lastIndexOf('/'));
    }

    // Uses simple path syntax from lodash.get function
    // path example: 'store.book[5].author'
    // use path '' for the root object
    public select(mask: string = '*', options: ReadOptions = {}): EtlObservable<PathDetails> {
        const observable = new EtlObservable<any>((subscriber) => {
            try {
                this.sendStartEvent();

                if (options.includeRootDir && (options.objectsToSearch == 'all' || options.objectsToSearch == 'foldersOnly' || !options.objectsToSearch)) {
                    let res = this.getRootFolderDetails();
                    this.sendReciveEvent(res);
                    subscriber.next(res);
                }
        
                glob(mask, {cwd: this.folderPath}, (err: Error, matches: string[]) => {
                    if (err) {
                        this.sendErrorEvent(err);
                        subscriber.error(err);
                        return;
                    }

                    (async () => {
                        for (let i = 0; i < matches.length; i++) {
                            const res: PathDetails = await this.getPathDetails(matches[i], options);
                            if ( (res.isFolder && (options.objectsToSearch == 'all' || options.objectsToSearch == 'foldersOnly' || !options.objectsToSearch))
                                || (!res.isFolder && (options.objectsToSearch == 'filesOnly' || options.objectsToSearch == 'all' || !options.objectsToSearch)) )
                            {
                                await this.waitWhilePaused();
                                this.sendReciveEvent(res);
                                subscriber.next(res);

                                if (i == matches.length - 1) {
                                    subscriber.complete();
                                    this.sendEndEvent();
                                }
                            }
                        };
                    })();
                });
            }
            catch(err) {
                this.sendErrorEvent(err);
                subscriber.error(err);
            }
        });
        return observable;
    }

    public async insert(pathDetails: PathDetails, data?: string | NodeJS.ArrayBufferView | Iterable<string | NodeJS.ArrayBufferView> | AsyncIterable<string | NodeJS.ArrayBufferView> | internal.Stream);
    public async insert(filePath: string, data?: string | NodeJS.ArrayBufferView | Iterable<string | NodeJS.ArrayBufferView> | AsyncIterable<string | NodeJS.ArrayBufferView> | internal.Stream, isFolder?: boolean);
    public async insert(fileInfo: string | PathDetails, data: string | NodeJS.ArrayBufferView | Iterable<string | NodeJS.ArrayBufferView> | AsyncIterable<string | NodeJS.ArrayBufferView> 
        | internal.Stream = '', isFolder: boolean = false) 
    {
        let pathDetails: PathDetails;
        if (typeof fileInfo == 'string') {
            const relativePath = fileInfo;
            const rootFolderPath = this.folderPath ? this.folderPath : '.';
            const fullPath = path.resolve(rootFolderPath + '/' + relativePath);
            pathDetails = {
                isFolder,
                name: extractFileName(relativePath),
                relativePath,
                fullPath,
                parentFolderRelativePath: extractParentFolderPath(relativePath),
                parentFolderFullPath: extractParentFolderPath(fullPath)
            };
        }
        else {
            pathDetails = fileInfo;
        }
        if (!data && pathDetails.content) data = pathDetails.content;

        super.insert(pathDetails);

        if (pathDetails.isFolder) {
            if (!fs.existsSync(pathDetails.fullPath)) {
                fs.mkdirSync(pathDetails.fullPath, { recursive: true });
            }
        }
        else {
            if (!fs.existsSync(pathDetails.parentFolderFullPath)) {
                fs.mkdirSync(pathDetails.parentFolderFullPath, { recursive: true });
            }
            await fs.promises.writeFile(pathDetails.fullPath, data);
        }
    }

    public async delete(mask: string = '*', options: ReadOptions = {}) {
        super.delete();

        if (options.includeRootDir && (options.objectsToSearch == 'all' || options.objectsToSearch == 'foldersOnly' || !options.objectsToSearch)) {
            fs.rmSync(this.folderPath, { recursive: true, force: true });
            return;
        }

        const matches = glob.sync(mask, {cwd: this.folderPath});
        for (let i = 0; i < matches.length; i++) {
            const matchPath = path.join(this.folderPath, matches[i]);
            const isFolder = (await fs.promises.lstat(matchPath)).isDirectory();

            if ( isFolder && (options.objectsToSearch == 'all' || options.objectsToSearch == 'foldersOnly' || !options.objectsToSearch) ) {
                fs.rmdirSync(matchPath, { recursive: true });
            }

            if ( !isFolder && (options.objectsToSearch == 'filesOnly' || options.objectsToSearch == 'all' || !options.objectsToSearch) ) {
                fs.rmSync(matchPath, { force: true });
            }
            
        };
    }

    protected isRootCur(): boolean {
        const r = path.join(this.folderPath.trim()).trim();
        return this.folderPath == '.' || this.folderPath == '';
    }

    protected getRootFolderDetails(): PathDetails {
        const fullPath = path.resolve(this.folderPath);
        return {
            isFolder: true,
            name: extractFileName(fullPath),
            relativePath: '',
            fullPath,
            parentFolderRelativePath: '..',
            parentFolderFullPath: path.resolve(this.folderPath + '/..')
        }
    }

    protected async getPathDetails(relativePath: string, options?: ReadOptions) {
        const rootFolderPath = this.folderPath ? this.folderPath : '.';
        const fullPath = path.resolve(rootFolderPath + '/' + relativePath);

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

    protected removeLeadingRoot(path: string): string {
        if (path.startsWith(this.folderPath)) path = path.substring(this.folderPath.length);
        if (path.startsWith('./')) path = path.substring(2);
        if (path.startsWith('/')) path = path.substring(1);
        return path;
    }
    
}


