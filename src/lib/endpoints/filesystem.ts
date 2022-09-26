import * as fs from "fs";
import { glob } from "glob";
import path = require("path");
import { Observable, Subscriber } from 'rxjs';
import internal = require("stream");
import { Endpoint, EndpointImpl } from "../core/endpoint";
import { EtlObservable } from "../core/observable";


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

export class FilesystemEndpoint extends EndpointImpl<PathDetails> {
    protected rootFolderPath: string;

    constructor(rootFolderPath: string) {
        super();
        this.rootFolderPath = rootFolderPath.trim();
        if (this.rootFolderPath.endsWith('/')) this.rootFolderPath.substring(0, this.rootFolderPath.lastIndexOf('/'));
    }

    // Uses simple path syntax from lodash.get function
    // path example: 'store.book[5].author'
    // use path '' for the root object
    public read(mask: string = '*', options: ReadOptions = {}): EtlObservable<PathDetails> {
        const observable = new EtlObservable<any>((subscriber) => {
            try {
                this.sendStartEvent();

                if (options.includeRootDir && (options.objectsToSearch == 'all' || options.objectsToSearch == 'foldersOnly' || !options.objectsToSearch)) {
                    let res = this.getRootFolderDetails();
                    this.sendDataEvent(res);
                    subscriber.next(res);
                }
        
                glob(mask, {cwd: this.rootFolderPath}, (err: Error, matches: string[]) => {
                    if (err) {
                        this.sendErrorEvent(err);
                        subscriber.error(err);
                        return;
                    }
        
                    for (let i = 0; i < matches.length; i++) {
                        this.getPathDetails(matches[i], options).then(res => {
                            if ( (res.isFolder && (options.objectsToSearch == 'all' || options.objectsToSearch == 'foldersOnly' || !options.objectsToSearch))
                                || (!res.isFolder && (options.objectsToSearch == 'filesOnly' || options.objectsToSearch == 'all' || !options.objectsToSearch)) )
                            {
                                this.sendDataEvent(res);
                                subscriber.next(res);

                                if (i == matches.length - 1) {
                                    subscriber.complete();
                                    this.sendEndEvent();
                                }
                            }
                        })
                    };
                });
            }
            catch(err) {
                this.sendErrorEvent(err);
                subscriber.error(err);
            }
        });
        return observable;
    }

    public async push(pathDetails: PathDetails, data?: string | NodeJS.ArrayBufferView | Iterable<string | NodeJS.ArrayBufferView> | AsyncIterable<string | NodeJS.ArrayBufferView> | internal.Stream);
    public async push(filePath: string, data?: string | NodeJS.ArrayBufferView | Iterable<string | NodeJS.ArrayBufferView> | AsyncIterable<string | NodeJS.ArrayBufferView> | internal.Stream, isFolder?: boolean);
    public async push(fileInfo: string | PathDetails, data: string | NodeJS.ArrayBufferView | Iterable<string | NodeJS.ArrayBufferView> | AsyncIterable<string | NodeJS.ArrayBufferView> 
        | internal.Stream = '', isFolder: boolean = false) 
    {
        let pathDetails: PathDetails;
        if (typeof fileInfo == 'string') {
            const relativePath = fileInfo;
            const rootFolderPath = this.rootFolderPath ? this.rootFolderPath : '.';
            const fullPath = path.resolve(rootFolderPath + '/' + relativePath);
            pathDetails = {
                isFolder,
                name: this.extractFileName(relativePath),
                relativePath,
                fullPath,
                parentFolderRelativePath: this.extractParentFolderPath(relativePath),
                parentFolderFullPath: this.extractParentFolderPath(fullPath)
            };
        }
        else {
            pathDetails = fileInfo;
        }
        if (!data && pathDetails.content) data = pathDetails.content;

        super.push(pathDetails);

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

    public async clear(mask: string = '*', options: ReadOptions = {}) {
        super.clear();

        if (options.includeRootDir && (options.objectsToSearch == 'all' || options.objectsToSearch == 'foldersOnly' || !options.objectsToSearch)) {
            fs.rmSync(this.rootFolderPath, { recursive: true, force: true });
            return;
        }

        const matches = glob.sync(mask, {cwd: this.rootFolderPath});
        for (let i = 0; i < matches.length; i++) {
            const matchPath = path.join(this.rootFolderPath, matches[i]);
            const isFolder = (await fs.promises.lstat(matchPath)).isDirectory();

            if ( isFolder && (options.objectsToSearch == 'all' || options.objectsToSearch == 'foldersOnly' || !options.objectsToSearch) ) {
                fs.rmdirSync(matchPath, { recursive: true });
            }

            if ( !isFolder && (options.objectsToSearch == 'filesOnly' || options.objectsToSearch == 'all' || !options.objectsToSearch) ) {
                fs.rmSync(matchPath, { force: true });
            }
            
        };
    }

    protected extractFileName(path: string): string {
        return path.substring(path.lastIndexOf('/') + 1);
    }

    protected extractParentFolderPath(path: string): string {
        return path.substring(0, path.lastIndexOf('/'));
    }

    protected removeLeadingRoot(path: string): string {
        if (path.startsWith(this.rootFolderPath)) path = path.substring(this.rootFolderPath.length);
        if (path.startsWith('./')) path = path.substring(2);
        if (path.startsWith('/')) path = path.substring(1);
        return path;
    }

    protected isRootCur(): boolean {
        const r = path.join(this.rootFolderPath.trim()).trim();
        return this.rootFolderPath == '.' || this.rootFolderPath == '';
    }

    protected getRootFolderDetails(): PathDetails {
        const fullPath = path.resolve(this.rootFolderPath);
        return {
            isFolder: true,
            name: this.extractFileName(fullPath),
            relativePath: '',
            fullPath,
            parentFolderRelativePath: '..',
            parentFolderFullPath: path.resolve(this.rootFolderPath + '/..')
        }
    }

    protected async getPathDetails(relativePath: string, options?: ReadOptions) {
        const rootFolderPath = this.rootFolderPath ? this.rootFolderPath : '.';
        const fullPath = path.resolve(rootFolderPath + '/' + relativePath);

        const res = {
            isFolder: (await fs.promises.lstat(fullPath)).isDirectory(),
            name: this.extractFileName(relativePath),
            relativePath,
            fullPath,
            parentFolderRelativePath: this.extractParentFolderPath(relativePath),
            parentFolderFullPath: this.extractParentFolderPath(fullPath)
        } as PathDetails;
        if (options && options.withContent) {
            res.content = fs.readFileSync(fullPath);
        }
        return res;
    }
    
}


