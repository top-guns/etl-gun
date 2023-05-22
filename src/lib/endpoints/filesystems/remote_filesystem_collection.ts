import { Readable } from "stream";
import { AlreadyExistsAction, FilesystemCollection } from "./filesystem_collection.js";


export abstract class RemoteFilesystemCollection<T> extends FilesystemCollection<T> { 
    public abstract insert(remotePath: string, fileContents?: string | Readable): Promise<void>;
    public abstract update(remoteFilePath: string, fileContents: string | Readable): Promise<void>;
    public abstract upsert(remoteFilePath: string, fileContents: string | Readable): Promise<boolean>;
    public abstract delete(remotePath: string): Promise<boolean>;

    public abstract append(remoteFilePath: string, fileContents: string | Readable): Promise<void>;
    public abstract clear(remotePath: string): Promise<void>;
    public abstract copy(remoteSrcPath: string, remoteDstPath: string): Promise<void>;
    public abstract move(remoteSrcPath: string, remoteDstPath: string): Promise<void>;

    public abstract isExists(remotePath: string): Promise<boolean>;
    public abstract isFolder(remotePath: string): Promise<boolean>;
    public abstract getInfo(remotePath: string): Promise<any>;

    public abstract download(remotePath: string, localPath: string): Promise<void>;
    public abstract upload(localPath: string, remotePath: string): Promise<void>;


    protected sendDownloadEvent(remotePath: string, localPath: string) {
        super.sendGetEvent(remotePath, localPath, 'download', { localPath, remotePath });
    }

    protected sendUploadEvent(localPath: string, remotePath: string) {
        super.sendUpsertEvent(remotePath, localPath, 'upload', { localPath, remotePath });
    }
}
  