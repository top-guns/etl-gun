import { FilesystemCollection } from "./filesystem_collection.js";


export abstract class RemoteFilesystemCollection extends FilesystemCollection { 
    public abstract copy(remoteSrcPath: string, remoteDestPath: string): Promise<void>; 
    public abstract move(remoteSrcPath: string, remoteDestPath: string): Promise<void>; 

    public abstract download(remotePath: string, localPath: string): Promise<void>;
    public abstract upload(localPath: string, remotePath: string): Promise<void>;


    public sendDownloadEvent(remotePath: string, localPath: string) {
        super.sendGetEvent(localPath, remotePath, 'download', { localPath, remotePath });
    }

    public sendUploadEvent(localPath: string, remotePath: string) {
        super.sendInsertEvent(remotePath, localPath, 'upload', { localPath, remotePath });
    }
}
  