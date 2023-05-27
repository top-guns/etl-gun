import * as fs from "fs";
import { tmpdir } from "os";
import path from "path";

export function deleteFileIfExists(path: string) {
    if (!fs.existsSync(path)) return;
    const isFolder = fs.lstatSync(path).isDirectory();
    fs.rmSync(path, { recursive: true, force: true });
};

export function loadFileContent(path: string) {
    if (fs.existsSync(path)) {
        const res = fs.readFileSync(path);
        return res.toString();
    }
    return '';
};

export function getTempFolder(folderName: string = '') {
    return path.join(tmpdir(), folderName);
};

export function getTempPath(filename: string) {
    return path.join(tmpdir(), filename);
};
