import * as fs from "fs";
import { tmpdir } from "os";
import path from "path";

export function deleteFileIfExists(path: string) {
    if (fs.existsSync(path)) fs.rmSync(path);
};

export function loadFileContent(path: string) {
    if (fs.existsSync(path)) {
        const res = fs.readFileSync(path);
        return res.toString();
    }
    return '';
};

export function getTempFolder() {
    return tmpdir();
};

export function getTempPath(filename: string) {
    return path.join(tmpdir(), filename);
};
