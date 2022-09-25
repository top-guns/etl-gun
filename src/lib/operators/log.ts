import { tap } from "rxjs";

export function log<T>(before: string = '', after: string = '', outStream: NodeJS.WritableStream = null) {
    const outConsole = outStream ? new console.Console(outStream) : console;
    return tap<T>(v => outConsole.log(before, v, after)); 
}