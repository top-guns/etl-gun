import { tap } from "rxjs";
import { GuiManager } from "../core";

export function log<T>(before: string = '', valuefn?: ((value) => string) | null, outStream: NodeJS.WritableStream = null) {
    const outConsole = outStream ? new console.Console(outStream) : console;
    return tap<T>(v => 
        GuiManager.instance 
        ? 
        GuiManager.instance.log(before + dumpObject(valuefn ? valuefn(v) : v)) 
        : 
        outConsole.log(before, valuefn ? valuefn(v) : v)); 
}

function dumpObject(obj: any, deep: number = 1): string {
    switch (typeof obj) {
        case 'number': return '' + obj;
        case 'string': return `"${obj}"`;
        case 'boolean': return '' + obj;
        case 'function': return '()';
        case 'object': {
            if (deep <= 0) return 'object';

            let res = '';
            for (let key in obj) {
                if (obj.hasOwnProperty(key)) {
                    if (res) res += `, `;
                    res += `${key}: ${dumpObject(obj[key], deep - 1)}`;
                }
            }
            return `{${res}}`;
        }
        default: return '' + obj;
    }
}