import { tap } from "rxjs";
import { GuiManager } from "../core";

export function log<T>(before: string = '', after: string = '', outStream: NodeJS.WritableStream = null) {
    const outConsole = outStream ? new console.Console(outStream) : console;
    return tap<T>(v => GuiManager.instance ? GuiManager.instance.log(before + v + after) : outConsole.log(before, v, after)); 
}