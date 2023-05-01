import { mergeMap, ObservableInput, from } from "rxjs";
import _ from 'lodash';
import { GuiManager } from "../core/gui.js";

type LogOptions<S> = {
    property?: string;
    value?: any;
    valueFn?: (value: S) => (any | Promise<any>);
}

export function log<S>(before: string = '', options?: LogOptions<S> | null, outStream: NodeJS.WritableStream = null) {
    const outConsole = outStream ? new console.Console(outStream) : console;
    return mergeMap((v: S) => {
        const f = async () => {
            const val = await getValue<S>(v, options);
            GuiManager.instance
            ? 
            GuiManager.instance.log(await val, before) 
            : 
            outConsole.log(before, val)
            return v;
        }
        return from(f());
    })
}

async function getValue<S>(streamValue: S, options?: LogOptions<S> | null): Promise<any> {
    if (!options) return streamValue;
    if (options.property) return _.get(streamValue, options.property);
    if (options.valueFn) return await options.valueFn(streamValue);
    return options.value;
}