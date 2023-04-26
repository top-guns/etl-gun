import { mergeMap, from } from "rxjs";
import { BaseCollection } from "../core/collection.js";
import { GuiManager } from "../index.js";


export function push<T>(collection: BaseCollection<T>, value?: T);
export function push<T, S = T>(collection: BaseCollection<T>, callback?: (value: S) => T | Promise<T>);
export function push<T, S = T>(collection: BaseCollection<T>, paramValue?: T | ((value: S) => T | Promise<T>)) {
    //return tap<T>(v => collection.insert(v, params));
    const f = async (v: S) => {
        // No wait
        let vv: T = await getValue<T, S>(v, paramValue);
        collection.insert(vv);
        return v;
    }
    const observable = (v: S) => from(f(v));
    return mergeMap((v: S)=> observable(v)); 
}

export function pushAndLog<T>(collection: BaseCollection<T>, value?: T);
export function pushAndLog<T, S = T>(collection: BaseCollection<T>, callback?: (value: S) => T | Promise<T>);
export function pushAndLog<T, S = T>(collection: BaseCollection<T>, paramValue?: T | ((value: S) => T | Promise<T>)) {
    const f = async (v: S) => {
        let vv: T = await getValue<T, S>(v, paramValue);
        const res = await collection.insert(vv);
        GuiManager.log(res);
        return v;
    }
    const observable = (v: S) => from(f(v));
    return mergeMap((v: S)=> observable(v)); 
}


export function pushAndGet<T>(collection: BaseCollection<T>, value?: T);
export function pushAndGet<T, S = T, R = T>(collection: BaseCollection<T>, callback?: (value: S) => T | Promise<T>);
export function pushAndGet<T, S = T, R = T>(collection: BaseCollection<T>, paramValue?: T | ((value: S) => T | Promise<T>)) {
    const f = async (v: S): Promise<R> => {
        let vv: T = await getValue<T, S>(v, paramValue);
        const res: R = await collection.insert(vv);
        return res;
    }
    const observable = (v: S) => from(f(v));
    return mergeMap((v: S)=> observable(v)); 
}

export function pushAndWait<T>(collection: BaseCollection<T>, value?: T);
export function pushAndWait<T, S = T>(collection: BaseCollection<T>, callback?: (value: S) => T | Promise<T>);
export function pushAndWait<T, S = T>(collection: BaseCollection<T>, paramValue?: T | ((value: S) => T | Promise<T>)) {
    const f = async (v: S) => {
        let vv: T = await getValue<T, S>(v, paramValue);
        await collection.insert(vv);
        return v;
    }
    const observable = (v: S) => from(f(v));
    return mergeMap((v: S)=> observable(v)); 
}


async function getValue<T, S = T>(streamValue: S, paramValue?: T | ((value: S) => T | Promise<T>)): Promise<T> {
    if (typeof paramValue == 'undefined') return streamValue as unknown as T;
    if (typeof paramValue == 'function') return await (paramValue as ((value: S) => T | Promise<T>))(streamValue);
    return paramValue;
}