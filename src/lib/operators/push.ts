import { mergeMap, from } from "rxjs";
import { BaseCollection } from "../core/collection.js";
import { GuiManager } from "../index.js";


export function push<T>(collection: BaseCollection<T>, value: T);
export function push<T>(collection: BaseCollection<T>, callback: (value: T) => Promise<any>);
export function push<T>(collection: BaseCollection<T>, value: any) {
    //return tap<T>(v => collection.insert(v, params));
    const f = async (v: T) => {
        // No wait
        let vv = await getValue(v, value);
        collection.insert(vv);
        return v;
    }
    const observable = (v: T) => from(f(v));
    return mergeMap((v: T)=> observable(v)); 
}

export function pushAndLog<T>(collection: BaseCollection<T>, value: T);
export function pushAndLog<T>(collection: BaseCollection<T>, callback: (value: T) => Promise<any>);
export function pushAndLog<T>(collection: BaseCollection<T>, value: any) {
    const f = async (v: T) => {
        let vv = await getValue(v, value);
        const res = await collection.insert(vv);
        GuiManager.log(res);
        return v;
    }
    const observable = (v: T) => from(f(v));
    return mergeMap((v: T)=> observable(v)); 
}


export function pushAndGet<T>(collection: BaseCollection<T>, value: T);
export function pushAndGet<T>(collection: BaseCollection<T>, callback: (value: T) => Promise<any>);
export function pushAndGet<T>(collection: BaseCollection<T>, value: any) {
    const f = async (v: T) => {
        let vv = await getValue(v, value);
        const res = await collection.insert(vv);
        return res;
    }
    const observable = (v: T) => from(f(v));
    return mergeMap((v: T)=> observable(v)); 
}

export function pushAndWait<T>(collection: BaseCollection<T>, value: T);
export function pushAndWait<T>(collection: BaseCollection<T>, callback: (value: T) => Promise<any>);
export function pushAndWait<T>(collection: BaseCollection<T>, value: any) {
    const f = async (v: T) => {
        let vv = await getValue(v, value);
        const res = await collection.insert(vv);
        return v;
    }
    const observable = (v: T) => from(f(v));
    return mergeMap((v: T)=> observable(v)); 
}

async function getValue<T = any>(streamValue: T, paramValue?: T | ((value: T) => Promise<any>) | any): Promise<any> {
    if (typeof paramValue == 'undefined') return streamValue;
    if (typeof paramValue == 'function') paramValue = await (paramValue as (value: T) => Promise<any>)(streamValue);
    return paramValue;
}