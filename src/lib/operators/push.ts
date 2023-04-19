import { tap, mergeMap, from } from "rxjs";
import { BaseCollection } from "../core/collection.js";
import { GuiManager } from "../index.js";

export function push<T>(collection: BaseCollection<T>, ...params: any[]) {
    //return tap<T>(v => collection.insert(v, params));
    const f = async (v: T) => {
        const res = await collection.insert(v, params);
        return v;
    }
    const observable = (v: T) => from(f(v));
    return mergeMap((v: T)=> observable(v)); 
}

export function pushAndLog<T>(collection: BaseCollection<T>, ...params: any[]) {
    //return tap<T>(v => collection.insert(v, params));
    const f = async (v: T) => {
        const res = await collection.insert(v, params);
        GuiManager.log(res);
        return v;
    }
    const observable = (v: T) => from(f(v));
    return mergeMap((v: T)=> observable(v)); 
}

export function pushAndGet<T>(collection: BaseCollection<T>, ...params: any[]) {
    //return tap<T>(v => collection.insert(v, params));
    const f = async (v: T) => {
        const res = await collection.insert(v, params);
        return res;
    }
    const observable = (v: T) => from(f(v));
    return mergeMap((v: T)=> observable(v)); 
}