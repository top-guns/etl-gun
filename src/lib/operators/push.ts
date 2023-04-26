import { mergeMap, from, OperatorFunction, Observable } from "rxjs";
import { BaseCollection } from "../core/collection.js";
import { GuiManager } from "../index.js";


export function push<S>(collection: BaseCollection<S>): OperatorFunction<S, S>;
export function push<S>(collection: BaseCollection<S>, value: S): OperatorFunction<S, S>;
export function push<S, T = S>(collection: BaseCollection<T>, callback: (value: S) => (T | Promise<T>)): OperatorFunction<S, S>;
export function push<S, T = S>(collection: BaseCollection<T>, paramValue?: T | ((value: S) => (T | Promise<T>))): OperatorFunction<S, S> {
    //return tap<T>(v => collection.insert(v, params));
    const f = async (v: S) => {
        // No wait
        let vv: T = await getValue<S, T>(v, paramValue);
        collection.insert(vv);
        return v;
    }
    const observable = (v: S) => from(f(v));
    return mergeMap((v: S)=> observable(v)); 
}

export function pushAndLog<S>(collection: BaseCollection<S>): OperatorFunction<S, S>;
export function pushAndLog<S>(collection: BaseCollection<S>, value: S): OperatorFunction<S, S>;
export function pushAndLog<S, T = S>(collection: BaseCollection<T>, callback: (value: S) => (T | Promise<T>)): OperatorFunction<S, S>;
export function pushAndLog<S, T = S>(collection: BaseCollection<T>, paramValue?: T | ((value: S) => (T | Promise<T>))): OperatorFunction<S, S> {
    const f = async (v: S) => {
        let vv: T = await getValue<S, T>(v, paramValue);
        const res = await collection.insert(vv);
        GuiManager.log(res);
        return v;
    }
    const observable = (v: S) => from(f(v));
    return mergeMap((v: S)=> observable(v)); 
}


export function pushAndGet<S>(collection: BaseCollection<S>): OperatorFunction<S, S>;
export function pushAndGet<S>(collection: BaseCollection<S>, value: S): OperatorFunction<S, S>;
export function pushAndGet<S, T = S, R = S>(collection: BaseCollection<T>, callback: (value: S) => (T | Promise<T>)): OperatorFunction<S, R>;
export function pushAndGet<S, T = S, R = S>(collection: BaseCollection<T>, paramValue?: T | ((value: S) => (T | Promise<T>))): OperatorFunction<S, R> {
    const f = async (v: S): Promise<R> => {
        let vv: T = await getValue<S, T>(v, paramValue);
        const res: R = await collection.insert(vv);
        return res;
    }

    const observable: (value: S) => Observable<R> = (v: S) => from(f(v));

    return mergeMap<S, Observable<R>>((v: S)=> observable(v)); 
}

export function pushAndWait<S>(collection: BaseCollection<S>): OperatorFunction<S, S>;
export function pushAndWait<S>(collection: BaseCollection<S>, value: S): OperatorFunction<S, S>;
export function pushAndWait<S, T = S>(collection: BaseCollection<T>, callback?: (value: S) => (T | Promise<T>)): OperatorFunction<S, S>;
export function pushAndWait<S, T = S>(collection: BaseCollection<T>, paramValue?: T | ((value: S) => (T | Promise<T>))): OperatorFunction<S, S> {
    const f = async (v: S) => {
        let vv: T = await getValue<S, T>(v, paramValue);
        await collection.insert(vv);
        return v;
    }
    const observable = (v: S) => from(f(v));
    return mergeMap((v: S)=> observable(v)); 
}


async function getValue<S, T = S>(streamValue: S, paramValue?: T | ((value: S) => (T | Promise<T>))): Promise<T> {
    if (typeof paramValue == 'undefined') return streamValue as unknown as T;
    if (typeof paramValue == 'function') return await (paramValue as ((value: S) => (T | Promise<T>)))(streamValue);
    return paramValue;
}