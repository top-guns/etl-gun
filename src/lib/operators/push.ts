import { mergeMap, from, OperatorFunction, Observable } from "rxjs";
import _ from 'lodash';
import { BaseCollection } from "../core/collection.js";
import { GuiManager } from "../index.js";


type PushOptions<S, T> = {
    fromProperty: string;
    value: S | ((value: S) => (T | Promise<T>));
}


export function push<S, T=S>(collection: BaseCollection<T>, options?: PushOptions<S, T>): OperatorFunction<S, S> {
    //return tap<T>(v => collection.insert(v, params));
    const f = async (v: S) => {
        // No wait
        let vv: T = await getValue<S, T>(v, options);
        collection.insert(vv);
        return v;
    }
    const observable = (v: S) => from(f(v));
    return mergeMap((v: S)=> observable(v)); 
}

export function pushAndLog<S, T=S>(collection: BaseCollection<T>, options?: PushOptions<S, T>): OperatorFunction<S, S> {
    const f = async (v: S) => {
        let vv: T = await getValue<S, T>(v, options);
        const res = await collection.insert(vv);
        GuiManager.log(res);
        return v;
    }
    const observable = (v: S) => from(f(v));
    return mergeMap((v: S)=> observable(v)); 
}


export function pushAndGet<S, T, R>(collection: BaseCollection<T>, options?: PushOptions<S, T> & {toProperty: string}): OperatorFunction<S, R> {
    const f = async (v: S): Promise<R> => {
        let vv: T = await getValue<S, T>(v, options);
        const res: R = await collection.insert(vv);
        return getOperatorResult(v, options.toProperty, res);
    }

    const observable: (value: S) => Observable<R> = (v: S) => from(f(v));
    return mergeMap<S, Observable<R>>((v: S)=> observable(v)); 
}

export function pushAndWait<S, T=S>(collection: BaseCollection<T>, options?: PushOptions<S, T>): OperatorFunction<S, S> {
    const f = async (v: S) => {
        let vv: T = await getValue<S, T>(v, options);
        await collection.insert(vv);
        return v;
    }
    const observable = (v: S) => from(f(v));
    return mergeMap((v: S)=> observable(v)); 
}


async function getValue<S, T>(streamValue: S, options?: PushOptions<S, T>): Promise<T> {
    if (typeof options == 'undefined') return streamValue as unknown as T;
    if (typeof options.fromProperty) return _.get(streamValue, options.fromProperty) as unknown as T;
    if (typeof options.value === 'function') return await (options.value as ((value: S) => (T | Promise<T>)))(streamValue);
    return options.value as unknown as T;
}

async function getOperatorResult<S, R>(val: S, toProperty: string, res: any): Promise<R> {
    if (typeof toProperty === 'undefined') return res;
    if (!toProperty) return val as unknown as R; 
    _.set(val, toProperty, res);
    return val as unknown as R;
}