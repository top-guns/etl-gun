import { mergeMap, from, OperatorFunction, Observable } from "rxjs";
import _ from 'lodash';
import { GuiManager } from "../index.js";
import { UpdatableCollection } from "../core/updatable_collection.js";


type PushOptions<S, T> = {
    property?: string;
    value?: T;
    valueFn?: (value: S) => (T | Promise<T>);
}


export function push<S, T=S>(collection: UpdatableCollection<T>, options?: PushOptions<S, T> | null): OperatorFunction<S, S> {
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

export function pushAndLog<S, T=S>(collection: UpdatableCollection<T>, options?: PushOptions<S, T> | null): OperatorFunction<S, S> {
    const f = async (v: S) => {
        let vv: T = await getValue<S, T>(v, options);
        const res = await collection.insert(vv);
        GuiManager.log(res);
        return v;
    }
    const observable = (v: S) => from(f(v));
    return mergeMap((v: S)=> observable(v)); 
}


export function pushAndGet<S, T, R>(collection: UpdatableCollection<T>, options?: PushOptions<S, T> & {toProperty: string} | null): OperatorFunction<S, R> {
    const f = async (v: S): Promise<R> => {
        let vv: T = await getValue<S, T>(v, options);
        const res: R = await collection.insert(vv);
        return getOperatorResult(v, options.toProperty, res);
    }

    const observable: (value: S) => Observable<R> = (v: S) => from(f(v));
    return mergeMap<S, Observable<R>>((v: S)=> observable(v)); 
}

export function pushAndWait<S, T=S>(collection: UpdatableCollection<T>, options?: PushOptions<S, T> | null): OperatorFunction<S, S> {
    const f = async (v: S) => {
        let vv: T = await getValue<S, T>(v, options);
        await collection.insert(vv);
        return v;
    }
    const observable = (v: S) => from(f(v));
    return mergeMap((v: S)=> observable(v)); 
}


async function getValue<S, T>(streamValue: S, options?: PushOptions<S, T>| null): Promise<T> {
    if (!options) return streamValue as unknown as T;
    if (options.property) return _.get(streamValue, options.property) as unknown as T;
    if (options.valueFn) return await options.valueFn(streamValue);
    return options.value;
}

async function getOperatorResult<S, R>(val: S, toProperty: string, res: any): Promise<R> {
    if (typeof toProperty === 'undefined') return res;
    if (!toProperty) return val as unknown as R; 
    _.set(val, toProperty, res);
    return val as unknown as R;
}