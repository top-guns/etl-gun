import { mergeMap, from, OperatorFunction } from "rxjs";
import _ from 'lodash';
import { UpdatableCollection } from "../core/updatable_collection.js";


type RemoveOptions<S, T> = {
    property?: string;
    value?: T;
    valueFn?: (value: S) => (T | Promise<T>);
    skipNotFoundErrors?: boolean;
}

/**
 * Performs collection.insert and go to the next pipe step without waiting for result.
 * @param collection  Destination collection.
 * @param options  Some options to specify the value to insert.
 */
export function remove<S, T=S>(collection: UpdatableCollection<T>, options?: RemoveOptions<S, T> | null): OperatorFunction<S, S> {
    //return tap<T>(v => collection.insert(v, params));
    const f = async (v: S) => {
        // No wait
        let vv: T = await getValue<S, T>(v, options);
        collection.delete(vv).catch(reason => {
            if (!options.skipNotFoundErrors) throw reason;
        });
        return v;
    }
    const observable = (v: S) => from(f(v));
    return mergeMap((v: S)=> observable(v)); 
}

/**
 * Performs collection.insert, wait for result, but skip the result value and go to the next pipe step.
 * @param collection  Destination collection.
 * @param options  Some options to specify the value to insert.
 */
export function removeAndWait<S, T=S>(collection: UpdatableCollection<T>, options?: RemoveOptions<S, T> | null): OperatorFunction<S, S> {
    const f = async (v: S) => {
        let vv: T = await getValue<S, T>(v, options);
        try {
            await collection.delete(vv);
        }
        catch (e) {
            if (!options.skipNotFoundErrors) throw e;
        }
        return v;
    }
    const observable = (v: S) => from(f(v));
    return mergeMap((v: S)=> observable(v)); 
}


async function getValue<S, T>(streamValue: S, options?: RemoveOptions<S, T>| null): Promise<T> {
    if (!options) return streamValue as unknown as T;
    if (options.property) return _.get(streamValue, options.property) as unknown as T;
    if (options.valueFn) return await options.valueFn(streamValue);
    return options.value;
}
