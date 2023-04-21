import { tap, mergeMap, from, Observable, iif, of, EMPTY, OperatorFunction } from "rxjs";
import { BaseCollection } from "../core/collection.js";
import { GuiManager } from "../index.js";

type FindErrorFunction<T = any> = (data: T) => boolean | null | EtlError<T>;

type EtlError<T = any> = {
    name: string;
    message: string;
    data: T;
    condition: FindErrorFunction<T> | Partial<T>;
}

// assertion
export function expect<T>(name: string, where: Partial<T>, errorsCollection?: BaseCollection<EtlError<T>>): OperatorFunction<T, T> 
export function expect<T>(name: string, findErrorFunction: FindErrorFunction<T>, errorsCollection?: BaseCollection<EtlError<T>>): OperatorFunction<T, T> 
export function expect<T>(name: string, condition: FindErrorFunction<T> | Partial<T>, errorsCollection: BaseCollection<EtlError<T>> = null): OperatorFunction<T, T> {
    const checkFunc = (data: T): boolean => {
        let error = (condition as FindErrorFunction<T>)(data);
        if (!error) return false;

        if (typeof error == "boolean") error = { name, message: 'Error: unexpected value', data, condition } as EtlError<T>;
        // Insert error without waiting
        if (errorsCollection) errorsCollection.insert(error);
        return true;
    }
    const checkObj = (data: T): boolean => {
        for (let key in condition) {
            if (!condition.hasOwnProperty(key)) continue;
            if (condition[key] != data[key]) {
                const error = { name, message: `Error: unexpected value. The field '${key}' is equals to '${data[key]}' but expected to be '${condition[key]}'`, data, condition } as EtlError<T>;
                // Insert error without waiting
                if (errorsCollection) errorsCollection.insert(error);
                return false;
            }
        }
        return true;
    }

    if (typeof condition === 'function') return mergeMap((data: T)=> iif(() => checkFunc(data), of(data), EMPTY)); 
    return mergeMap((data: T)=> iif(() => checkObj(data), of(data), EMPTY)); 
}
