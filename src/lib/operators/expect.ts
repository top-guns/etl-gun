import { tap, mergeMap, from, Observable, iif, of, EMPTY, OperatorFunction } from "rxjs";
import { BaseCollection } from "../core/collection.js";
import { BaseObservable } from "../core/observable.js";
import { EtlError, EtlErrorData } from "../endpoints/errors.js";

type FindErrorFunction<T = any> = (data: T) => boolean | null | EtlErrorDataUnexpected<T>;

type EtlErrorDataUnexpected<T = any> = EtlErrorData & {
    condition: FindErrorFunction<T> | Partial<T>;
}

// assertion
export function expect<T>(name: string, where: Partial<T>, errorsCollection?: BaseCollection<EtlError> | null): OperatorFunction<T, T> 
export function expect<T>(name: string, findErrorFunction: FindErrorFunction<T>, errorsCollection?: BaseCollection<EtlError> | null): OperatorFunction<T, T> 
export function expect<T>(name: string, condition: FindErrorFunction<T> | Partial<T>, errorsCollection?: BaseCollection<EtlError> | null): OperatorFunction<T, T> {

    const checkFunc = (data: T): EtlErrorDataUnexpected<T> => {
        let error = (condition as FindErrorFunction<T>)(data);
        if (!error) return null;

        if (typeof error == "boolean") return { name, message: 'Error: unexpected value', data, condition } as EtlErrorDataUnexpected<T>;

        return error;
    }
    
    const checkObj = (data: T): EtlErrorDataUnexpected<T> => {
        for (let key in condition) {
            if (!condition.hasOwnProperty(key)) continue;

            if (condition[key] != data[key]) {
                return { name, message: `Error: unexpected value. The field '${key}' is equals to '${data[key]}' but expected to be '${condition[key]}'`, data, condition } as EtlErrorDataUnexpected<T>;
            }
        }
        return null;
    }


    return (function doObserve(observable: Observable<T>): Observable<T> {
        const pipeObservable: BaseObservable<T> = this;
        const collection = pipeObservable && (pipeObservable.collection as BaseCollection<T>);

        return new Observable<T>((subscriber) => {
            // this function will be called each time this Observable is subscribed to.
            const subscription = observable.subscribe({
                next: (value) => {
                    const err = (typeof condition === 'function') && checkFunc(value) || checkObj(value);
                    if (!err) {
                        subscriber.next(value);
                        return;
                    }

                    if (typeof errorsCollection === 'undefined' && collection) errorsCollection = collection.errors;

                    // Insert error without waiting
                    if (errorsCollection) errorsCollection.insert(err);
                    if (collection) collection.sendErrorEvent(err);
                    //subscriber.error(err);
                },
                error: (err) => {
                    subscriber.error(err);
                },
                complete: () => {
                    subscriber.complete();
                },
            });
        
            // Return the finalization logic. This will be invoked when
            // the result errors, completes, or is unsubscribed.
            return () => {
                subscription.unsubscribe();
            };
        })
    }) as OperatorFunction<T, T>;
}
