import { Observable, OperatorFunction } from "rxjs";
import { BaseCollection_I } from "../core/base_collection_i.js";
import { BaseObservable } from "../core/observable.js";
import { ErrorsQueue, EtlError, EtlErrorData } from "../endpoints/errors.js";
import { Condition, findDifference } from "../index.js";

type EtlErrorDataUnexpected<T> = EtlErrorData & {
    condition: Condition<T>;
}

// assertion
export function expect<T>(name: string, condition: Condition<T>, errorsCollection?: ErrorsQueue | null): OperatorFunction<T, T> { 
    const getError = (data: T): EtlErrorDataUnexpected<T> => {
        let diff = findDifference<T>(data, condition);
        if (diff) return { name, message: `Unexpected value: ${diff}`, data, condition } as EtlErrorDataUnexpected<T>;
        return null;
    }


    return (function doObserve(observable: Observable<T>): Observable<T> {
        const pipeObservable: BaseObservable<T> = this;
        const collection = pipeObservable && (pipeObservable.collection as BaseCollection_I<T>);

        return new Observable<T>((subscriber) => {
            // this function will be called each time this Observable is subscribed to.
            const subscription = observable.subscribe({
                next: (value) => {
                    const err = getError(value);
                    if (!err) {
                        subscriber.next(value);
                        return;
                    }

                    if (!errorsCollection && collection) errorsCollection = collection.errors;

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
