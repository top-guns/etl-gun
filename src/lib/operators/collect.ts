import * as rx from "rxjs";

export function collect<T>(maxLength: number): rx.OperatorFunction<T, T[]>;
export function collect<T>(startNewBuf: (value: T, buffer: T[]) => boolean): rx.OperatorFunction<T, T[]>;
export function collect<T>(condition: number | ((value: T, buffer: T[]) => boolean)): rx.OperatorFunction<T, T[]> { 
    let buf: T[] = [];

    const checkStartNewBuf = (v: T) => {
        if (typeof condition === 'number') return buf.length >= condition;
        return condition(v, buf);
    }

    const observer = ((observable: rx.Observable<T>): rx.Observable<T[]> => {
        return new rx.Observable<T[]>((subscriber) => {
            const subscription = observable.subscribe({
                next: (v) => {
                    const isNewBuf = checkStartNewBuf(v);
                    if (buf.length && isNewBuf) {
                        const arr = buf;
                        buf = [v];
                        subscriber.next(arr);
                    }
                    else {
                        buf.push(v);
                    }
                },
                error: (e) => {
                    subscriber.error(e);
                },
                complete: () => {
                    subscriber.next(buf);
                    subscriber.complete();
                },
            })
        
            // Return the finalization logic. This will be invoked when
            // the result errors, completes, or is unsubscribed.
            return () => {
                subscription.unsubscribe();
            }
        })
    });

    return observer;
}
