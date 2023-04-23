import { defer, Observable } from "rxjs";

export function onSubscribe<T>(handler: () => void): (source: Observable<T>) =>  Observable<T> {
    return function inner(source: Observable<T>): Observable<T> {
        return defer(() => {
            handler();
          return source;
        });
    };
}