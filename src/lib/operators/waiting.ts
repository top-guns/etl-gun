import * as rx from "rxjs";

export function waiting<T>(ms: number = 0): rx.OperatorFunction<T, T> {
    return rx.concatMap(val => rx.of(val).pipe(rx.delay(ms)))
}
