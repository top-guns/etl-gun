import * as rx from "rxjs";

export function waiting<T>(ms?: number): rx.OperatorFunction<T, T>;
export function waiting<T>(getDelay: (v: T) => number): rx.OperatorFunction<T, T>;
export function waiting<T>(delay?: number | ((v: T) => number)): rx.OperatorFunction<T, T> {
    return rx.concatMap(val => rx.of(val).pipe(
        rx.delay( !delay ? 0 : (typeof delay === 'number' ? delay : delay(val)) )
    ))
}
