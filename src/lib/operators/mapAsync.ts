import { mergeMap, OperatorFunction, from } from "rxjs";

export function mapAsync<T, R = T>(asyncFn: (value: T) => Promise<R>): OperatorFunction<T, R> {
    const observable = (v: T) => from(asyncFn(v));
    return mergeMap((v: T)=> observable(v)); 
}