import { mergeMap, OperatorFunction, from } from "rxjs";

export function sleep<T>(ms: number): OperatorFunction<T, T> {
    const f = async (v: T) => {
        await new Promise((r) => setTimeout(r, ms));
        return v;
    }
    const observable = (v: T) => from(f(v));
    return mergeMap((v: T)=> observable(v)); 
}
