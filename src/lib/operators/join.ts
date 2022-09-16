import { map, mergeMap, Observable, OperatorFunction, tap } from "rxjs";

export function joinObjects<T extends Record<string, any>, J extends Record<string, any>>(joinable: Observable<J>): OperatorFunction<T, T & J> {
    let buf: any;

    return (source: Observable<T>): Observable<T & J> => {
        let res = source.pipe(
            tap(v => buf = v),
            mergeMap(v => joinable),
            map(v => {
                return Object.assign({}, buf, v);
            })
        );
        return res;
    }
}

export function joinArrays<T extends any[], J extends any[]>(joinable: Observable<J>): OperatorFunction<T, any[]> {
    let buf: any;

    return (source: Observable<T>): Observable<any[]> => {
        let res = source.pipe(
            tap(v => buf = v),
            mergeMap(v => joinable),
            map(v => {
                return [...buf, ...v];
            })
        );
        return res;
    }
}

// export function join<T extends Record<string, any>>(joinable: Observable<Record<string, any>>): OperatorFunction<T, Record<string, any>>;
// export function join<T extends any[]>(joinable: Observable<any[]>): OperatorFunction<T, any[]>;
export function join<R = any>(joinable: Observable<any>): OperatorFunction<any, R> {
    let buf: any;
    return (source: Observable<any>): Observable<any> => {
        return source.pipe(
            tap(v => buf = v),
            mergeMap(v => joinable),
            map(v => {
                if (Array.isArray(buf)) {
                    if (Array.isArray(v)) return [...buf, ...v];
                    if (typeof v == "object") return [...buf, ...objectToArray(v)];
                    return [...buf, v];
                }
                if (Array.isArray(v)) {
                    if (typeof buf == "object") return [...objectToArray(buf), ...v];
                    return [buf, ...v];
                }
                if (typeof buf == "object") {
                    if (typeof v == "object") return Object.assign({}, buf, v);
                    return [...objectToArray(buf), v];
                }
                if (typeof v == "object") return [buf, ...objectToArray(v)];
                return [buf, v];
            })
        );
    };
}

function objectToArray(obj: Record<string, any>) {
    let arr = [];  
    for(let key in obj){  
        if(obj.hasOwnProperty(key)){  
            arr.push(obj[key]);  
        }  
    }  
    return arr;
}