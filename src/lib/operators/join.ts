import { concatMap, map, Observable, OperatorFunction } from "rxjs";

export function joinObjects<T extends Record<string, any>, J extends Record<string, any>>(joinable: Observable<J>): OperatorFunction<T, T & J> {
    let buf: any;

    return (source: Observable<T>): Observable<T & J> => {
        let res = source.pipe(
            concatMap(v => {buf = v; return joinable}),
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
            concatMap(v => {buf = v; return joinable}),
            map(v => {
                return [...buf, ...v];
            })
        );
        return res;
    }
}

export function join<R = any>(joinable: Observable<any>): OperatorFunction<any, R>;
export function join<R = any>(joinable: Observable<any>, fieldName: string): OperatorFunction<any, R>;
export function join<R = any>(joinable: Observable<any>, fieldName: string = ''): OperatorFunction<any, R> {
    return (source: Observable<any>): Observable<any> => {
        let buf: any;
        return source.pipe(
            concatMap(v => {buf = v; return joinable}),
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
                    if (fieldName) {
                        let res = Object.assign({}, buf);
                        res[fieldName] = v;
                        return res;
                    }
                    return [...objectToArray(buf), v];
                }
                if (typeof v == "object") {
                    if (fieldName) {
                        let res = Object.assign({}, v);
                        res[fieldName] = buf;
                        return res;
                    }
                    return [buf, ...objectToArray(v)];
                }
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