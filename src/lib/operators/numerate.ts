import { map, OperatorFunction } from "rxjs";

export function numerateArrays<T extends {push: any}>(fromIndex: number = 0): OperatorFunction<T, T> {
    let index = fromIndex - 1;

    return map<T, T>(value => {
        index++;
        value.push(index);
        return value;
    });
}

export function numerateObjects<T extends Record<string, any>>(fromIndex: number = 0): OperatorFunction<T, T & {index: number}> {
    let index = fromIndex - 1;

    return map<T, T & {index: number}>(value => {
        let v: T & {index: number} = value as T & {index: number};
        v.index = index;
        return v;
    });
}

export function numerate<T, R = any>(indexField: string = "", valueField: string = "", fromIndex: number = 0): OperatorFunction<T, R> {
    let index = fromIndex - 1;

    return map<T, R>(value => {
        index++;

        if (Array.isArray(value)) {
            value.push(index);
            return value as any;
        }

        if (indexField) {
            if (valueField) {
                let res: any = {};
                res[indexField] = index;
                res[valueField] = value;
                return res;
            }

            if (typeof value === 'object') {
                (value as any)[indexField] = index;
                return value as any;
            }

            throw new Error("You should specify value field name in the numerate.");
        }

        return [value, index] as any;
    });
}