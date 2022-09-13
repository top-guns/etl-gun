import { map, OperatorFunction } from "rxjs";

export function pushIndex<T extends {push: any}>(fromIndex: number = 0): OperatorFunction<T, T> {
    let index = fromIndex - 1;

    return map<T, T>(value => {
        index++;
        value.push(index);
        return value;
    });
}

export function addIndex<T extends object>(fromIndex: number = 0): OperatorFunction<T, T & {index: number}> {
    let index = fromIndex - 1;

    return map<T, T & {index: number}>(value => {
        let v: T & {index: number} = value as T & {index: number};
        v.index = index;
        return v;
    });
}

export function numerate<T extends object, R = T & {index: number}>(fromIndex: number = 0): OperatorFunction<T, R> {
    let index = fromIndex - 1;

    return map<T, R>(value => {
        if (Array.isArray(value)) {
            index++;
            value.push(index);
            return value as T & {index: number};
        }

        if (typeof value === 'object') {
            let v: any = value as any;
            v.index = index;
            return v;
        }

        throw new Error("Operator 'numerate' cannot be used with scalar type operand. Type of operand is '" + typeof value + "'.");
    });
}