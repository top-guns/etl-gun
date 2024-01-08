import { map, OperatorFunction } from "rxjs";
import _ from 'lodash';

export function propMove<R, T = any>(options: {from?: (keyof T) | string, to?: (keyof R) | string}): OperatorFunction<T, R> {
    return map<T, R>(value => {
        if (!options.from && !options.to) throw new Error('Error: fromPropertyPath and toPropertyPath in operator move() cannot be empty at the same time');
        if (options.from == options.to) return value as unknown as R;

        let val = value;
        if (options.from) val = _.get(value, options.from);

        if (!options.to) return val as unknown as R;

        if (options.from) _.unset(value, options.from);
        else value = {} as unknown as T;

        _.set(value as any, options.to, val);

        return value as unknown as R;
    });
}

export function propCopy<R, T = any>(fromPropertyPath: (keyof T) | string, toPropertyPath: (keyof R) | string): OperatorFunction<T, R> {
    return map<T, R>(value => {
        if (!fromPropertyPath || !toPropertyPath) throw new Error('Error: fromPropertyPath and toPropertyPath in operator copy() cannot be empty');
        if (fromPropertyPath == toPropertyPath) return value as unknown as R;

        let val = value;

        if (fromPropertyPath) val = _.get(value, fromPropertyPath);

        _.set(value as any, toPropertyPath, val);
        return value as unknown as R;
    });
}

export function propUpdate<T>(value: Partial<T> | Record<(keyof T) | string, any>): OperatorFunction<T, T> {
    return map<T, T>(value => {
        for (let key in value) {
            if (!value.hasOwnProperty(key)) continue;
            _.set(value as any, key, value[key]);
        }
        return value;
    });
}

export function propRemove<T>(...propertyPaths: ((keyof T) | string)[]): OperatorFunction<T, T> {
    return map<T, T>(value => {
        for (let path of propertyPaths) _.unset(value, path);
        return value;
    });
}

function getPropParent(path: string) {
    if (!path) return undefined;
    const i1 = path.lastIndexOf('.');
    const i2 = path.lastIndexOf('[');
    const i = Math.max(i1, i2);
    if (i < 0) return undefined;
    return path.substring(0, i);
}

function getPropName(path: string) {
    if (!path) return undefined;
    const i1 = path.lastIndexOf('.');
    const i2 = path.lastIndexOf('[');
    if (Math.max(i1, i2) < 0) return path;

    if (i1 > i2) return path.substring(i1 + 1);

    return path.substring(i2);
}
