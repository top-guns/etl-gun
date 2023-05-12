import { map, OperatorFunction } from "rxjs";
import _ from 'lodash';

export function move<R, T = any>(options: {fromPropertyPath?: string, toPropertyPath?: string}): OperatorFunction<T, R> {
    return map<T, R>(value => {
        if (!options.fromPropertyPath && !options.toPropertyPath) throw new Error('Error: fromPropertyPath and toPropertyPath in operator move() cannot be empty at the same time');
        if (options.fromPropertyPath == options.toPropertyPath) return value as unknown as R;

        let val = value;
        if (options.fromPropertyPath) val = _.get(value, options.fromPropertyPath);

        if (!options.toPropertyPath) return val as unknown as R;

        const parent = getPropParent(options.fromPropertyPath);
        const name = getPropName(options.fromPropertyPath);
        if (!parent) delete value[options.fromPropertyPath];
        else {
            const parentVal = _.get(value, parent);
            delete parentVal[name];
            _.set(value, parent, parentVal);
        }

        _.set(value, options.toPropertyPath, val);

        return value as unknown as R;
    });
}

export function copy<R, T = any>(fromPropertyPath: string, toPropertyPath: string): OperatorFunction<T, R> {
    return map<T, R>(value => {
        if (!fromPropertyPath || !toPropertyPath) throw new Error('Error: fromPropertyPath and toPropertyPath in operator copy() cannot be empty');
        if (fromPropertyPath == toPropertyPath) return value as unknown as R;

        let val = value;

        if (fromPropertyPath) val = _.get(value, fromPropertyPath);

        _.set(value, toPropertyPath, val);
        return value as unknown as R;
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
