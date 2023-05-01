import { map, OperatorFunction } from "rxjs";
import _ from 'lodash';

export function move<R, T = any>(toPropertyPath: string): OperatorFunction<T, R>;
export function move<R, T = any>(fromPropertyPath: string, toPropertyPath: string): OperatorFunction<T, R>;
export function move<R, T = any>(path1: string, path2?: string): OperatorFunction<T, R> {
    return map<T, R>(value => {
        let val = value;
        let res: any = value;
        let toPropertyPath = path1;

        if (path2) {
            val = _.get(value, path1);
            toPropertyPath = path2;

            const parent = getPropParent(path1);
            const name = getPropName(path1);
            if (!parent) delete value[path1];
            else delete value[parent][name];
        }
        else {
            res = {};
        }

        _.set(res, toPropertyPath, val);
        return res;
    });
}

export function copy<R, T = any>(fromPropertyPath: string, toPropertyPath: string): OperatorFunction<T, R> {
    return map<T, R>(value => {
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
