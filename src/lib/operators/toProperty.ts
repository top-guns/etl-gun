import { map, OperatorFunction } from "rxjs";

export function toProperty<R, T = any>(toProperty: string, fromProperty?: string, operation: 'move' | 'copy' = 'copy'): OperatorFunction<T, R> {
    return map<T, R>(value => {
        let val = value;
        let res: any = value;
        if (fromProperty) {
            val = value[fromProperty];
            if (operation == 'move') delete value[fromProperty];
        }
        else res = {};
        res[toProperty] = val;
        return res;
    });
}
