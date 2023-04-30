import { map, OperatorFunction } from "rxjs";

export function toProperty<T, R = T>(toProperty: string, fromProperty?: string): OperatorFunction<T, R> {
    return map<T, R>(value => {
        let val = value;
        let res: any = value;
        if (fromProperty) val = value[fromProperty];
        else res = {};
        res[toProperty] = val;
        return res;
    });
}
