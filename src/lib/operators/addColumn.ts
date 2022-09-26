import { map, OperatorFunction } from "rxjs";

export function addColumn<T, R = T>(calcColumnValueFn: (value: T) => any): OperatorFunction<T, R> {
    return map<T, R>(value => {
        const fieldValue = calcColumnValueFn(value);

        if (Array.isArray(value)) {
            value.push(fieldValue);
            return value as any;
        }

        return [value, fieldValue] as any;
    });
}
