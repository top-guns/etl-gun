import { map, OperatorFunction } from "rxjs";

export function addField<T, R extends T = T>(fieldName: string, calcFieldValueFn: (value: T) => any): OperatorFunction<T, R> {
    return map<T, R>(value => {
        const fieldValue = calcFieldValueFn(value);

        if (typeof value === 'object') {
            (value as any)[fieldName] = fieldValue;
            return value as any;
        }

        throw new Error("addField: You cannot use addField for scalar input stream")
    });
}
