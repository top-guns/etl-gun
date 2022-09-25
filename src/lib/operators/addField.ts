import { map, OperatorFunction } from "rxjs";

export function addField<T, R = T>(calcFieldValueFn: (value: T) => any): OperatorFunction<T, R>;
export function addField<T, R extends T = T>(fieldName: string, calcFieldValueFn: (value: T) => any): OperatorFunction<T, R>;
export function addField<T, R>(fieldNameOrCalcFn: any, calcFieldValueFn?: (value: T) => any): OperatorFunction<T, R> {
    return map<T, R>(value => {
        const calcFn = typeof calcFieldValueFn === 'function' ? calcFieldValueFn : fieldNameOrCalcFn;
        const fieldName = typeof calcFieldValueFn === 'function' ? fieldNameOrCalcFn : undefined;

        const fieldValue = calcFn(value);

        if (Array.isArray(value)) {
            value.push(fieldValue);
            return value as any;
        }

        if (fieldName) {
            if (typeof value === 'object') {
                (value as any)[fieldName] = fieldValue;
                return value as any;
            }
            throw new Error("addField: You cannot specify the fieldName when the input stream contains scalar value. Remove fieldName or convert stream values to object.");
        }

        return [value, fieldValue] as any;
    });
}