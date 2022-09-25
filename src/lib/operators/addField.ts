import { map, OperatorFunction } from "rxjs";

export function addFieldToArray<T extends {push: any}>(calcFieldValueFn: (value: T) => any): OperatorFunction<T, T> {
    return map<T, T>(value => {
        const fieldValue = calcFieldValueFn(value);
        value.push(fieldValue);
        return value;
    });
}

export function addFieldToObject<T extends Record<string, any>, R extends T = T>(fieldName: string, calcFieldValueFn: (value: T) => any): OperatorFunction<T, R> {
    return map<T, R>(value => {
        const fieldValue = calcFieldValueFn(value);
        (value as any)[fieldName] = fieldValue;
        return value as R;
    });
}

export function addField<T, R extends T = T>(calcFieldValueFn: (value: T) => any): OperatorFunction<T, R>;
export function addField<T, R extends T = T>(fieldName: string, calcFieldValueFn: (value: T) => any): OperatorFunction<T, R>;
export function addField<T, R extends T = T>(fieldNameOrCalcFn: any, calcFieldValueFn?: (value: T) => any): OperatorFunction<T, R> {
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