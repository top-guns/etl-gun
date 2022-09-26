import { map, OperatorFunction } from "rxjs";

export function numerate<TT>(fromIndex?: number): OperatorFunction<TT[] | TT, (TT | number)[]>;
export function numerate<T extends Record<string, any>, R extends T>(fromIndex?: number, indexField?: string): OperatorFunction<T, R>;
export function numerate<T, R extends T>(fromIndex: number = 0, indexField: string = ""): OperatorFunction<T, R> {
    let index = fromIndex - 1;

    return map<T, R>(value => {
        index++;

        if (!indexField) {
            if (Array.isArray(value)) {
                value.push(index);
                return value;
            }

            if (typeof value !== 'object') return [value, index] as any;

            throw new Error("Operator numerate: you should specify indexField for object value type.");
        }

        if (indexField) {
            if (Array.isArray(value)) throw new Error('Operator numerate: you cannot specify the indexField for array value type.');

            if (typeof value === 'object') {
                value[indexField] = index;
                return value;
            }
            
            throw new Error("Operator numerate: you cannot specify the indexField for scalar value type.");
        }
    });
}