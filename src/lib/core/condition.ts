import _ from 'lodash';

type ConditionOperations = {
    '==': (val: any) => (v: any) => boolean;
    '!=': (val: any) => (v: any) => boolean;
    '>': (val: any) => (v: any) => boolean;
    '<': (val: any) => (v: any) => boolean;
    '>=': (val: any) => (v: any) => boolean;
    '<=': (val: any) => (v: any) => boolean;

    in: (hash: any) => (v: any) => boolean;
    has: (key: string) => (v: any) => boolean; // like hasOwnProperty

    of: (arr: any[]) => (v: any) => boolean;
    includes: (val: any) => (v: any) => boolean;

    match: (regexp: RegExp) => (v: any) => boolean;

    isNull: () => (v: any) => boolean;
    isUndefined: () => (v: any) => boolean;
    isArray: () => (v: any) => boolean;
    isEmpty: () => (v: any) => boolean;
}

const operations: ConditionOperations & { not: ConditionOperations } = {
    '==': (val: any) => (v: any) => v == val,
    '!=': (val: any) => (v: any) => v != val,
    '>': (val: any) => (v: any) => v > val,
    '<': (val: any) => (v: any) => v < val,
    '>=': (val: any) => (v: any) => v >= val,
    '<=': (val: any) => (v: any) => v <= val,

    in: (hash: any) => (propertyPath: string) => typeof _.get(hash, propertyPath) != 'undefined',
    has: (propertyPath: string) => (hash: any) => typeof _.get(hash, propertyPath) != 'undefined',

    of: (arr: any[]) => (val: any) => arr.includes(val),
    includes: (val: any) => (arr: any[]) => arr.includes(val),

    match: (regexp: RegExp) => (v: any) => regexp.test(v),

    isNull: () => (v: any) => v === null,
    isUndefined: () => (v: any) => typeof v === 'undefined',
    isArray: () => (v: any) => Array.isArray(v),
    isEmpty: () => (v: any) => !!v,

    not: null
}

if (!operations.not) {
    operations.not = {} as any;
    for (let key in operations) {
        if (!operations.hasOwnProperty(key)) continue;
        if (key === 'not') continue;
    
        operations.not[key] = (value: any) => {
            return (v: any) => {
                return !operations[key](value)(v);
            }
        }
    }
}


export const Value: ConditionOperations & { not: ConditionOperations } = operations;

export type Condition<T> = 
    | Record<string, any | ((value: any) => (v: T) => string | boolean)> 
    | ((v: T) => string | boolean);


export function findDifference<T>(value: T, condition: Condition<T>): string | null {
    if (typeof condition === 'function') {
        const res = condition(value);
        if (res === true) return null;
        if (res === '') return null;
        if (typeof res === 'string') return res;
        return 'functional condition is not satisfied';
    }
    
    for (const key in condition) {
        if (!condition.hasOwnProperty(key)) continue;
        
        const val: any = _.get(value, key);
        const cond: any = (condition as any)[key];

        let diff: string = '';
        switch (typeof cond) {
            case 'function': {
                if (!cond(val)) diff = `field '${key}' has unexpected value '${val}'`;
                break;
            }
            case 'string': 
            case 'boolean':
            case 'number': {
                if (cond != val) diff = `field '${key}' is equals to '${val}' but expected to be '${(condition as any)[key]}'`;
                break;
            }
            case 'object': {
                diff = findDifference(val, cond);
                break;
            }
            default:
                if (cond === null) {
                    if (val !== null) diff = `field '${key}' is equals to '${val}' but expected to be null`;
                    break;
                }
                throw new Error(`Unexpected condition type: condition ${cond} has type ${typeof cond} but should has one of [string, number, boolean, null, function, object]`)
        }

        if (diff) return diff;
    }

    return null;
}

export function isMatch<T>(value: T, condition: Condition<T>): boolean {
    return !findDifference<T>(value, condition);
}