import _ from 'lodash';

type Value = {
    '==': (val: any) => (v: any) => boolean;
    '!=': (val: any) => (v: any) => boolean;
    '>': (val: any) => (v: any) => boolean;
    '<': (val: any) => (v: any) => boolean;
    '>=': (val: any) => (v: any) => boolean;
    '<=': (val: any) => (v: any) => boolean;

    in: (hash: any) => (v: any) => boolean;
    hasProperty: (key: string) => (v: any) => boolean; // like hasOwnProperty

    of: (arr: any[]) => (v: any) => boolean;
    includes: (val: any) => (v: any) => boolean;

    match: (regexp: RegExp) => (v: any) => boolean;
}

export const VALUE: Value & { not: Value } = {
    '==': (val: any) => (v: any) => v == val,
    '!=': (val: any) => (v: any) => v != val,
    '>': (val: any) => (v: any) => v > val,
    '<': (val: any) => (v: any) => v < val,
    '>=': (val: any) => (v: any) => v >= val,
    '<=': (val: any) => (v: any) => v <= val,

    in: (hash: any) => (key: string) => hash.hasOwnProperty(key),
    hasProperty: (key: string) => (hash: any) => hash.hasOwnProperty(key),

    of: (arr: any[]) => (val: any) => arr.includes(val),
    includes: (val: any) => (arr: any[]) => arr.includes(val),

    match: (regexp: RegExp) => (v: any) => regexp.test(v),

    not: null
}

if (!VALUE.not) {
    VALUE.not = {} as any;
    for (let key in VALUE) {
        if (!VALUE.hasOwnProperty(key)) continue;
        if (key === 'not') continue;
    
        VALUE.not[key] = (value: any) => {
            return (v: any) => {
                return !VALUE[key](value)(v);
            }
        }
    }
}


export type Condition<T> = Record<keyof T, any | ((value: any) => (v: T) => string | boolean)> | ((v: T) => string | boolean);


export function findDifference<T>(value: T, condition: Condition<T>, rootProperty?: string): string | null {
    if (typeof condition === 'function') {
        const res = condition(value);
        if (res === true) return null;
        if (res === '') return null;
        if (typeof res === 'string') return res;
        return 'functional condition is not satisfied';
    }
    
    for (const key in condition) {
        if (!condition.hasOwnProperty(key)) continue;
        
        let prop: string = key;
        if (rootProperty) prop = rootProperty + prop;
        const val: T = _.get(value, prop);

        const isOk: boolean = typeof (condition as any)[key] === 'function' ? (condition as any)[key](val) : (condition as any)[key] == val;
        if (!isOk) {
            if (typeof (condition as any)[key] === 'function') return `field '${key}' has unexpected value '${val}'`;
            return `field '${key}' is equals to '${val}' but expected to be '${(condition as any)[key]}'`;
        }
    }

    return null;
}

export function isMatch<T>(value: T, condition: Condition<T>, rootProperty?: string): boolean {
    return !findDifference<T>(value, condition, rootProperty);
}