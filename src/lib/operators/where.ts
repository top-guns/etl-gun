import { filter } from "rxjs";

// Synonym for operator "filter" 
export function where<T>(criteria: {});
export function where<T>(predicate: (value: T, index: number) => boolean);
export function where<T>(predicate: {} | ((value: T, index: number) => boolean)) {
    if (typeof predicate === 'function') return filter<T>(predicate as (value: T, index: number) => boolean);
    else {
        return filter<T>((value: T) => {
            for (const key in predicate) {
                if (predicate.hasOwnProperty(key)) {
                    if (predicate[key] != value[key]) return false;
                }
            }
            return true;
        });
    }
}