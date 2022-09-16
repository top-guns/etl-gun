import { filter } from "rxjs";

// Synonym for operator "filter" 
export function where<T>(predicate: (value: T, index: number) => boolean) {
    return filter<T>(predicate);
}