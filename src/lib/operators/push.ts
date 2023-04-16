import { tap } from "rxjs";
import { Collection } from "../core/collection.js";

export function push<T>(collection: Collection<T>, ...params: any[]) {
    return tap<T>(v => collection.push(v, params));
}