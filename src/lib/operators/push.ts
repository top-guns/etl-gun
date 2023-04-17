import { tap } from "rxjs";
import { BaseCollection } from "../core/collection.js";

export function push<T>(collection: BaseCollection<T>, ...params: any[]) {
    return tap<T>(v => collection.insert(v, params));
}