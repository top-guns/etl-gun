import { tap } from "rxjs";

export function log<T>(before: string = '', after: string = '') {
    return tap<T>(v => console.log(before, v, after));
}