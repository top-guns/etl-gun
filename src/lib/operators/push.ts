import { tap } from "rxjs";
import { Endpoint } from "../core/endpoint";

export function push<T>(endpoint: Endpoint<T>, ...params: any[]) {
    return tap<T>(v => endpoint.push(v, params));
}