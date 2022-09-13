import { tap } from "rxjs";
import { Endpoint } from "../core/endpoint";

export function push<T>(endpoint: Endpoint<T>) {
    return tap<T>(v => endpoint.push(v));
}