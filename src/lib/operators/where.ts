import * as rx from "rxjs";
import { OperatorFunction } from "rxjs";
import { Condition, isMatch } from "../core/condition.js";

// Analog of the operator "filter" 
export function where<T>(condition: Condition<T>): OperatorFunction<T, T> {
    return rx.mergeMap<T, rx.ObservableInput<T>>(v => isMatch<T>(v, condition) ? rx.of(v) : rx.EMPTY);
}