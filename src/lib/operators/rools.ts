import * as rx from "rxjs";
import { Rools, EvaluateResult } from "rools";

export type EtlRoolsResult = {
    etl?: {
        skip?: boolean;
        stop?: boolean;
        error?: string;
    }
}

export function rools<T, R = T>(rools: Rools): rx.OperatorFunction<T, R> {

    const stopSignal$ = new rx.Subject();

    const process = async (v: T): Promise<R & EtlRoolsResult> => {
        const r: EvaluateResult = await rools.evaluate(v);
        return v as unknown as R & EtlRoolsResult;
    }

    const observable = (v: T): rx.Observable<R> => 
        rx.from(process(v))
        .pipe(
            rx.mergeMap<R, rx.ObservableInput<R>>((r: R & EtlRoolsResult) => {
                if (!r.etl) return rx.of(r);

                if (r.etl.skip) return rx.EMPTY;
                if (r.etl.error) rx.throwError(() => new Error(r.etl.error));
                if (r.etl.stop) stopSignal$.next(true);

                delete r.etl;
                return rx.of(r);
            }),
            rx.takeUntil(stopSignal$)
        );
        
    const res = rx.mergeMap<T, rx.ObservableInput<R>>(v => observable(v));

    return res;
}
