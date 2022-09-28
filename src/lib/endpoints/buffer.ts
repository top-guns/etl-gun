import { from, Observable } from "rxjs";
import { Endpoint, EndpointGuiOptions, EndpointImpl } from "../core/endpoint";
import { EtlObservable } from "../core/observable";

export class BufferEndpoint<T = any> extends EndpointImpl<T> {
    protected static instanceNo = 0;
    protected _buffer: T[];

    get buffer() {
        return this._buffer;
    }

    constructor(values: T[] = [], guiOptions: EndpointGuiOptions<T> = {}) {
        guiOptions.displayName = guiOptions.displayName ?? `Buffer ${++BufferEndpoint.instanceNo}`;
        super(guiOptions);
        this._buffer = [...values];
    }

    public read(): EtlObservable<T> {
        const observable = new EtlObservable<T>((subscriber) => {
            try {
                this.sendStartEvent();
                this._buffer.forEach(value => {
                    this.sendDataEvent(value);
                    subscriber.next(value);
                });
                subscriber.complete();
                this.sendEndEvent();
            }
            catch(err) {
                this.sendErrorEvent(err);
                subscriber.error(err);
            }
        });
        return observable;
    }

    public async push(value: T) {
        super.push(value);
        this._buffer.push(value); 
    }

    public async clear() {
        super.clear();
        this._buffer = [];
    }

    public sort(compareFn: ((v1: T, v2: T) => number | boolean) | undefined = undefined): void {
        if (compareFn === undefined) {
            this._buffer.sort();
            return;
        }

        this._buffer.sort((v1: T, v2: T) => {
            let res = compareFn(v1, v2);
            if (typeof res === "boolean") res = res? 1 : -1;
            return res;
        })
    }

    public forEach(callbackfn: (value: T, index: number, array: T[]) => void) {
        this._buffer.forEach(callbackfn);
    }
}
