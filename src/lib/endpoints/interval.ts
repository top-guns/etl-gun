import { Subscriber } from "rxjs";
import { GuiManager } from "../core";
import { EndpointImpl } from "../core/endpoint";
import { EtlObservable } from "../core/observable";

export class IntervalEndpoint extends EndpointImpl<number> {
    protected interval: number;
    protected intervalId: NodeJS.Timer;
    protected counter: number = 0;
    protected subscriber: Subscriber<number>;
    
    constructor(interval: number, displayName: string = '') {
        super(displayName ? displayName : `Interval (${interval}ms)`);
        this.interval = interval;
    }

    public read(): EtlObservable<number> {
        const observable = new EtlObservable<number>((subscriber) => {
            try {
                this.subscriber = subscriber;
                this.sendStartEvent();
                this.intervalId = setInterval(this.onTick.bind(this), this.interval);
            }
            catch(err) {
                this.sendErrorEvent(err);
                subscriber.error(err);
            }
        });
        return observable;
    }

    public async push(value: number) {
        super.push(value);
        this.counter = value;
    }

    public async clear() {
        super.clear();
        this.counter = 0;
    }

    protected onTick() {
        if (this.isPaused) return;
        try {
            this.sendDataEvent(this.counter);
            this.subscriber.next(this.counter);
            this.counter++;
        }
        catch(err) {
            console.log(err)
            this.sendErrorEvent(err);
            this.subscriber.error(err);
        }
    }

    public async stop() {
        try {
            clearInterval(this.intervalId);
            this.subscriber.complete();
            this.sendEndEvent();
            this.subscriber = null;
        }
        catch(err) {
            this.sendErrorEvent(err);
            this.subscriber.error(err);
        }
    }
}
