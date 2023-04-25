import { Observable, Subscriber, tap } from "rxjs";
import { GuiManager } from "./gui.js";
import { BaseEndpoint } from "./endpoint.js";
import { Errors, Memory } from "../index.js";
import { push, run } from "../operators/index.js";
import { ErrorsQueue, EtlError, EtlErrorData } from "../endpoints/errors.js";
import { BaseObservable } from "./observable.js";

export type CollectionOptions<T> = {
    displayName?: string;
    watch?: (value: T) => string;
    disableErrorsCollectionCreation?: boolean;
}

export type CollectionEvent = 
    "select.start" |
    "select.end" |
    "select.sleep" |
    "select.recive" |
    "select.error" |
    "select.skip" |
    "select.up" |
    "select.down" |
    "pipe.start" |
    "pipe.end" |
    "insert" |
    "update" |
    "upsert" |
    "delete";


type EventListener = (...data: any[]) => void;

export class BaseCollection<T> {
    protected listeners: Record<CollectionEvent, EventListener[]> = {
        "insert": [],
        "update": [],
        "upsert": [],
        "delete": [],
        "pipe.start": [],
        "pipe.end": [],
        "select.start": [],
        "select.end": [],
        "select.sleep": [],
        "select.recive": [],
        "select.error": [],
        "select.skip": [],
        "select.up": [],
        "select.down": []
    };

    protected _endpoint: BaseEndpoint;
    get endpoint(): BaseEndpoint {
        return this._endpoint;
    }

    public errors: BaseCollection<EtlError> = null;

    protected options: CollectionOptions<T>;

    protected _isPaused: boolean = false;

    constructor(endpoint: BaseEndpoint, collectionName: string, options: CollectionOptions<T> = {}) {
        this._endpoint = endpoint;
        this.options = options;
        if (GuiManager.instance) GuiManager.instance.registerCollection(this, options);
        if (!options.disableErrorsCollectionCreation && this.constructor.name != 'ErrorsQueue') {
            this.errors = Errors.Endpoint.instance.getCollection(`${collectionName}`, {disableErrorsCollectionCreation: true});
        }
    }

    public pause() {
        this._isPaused = true;
    }

    public resume() {
        this._isPaused = false;
    }

    get isPaused(): boolean {
        if (this._isPaused) return true;
        if (!GuiManager.instance) return false;

        if (GuiManager.instance.stepByStepMode) {
            if (GuiManager.instance.makeStepForward) {
                GuiManager.instance.makeStepForward = false;
                GuiManager.instance.processStatus = 'paused';
                return false;
            }
            return true;
        }

        return GuiManager.instance.processStatus == 'paused';
    }

    public waitWhilePaused() {
        if (!GuiManager.isGuiStarted()) return;

        const doWait = (resolve) => {
            if (!this.isPaused) {
                resolve();
                return;
            }
            setTimeout(doWait, 50, resolve);
        }

        //return new Promise<void>(resolve => doWait(resolve));
        return new Promise<void>(resolve => setTimeout(doWait, 10, resolve));
    }

    public select(...params: any[]): BaseObservable<T> {
        throw new Error("Method not implemented.");
    }

    // public selectOneByOne(delay: number = 0, ...params: any[]): Observable<CollectionItem<T>> {
    //     let timestamp = null;

    //     const memory = Memory.getEndpoint();
    //     const queue = memory.getQueue<T>(`${this.options.displayName}-queue`);

    //     this.select(params).pipe(
    //         push(queue)
    //     ).subscribe();

    //     return queue.select(false, delay);
    // }

    public async insert(value: T | any, ...params: any[]): Promise<any> {
        this.sendEvent("insert", { value });
    }

    public async update(where: any, value: T, ...params: any[]): Promise<any> {
        this.sendEvent("update", { where, value });
    }

    public async upsert(value: T, ...params: any[]): Promise<any> {
        this.sendEvent("upsert", { value });
    }

    public async delete(where?: any): Promise<any> {
        this.sendEvent("delete", { where });
    }
 
    public stop() {
        throw new Error("Method not implemented.");
    }
  
    public on(event: CollectionEvent, listener: EventListener): BaseCollection<T> {
        if (!this.listeners[event]) this.listeners[event] = [];
        this.listeners[event].push(listener); 
        return this;
    }

   
    public sendEvent(event: CollectionEvent, ...data: any[]) {
        if (!this.listeners[event]) this.listeners[event] = [];
        this.listeners[event].forEach(listener => listener(...data));
    }
  
    public sendStartEvent() {
        this.sendEvent("select.start");
    }
  
    public sendEndEvent() {
        this.sendEvent("select.end");
    }

    public sendSleepEvent() {
        this.sendEvent("select.sleep");
    }
  
    public sendErrorEvent(error: any) {
        this.sendEvent("select.error", {error});
    }
  
    public sendReciveEvent(value: T) {
        this.sendEvent("select.recive", {value});
    }
  
    public sendSkipEvent(value: T) {
        this.sendEvent("select.skip", {value});
    }
  
    public sendUpEvent() {
      this.sendEvent("select.up");
    }
  
    public sendDownEvent() {
        this.sendEvent("select.down");
    }

    public sendPipeStartEvent(value: T) {
        this.sendEvent("pipe.start", {value});
    }
  
    public sendPipeEndEvent(value: T) {
        this.sendEvent("pipe.end", {value});
    }
}
  