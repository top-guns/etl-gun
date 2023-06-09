import * as rx from "rxjs";
import * as ix from 'ix';
import * as internal from "stream";
import { GuiManager } from "./gui.js";
import { BaseEndpoint } from "./endpoint.js";
import { Errors, run } from "../index.js";
import { EtlError } from "../endpoints/errors.js";
import { BaseObservable } from "./observable.js";
import { generator2Iterable, observable2Stream } from "../utils/flows.js";


export type CollectionOptions<T> = {
    displayName?: string;
    watch?: (value: T) => string;
    disableErrorsCollectionCreation?: boolean;
}

export type BaseCollectionEvent = 
    "select.start" |
    "select.end" |
    "select.sleep" |
    "select.error" |
    "select.skip" |
    "select.up" |
    "select.down" |
    "pipe.start" |
    "pipe.end" |
    "recive" |
    string;


export type CollectionEventListener = (...data: any[]) => void;

export abstract class BaseCollection<T> {
    protected _listeners: Record<BaseCollectionEvent, CollectionEventListener[]>;
    protected get listeners(): Record<BaseCollectionEvent, CollectionEventListener[]> {
        return this._listeners;
    }

    protected _endpoint: BaseEndpoint;
    get endpoint(): BaseEndpoint {
        return this._endpoint;
    }

    protected _errors: Errors.ErrorsQueue | null = null;
    get errors(): Errors.ErrorsQueue | null {
        if (this._errors) return this._errors;
        if (!this.options.disableErrorsCollectionCreation && !['ErrorsQueue'].includes(this.constructor.name)) {
            // TODO
            //this._errors = Errors.Endpoint.instance.getCollection(`${this.collectionName}`, {disableErrorsCollectionCreation: true});
        }
        return this._errors;
    }
    set errors(errorsQueue: Errors.ErrorsQueue) {
        this._errors = errorsQueue;
    }

    protected collectionName: string;
    protected options: CollectionOptions<T>;

    protected _isPaused: boolean = false;

    constructor(endpoint: BaseEndpoint, collectionName: string, options: CollectionOptions<T> = {}) {
        this._endpoint = endpoint;
        this.collectionName = collectionName;
        this.options = options;

        this._listeners = {
            "select.start": [],
            "select.end": [],
            "select.sleep": [],
            "select.error": [],
            "select.skip": [],
            "select.up": [],
            "select.down": [],
            "pipe.start": [],
            "pipe.end": [],
            "recive": []
        };

        if (GuiManager.instance) GuiManager.instance.registerCollection(this, options);
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

    
    public abstract select(...params: any[]): Promise<T[]>;
    public abstract selectGen(...params: any[]): AsyncGenerator<T, void, void>;
    public abstract selectRx(...params: any[]): BaseObservable<T>;
    public abstract selectIx(...params: any[]): ix.AsyncIterable<T>;
    public abstract selectStream(...params: any[]): ReadableStream<T>;
    public abstract selectOne(...params: any[]): Promise<T | null>;

    // public selectReadable(...params: any[]): internal.Readable {
    //     return rx2Stream(this.selectRx(...params));
    // }

    // public abstract selectStream(...params: any[]): NodeJS.ReadStream; 
    //public abstract selectStream(...params: any[]): NodeJS.ReadableStream; 
    // public abstract selectStream(...params: any[]): NodeJS.ArrayBufferView; 
    

 
    public stop() {
        throw new Error("Method not implemented.");
    }
  
    public on(event: BaseCollectionEvent, listener: CollectionEventListener): BaseCollection<T> {
        if (!this.listeners[event]) this.listeners[event] = [];
        this.listeners[event].push(listener); 
        return this;
    }

   
    public sendEvent(event: BaseCollectionEvent, ...data: any[]) {
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
        this.sendEvent("select.error", { error });
    }
  
    public sendReciveEvent(value: T) {
        this.sendEvent("recive", { value });
    }
  
    public sendSkipEvent(value: T) {
        this.sendEvent("select.skip", { value });
    }
  
    public sendUpEvent() {
      this.sendEvent("select.up");
    }
  
    public sendDownEvent() {
        this.sendEvent("select.down");
    }

    public sendPipeStartEvent(value: T) {
        this.sendEvent("pipe.start", { value });
    }
  
    public sendPipeEndEvent(value: T) {
        this.sendEvent("pipe.end", { value });
    }

    public sendSelectOneEvent(value: T | null) {
        this.sendEvent("recive", { value });
    }

    public sendSelectEvent(values: T[]) {
        this.sendEvent("recive", { value: values });
    }
}
  