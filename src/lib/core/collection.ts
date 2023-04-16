import { Observable } from "rxjs";
import { GuiManager } from "./gui";
import { Endpoint } from "./endpoint";

export type CollectionGuiOptions<T> = {
    displayName?: string;
    watch?: (value: T) => string;
}

export type CollectionEvent = 
    "list.start" |
    "list.end" |
    "list.data" |
    "list.error" |
    "list.skip" |
    "list.up" |
    "list.down" |
    "push" |
    "clear";

export interface Collection<T> {

    //public createReadStream(): Observable<T> {
    list(): Observable<T>;

    push(value: T, ...params: any[]): Promise<any>;
    clear(): Promise<void>;

    on(event: CollectionEvent, listener: (...data: any[]) => void): Collection<T>;

    stop(): void;
    pause(): void;
    resume(): void;

    get isPaused(): boolean;

    get endpoint(): Endpoint;

    // public async delete(where: any) {
    //     throw new Error("Method not implemented.");
    // }

    // public async find(where: any): Promise<T[]> {
    //     throw new Error("Method not implemented.");
    // }

    // public async pop(): Promise<T> {
    //     throw new Error("Method not implemented.");
    // }
    
    // public async updateCurrent(value) {
    //     throw new Error("Method not implemented.");
    // }
    
}

type EventListener = (...data: any[]) => void;

export class CollectionImpl<T> implements Collection<T> {
    protected listeners: Record<CollectionEvent, EventListener[]> = {
        "push": [],
        "clear": [],
        "list.start": [],
        "list.end": [],
        "list.data": [],
        "list.error": [],
        "list.skip": [],
        "list.up": [],
        "list.down": []
    };

    protected _endpoint: Endpoint;
    get endpoint(): Endpoint {
        return this._endpoint;
    }
    protected _isPaused: boolean = false;

    constructor(endpoint: Endpoint, guiOptions: CollectionGuiOptions<T> = {}) {
        this._endpoint = endpoint;
        if (GuiManager.instance) GuiManager.instance.registerCollection(this, guiOptions);
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

    public list(): Observable<T> {
        throw new Error("Method not implemented.");
    }

    public stop() {
        throw new Error("Method not implemented.");
    }
  
    public on(event: CollectionEvent, listener: EventListener): Collection<T> {
        if (!this.listeners[event]) this.listeners[event] = [];
        this.listeners[event].push(listener); 
        return this;
    }

    public async push(value: any, ...params: any[]): Promise<any> {
        this.sendEvent("push", value);
    }

    public async clear() {
        this.sendEvent("clear");
    }
  
    public sendEvent(event: CollectionEvent, ...data: any[]) {
        if (!this.listeners[event]) this.listeners[event] = [];
        this.listeners[event].forEach(listener => listener(...data));
    }
  
    public sendStartEvent() {
        this.sendEvent("list.start");
    }
  
    public sendEndEvent() {
        this.sendEvent("list.end");
    }
  
    public sendErrorEvent(error: any) {
        this.sendEvent("list.error", error);
    }
  
    public sendDataEvent(data: any) {
        this.sendEvent("list.data", data);
    }
  
    public sendSkipEvent(data: any) {
        this.sendEvent("list.skip", data);
    }
  
    public sendUpEvent() {
      this.sendEvent("list.up");
    }
  
    public sendDownEvent() {
        this.sendEvent("list.down");
    }
}
  