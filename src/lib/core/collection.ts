import { Observable } from "rxjs";
import { GuiManager } from "./gui.js";
import { BaseEndpoint } from "./endpoint.js";

export type CollectionGuiOptions<T> = {
    displayName?: string;
    watch?: (value: T) => string;
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
    get type(): string {
        throw new Error("Method not implemented.");
    }
    protected _isPaused: boolean = false;

    constructor(endpoint: BaseEndpoint, guiOptions: CollectionGuiOptions<T> = {}) {
        this._endpoint = endpoint;
        console.log('||||' + guiOptions.displayName)
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

    public select(): Observable<T> {
        throw new Error("Method not implemented.");
    }

    public async insert(value: any, ...params: any[]): Promise<any> {
        this.sendEvent("insert", { value });
    }

    public async update(where: any, value: any, ...params: any[]): Promise<any> {
        this.sendEvent("update", { where, value });
    }

    public async upsert(value: any, ...params: any[]): Promise<any> {
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
  
    public sendReciveEvent(value: any) {
        this.sendEvent("select.recive", {value});
    }
  
    public sendSkipEvent(value: any) {
        this.sendEvent("select.skip", {value});
    }
  
    public sendUpEvent() {
      this.sendEvent("select.up");
    }
  
    public sendDownEvent() {
        this.sendEvent("select.down");
    }
}
  