import { Observable } from "rxjs";

export type EndpointEvent = 
    "read.start" |
    "read.end" |
    "read.data" |
    "read.error" |
    "read.skip" |
    "read.up" |
    "read.down" |
    "push" |
    "clear";

export class Endpoint<T> {

    //public createReadStream(): Observable<T> {
    public read(): Observable<T> {
        throw new Error("Method not implemented.");
    }

    public async push(value: T, ...params: any[]) {
        throw new Error("Method not implemented.");
    }

    public async clear() {
        throw new Error("Method not implemented.");
    }

    public on(event: EndpointEvent, listener: (...data: any[]) => void): Endpoint<T> {
        throw new Error("Method not implemented.");
    }

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

export class EndpointImpl<T> extends Endpoint<T> {
    protected listeners: Record<EndpointEvent, EventListener[]> = {
        "push": [],
        "clear": [],
        "read.start": [],
        "read.end": [],
        "read.data": [],
        "read.error": [],
        "read.skip": [],
        "read.up": [],
        "read.down": []
    };
  
    public on(event: EndpointEvent, listener: EventListener): Endpoint<T> {
        if (!this.listeners[event]) this.listeners[event] = [];
        this.listeners[event].push(listener); 
        return this;
    }

    public async push(value: T, ...params: any[]) {
        this.sendEvent("push", value);
    }

    public async clear() {
        this.sendEvent("clear");
    }
  
    public sendEvent(event: EndpointEvent, ...data: any[]) {
        if (!this.listeners[event]) this.listeners[event] = [];
        this.listeners[event].forEach(listener => listener(...data));
    }
  
    public sendStartEvent() {
        this.sendEvent("read.start");
    }
  
    public sendEndEvent() {
        this.sendEvent("read.end");
    }
  
    public sendErrorEvent(error: any) {
        this.sendEvent("read.error", error);
    }
  
    public sendDataEvent(data: any) {
        this.sendEvent("read.data", data);
    }
  
    public sendSkipEvent(data: any) {
        this.sendEvent("read.skip", data);
    }
  
    public sendUpEvent() {
      this.sendEvent("read.up");
    }
  
    public sendDownEvent() {
        this.sendEvent("read.down");
    }
}
  