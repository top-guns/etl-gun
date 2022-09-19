import { Observable } from "rxjs";

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

    public on(event: string, listener: (...data: any[]) => void, eventGroupName: string = ''): Endpoint<T> {
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
    protected listeners: Record<string, EventListener[]> = {};
  
    public on(event: string, listener: EventListener): Endpoint<T> {
        if (!this.listeners[event]) this.listeners[event] = [];
        this.listeners[event].push(listener); 
        return this;
    }
  
    public sendEvent(event: string, ...data: any[]) {
        if (!this.listeners[event]) this.listeners[event] = [];
        this.listeners[event].forEach(listener => listener(...data));
    }
  
    public sendStartEvent() {
        this.sendEvent("start");
    }
  
    public sendEndEvent() {
        this.sendEvent("end");
    }
  
    public sendErrorEvent(error: any) {
        this.sendEvent("error", error);
    }
  
    public sendDataEvent(data: any) {
        this.sendEvent("data", data);
    }
  
    public sendSkipEvent(data: any) {
        this.sendEvent("skip", data);
    }
  
    public sendUpEvent() {
      this.sendEvent("up");
    }
  
    public sendDownEvent() {
        this.sendEvent("down");
    }
}
  