import { Observable } from "rxjs";

export class Endpoint<T> {

    public createReadStream(): Observable<T> {
        throw new Error("Method not implemented.");
    }

    public async push(value: T) {
        throw new Error("Method not implemented.");
    }

    public async clear() {
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