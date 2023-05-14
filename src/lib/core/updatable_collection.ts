import { CollectionEvent, ReadonlyCollection } from "./readonly_collection.js";


export class UpdatableCollection<T> extends ReadonlyCollection<T> {
    public async insert(value: T | any, ...params: any[]): Promise<any> {
        super.sendEvent("insert", { value });
    }

    public async update(value: T, where: any, ...params: any[]): Promise<any> {
        super.sendEvent("update", { value, where });
    }

    public async upsert(value: T, where?: any, ...params: any[]): Promise<any> {
        super.sendEvent("upsert", { value });
    }

    public async delete(where?: any): Promise<any> {
        super.sendEvent("delete", { where });
    }

    public on(event: CollectionEvent, listener: EventListener): UpdatableCollection<T> {
        return super.on(event, listener) as UpdatableCollection<T>;
    }

}
  