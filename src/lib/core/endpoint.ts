import { BaseCollection } from "./base_collection.js";
import { GuiManager } from "./gui.js";

export class BaseEndpoint {
    protected static instanceCount = 0;
    protected instanceNo: number;
    
    constructor(hidden: boolean = false, first: boolean = false) {
        BaseEndpoint.instanceCount++;
        this.instanceNo = BaseEndpoint.instanceCount;
        if (GuiManager.instance && !hidden) GuiManager.instance.registerEndpoint(this, first);
    }

    async releaseEndpoint() {
        for (let name in this.collections) this._removeCollection(name);
    }

    get displayName(): string {
        return `Endpoint ${this.instanceNo}`;
    }


    protected collections: Record<string, BaseCollection<any>> = {};

    protected _addCollection<T extends BaseCollection<any>>(collectionName: string, collection: T): T {
        this.collections[collectionName] = collection;
        return collection;
    }

    protected _removeCollection(collectionName: string) {
        if (!this.collections[collectionName]) throw new Error(`Collection with name ${collectionName} does not exists`);
        delete this.collections[collectionName];
    }
}
  