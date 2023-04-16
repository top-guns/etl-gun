import { Observable } from "rxjs";
import { Collection, CollectionGuiOptions } from "./collection.js";
import { GuiManager } from "./gui.js";

export class Endpoint {
    protected collections: Record<string, Collection<any>> = {};

    protected _addCollection<T extends Collection<any>>(collectionName: string, collection: T): T {
        this.collections[collectionName] = collection;
        return collection;
    }

    protected _removeCollection(collectionName: string) {
        if (!this.collections[collectionName]) throw new Error(`Collection with name ${collectionName} does not exists`);
        delete this.collections[collectionName];
    }

    protected static instanceCount = 0;
    protected instanceNo: number;

    constructor() {
        Endpoint.instanceCount++;
        this.instanceNo = Endpoint.instanceCount;
        if (GuiManager.instance) GuiManager.instance.registerEndpoint(this);
    }

    get displayName(): string {
        return `Endpoint ${this.instanceNo}`;
    }
}
  