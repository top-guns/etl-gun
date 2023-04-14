import { Observable } from "rxjs";
import { Collection, CollectionGuiOptions } from "./collection";
import { GuiManager } from "./gui";

export class Endpoint {
    protected collections: Record<string, Collection<any>> = {};

    protected _addCollection<T extends Collection<any>>(name: string, collection: T): T {
        this.collections[name] = collection;
        return collection;
    }

    protected _removeCollection(name: string) {
        if (!this.collections[name]) throw new Error(`Collection with name ${name} does not exists`);
        delete this.collections[name];
    }
}
  