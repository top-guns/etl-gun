import { BaseObservable } from "../../core/observable.js";
import { CollectionOptions } from "../../core/readonly_collection.js";
import { UpdatableCollection } from "../../core/updatable_collection.js";
import { Endpoint } from './endpoint.js';


export type List = {
    id: string;
    name: string;
    closed: boolean;
    idBoard: string;
    pos: number;
    subscribed: boolean;
    softLimit: any;
}


export class ListsCollection extends UpdatableCollection<Partial<List>> {
    protected static instanceNo = 0;
    protected boardId: string;

    constructor(endpoint: Endpoint, collectionName: string, boardId: string, options: CollectionOptions<Partial<List>> = {}) {
        ListsCollection.instanceNo++;
        super(endpoint, collectionName, options);
        this.boardId = boardId;
    }

    public select(where: Partial<List> = {}, fields: (keyof List)[] = null): BaseObservable<Partial<List>> {
        const observable = new BaseObservable<Partial<List>>(this, (subscriber) => {
            (async () => {
                try {
                    if (!where) where = {};
                    const params = 
                    fields && fields.length ? {
                        fields,
                        ...where
                    }
                    : {
                        ...where
                    }
                    const lists = await this.endpoint.fetchJson(`/1/boards/${this.boardId}/lists`, params) as Partial<List>[];

                    this.sendStartEvent();
                    for (const obj of lists) {
                        if (subscriber.closed) break;
                        await this.waitWhilePaused();
                        this.sendReciveEvent(obj);
                        subscriber.next(obj);
                    }
                    subscriber.complete();
                    this.sendEndEvent();
                }
                catch(err) {
                    this.sendErrorEvent(err);
                    subscriber.error(err);
                }
            })();
        });
        return observable;
    }

    async get(): Promise<List[]>;
    async get(listId?: string): Promise<List>;
    async get(listId?: string) {
        if (listId) return this.endpoint.fetchJson(`1/lists/${listId}`);
        return await this.endpoint.fetchJson(`/1/boards/${this.boardId}/lists`);
    }

    public async insert(value: Omit<Partial<List>, 'id'>) {
        await super.insert(value as Partial<List>);
        return await this.endpoint.fetchJson('1/lists', {}, 'POST', value);
    }

    public async update(value: Omit<Partial<List>, 'id'>, listId: string) {
        await super.update(value as Partial<List>, listId);
        return await this.endpoint.fetchJson(`1/lists/${listId}`, {}, 'PUT', value);
    }

    // public async updateField(listId: string, fieldName: string, fieldValue: any) {
    //     return await this.endpoint.fetchJson(`1/lists/${listId}/${fieldName}`, {}, 'PUT', value);
    // }

    // Archive or unarchive a list
    public async switchClosed(listId: string) {
        return await this.endpoint.fetchJson(`1/lists/${listId}/closed`, {}, 'PUT');
    }

    public async move(listId: string, destBoardId: string) {
        return await this.endpoint.fetchJson(`1/lists/${listId}/idBoard`, {value: destBoardId}, 'PUT');
    }

    public async getActions(listId: string): Promise<any> {
        return await this.endpoint.fetchJson(`1/lists/${listId}/actions`, {}, 'GET');
    }

    get endpoint(): Endpoint {
        return super.endpoint as Endpoint;
    }
}
