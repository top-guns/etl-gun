import { Collection, CollectionGuiOptions, CollectionImpl } from "../../core/collection";
import { EtlObservable } from '../../core/observable';
import { TrelloEndpoint } from './endpoint';


export type List = {
    id: string;
    name: string;
    closed: boolean;
    idBoard: string;
    pos: number;
    subscribed: boolean;
    softLimit: any;
}


export class ListsCollection extends CollectionImpl<Partial<List>> {
    protected static instanceNo = 0;
    protected boardId: string;

    constructor(endpoint: TrelloEndpoint, boardId: string, guiOptions: CollectionGuiOptions<Partial<List>> = {}) {
        ListsCollection.instanceNo++;
        super(endpoint, guiOptions);
        this.boardId = boardId;
    }

    public list(where: Partial<List> = {}, fields: (keyof List)[] = null): EtlObservable<Partial<List>> {
        const observable = new EtlObservable<Partial<List>>((subscriber) => {
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
                        await this.waitWhilePaused();
                        this.sendDataEvent(obj);
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

    public async push(value: Omit<Partial<List>, 'id'>) {
        super.push(value as Partial<List>);
        return await this.endpoint.fetchJson('1/lists', {}, 'POST', value);
    }

    public async update(listId: string, value: Omit<Partial<List>, 'id'>) {
        super.push(value as Partial<List>);
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

    get endpoint(): TrelloEndpoint {
        return super.endpoint as TrelloEndpoint;
    }
}
