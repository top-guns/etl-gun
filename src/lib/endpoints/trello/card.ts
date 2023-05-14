import { BaseObservable } from "../../core/observable.js";
import { CollectionOptions } from "../../core/readonly_collection.js";
import { UpdatableCollection } from "../../core/updatable_collection.js";
import { Endpoint } from './endpoint.js';


export type Card = {
    id: string;
    idBoard: string;
    idList: string;
    idShort: number;
    name: string;
    pos: number;
    shortLink: string;
    shortUrl: string;
    url: string;
}


export class CardsCollection extends UpdatableCollection<Partial<Card>> {
    protected static instanceNo = 0;
    //protected boardId: string;
    protected listId: string;

    constructor(endpoint: Endpoint, collectionName: string, listId: string, options: CollectionOptions<Partial<Card>> = {}) {
        CardsCollection.instanceNo++;
        super(endpoint, collectionName, options);
        //this.boardId = boardId;
        this.listId = listId;
    }

    public select(where: Partial<Card> = {}, fields: (keyof Card)[] = []): BaseObservable<Partial<Card>> {
        if (this.listId) return this.listByUrl(`/1/lists/${this.listId}/cards`, {}, fields);
        //let filter = '';
        //if (this.boardId) return this.listByUrl(`1/boards/${this.boardId}/cards${filter ? '/' + filter : ''}`);
        throw new Error('Error: listId cannot be empty in CardsCollection');
    }

    protected listByUrl(url: string, params: {} = {}, fields: (keyof Card)[] = []): BaseObservable<Partial<Card>> {
        const observable = new BaseObservable<Partial<Card>>(this, (subscriber) => {
            (async () => {
                try {
                    const getParams = 
                        fields.length 
                        ? {
                            fields,
                            ...params
                        }
                        : {
                            ...params
                        }
                    const cards = await this.endpoint.fetchJson(url, getParams) as Partial<Card>[];

                    this.sendStartEvent();
                    for (const obj of cards) {
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

    public async insert(value: Omit<Partial<Card>, 'id'>) {
        super.insert(value as Partial<Card>);
        return await this.endpoint.fetchJson('1/cards', {}, 'POST', value);
    }

    public async update(value: Omit<Partial<Card>, 'id'>, cardId: string) {
        super.update(value as Partial<Card>, cardId);
        return await this.endpoint.fetchJson(`1/cards/${cardId}`, {}, 'PUT', value);
    }

    async get(): Promise<Card[]>;
    async get(cardId?: string): Promise<Card>;
    async get(cardId?: string) {
        if (cardId) return this.endpoint.fetchJson(`1/cards/${cardId}`);
        return await this.endpoint.fetchJson(`1/lists/${this.listId}/cards`);
    }

    // public async getBoardCard(boardId: string, cardId: string): Promise<Card> {
    //     return await this.endpoint.fetchJson(`1/boards/${boardId}/cards/${cardId}`, {}, 'GET');
    // }
    // public async getBoardCards(boardId: string, filter?: string): Promise<Card[]> {
    //     return await this.endpoint.fetchJson(`1/boards/${boardId}/cards${filter ? '/' + filter : ''}`, {}, 'GET');
    // }

    // public async getListCards(listId: string): Promise<Card[]> {
    //     return await this.endpoint.fetchJson(`1/lists/${listId}/cards`, {}, 'GET');
    // }

    public async archiveListCards() {
        return await this.endpoint.fetchJson(`1/lists/${this.listId}/archiveAllCards`, {}, 'POST');
    }

    public async moveListCards(destBoardId: string, destListId: string) {
        return await this.endpoint.fetchJson(`1/lists/${this.listId}/moveAllCards`, {idBoard: destBoardId, idList: destListId}, 'POST');
    }

    get endpoint(): Endpoint {
        return super.endpoint as Endpoint;
    }
}
