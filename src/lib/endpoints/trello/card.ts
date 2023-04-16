import { Collection, CollectionGuiOptions, CollectionImpl } from "../../core/collection.js";
import { EtlObservable } from '../../core/observable.js';
import { TrelloEndpoint } from './endpoint.js';


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


export class CardsCollection extends CollectionImpl<Partial<Card>> {
    protected static instanceNo = 0;
    //protected boardId: string;
    protected listId: string;

    constructor(endpoint: TrelloEndpoint, listId: string, guiOptions: CollectionGuiOptions<Partial<Card>> = {}) {
        CardsCollection.instanceNo++;
        super(endpoint, guiOptions);
        //this.boardId = boardId;
        this.listId = listId;
    }

    public list(where: Partial<Card> = {}, fields: (keyof Card)[] = []): EtlObservable<Partial<Card>> {
        if (this.listId) return this.listByUrl(`/1/lists/${this.listId}/cards`, {}, fields);
        //let filter = '';
        //if (this.boardId) return this.listByUrl(`1/boards/${this.boardId}/cards${filter ? '/' + filter : ''}`);
        throw new Error('Error: listId cannot be empty in CardsCollection');
    }

    protected listByUrl(url: string, params: {} = {}, fields: (keyof Card)[] = []): EtlObservable<Partial<Card>> {
        const observable = new EtlObservable<Partial<Card>>((subscriber) => {
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

    public async push(value: Omit<Partial<Card>, 'id'>) {
        super.push(value as Partial<Card>);
        return await this.endpoint.fetchJson('1/cards', {}, 'POST', value);
    }

    public async update(cardId: string, value: Omit<Partial<Card>, 'id'>) {
        super.push(value as Partial<Card>);
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

    get endpoint(): TrelloEndpoint {
        return super.endpoint as TrelloEndpoint;
    }
}