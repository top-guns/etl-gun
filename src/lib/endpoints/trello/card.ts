import { CollectionOptions } from "../../core/base_collection.js";
import { BaseObservable } from "../../core/observable.js";
import { Endpoint } from './endpoint.js';
import { TrelloCollection } from "./trello_collection.js";


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


export class CardsCollection extends TrelloCollection<Card> {
    protected static instanceNo = 0;
    //protected boardId: string;
    protected listId: string;

    constructor(endpoint: Endpoint, collectionName: string, listId: string, options: CollectionOptions<Card> = {}) {
        CardsCollection.instanceNo++;
        super(endpoint, collectionName, 'card', 'cards', options);
        //this.boardId = boardId;
        this.listId = listId;
    }

    protected getSearchUrl(): string {
        return `lists/${this.listId}/${this.resourceNameS}`;
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
        return await this.endpoint.fetchJson(`lists/${this.listId}/archiveAllCards`, 'POST');
    }

    public async moveListCards(destBoardId: string, destListId: string) {
        return await this.endpoint.fetchJson(`lists/${this.listId}/moveAllCards?idBoard=${destBoardId}&idList=${destListId}`,'POST');
    }

    get endpoint(): Endpoint {
        return super.endpoint as Endpoint;
    }
}
