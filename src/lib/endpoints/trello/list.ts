import { CollectionOptions } from "../../core/base_collection.js";
import { BaseObservable } from "../../core/observable.js";
import { Endpoint } from './endpoint.js';
import { TrelloCollection } from "./trello_collection.js";


export type List = {
    id: string;
    name: string;
    closed: boolean;
    idBoard: string;
    pos: number;
    subscribed: boolean;
    softLimit: any;
}

export class ListsCollection extends TrelloCollection<List> {
    protected static instanceNo = 0;
    protected boardId: string;

    constructor(endpoint: Endpoint, collectionName: string, boardId: string, options: CollectionOptions<List> = {}) {
        ListsCollection.instanceNo++;
        super(endpoint, collectionName, 'list', 'lists', options);
        this.boardId = boardId;
    }

    protected getSearchUrl(): string {
        return `boards/${this.boardId}/${this.resourceNameS}`;
    }

    // public async updateField(listId: string, fieldName: string, fieldValue: any) {
    //     return await this.endpoint.fetchJson(`1/lists/${listId}/${fieldName}`, {}, 'PUT', value);
    // }

    // Archive or unarchive a list
    public async switchClosed(listId: string) {
        return await this.endpoint.fetchJson(`${this.getResourceUrl(listId)}/closed`, 'PUT');
    }

    public async move(listId: string, destBoardId: string) {
        return await this.endpoint.fetchJson(`${this.getResourceUrl(listId)}/idBoard?value=${destBoardId}`, 'PUT');
    }

    public async getActions(listId: string): Promise<any> {
        return await this.endpoint.fetchJson(`${this.getResourceUrl(listId)}/actions`, 'GET');
    }

    get endpoint(): Endpoint {
        return super.endpoint as Endpoint;
    }
}
