import { CollectionOptions } from "../../core/base_collection.js";
import { BaseObservable } from "../../core/observable.js";
import { UpdatableCollection } from "../../core/updatable_collection.js";
import { Endpoint } from './endpoint.js';
import { TrelloCollection } from "./trello_collection.js";


export type Comment = {
    id: string
    type: 'commentCard' | string
    date: string
    idMemberCreator: string
    data: {
        text: string
        textData?: { 
            emoji: any
        }
        card?: {
            id: string
            name: string
            idShort: number
            shortLink: string
        }
        board?: {
            id: string
            name: string
            shortLink: string
        }
        list?: {
            id: string
            name: string
        }
    }
    appCreator: any
    limits: { 
        reactions: { 
            perAction: any
            uniquePerAction: any
        } 
    }
    memberCreator: {
        id: string
        activityBlocked: boolean
        avatarHash: string
        avatarUrl: string
        fullName: string
        idMemberReferrer: string
        initials: string
        nonPublic: any
        nonPublicAvailable: boolean
        username: string
    }
}


export class CommentsCollection extends TrelloCollection<Comment> {
    protected static instanceNo = 0;
    //protected boardId: string;
    protected cardId: string;
    
    constructor(endpoint: Endpoint, collectionName: string, cardId: string, options: CollectionOptions<Comment> = {}) {
        CommentsCollection.instanceNo++;
        super(endpoint, collectionName, 'commentCard', 'comments', options);
        //this.boardId = boardId;
        this.cardId = cardId;
    }

    protected getResourceUrl(id: string): string {
        return `cards/${this.cardId}/actions/${this.resourceNameS}/${id}`;
    }

    protected getResourceListUrl(): string {
        return `cards/${this.cardId}/actions/${this.resourceNameS}`;
    }

    protected getSearchUrl(): string {
        // `cards/${this.cardId}/actions/${this.resourceNameS}`
        return `cards/${this.cardId}/actions?filter=${this.resourceName}`;
    }

    protected async _insert(text: string) {
        return await super._insert({ text });
        //return await this.endpoint.fetchJson(`/1/cards/${this.cardId}/actions/comments`, {text}, 'POST');
    }

    get endpoint(): Endpoint {
        return super.endpoint as Endpoint;
    }
}
