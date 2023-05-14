import fetch, { RequestInit } from 'node-fetch';
import https from 'node:https';
import { BaseEndpoint} from "../../core/endpoint.js";
import { pathJoin } from '../../utils/index.js';
import { Board, BoardsCollection } from './board.js';
import { List, ListsCollection } from './list.js';
import { Card, CardsCollection } from './card.js';
import { CommentsCollection, Comment } from './comment.js';
import { CollectionOptions } from '../../core/readonly_collection.js';

export type User = {
    id: string;
    username: string;
    email: string;
    fullName: string;
    status: string;
    url: string;

    prefs: {};
    limits: {};
    
    aaBlockSyncUntil: any;
    aaEmail: string;
    aaEnrolledDate: string;
    aaId: string;
    activityBlocked: boolean;
    avatarHash: string;
    avatarSource: "none" | string;
    avatarUrl: string;
    bio: string;
    bioData: {emoji: {}};
    confirmed: boolean;
    credentialsRemovedCount: number;
    domainClaimed: any;
    
    gravatarHash: string;
    
    idBoards: string[];
    idEnterprise: string;
    idEnterprisesAdmin: [];
    idEnterprisesDeactivated: [];
    idMemberReferrer: any;
    idOrganizations: string[];
    idPremOrgsAdmin: [];
    initials: string;
    isAaMastered: boolean;
    ixUpdate: number; // string?
    
    loginTypes: string[];
    marketingOptIn: {optedIn: boolean, date: string};
    memberType: "normal" | string;
    messagesDismissed: [];
    nonPublic: {fullName: string, initials: string, avatarHash: string};
    nonPublicAvailable: boolean;
    oneTimeMessagesDismissed: string[];
    
    premiumFeatures: string[];
    products: [];
    
    trophies: [];
    uploadedAvatarHash: string;
    uploadedAvatarUrl: string;
}


export class Endpoint extends BaseEndpoint {
    protected url: string;
    protected apiKey: string;
    protected authToken: string;
    protected token: string;
    protected rejectUnauthorized: boolean;
    protected agent: https.Agent;

    constructor(apiKey: string, authToken: string, url: string = "https://trello.com", rejectUnauthorized: boolean = true) {
        super();

        this.url = url;
        this.apiKey = apiKey;
        this.authToken = authToken;

        this.rejectUnauthorized = rejectUnauthorized;
        this.agent = rejectUnauthorized ? null : new https.Agent({
            rejectUnauthorized
        });
    }

    async fetchJson<T = any>(url: string, params: {} = {}, method: 'GET' | 'PUT' | 'POST' = 'GET', body?: any): Promise<T> {
        const init: RequestInit = {
            method,
            agent: this.agent,
            headers: {
                "Content-Type": "application/json"
            }
        }
        if (body) init.body = JSON.stringify(body);

        const  getParams = new URLSearchParams(params).toString();

        let fullUrl = url.startsWith('http') ? url : pathJoin([this.url, url], '/');
        fullUrl += `?key=${this.apiKey}&token=${this.authToken}${getParams ? '&' + getParams : ''}`;
        //console.log(fullUrl);
        const res = await fetch(fullUrl, init);
        //console.log(res)
        return (await res.json()) as T;
    }

    async getUser(username: string): Promise<User>;
    async getUser(userid: number): Promise<User>;
    async getUser(user: any): Promise<User> {
        return this.fetchJson(`1/members/${user}`);
    }

    getUserBoards(username: string = 'me', collectionName: string = 'Boards', options: CollectionOptions<Partial<Board>> = {}): BoardsCollection {
        options.displayName ??= `Boards`;
        const collection = new BoardsCollection(this, collectionName, username, options);
        this._addCollection(collectionName, collection);
        return collection;
    }

    getBoardLists(boardId: string, collectionName: string = 'Lists', options: CollectionOptions<Partial<List>> = {}): ListsCollection {
        options.displayName ??= `Lists`;
        const collection = new ListsCollection(this, collectionName, boardId, options);
        this._addCollection(collectionName, collection);
        return collection;
    }

    getListCards(listId: string, collectionName: string = 'Cards', options: CollectionOptions<Partial<Card>> = {}): CardsCollection {
        options.displayName ??= `Cards`;
        const collection = new CardsCollection(this, collectionName, listId, options);
        this._addCollection(collectionName, collection);
        return collection;
    }
    // getBoardCards(boardId: string, collectionName: string = 'Cards', guiOptions: CollectionGuiOptions<Partial<Card>> = {}): CardsCollection {
    //     guiOptions.displayName ??= `Cards`;
    //     const collection = new CardsCollection(this, boardId, null, guiOptions);
    //     this._addCollection(COLLECTIONS_NAMES.cards, collection);
    //     return collection;
    // }

    getCardComments(cardId: string, collectionName: string = 'Comments', options: CollectionOptions<Partial<Comment>> = {}): CommentsCollection {
        options.displayName ??= `Cards`;
        const collection = new CommentsCollection(this, collectionName, cardId, options);
        this._addCollection(collectionName, collection);
        return collection;
    }

    releaseCollection(collectionName: string) {
        this._removeCollection(collectionName);
    }

    get displayName(): string {
        return `Trello (${this.url})`;
    }
}

export function getEndpoint(apiKey: string, authToken: string, url: string = "https://trello.com", rejectUnauthorized: boolean = true): Endpoint {
    return new Endpoint(apiKey, authToken, url, rejectUnauthorized);
}