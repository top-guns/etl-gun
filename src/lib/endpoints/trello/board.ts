import { CollectionOptions } from "../../core/base_collection.js";
import { Endpoint } from './endpoint.js';
import { TrelloCollection } from "./trello_collection.js";



export type Board = {
    id: string;
    name: string;
    url: string;
    shortUrl: string;
    pinned: boolean;
    closed: boolean;

    desc: string;
    descData: any;

    idOrganization: string;
    idEnterprise: string;

    prefs: any;
    labelNames: any;

    // Optional parameters

    dateClosed?: string;

    starred?: boolean;
    subscribed?: boolean;
    idMemberCreator?: string;
    idBoardSource?: string;
    nodeId?: string;
    shortLink?: string;
    templateGallery?: any;

    dateLastActivity?: string;
    dateLastView?: string;

    limits?: any;

    idTags?: string[];
    ixUpdate?: number;
    memberships?: any[];
    powerUps?: any[];
    premiumFeatures?: string[];
}

export class BoardsCollection extends TrelloCollection<Board> {
    protected static instanceNo = 0;
    protected username: string;

    constructor(endpoint: Endpoint, collectionName: string, username: string = 'me', options: CollectionOptions<Board> = {}) {
        BoardsCollection.instanceNo++;
        super(endpoint, collectionName, 'board', 'boards', options);
        this.username = username;
    }

    protected getSearchUrl(): string {
        return `members/${this.username}/${this.resourceNameS}`;
    }

    async getByBrowserUrl(url: string): Promise<Board> {
        return this.endpoint.fetchJson(url.endsWith('.json') ? url : url + ".json");
    }

    // public async getListBoard(listId: string): Promise<Board> {
    //     return await this.endpoint.fetchJson(`1/lists/${listId}/board`, {}, 'GET');
    // }

    get endpoint(): Endpoint {
        return super.endpoint as Endpoint;
    }
}
