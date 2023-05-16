import { BaseObservable } from "../../core/observable.js";
import { CollectionOptions } from "../../core/readonly_collection.js";
import { UpdatableCollection } from "../../core/updatable_collection.js";
import { Endpoint } from './endpoint.js';



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

export class BoardsCollection extends UpdatableCollection<Partial<Board>> {
    protected static instanceNo = 0;
    protected username: string;

    constructor(endpoint: Endpoint, collectionName: string, username: string = 'me', options: CollectionOptions<Partial<Board>> = {}) {
        BoardsCollection.instanceNo++;
        super(endpoint, collectionName, options);
        this.username = username;
    }

    public select(where: Partial<Board> = {}, fields: (keyof Board)[] = null): BaseObservable<Partial<Board>> {
        const observable = new BaseObservable<Partial<Board>>(this, (subscriber) => {
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
                    const boards = await this.endpoint.fetchJson(`/1/members/${this.username}/boards`, params) as Partial<Board>[];

                    this.sendStartEvent();
                    for (const obj of boards) {
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

    async get(): Promise<Board[]>;
    async get(boardId?: string): Promise<Board>;
    async get(boardId?: string) {
        if (boardId) return this.endpoint.fetchJson(`1/boards/${boardId}`);
        return await this.endpoint.fetchJson(`/1/members/${this.username}/boards`);
    }

    async getByBrowserUrl(url: string): Promise<Board> {
        return this.endpoint.fetchJson(url.endsWith('.json') ? url : url + ".json");
    }
    public async insert(value: Omit<Partial<Board>, 'id'>) {
        await super.insert(value as Partial<Board>);
        return await this.endpoint.fetchJson('1/boards', {}, 'POST', value);
    }

    public async update(value: Omit<Partial<Board>, 'id'>, boardId: string) {
        await super.update(value as Partial<Board>, boardId);
        return await this.endpoint.fetchJson(`1/boards/${boardId}`, {}, 'PUT', value);
    }

    // public async getListBoard(listId: string): Promise<Board> {
    //     return await this.endpoint.fetchJson(`1/lists/${listId}/board`, {}, 'GET');
    // }

    get endpoint(): Endpoint {
        return super.endpoint as Endpoint;
    }
}
