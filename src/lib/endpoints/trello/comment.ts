import { Observable } from "rxjs";
import { BaseCollection, CollectionGuiOptions } from "../../core/collection.js";
import { Endpoint } from './endpoint.js';


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


export class CommentsCollection extends BaseCollection<Partial<Comment>> {
    protected static instanceNo = 0;
    //protected boardId: string;
    protected cardId: string;
    
    constructor(endpoint: Endpoint, cardId: string, guiOptions: CollectionGuiOptions<Partial<Comment>> = {}) {
        CommentsCollection.instanceNo++;
        super(endpoint, guiOptions);
        //this.boardId = boardId;
        this.cardId = cardId;
    }

    public select(where: Partial<Comment> = {}, fields: (keyof Comment)[] = null): Observable<Partial<Comment>> {
        const observable = new Observable<Partial<Comment>>((subscriber) => {
            (async () => {
                try {
                    if (!where) where = {};
                    const params =
                        fields && fields.length ? {
                            fields,
                            ...where,
                            filter: 'commentCard'
                        }
                        : {
                            ...where,
                            filter: 'commentCard'
                        }
                    const comments = await this.endpoint.fetchJson(`/1/cards/${this.cardId}/actions`, params) as Partial<Comment>[];

                    this.sendStartEvent();
                    for (const obj of comments) {
                        if (subscriber.closed) break;
                        await this.waitWhilePaused();
                        this.sendReciveEvent(obj);
                        subscriber.next(obj);
                    }
                    subscriber.complete();
                    this.sendEndEvent();
                }
                catch (err) {
                    this.sendErrorEvent(err);
                    subscriber.error(err);
                }
            })();
        });
        return observable;
    }

    public async insert(text: string) {
        super.insert(text);
        return await this.endpoint.fetchJson(`/1/cards/${this.cardId}/actions/comments`, {text}, 'POST');
    }

    public async update(commentId: string, value: Omit<Partial<Comment>, 'id'>) {
        super.insert(value as Partial<Comment>);
        return await this.endpoint.fetchJson(`/1/cards/${this.cardId}/actions/comments/${commentId}`, {filter: 'commentCard'}, 'PUT', value);
    }

    async get(): Promise<Comment[]>;
    async get(commentId?: string): Promise<Comment>;
    async get(commentId?: string) {
        if (commentId) return this.endpoint.fetchJson(`1/cards/${this.cardId}/actions/comments/${commentId}`);
        return await this.endpoint.fetchJson(`1/cards/${this.cardId}/actions/comments`);
    }

    get endpoint(): Endpoint {
        return super.endpoint as Endpoint;
    }
}
