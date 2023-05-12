import { BaseCollection, CollectionOptions } from "../../core/collection.js";
import { BaseObservable } from "../../core/observable.js";
import { Endpoint } from './endpoint.js';



export type Ticket = {
    id: number;
    url: string;
    external_id: string | null;
    via: {
        channel: 'web' | string;
        source: { 
            to: any; 
            from: any[];
            rel: 'follow_up' | string;
        }
    }
    created_at: Date; //'2022-12-29T21:02:16Z'
    updated_at: Date;
    type: null | any;
    subject: string;
    raw_subject: string;
    comment: string;
    description: string;
    priority: null | any;
    status: 'open' | 'closed' | 'solved' | string;
    recipient: string;
    requester_id: number;
    submitter_id: number;
    assignee_id: number;
    organization_id: null | number;
    group_id: number;
    collaborator_ids: any[];
    follower_ids: any[];
    email_cc_ids: any[];
    forum_topic_id: null | number;
    problem_id: null | number;
    has_incidents: boolean;
    is_public: boolean;
    due_at: null | any;
    tags: string[];
    custom_fields: { id: number; value: string | null }[];
    satisfaction_rating: { score: 'offered' | string };
    sharing_agreement_ids: any[];
    fields: { id: number; value: string | null }[];
    followup_ids: any[];
    ticket_form_id: number;
    brand_id: number;
    allow_channelback: boolean;
    allow_attachments: boolean;
    from_messaging_channel: boolean;
}

export class TicketsCollection extends BaseCollection<Partial<Ticket>> {
    protected static instanceNo = 0;

    constructor(endpoint: Endpoint, collectionName: string, options: CollectionOptions<Partial<Ticket>> = {}) {
        TicketsCollection.instanceNo++;
        super(endpoint, collectionName, options);
    }

    public select(where: Partial<Ticket> = {}): BaseObservable<Partial<Ticket>> {
        const observable = new BaseObservable<Partial<Ticket>>(this, (subscriber) => {
            (async () => {
                try {
                    let tickets: Partial<Ticket>[];
                    if (!where) tickets = (await this.endpoint.fetchJson(`/tickets`, where)).tickets as Partial<Ticket>[];
                    else {
                        let query = "type%3Aticket";
                        for (let key in where) {
                            if (!where.hasOwnProperty(key)) continue;
                            query += '+' + key + '%3A' + where[key];
                        }
                        tickets = (await this.endpoint.fetchJson(`/search.json?query=${query}`)).results as Partial<Ticket>[];
                    }

                    this.sendStartEvent();
                    for (const obj of tickets) {
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

    async get(): Promise<Ticket[]>;
    async get(ticketId: number): Promise<Ticket>;
    async get(ticketId?: number) {
        if (ticketId) return (await this.endpoint.fetchJson(`/tickets/${ticketId}`)).ticket;
        return (await this.endpoint.fetchJson(`/tickets`)).tickets;
    }

    public async insert(value: Omit<Partial<Ticket>, 'id'>) {
        super.insert(value as Partial<Ticket>);
        return await this.endpoint.fetchJson('/tickets', {}, 'POST', { ticket: value });
    }

    public async update(ticketId: number, value: Omit<Partial<Ticket>, 'id'>) {
        super.insert(value as Partial<Ticket>);
        return await this.endpoint.fetchJson(`/tickets/${ticketId}`, {}, 'PUT', { ticket: value });
    }

    get endpoint(): Endpoint {
        return super.endpoint as Endpoint;
    }
}
