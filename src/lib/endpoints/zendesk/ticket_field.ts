import { BaseObservable } from "../../core/observable.js";
import { CollectionOptions } from "../../core/readonly_collection.js";
import { UpdatableCollection } from "../../core/updatable_collection.js";
import { Endpoint } from './endpoint.js';



export type Field = {
    id: number;
    url: string;
    type: 'integer' | 'text' | 'textarea' | 'date' | 'assignee' | 'group' | 'tickettype' | 'priority' | 'status' | 'custom_status' | 'subject' | 'description' | 'tagger' | string;
    title: string;
    raw_title: string;
    description: string;
    raw_description: string;
    position: number;
    active: boolean;
    required: boolean;
    collapsed_for_agents: boolean;
    regexp_for_validation: null | any;
    title_in_portal: string;
    raw_title_in_portal: string;
    visible_in_portal: boolean;
    editable_in_portal: boolean;
    required_in_portal: boolean;
    tag: null | any;
    created_at: Date; // '2022-09-06T18:00:37Z'
    updated_at: Date; 
    removable: boolean;
    key: null | any;
    agent_description: null | any;
    system_field_options?: { 
        name: string;
        value: string;
        id?: number;
        raw_name?: string;
        default?: boolean;
    }[];
    sub_type_id?: number;
}

export class TicketFieldsCollection extends UpdatableCollection<Partial<Field>> {
    protected static instanceNo = 0;

    constructor(endpoint: Endpoint, collectionName: string, options: CollectionOptions<Partial<Field>> = {}) {
        TicketFieldsCollection.instanceNo++;
        super(endpoint, collectionName, options);
    }

    public select(): BaseObservable<Partial<Field>> {
        const observable = new BaseObservable<Partial<Field>>(this, (subscriber) => {
            (async () => {
                try {
                    const tickets = (await this.endpoint.fetchJson(`/ticket_fields`)).ticket_fields as Partial<Field>[];

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

    async get(): Promise<Field[]>;
    async get(fieldId: number): Promise<Field>;
    async get(fieldId?: number) {
        if (fieldId) return (await this.endpoint.fetchJson(`/ticket_fields/${fieldId}`)).ticket_field;
        return (await this.endpoint.fetchJson(`/ticket_fields`)).ticket_fields;
    }

    public async insert(value: Omit<Partial<Field>, 'id'>) {
        await super.insert(value as Partial<Field>);
        return await this.endpoint.fetchJson('/ticket_fields', {}, 'POST', { ticket_field: value });
    }

    public async update(value: Omit<Partial<Field>, 'id'>, fieldId: number) {
        await super.update(value as Partial<Field>, fieldId);
        return await this.endpoint.fetchJson(`/ticket_fields/${fieldId}`, {}, 'PUT', { ticket_field: value });
    }

    get endpoint(): Endpoint {
        return super.endpoint as Endpoint;
    }
}
