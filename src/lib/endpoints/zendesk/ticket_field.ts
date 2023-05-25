import { CollectionOptions } from "../../core/base_collection.js";
import { BaseObservable } from "../../core/observable.js";
import { UpdatableCollection } from "../../core/updatable_collection.js";
import { Endpoint } from './endpoint.js';
import { ZendeskCollection } from "./zendesk_collection.js";



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

export class TicketFieldsCollection extends ZendeskCollection<Field> {
    protected static instanceNo = 0;

    constructor(endpoint: Endpoint, collectionName: string, options: CollectionOptions<Field> = {}) {
        TicketFieldsCollection.instanceNo++;
        super(endpoint, collectionName, 'ticket_field', 'ticket_fields', options);
    }
}

