import { CollectionOptions } from "../../core/base_collection.js";
import { Endpoint } from './endpoint.js';
import { ZendeskCollection } from "./zendesk_collection.js";



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

export class TicketsCollection extends ZendeskCollection<Ticket> {
    protected static instanceNo = 0;

    constructor(endpoint: Endpoint, collectionName: string, options: CollectionOptions<Ticket> = {}) {
        TicketsCollection.instanceNo++;
        super(endpoint, collectionName, 'ticket', 'tickets', options);
    }
}
