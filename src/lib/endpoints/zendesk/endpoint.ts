import fetch, { RequestInit } from 'node-fetch';
import https from 'node:https';
import { CollectionOptions } from '../../core/base_collection.js';
import { pathJoin } from '../../utils/index.js';
import { BasicAuthEndpoint } from '../rest/basic_auth_endpoint.js';
import { Ticket, TicketsCollection } from './ticket.js';
import { Field, TicketFieldsCollection } from './ticket_field.js';


export class Endpoint extends BasicAuthEndpoint {
    constructor(zendeskUrl: string, username: string, token: string, rejectUnauthorized: boolean = true) {
        let apiUrl = zendeskUrl;
        if (!apiUrl.endsWith('/v2')) {
            if (!apiUrl.endsWith('/api')) apiUrl = pathJoin([apiUrl, 'api']);
            apiUrl = pathJoin([apiUrl, 'v2']);
        }

        super(apiUrl, username, token, rejectUnauthorized);
    }

    getTickets(collectionName: string = 'Tickets', options: CollectionOptions<Partial<Ticket>> = {}): TicketsCollection {
        options.displayName ??= `Tickets`;
        const collection = new TicketsCollection(this, collectionName, options);
        this._addCollection(collectionName, collection);
        return collection;
    }

    getTicketFields(collectionName: string = 'TicketFields', options: CollectionOptions<Partial<Field>> = {}): TicketFieldsCollection {
        options.displayName ??= `TicketFields`;
        const collection = new TicketFieldsCollection(this, collectionName, options);
        this._addCollection(collectionName, collection);
        return collection;
    }

    releaseCollection(collectionName: string) {
        this._removeCollection(collectionName);
    }

    get displayName(): string {
        return `Zendesk (${this.apiUrl.slice(0, -'/api/v2'.length)})`;
    }
}

export function getEndpoint(zendeskUrl: string, username: string, token: string, rejectUnauthorized: boolean = true): Endpoint {
    return new Endpoint(zendeskUrl, username, token, rejectUnauthorized);
}