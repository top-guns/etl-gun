import fetch, { RequestInit } from 'node-fetch';
import https from 'node:https';
import { BaseEndpoint} from "../../core/endpoint.js";
import { CollectionOptions } from '../../core/readonly_collection.js';
import { pathJoin } from '../../utils/index.js';
import { Ticket, TicketsCollection } from './ticket.js';
import { Field, TicketFieldsCollection } from './ticket_field.js';


export class Endpoint extends BaseEndpoint {
    protected apiUrl: string;
    protected username: string;
    protected token: string;
    protected agent: https.Agent;

    constructor(zendeskUrl: string, username: string, token: string, rejectUnauthorized: boolean = true) {
        super();

        this.apiUrl = zendeskUrl;
        if (!this.apiUrl.endsWith('/v2')) {
            if (!this.apiUrl.endsWith('/api')) this.apiUrl = pathJoin([this.apiUrl, 'api']);
            this.apiUrl = pathJoin([this.apiUrl, 'v2']);
        }

        this.username = username.endsWith('/token') ? username : pathJoin([username, 'token']);
        this.token = token;

        this.agent = rejectUnauthorized ? null : new https.Agent({
            rejectUnauthorized
        });
    }

    async fetchJson<T = any>(url: string, params: {} = {}, method: 'GET' | 'PUT' | 'POST' = 'GET', body?: any): Promise<T> {
        const init: RequestInit = {
            method,
            agent: this.agent,
            headers: {
                "Content-Type": "application/json",
                'Authorization': 'Basic ' + Buffer.from(this.username + ":" + this.token).toString('base64')
            }
        }
        if (body) init.body = JSON.stringify(body);

        const  getParams = new URLSearchParams(params).toString();

        let fullUrl = url.startsWith('http') ? url : pathJoin([this.apiUrl, url], '/');
        fullUrl += `${getParams ? '?' + getParams : ''}`;
        //console.log(fullUrl)
        const res = await fetch(fullUrl, init);
        //console.log(res)
        const jsonRes = await res.json();
        //console.log(jsonRes);
        //for (let key in jsonRes) console.log(key)
        return jsonRes as T;
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