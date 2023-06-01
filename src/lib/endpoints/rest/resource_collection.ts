import _ from 'lodash';
import { Subscriber } from 'rxjs';
import { CollectionOptions } from "../../core/base_collection.js";
import { BaseObservable } from "../../core/observable.js";
import { UpdatableCollection } from "../../core/updatable_collection.js";
import { RestEndpoint } from './endpoint.js';


export class RestResourceCollection<T> extends UpdatableCollection<T> {
    protected static instanceNo = 0;
    protected resourceNameS: string;
    protected resourceName: string;

    constructor(endpoint: RestEndpoint, collectionName: string, resourceName: string, resourceNameS: string, options: CollectionOptions<T> = {}) {
        RestResourceCollection.instanceNo++;
        super(endpoint, collectionName, options);
        this.resourceNameS = resourceNameS;
        this.resourceName = resourceName;
    }

    public select(where: Partial<T> = {}): BaseObservable<T>{
        const observable = new BaseObservable<T>(this, (subscriber) => {
            this._select(this._findWithoutEvent(where), subscriber);
        });
        return observable;
    }

    protected async _select(itemsPromise: Promise<T[]>, subscriber: Subscriber<T>) {
        try {
            this.sendStartEvent();
            const items = await itemsPromise;
            for (const obj of items) {
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
    }

    protected getResourceUrl(id: string): string {
        return `${this.resourceNameS}/${id}`;
    }

    protected getResourceListUrl(): string {
        return `${this.resourceNameS}`;
    }

    protected getSearchUrl(): string {
        return `${this.resourceNameS}`;
    }

    async get(id: string): Promise<T> {
        const res = await this._getWithoutEvent(id);
        this.sendGetEvent(res, id);
        return res;
    }

    async _getWithoutEvent(id: string): Promise<T> {
        const res = await this.endpoint.fetchJson(this.getResourceUrl(id));
        return res ? (typeof res[this.resourceName] === 'undefined' ? res : res[this.resourceName]) : null;
    }

    async find(where?: Partial<T>): Promise<T[]> {
        const res = await this._findWithoutEvent(where);
        this.sendListEvent(res, where);
        return res;
    }

    async _findWithoutEvent(where?: Partial<T>): Promise<T[]> {
        const res = await this.endpoint.fetchJson(this.getSearchUrl(), 'GET', { params: where });
        return res ? (typeof res[this.resourceNameS] === 'undefined' ? res : res[this.resourceNameS]) : null;
    }

    public async isExists(id: string): Promise<boolean> {
        const res = await this._getWithoutEvent(id);
        const exists = !!res;
        return exists;
    }

    protected async _insert(value: Partial<Omit<T, 'id'>> | any): Promise<any> {
        const body = {};
        body[this.resourceName] = value;
        return await this.endpoint.fetchJson(this.getResourceListUrl(), 'POST', body);
    }

    public async update(value: Partial<Omit<T, 'id'>>, id: string): Promise<void> {
        this.sendUpdateEvent(value, id);
        const body = {};
        body[this.resourceName] = value;
        await this.endpoint.fetchJson(this.getResourceUrl(id), 'PUT', body);
    }

    public async upsert(value: Partial<Omit<T, 'id'>>, id: string): Promise<boolean> {
        const exists = await this.isExists(id);
        if (exists) this.update(value, id);
        else this.insert(value);
        return exists;
    }

    public async delete(id: string): Promise<boolean> {
        const exists = await this.isExists(id);
        if (exists) await this.endpoint.fetchJson(this.getResourceUrl(id), 'DELETE');
        return exists;
    }

    get endpoint(): RestEndpoint {
        return super.endpoint as RestEndpoint;
    }
}
