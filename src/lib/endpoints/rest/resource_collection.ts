import * as ix from 'ix';
import _ from 'lodash';
import { CollectionOptions } from "../../core/base_collection.js";
import { BaseObservable } from "../../core/observable.js";
import { UpdatableCollection } from "../../core/updatable_collection.js";
import { RestEndpoint } from './endpoint.js';
import { generator2Iterable, observable2Stream, promise2Generator, promise2Observable, selectOne_from_Promise, wrapGenerator, wrapObservable } from '../../utils/flows.js';


export class ResourceCollection<T> extends UpdatableCollection<T> {
    protected static instanceNo = 0;
    protected resourceNameS: string;
    protected resourceName: string;

    constructor(endpoint: RestEndpoint, collectionName: string, resourceName: string, resourceNameS: string, options: CollectionOptions<T> = {}) {
        ResourceCollection.instanceNo++;
        super(endpoint, collectionName, options);
        this.resourceNameS = resourceNameS;
        this.resourceName = resourceName;
    }

    protected async _select(where?: any): Promise<T[]> {
        const params: Partial<T> = where;
        const res = await this.endpoint.fetchJson(this.getSearchUrl(), 'GET', { params });
        if (!res) return null;
        if (typeof res[this.resourceNameS] !== 'undefined' && Array.isArray(res[this.resourceNameS])) return res[this.resourceNameS];
        if (typeof res['items'] !== 'undefined' && Array.isArray(res['items'])) return res['items'];
        if (typeof res[this.resourceName] === 'object') return [res[this.resourceName]];
        return [res];
    }

    public async select(where?: any): Promise<T[]> {
        const values = await this._select(where);
        this.sendSelectEvent(values);
        return values;
    }

    public async* selectGen(where?: any): AsyncGenerator<T, void, void> {
        const values = this._select(where);
        const generator = wrapGenerator(promise2Generator(values), this);
        for await (const value of generator) yield value;
    }

    public selectRx(where?: any): BaseObservable<T> {
        const values = this._select(where);
        return wrapObservable(promise2Observable(values), this);
    }

    public selectIx(where?: any): ix.AsyncIterable<T> {
        return generator2Iterable(this.selectGen(where));
    }

    public selectStream(where?: any): ReadableStream<T> {
        return observable2Stream(this.selectRx(where));
    }

    protected async _selectOne(id: string | number): Promise<T> {
        const res = await this.endpoint.fetchJson(this.getResourceUrl(encodeURIComponent(id.toString())));
        const result = res ? (typeof res[this.resourceName] === 'undefined' ? res : res[this.resourceName]) : null;
        return result;
    }

    public async selectOne(id: string | number): Promise<T> {
        const result = await this._selectOne(id);
        this.sendSelectOneEvent(result);
        return result;
    }

    protected getResourceUrl(id: string): string {
        const url = `${this.resourceNameS}/${id}`;
        return url;
    }

    protected getResourceListUrl(): string {
        return `${this.resourceNameS}`;
    }

    protected getSearchUrl(): string {
        return `${this.resourceNameS}`;
    }

    public async isExists(id: string): Promise<boolean> {
        const res = await this._selectOne(id);
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
