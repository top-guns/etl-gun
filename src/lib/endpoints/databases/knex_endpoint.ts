//import { Knex, knex } from 'knex';
//import parseDbUrl from "parse-database-url";
import * as pkg from 'knex';
import * as ix from 'ix';

import { BaseEndpoint} from "../../core/endpoint.js";
import { BaseObservable } from '../../core/observable.js';
import { conditionToSql, SqlCondition } from './condition.js';
import { UpdatableCollection } from '../../core/updatable_collection.js';
import { BaseCollection, CollectionOptions } from '../../core/base_collection.js';
import { generator2Iterable, promise2Generator, promise2Observable, observable2Stream, selectOne_from_Promise, wrapObservable, wrapGenerator } from '../../utils/flows.js';


type ClientType = 'pg'  // pg for PostgreSQL, CockroachDB and Amazon Redshift
                        // pg-native for PostgreSQL with native C++ libpq bindings (requires PostgresSQL installed to link against)
    | 'mssql'           // tedious for MSSQL
    | 'mysql'           // mysql for MySQL or MariaDB
    | 'mysql2'           // mysql for MySQL or MariaDB
    | 'oracledb' 
    | 'sqlite3'         // sqlite3 for SQLite3
    | 'better-sqlite3'


export interface ConnectionConfig {
    host?: string;
    port?: number;
    user?: string;
    password?: string;
    database?: string;

    filename?: ":memory:" | string;
}

export interface PoolConfig {
    min?: number;
    max?: number;
}


export class KnexEndpoint extends BaseEndpoint {
    protected config: pkg.Knex.Config;
    protected _database: pkg.Knex<any, unknown[]>;
    get database(): pkg.Knex<any, unknown[]> {
        return this._database;
    }

    constructor(client: ClientType, connectionString: string, pool?: PoolConfig);
    constructor(client: ClientType, connectionConfig: ConnectionConfig, pool?: PoolConfig);
    constructor(knexConfig: pkg.Knex.Config);
    constructor(p1: any, connection?: string | ConnectionConfig, pool?: PoolConfig) {
        super();

        if (typeof p1 == 'string') {
            this.config = {
                client: p1,
                connection: connection,
                pool: pool ? pool : undefined
            }
        }
        else {
            this.config = p1;
        }

        this._database = pkg.knex(this.config);
    }

    getTable<T = Record<string, any>>(table: string, options: CollectionOptions<string[]> = {}): KnexTableCollection<T> {
        options.displayName ??= `${table}`;
        const c = this._addCollection(table, new KnexTableCollection(this, table, table, options));
        return c as unknown as KnexTableCollection<T>;
    }

    getQuery<T = Record<string, any>>(collectionName: string, sqlQuery: string, options: CollectionOptions<string[]> = {}): KnexQueryCollection<T> {
        options.displayName ??= `${collectionName}`;
        const c = this._addCollection(collectionName, new KnexQueryCollection(this, collectionName, sqlQuery, options));
        return c as unknown as KnexQueryCollection<T>;
    }

    releaseCollection(collectionName: string) {
        this._removeCollection(collectionName);
    }

    async releaseEndpoint() {
        await super.releaseEndpoint();
        await this.database.destroy();
    }

    get displayName(): string {
        if (typeof this.config.connection === 'string') return `${this.config.client} (${this.config.connection})`;
        if (typeof this.config.connection === 'function') return `${this.config.client} (${this.instanceNo})`;
        if (typeof this.config.connection === 'undefined') return `${this.config.client} (${this.instanceNo})`;

        const connection = this.config.connection as ConnectionConfig;

        if (typeof connection.host !== 'undefined') return `${this.config.client} (${connection.host}:${connection.port}/${connection.database})`;
        if (typeof connection.filename !== 'undefined') return `${this.config.client} (${connection.filename})`;

        return `${this.config.client} (${this.instanceNo})`;
    }
}

export function getEndpoint(client: ClientType, connectionString: string, pool?: PoolConfig): KnexEndpoint;
export function getEndpoint(client: ClientType, connectionConfig: ConnectionConfig, pool?: PoolConfig): KnexEndpoint;
export function getEndpoint(knexConfig: pkg.Knex.Config): KnexEndpoint;
export function getEndpoint(p1: any, connection?: string | ConnectionConfig, pool?: PoolConfig): KnexEndpoint {
    return new KnexEndpoint(p1, connection as any, pool);
}


export class KnexTableCollection<T = Record<string, any>> extends UpdatableCollection<T> {
    protected static instanceNo = 0;

    protected table: string;

    constructor(endpoint: KnexEndpoint, collectionName: string, table: string, options: CollectionOptions<T> = {}) {
        KnexTableCollection.instanceNo++;
        super(endpoint, collectionName, options);
        this.table = table;
    }

    protected async _select(p1?: SqlCondition<T> | string, p2?: string[] | any[], fields?: string[]): Promise<T[]> {
        const where: SqlCondition<T> | undefined = typeof p1 === 'string' ? undefined : p1;
        const whereSql: string | undefined = typeof p1 === 'string' ? p1 : undefined;
        const whereParams: any[] | undefined = p2;
        if (typeof p1 !== 'string') fields = p2;

        let result: any[];
        if (whereSql) {
            result = await this.endpoint.database(this.table)
                .whereRaw(whereSql, whereParams)
                .select(...fields!);
        }
        else {
            result = await this.endpoint.database(this.table)
                .whereRaw(where!.expression, whereParams)
                .select(...fields!);
                // .where(where)
                // .select(...params);
        } 
        return result;
    }

    public async select(where: SqlCondition<T>, fields?: string[]): Promise<T[]>;
    public async select(whereSql?: string, whereParams?: any[], fields?: string[]): Promise<T[]>;
    public async select(p1?: any, p2?: any, fields?: string[]): Promise<T[]> {
        let result: any[] = await this._select(p1, p2, fields);
        this.sendSelectEvent(result);
        return result;
    }

    public selectGen(where: SqlCondition<T>, fields?: string[]): AsyncGenerator<T, void, void>;
    public selectGen(whereSql?: string, whereParams?: any[], fields?: string[]): AsyncGenerator<T, void, void>;
    public async* selectGen(p1?: any, p2?: any, fields?: string[]): AsyncGenerator<T, void, void> {
        const values = this._select(p1, p2, fields);
        const generator = wrapGenerator(promise2Generator(values), this);
        for await (const value of generator) yield value;
    }

    public selectRx(where: SqlCondition<T>, fields?: string[]): BaseObservable<T>;
    public selectRx(whereSql?: string, whereParams?: any[], fields?: string[]): BaseObservable<T>;
    public selectRx(p1?: any, p2?: any, fields?: string[]): BaseObservable<T> {
        const values = this._select(p1, p2, fields);
        return wrapObservable(promise2Observable(values), this);
    }

    public async selectOne(where: SqlCondition<T>, fields?: string[]): Promise<T | null>;
    public async selectOne(whereSql?: string, whereParams?: any[], fields?: string[]): Promise<T | null>;
    public async selectOne(p1?: any, p2?: any, fields?: string[]): Promise<T | null> {
        const values = this._select(p1, p2, fields);
        const value = await selectOne_from_Promise(values);
        this.sendSelectOneEvent(value);
        return value;
    }

    public selectIx(where: SqlCondition<T>, fields?: string[]): ix.AsyncIterable<T>;
    public selectIx(whereSql?: string, whereParams?: any[], fields?: string[]): ix.AsyncIterable<T>;
    public selectIx(p1?: any, p2?: any, fields?: string[]): ix.AsyncIterable<T> {
        return generator2Iterable(this.selectGen(p1, p2, fields));
    }

    public selectStream(where: SqlCondition<T>, fields?: string[]): ReadableStream<T>;
    public selectStream(whereSql?: string, whereParams?: any[], fields?: string[]): ReadableStream<T>;
    public selectStream(p1?: any, p2?: any, fields?: string[]): ReadableStream<T> {
        return observable2Stream(this.selectRx(p1, p2, fields));
    }

    protected async _insert(value: T): Promise<void>;
    protected async _insert(values: T[]): Promise<void>;
    protected async _insert(value: T | T[]): Promise<void>{
        await this.endpoint.database(this.table).insert(value);
    }

    public async insert(value: T): Promise<void>;
    public async insert(values: T[]): Promise<void>;
    public async insert(value: T | T[]): Promise<void>{
        this.sendInsertEvent(value);
        await this._insert(value as any);
    }

    public async update(value: T, where: SqlCondition<T>): Promise<void>;
    public async update(value: T, whereSql?: string, whereParams?: any[]): Promise<void>;
    public async update(value: T, where: string | SqlCondition<T> = '', whereParams?: any[]): Promise<void> {
        this.sendUpdateEvent(value, where);

        if (typeof where === 'string') {
            await this.endpoint.database(this.table)
                .whereRaw(where, whereParams)
                .update(value);
        }
        else {
            const whereExpr = conditionToSql(where);
            console.log(whereExpr);
            
            await this.endpoint.database(this.table)
                .whereRaw(whereExpr.expression, whereExpr.params)
                .update(value);
        } 
    }

    public async upsert(value: T): Promise<boolean> {
        const res = await this.endpoint.database(this.table).upsert(value);
        if (res.length > 0) this.sendInsertEvent(value);
        else this.sendUpdateEvent(value, {});
        return res.length > 0;
    }

    public async delete(where: SqlCondition<T>): Promise<boolean>;
    public async delete(whereSql?: string, whereParams?: any[]): Promise<boolean>;
    public async delete(where: string | SqlCondition<T> = '', whereParams?: any[]): Promise<boolean> {
        this.sendDeleteEvent(where, whereParams);

        if (typeof where === 'string') {
            return await this.endpoint.database(this.table)
                .whereRaw(where, whereParams)
                .del();
        }
        else {
            const whereExpr = conditionToSql(where);
            return await this.endpoint.database(this.table)
                .whereRaw(whereExpr.expression, whereExpr.params)
                .del();
        } 
    }

    get endpoint(): KnexEndpoint {
        return super.endpoint as KnexEndpoint;
    }

}


export class KnexQueryCollection<T = Record<string, any>> extends BaseCollection<T> {
    protected static instanceNo = 0;

    protected query: string;

    constructor(endpoint: KnexEndpoint, collectionName: string, sqlQuery: string, options: CollectionOptions<T> = {}) {
        KnexQueryCollection.instanceNo++;
        super(endpoint, collectionName, options);
        this.query = sqlQuery;
    }

    protected async _select(params?: any[]): Promise<T[]> {
        const result = await this.endpoint.database.raw(this.query, ...params!);
        return result.rows;
    }

    public async select(params?: any[]): Promise<T[]> {
        const values = await this._select(params);
        this.sendSelectEvent(values);
        return values;
    }

    public async* selectGen(params?: any[]): AsyncGenerator<T, void, void> {
        const values = this._select(params);
        const generator = wrapGenerator(promise2Generator(values), this);
        for await (const value of generator) yield value;
    }

    public selectRx(params?: any[]): BaseObservable<T> {
        const values = this._select(params);
        return wrapObservable(promise2Observable(values), this);
    }

    public selectIx(params?: any[]): ix.AsyncIterable<T> {
        return generator2Iterable(this.selectGen(params));
    }

    public selectStream(params?: any[]): ReadableStream<T> {
        return observable2Stream(this.selectRx(params));
    }

    public async selectOne(params?: any[]): Promise<T | null> {
        const values = this._select(params);
        const value = await selectOne_from_Promise(values);
        this.sendSelectOneEvent(value);
        return value;
    }
    
    get endpoint(): KnexEndpoint {
        return super.endpoint as KnexEndpoint;
    }
}

