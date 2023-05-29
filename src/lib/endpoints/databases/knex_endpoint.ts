//import { Knex, knex } from 'knex';
//import parseDbUrl from "parse-database-url";
import pkg from 'knex';

import { BaseEndpoint} from "../../core/endpoint.js";
import { BaseObservable } from '../../core/observable.js';
import { conditionToSql, SqlCondition } from './condition.js';
import { UpdatableCollection } from '../../core/updatable_collection.js';
import { BaseCollection, CollectionOptions } from '../../core/base_collection.js';


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

    public select(where: SqlCondition<T>, fields?: string[]): BaseObservable<T>;
    public select(whereSql?: string, whereParams?: any[], fields?: string[]): BaseObservable<T>;
    public select(where: string | SqlCondition<T> = '', params?: any[], fields: string[] = []): BaseObservable<T> {
        const observable = new BaseObservable<T>(this, (subscriber) => {
            (async () => {
                try {
                    this.sendStartEvent();
                    
                    let result: any[];
                    if (typeof where === 'string') {
                        result = await this.endpoint.database(this.table)
                            .whereRaw(where, params)
                            .select(...fields);
                    }
                    else {
                        result = await this.endpoint.database(this.table)
                            .whereRaw(where.expression, where.params)
                            .select(...fields);
                            // .where(where)
                            // .select(...params);
                    } 

                    for (const row of result) {
                        if (subscriber.closed) break;
                        await this.waitWhilePaused();
                        this.sendReciveEvent(row);
                        subscriber.next(row);
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

    public async get(where: SqlCondition<T>): Promise<any>;
    public async get(whereSql?: string, whereParams?: any[]): Promise<any>;
    public async get(where: string | SqlCondition<T> = '', params?: any[]): Promise<any> {
        const results = await this._find(where, params);
        const result = results.length > 0 ? results[0] : undefined;
        this.sendGetEvent(result, where, params);
        return result;
    }

    public async find(where: SqlCondition<T>, fields?: string[]): Promise<any[]>;
    public async find(whereSql?: string, whereParams?: any[], fields?: string[]): Promise<any[]>;
    public async find(where: string | SqlCondition<T> = '', params?: any[], fields: string[] = []): Promise<any[]> {
        const result = await this._find(where, params, fields);
        this.sendListEvent(result, where, params);
        return result;
    }

    public async _find(where: string | SqlCondition<T> = '', params?: any[], fields: string[] = []): Promise<any[]> {
        let result: any[];
        if (typeof where === 'string') {
            result = await this.endpoint.database(this.table)
                .whereRaw(where, params)
                .select(...fields);
        }
        else {
            result = await this.endpoint.database(this.table)
                .whereRaw(where.expression, where.params)
                .select(...fields);
                // .where(where)
                // .select(...params);
        } 
        return result;
    }

    public async insert(value: T): Promise<void>;
    public async insert(values: T[]): Promise<void>;
    public async insert(value: T | T[]): Promise<void>{
        this.sendInsertEvent(value);
        await this.endpoint.database(this.table).insert(value);
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

    public select(params?: any[]): BaseObservable<T> {
        const observable = new BaseObservable<T>(this, (subscriber) => {
            (async () => {
                try {
                    this.sendStartEvent();

                    const myArray = [1,2,3]
                    this.endpoint.database.raw('select * from users where id in (' + myArray.map(_ => '?').join(',') + ')', [...myArray]);


                    const result: any[] = await this.endpoint.database.raw('select * from users where id = ?', params);
                    
                    for (const row of result) {
                        if (subscriber.closed) break;
                        await this.waitWhilePaused();
                        this.sendReciveEvent(row);
                        subscriber.next(row);
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

    get endpoint(): KnexEndpoint {
        return super.endpoint as KnexEndpoint;
    }
}

