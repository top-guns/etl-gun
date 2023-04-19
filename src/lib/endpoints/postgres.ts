import * as pg from 'pg'
import { Observable } from "rxjs";
import { GuiManager } from '../core/gui.js';
import { BaseEndpoint} from "../core/endpoint.js";
import { BaseCollection, CollectionGuiOptions } from "../core/collection.js";

export class Endpoint extends BaseEndpoint {
    protected connectionString: string = null;
    protected _connectionPool: pg.Pool = null;
    get connectionPool(): pg.Pool {
        return this._connectionPool;
    }

    constructor(connectionString: string);
    constructor(connectionPool: pg.Pool);
    constructor(connection: any) {
        super();
        if (typeof connection == 'string') {
            const config = { connectionString: connection };
            this._connectionPool = new pg.Pool(config);
            this.connectionString = connection;
        }
        else this._connectionPool = connection;
    }

    getTable(table: string, guiOptions: CollectionGuiOptions<string[]> = {}): TableCollection {
        guiOptions.displayName ??= `${table}`;
        return this._addCollection(table, new TableCollection(this, table, guiOptions));
    }

    releaseTable(table: string) {
        this._removeCollection(table);
    }

    releaseEndpoint() {
        super.releaseEndpoint();
        this._connectionPool.end();
    }

    get displayName(): string {
        return this.connectionString ? `PostgreSQL (${this.connectionString})` : `PostgreSQL (${this.instanceNo})`;
    }
}

export class TableCollection<T = Record<string, any>> extends BaseCollection<T> {
    protected static instanceNo = 0;
    protected table: string;

    constructor(endpoint: Endpoint, table: string, guiOptions: CollectionGuiOptions<T> = {}) {
        TableCollection.instanceNo++;
        super(endpoint, guiOptions);
        this.table = table;
    }

    public select(where: string | {} = ''): Observable<T> {
        const observable = new Observable<T>((subscriber) => {
            (async () => {
                try {
                    this.sendStartEvent();
                    let params = [];
                    if (where && typeof where === 'object') {
                        where = this.getWhereAsString(where);
                        params = this.getCommaSeparatedValues(where);
                    }

                    const query = `select * from ${this.table} ${where ? 'where ' + where : ''}`;
                    const results = await this.endpoint.connectionPool.query(query, params);
                    for (const row of results.rows) {
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

    public async insert(value: T) {
        super.insert(value);
        const query = `insert into ${this.table}(${this.getCommaSeparatedFields(value)}) values ${this.getCommaSeparatedParams(value)}`;
        await this.endpoint.connectionPool.query(query, this.getCommaSeparatedValues(value));
        // TODO: returning id or the whole T-value
    }

    public async delete(where: string | {} = '') {
        super.delete();
        let params = [];
        if (where && typeof where === 'object') {
            where = this.getWhereAsString(where);
            params = this.getCommaSeparatedValues(where);
        }

        const query = `delete from ${this.table} ${where ? 'where ' + where : ''}`;
        await this.endpoint.connectionPool.query(query, params);
    }

    protected getCommaSeparatedFields(value: T) {
        let res = '';
        for (let key in value) {
            if (value.hasOwnProperty(key)) {
                if (res) res += ", ";
                res += key;
            }
        }
        return res;
    }

    protected getCommaSeparatedParams(value: T) {
        let res = '', n = 1;
        for (let key in value) {
            if (value.hasOwnProperty(key)) {
                if (res) res += ", ";
                res += "$" + n;
                n++;
            }
        }
        return res;
    }

    protected getCommaSeparatedValues(value: {}) {
        let res = [];
        for (let key in value) {
            if (value.hasOwnProperty(key)) {
                res.push(value[key]);
            }
        }
        return res;
    }

    protected getWhereAsString(where: {}) {
        let res = '', n = 1;
        for (let key in where) {
            if (where.hasOwnProperty(key)) {
                if (res) res += " and ";
                res += key + " = $" + n;
                n++;
            }
        }
        return res;
    }

    get endpoint(): Endpoint {
        return super.endpoint as Endpoint;
    }

}
