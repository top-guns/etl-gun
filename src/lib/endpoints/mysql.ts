import Db, { DbConfig } from 'mysql2-async';
import parseDbUrl from "parse-database-url";
import { BaseEndpoint} from "../core/endpoint.js";
import { CollectionGuiOptions, BaseCollection } from "../core/collection.js";
import { EtlObservable } from '../core/observable.js';

class DbExt extends Db {
    close() {
        this.pool.end();
    }
}

interface canBeStringed {
    toString: () => string;
}
type BindParam = boolean | number | string | null | Date | Buffer | canBeStringed | BindObject;
interface BindObject {
    [keys: string]: BindParam;
}

function toBindObject(value: Record<string, any>): BindObject {
    const res: Record<string, string> = {};
    for (const key in value) {
        if (!value.hasOwnProperty(key)) continue;
        if (value[key] == null) {
            res[key] = null;
            continue;
        }
        switch (typeof value[key]) {
            case 'undefined':
                res[key] = null;
                break;
            case 'number':
                res[key] = value[key];
                break;
            case 'boolean':
                res[key] = value[key] ? 'true' : 'false';
                break;
            default:
                res[key] = '' + value[key];
                break;
        }
    }
    return res;
}

export class Endpoint extends BaseEndpoint {
    protected connectionString: string = null;
    protected _db: DbExt = null;
    get db(): DbExt {
        return this._db;
    }

    constructor(connectionString: string);
    constructor(connection: Db);
    constructor(connection: any) {
        super();
        if (typeof connection == 'string') {
            const config = this.parseConnectionString(connection);
            this._db = new DbExt(config);
            this.connectionString = connection;
        }
        else this._db = connection;
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
        this._db.close();
    }

    get displayName(): string {
        return this.connectionString ? `MySQL (${this.connectionString})` : `MySQL (${this.instanceNo})`;
    }

    protected parseConnectionString(connectionString: string): DbConfig {
        return parseDbUrl(connectionString);
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

    public select(where: string | {} = ''): EtlObservable<T> {
        const observable = new EtlObservable<T>((subscriber) => {
            (async () => {
                try {
                    this.sendStartEvent();
                    let params = [];
                    if (where && typeof where === 'object') {
                        where = this.getWhereAsString(where);
                        params = this.getCommaSeparatedValues(where);
                    }

                    const query = `select * from ${this.table} ${where ? 'where ' + where : ''}`;
                    const stream = this.endpoint.db.stream(query, params);

                    for await (const row of stream) {
                        await this.waitWhilePaused();
                        this.sendValueEvent(row);
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
        const query = `insert into ${this.table}(${this.getCommaSeparatedKeys(value)}) values (${this.getCommaSeparatedKeys(value, ':')})`;
        return await this.endpoint.db.insert(query, toBindObject(value));
    }

    public async update(where: string | {} = '', value: T) {
        super.update(where, value);
        const query = `update ${this.table} set :value where ${where ? (typeof where == 'string' ? where : ':where') : '1 = 1'}`;
        return await this.endpoint.db.update(query, {value: toBindObject(value), where: toBindObject(where as {})});
    }

    public async delete(where: string | {} = '') {
        super.delete(where);
        const query = `delete from ${this.table} where ${where ? (typeof where == 'string' ? where : ':where') : '1 = 1'}`;
        return await this.endpoint.db.delete(query, {where: toBindObject(where as {})});
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

    



    protected getCommaSeparatedKeys(value: T, withPrefix: string = '') {
        let res = '';
        for (let key in value) {
            if (value.hasOwnProperty(key)) {
                if (res) res += ", ";
                res += withPrefix + key;
            }
        }
        return res;
    }

    protected getSetAsString(value: T) {
        let res = '';
        for (let key in value) {
            if (value.hasOwnProperty(key)) {
                if (res) res += ", ";
                res += `${key} = :${key}`;
            }
        }
        return res;
    }

    protected getWhereAsString(where: {}) {
        let res = '';
        for (let key in where) {
            if (where.hasOwnProperty(key)) {
                if (res) res += " and ";
                res += `${key} = :w.${key}`;
            }
        }
        return res;
    }



    get endpoint(): Endpoint {
        return super.endpoint as Endpoint;
    }

}
