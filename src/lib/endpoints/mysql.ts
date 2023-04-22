import Db, { DbConfig } from 'mysql2-async';
import parseDbUrl from "parse-database-url";
import { BaseEndpoint} from "../core/endpoint.js";
import { CollectionGuiOptions, BaseCollection } from "../core/collection.js";
import { Observable } from 'rxjs';

class DbExt extends Db {
    async close() {
        await this.wait();
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

        if (value[key] === null) {
            res[key] = null;
            continue;
        }

        switch (typeof value[key]) {
            case 'undefined':
                break;
            case 'number':
            case 'bigint':
                res[key] = value[key];
                break;
            case 'boolean':
                res[key] = value[key] ? 'true' : 'false';
                break;
            case 'string':
            case 'symbol':
                res[key] = '' + value[key];
                break;
            case 'object':
                res[key] = value[key].toString();
                break;
            default:
                throw new Error(`Error in convertion of properti '${key}' value '${value[key]}': type '${typeof value[key]}' cannot be converted to database field value`);
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

    getTable<T = Record<string, any>>(table: string, guiOptions: CollectionGuiOptions<string[]> = {}): TableCollection<T> {
        guiOptions.displayName ??= `${table}`;
        const c = this._addCollection(table, new TableCollection(this, table, guiOptions));
        return c as unknown as TableCollection<T>;
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

    get type(): string {
        return 'Mysql.TableCollection';
    }

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
                    const stream = this.endpoint.db.stream(query, params);

                    for await (const row of stream) {
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

    public async insert(value: T) {
        super.insert(value);
        const val = toBindObject(value);
        const query = `insert into ${this.table}(${this.getCommaSeparatedKeys(val)}) values (${this.getCommaSeparatedKeys(val, ':')})`;
        return await this.endpoint.db.insert(query, toBindObject(val));
    }

    public async update(where: string | {} = '', value: T) {
        super.update(where, value);
        const query = `update ${this.table} set :value where ${where ? (typeof where == 'string' ? where : ':where') : '1 = 1'}`;
        return await this.endpoint.db.update(query, {value: toBindObject(value), where: toBindObject(where as {})});
    }

    public async upsert(value: T) {
        super.upsert(value);

        const val = toBindObject(value);
        const query = `
            insert into ${this.table}(${this.getCommaSeparatedKeys(val)}) 
            values (${this.getCommaSeparatedKeys(val, ':')})
            on duplicate key update :value_with_all_object_
        `;

        return await this.endpoint.db.execute(query, {...val, value_with_all_object_: val});
    }

    public async delete(where: string | {} = '') {
        super.delete(where);
        const query = `delete from ${this.table} where ${where ? (typeof where == 'string' ? where : ':where') : '1 = 1'}`;
        return await this.endpoint.db.delete(query, {where: toBindObject(where as {})});
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

    protected getCommaSeparatedKeys(value: BindObject, withPrefix: string = '') {
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
