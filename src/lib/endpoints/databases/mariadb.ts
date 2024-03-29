import { ConnectionConfig, KnexEndpoint, PoolConfig } from './knex_endpoint.js';

export class Endpoint extends KnexEndpoint {
    constructor(connectionString: string, pool?: PoolConfig, driver?: 'mysql' | 'mysql2');
    constructor(connectionConfig: ConnectionConfig, pool?: PoolConfig, driver?: 'mysql' | 'mysql2');
    constructor(connection: any, pool?: PoolConfig, driver: 'mysql' | 'mysql2' = 'mysql') {
        super(driver, connection, pool);
    }

    get displayName(): string {
        if (typeof this.config.connection === 'string') return `MariaDB (${this.config.connection})`;

        const connection = this.config.connection as ConnectionConfig;
        if (typeof connection.host !== 'undefined') return `MariaDB (${connection.host}:${connection.port}/${connection.database})`;

        return `MariaDB (${this.instanceNo})`;
    }
}

export function getEndpoint(connectionString: string, pool?: PoolConfig): Endpoint;
export function getEndpoint(connectionConfig: ConnectionConfig, pool?: PoolConfig): Endpoint;
export function getEndpoint(connection: any, pool?: PoolConfig): Endpoint {
    return new Endpoint(connection as any, pool);
}

