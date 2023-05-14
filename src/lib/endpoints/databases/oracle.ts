import { ConnectionConfig, KnexEndpoint, PoolConfig } from './knex_endpoint.js';

export class Endpoint extends KnexEndpoint {
    constructor(connectionString: string, pool?: PoolConfig);
    constructor(connectionConfig: ConnectionConfig, pool?: PoolConfig);
    constructor(connection: any, pool?: PoolConfig) {
        super('oracledb', connection, pool);
    }

    get displayName(): string {
        if (typeof this.config.connection === 'string') return `Oracle DB (${this.config.connection})`;

        const connection = this.config.connection as ConnectionConfig;
        if (typeof connection.host !== 'undefined') return `Oracle DB (${connection.host}:${connection.port}/${connection.database})`;
        
        return `Oracle DB (${this.instanceNo})`;
    }
}

export function getEndpoint(connectionString: string, pool?: PoolConfig): Endpoint;
export function getEndpoint(connectionConfig: ConnectionConfig, pool?: PoolConfig): Endpoint;
export function getEndpoint(connection: any, pool?: PoolConfig): Endpoint {
    return new Endpoint(connection as any, pool);
}

