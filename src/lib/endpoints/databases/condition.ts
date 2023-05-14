import _ from 'lodash';

type SqlExpression = {expression: string, params?: any[]};

type SqlConditionOperations = {
    '==': (val: any) => SqlExpression;
    '!=': (val: any) => SqlExpression;
    '>': (val: any) => SqlExpression;
    '<': (val: any) => SqlExpression;
    '>=': (val: any) => SqlExpression;
    '<=': (val: any) => SqlExpression;

    of: (arr: any[]) => SqlExpression;

    isNull: () => SqlExpression;
    isNotNull: () => SqlExpression;

    or: (...conditions: SqlCondition<any>[]) => SqlExpression;
    and: (...conditions: SqlCondition<any>[]) => SqlExpression;
}

const operations: SqlConditionOperations & { not: SqlConditionOperations } = {
    '==': (val: any) => ({ expression: '#{{field}} = ?', params: [val] }),
    '!=': (val: any) => ({ expression: '#{{field}} <> ?', params: [val] }),
    '>': (val: any) => ({ expression: '#{{field}} > ?', params: [val] }),
    '<': (val: any) => ({ expression: '#{{field}} < ?', params: [val] }),
    '>=': (val: any) => ({ expression: '#{{field}} >= ?', params: [val] }),
    '<=': (val: any) => ({ expression: '#{{field}} <= ?', params: [val] }),

    of: (arr: any[]) => ({
        expression: `#{{field}} in (${arr.reduce(p => p ? p + ', ?' : '?', '')})`,
        arr
    }),

    isNull: () => ({ expression: '#{{field}} is null '}),
    isNotNull: () => ({ expression: '#{{field}} is not null '}),

    or: (...conditions: SqlCondition<any>[]) => {
        let expression = '';
        let params = [];

        for (let condition of conditions) {
            let sqlExpr = conditionToSql<any>(condition);
            if (expression) expression += ` or `;
            expression += `(${sqlExpr.expression})`;
            if (sqlExpr.params) params.push(...sqlExpr.params);
        }
        return { expression, params };
    },

    and: (...conditions: SqlCondition<any>[]) => {
        let expression = '';
        let params = [];

        for (let condition of conditions) {
            let sqlExpr = conditionToSql<any>(condition);
            if (expression) expression += ` and `;
            expression += `(${sqlExpr.expression})`;
            if (sqlExpr.params) params.push(...sqlExpr.params);
        }
        return { expression, params };
    },

    not: null
}

if (!operations.not) {
    operations.not = {} as any;
    for (let key in operations) {
        if (!operations.hasOwnProperty(key)) continue;
        if (key === 'not') continue;
    
        operations.not[key] = (val: any) => {
            const sqlExpr = operations[key](val);
            sqlExpr.expression = `not (${sqlExpr.expression})`
            return sqlExpr;
        }
    }
}


export const sqlvalue: SqlConditionOperations & { not: SqlConditionOperations } = operations;


export type SqlCondition<T> = Record<string, any | ((value: any) => SqlExpression)>;


export function conditionToSql<T>(condition: SqlCondition<T>): SqlExpression {
    let expression = '';
    let params = [];

    for (const key in condition) {
        if (!condition.hasOwnProperty(key)) continue;

        if (expression) expression += ` and `;
        
        const cond: any = (condition as any)[key];

        if (cond === null) {
            expression += `${key} is null`;
            continue;
        }

        switch (typeof cond) {
            case 'string': 
            case 'boolean':
            case 'number': {
                expression += `${key} = ?`;
                params.push(cond);
                break;
            }
            case 'object': {
                if (Array.isArray(cond)) {
                    expression += `${key} in (${cond.reduce(p => p ? p + ', ?' : '?', '')})`;
                    params.push(...cond);
                    break;
                }

                expression += `(${cond.expression})`.replaceAll('#{{field}}', key);
                if (cond.params) params.push(...cond.params);
                break;
            }
            default:
                throw new Error(`Unexpected condition type: condition ${cond} has type ${typeof cond} but should has one of [string, number, boolean, null, array, {expression: string, params: any[]}]`)
        }
    }

    return {expression, params};
}