export class Header {
    protected fields: (string | Record<string, 'string'|'number'|'boolean'|'object'>)[];

    constructor(fields: (string | Record<string, 'string'|'number'|'boolean'|'object'>)[]);
    constructor(object: Record<string, any>);
    constructor(value: any) {
        if (typeof value.length !== undefined) this.fields = [...(value as any[])];
        else {
            for (let key in value) {
                if (!value.hasOwnProperty(key)) continue;
                const v = {};
                switch (typeof value[key]) {
                    case 'string': {
                        v[key] = 'string';
                        break;
                    }
                    case 'number': {
                        v[key] = 'number';
                        break;
                    }
                    case 'boolean': {
                        v[key] = 'boolean';
                        break;
                    }
                    default:
                        v[key] = 'object';
                        break;
                }
                this.fields.push(v);
            }
        }
    }

    arrToObj(values: any[]) {
        let res: Record<string, any> = {};
        for (let i = 0; i < values.length && i < this.fields.length; i++) {
            res[this.getFieldName(i)] = values[i];
        }
        return res;
    }

    objToArr(values: Record<string, any>) {
        let res: any[] = [];
        for (let i = 0; i < this.fields.length; i++) {
            const fieldName = this.getFieldName(i);
            const value = values.hasOwnProperty(fieldName) ? values[fieldName] : undefined;
            res.push(value);
        }
        return res;
    }

    getFieldName(i: number): string {
        if (typeof this.fields[i] === 'string') return this.fields[i] as string; 
        for (let key in this.fields[i] as any) {
            if (!this.fields[i].hasOwnProperty(key)) continue;
            return key; 
        }
    }

    protected getFieldNullValue(i: number): string {
        if (typeof this.fields[i] === 'string' || this.getFieldType(i) == 'string') return 'null'; 
        return typeof this.fields[i]['nullValue'] == 'undefined' ? 'null' : this.fields[i]['nullValue'];
    }

    protected getFieldUndefinedValue(i: number): string {
        if (typeof this.fields[i] === 'string' || this.getFieldType(i) == 'string') return 'undefined'; 
        return typeof this.fields[i]['undefinedValue'] == 'undefined' ? 'undefined' : this.fields[i]['undefinedValue'];
    }

    getFieldType(i: number): 'string' | 'number' | 'boolean' | 'object' {
        if (typeof this.fields[i] === 'string') return 'string'; 
        for (let key in this.fields[i] as any) {
            if (!this.fields[i].hasOwnProperty(key)) continue;
            return this.fields[i][key]; 
        }
    }

    valToStr(i: number, val: any): string {
        if (val === null) return this.getFieldNullValue(i);
        if (typeof val === 'undefined') return this.getFieldUndefinedValue(i);

        const fieldType = this.getFieldType(i);
        if (fieldType == 'string') return '' + val;
        if (fieldType == 'number') return val.toString();
        if (fieldType == 'boolean') return val ? 'true' : 'false';
        if (fieldType == 'object') return JSON.stringify(val);

        throw new Error(`Error: unknown type '${fieldType}'`)
    }

    strToVal(i: number, val: string): any {
        if (val == this.getFieldNullValue(i)) return null;
        if (val == this.getFieldUndefinedValue(i)) return undefined;

        const fieldType = this.getFieldType(i);
        if (fieldType == 'string') return val;
        if (fieldType == 'number') {
            if (!val) return null;
            let r = parseFloat(val); 
            if (r === NaN) throw new Error(`Error: value '${val}' cannot be converted to number`);
            return r;
        }
        if (fieldType == 'boolean') {
            if (!val) return null;
            if (val == 'true') return true;
            if (val == 'false') return false;
            throw new Error(`Error: value '${val}' cannot be converted to boolean`);
        }
        if (fieldType == 'object') {
            if (!val) return null;
            return JSON.parse(val);
        }

        throw new Error(`Error: unknown type '${fieldType}'`)
    }

    getFieldsCount() {
        return this.fields.length;
    }
}