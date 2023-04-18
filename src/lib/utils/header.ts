type HeaderItemType = 'string'|'number'|'boolean'|'object'|string;

type HeaderItem = {
    [key: string]: HeaderItemType;
    nullValue?: string;
    undefinedValue?: string;
    trueValue?: string;
    falseValue?: string;
    name?: string;
    type?: HeaderItemType;
}

export class Header {
    protected fields: HeaderItem[] = [];

    constructor(fields: (string | HeaderItem)[]);
    constructor(object: Record<string, any>);
    constructor(value: any) {
        if (typeof value.length !== undefined) {
            for (let v of value) {
                if (typeof v == 'string') v = {name: v, type: 'string'};
                else {
                    if (!v.name || !v.type) {
                        let name = this.extractFieldName(v);
                        v = {...v, type: v[name], name};
                    }
                }
                this.fields.push(v);
            }
        }
        else {
            for (let key in value) {
                if (!value.hasOwnProperty(key)) continue;
                const v: HeaderItem = {};
                switch (typeof value[key]) {
                    case 'string':
                    case 'number': 
                    case 'boolean':
                        this.fields.push({name: key, type: typeof value[key]});
                        break;
                    default:
                        this.fields.push({name: key, type: 'object'});
                }
                
            }
        }
    }

    arrToObj(values: any[]) {
        let res: Record<string, any> = {};
        for (let i = 0; i < values.length && i < this.fields.length; i++) {
            res[this.fields[i].name] = values[i];
        }
        return res;
    }

    objToArr(values: Record<string, any>) {
        let res: any[] = [];
        for (let i = 0; i < this.fields.length; i++) {
            const fieldName = this.fields[i].name;
            const value = values.hasOwnProperty(fieldName) ? values[fieldName] : undefined;
            res.push(value);
        }
        return res;
    }

    protected extractFieldName(val: HeaderItem): string {
        //if (val.name && val.type) return val.name;
        for (let key in val) {
            if (!val.hasOwnProperty(key)) continue;
            if (['nullValue', 'undefinedValue', 'trueValue', 'falseValue'].includes(key)) continue;
            return key; 
        }
    }

    protected getFieldNullValue(i: number): string {
        return typeof this.fields[i].nullValue === 'undefined' ? 'null' : this.fields[i].nullValue;
    }

    protected getFieldUndefinedValue(i: number): string {
        return typeof this.fields[i].undefinedValue === 'undefined' ? 'undefined' : this.fields[i].undefinedValue;
    }

    valToStr(i: number, val: any): string {
        if (typeof val === 'undefined') return this.getFieldUndefinedValue(i);
        if (val === null) return this.getFieldNullValue(i);

        if (this.fields[i].type == 'string') return '' + val;
        if (this.fields[i].type == 'number') return val.toString();
        if (this.fields[i].type == 'boolean') return val ? 'true' : 'false';
        if (this.fields[i].type == 'object') return JSON.stringify(val);

        throw new Error(`Error: unknown type '${this.fields[i].type}'`)
    }

    strToVal(i: number, val: string): any {
        if (val == this.getFieldUndefinedValue(i)) return undefined;
        if (val == this.getFieldNullValue(i)) return null;

        if (this.fields[i].type == 'string') return val;
        if (this.fields[i].type == 'number') {
            if (val === null) return null;
            if (typeof val === 'undefined' || val === '') return undefined;
            let r = parseFloat(val); 
            if (isNaN(r)) throw new Error(`Error: value '${val}' cannot be converted to number`);
            return r;
        }
        if (this.fields[i].type == 'boolean') {
            if (typeof this.fields[i].trueValue !== 'undefined' && val == this.fields[i].trueValue) return true;
            if (typeof this.fields[i].falseValue !== 'undefined' && val == this.fields[i].falseValue) return false;
            if (typeof this.fields[i].trueValue !== 'undefined' && typeof this.fields[i].falseValue !== 'undefined') throw new Error(`Error: unknown boolean value '${val}'`);
            if (typeof this.fields[i].trueValue !== 'undefined') return false;
            if (typeof this.fields[i].falseValue !== 'undefined') return true;

            if (val === null) return null;
            if (typeof val === 'undefined' || val === '') return undefined;
            if (val == 'true') return true;
            if (val == 'false') return false;

            console.log('this.fields[i] = ', this.fields[i]);
            throw new Error(`Error: value '${val}' cannot be converted to boolean`);
        }
        if (this.fields[i].type == 'object') {
            if (val === null) return null;
            if (typeof val === 'undefined' || val === '') return undefined;
            return JSON.parse(val);
        }

        throw new Error(`Error: unknown type '${this.fields[i].type}'`);
    }

    getFieldsCount() {
        return this.fields.length;
    }
}