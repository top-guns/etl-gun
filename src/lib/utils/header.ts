export class Header {
    protected fieldNames: string[];

    constructor(fieldNames: string[]) {
        this.fieldNames = fieldNames;
    }

    arrToObj(values: any[]) {
        let res: Record<string, any> = {};
        for (let i = 0; i < values.length && i < this.fieldNames.length; i++) {
            res[this.fieldNames[i]] = values[i];
        }
        return res;
    }

    objToArr(values: Record<string, any>) {
        let res: any[] = [];
        for (let i = 0; i < this.fieldNames.length; i++) {
            const fieldName = this.fieldNames[i];
            const value = values.hasOwnProperty(fieldName) ? values[fieldName] : undefined;
            res.push(value);
        }
        return res;
    }

}