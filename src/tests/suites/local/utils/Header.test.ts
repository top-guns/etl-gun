import { describe, test } from 'node:test';
import { should, expect, assert } from "chai";
import * as rx from 'rxjs';
import * as etl from '../../../../lib/index.js';

describe('Header', () => {
    test('arrToObj', async () => {
        const header = new etl.Header(['f1', 'f2']);
        const arr = [1, 2];
        const res = header.arrToObj(arr);
        assert.deepStrictEqual(res, {f1: 1, f2: 2});
    });

    test('objToArr', async () => {
        const header = new etl.Header(['f1', 'f2']);
        const obj = {f1: 1, f2: 2};
        const res = header.objToArr(obj);
        assert.deepStrictEqual(res, [1, 2]);
    });
});