import * as rx from 'rxjs';
import * as etl from '../../lib';

describe('Header', () => {
    test('arrToObj', async () => {
        const header = new etl.Header('f1', 'f2');
        const arr = [1, 2];
        const res = header.arrToObj(arr);
        expect(res).toEqual({f1: 1, f2: 2});
    });

    test('objToArr', async () => {
        const header = new etl.Header('f1', 'f2');
        const obj = {f1: 1, f2: 2};
        const res = header.objToArr(obj);
        expect(res).toEqual([1, 2]);
    });
});