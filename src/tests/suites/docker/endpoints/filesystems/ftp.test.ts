import { describe, test, before, after, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert';
import * as rx from 'rxjs';
import * as etl from '../../../../../lib/index.js';
import { strictNullish, strictUndefined } from '../../../../utils.js';

const ROOT_FOLDER = 'ftp';

describe('FTP endpoint', () => {
    let ftp: etl.filesystems.Ftp.Endpoint;
    let folder: etl.filesystems.Ftp.Collection;

    before(() => {
        ftp = new etl.filesystems.Ftp.Endpoint({host: "localhost", user: process.env.DOCKER_FTP_USER, password: process.env.DOCKER_FTP_PASSWORD});
    })

    after(async () => {
        if (ftp) await ftp.releaseEndpoint();
    })

    beforeEach(() => {
        folder = ftp.getFolder(ROOT_FOLDER);
    })

    afterEach(() => {
        ftp.releaseFolder(ROOT_FOLDER);
        folder = null;
    })

    test('selectOne() missing path', async () => {
        const value = await folder.selectOne('-------------------');
        strictUndefined(value);
    });

    test('selectOne() existing empty folder', async () => {
        const value = await folder.selectOne('dir-2');
        assert.deepStrictEqual(value.path, 'dir-2');
    });

    test('selectOne() existing not empty folder', async () => {
        const value = await folder.selectOne('dir-1/child-dir-1');
        assert.deepStrictEqual(value.path, 'dir-1/child-dir-1');
    });

    test('selectOne() existing file', async () => {
        const value = await folder.selectOne('dir-1/file-1.txt');
        assert.deepStrictEqual(value.path, 'dir-1/file-1.txt');
    });

    test('select()', async () => {
        const values = await folder.select('dir-1');
        const res: any[] = values.map(v => v.name);

        assert.deepStrictEqual(res, ['child-dir-1', 'file-1.txt', 'file-2.txt']);
    });

    test('selectGen()', async () => {
        const res: any[] = [];

        for await (const v of folder.selectGen('dir-1')) res.push(v.name);

        assert.deepStrictEqual(res, ['child-dir-1', 'file-1.txt', 'file-2.txt']);
    });

    test('selectIx()', async () => {
        const res: any[] = [];

        await folder.selectIx('dir-1').forEach(v => {
            res.push(v.name);
        })

        assert.deepStrictEqual(res, ['child-dir-1', 'file-1.txt', 'file-2.txt']);
    });

    test('selectRx()', async () => {
        const res: any[] = [];

        let stream$ = folder.selectRx('dir-1').pipe(
            rx.tap(v => res.push(v.name))
        );
        await etl.run(stream$);

        assert.deepStrictEqual(res, ['child-dir-1', 'file-1.txt', 'file-2.txt']);
    });

    test('selectStream()', async () => {
        const res: any[] = [];

        const reader = folder.selectStream('dir-1').getReader();
        let v;
        while(!(v = await reader.read()).done) res.push(v.value.name);

        assert.deepStrictEqual(res, ['child-dir-1', 'file-1.txt', 'file-2.txt']);
    });
});