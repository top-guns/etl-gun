import { describe, test, before, after } from 'node:test';
import assert from 'node:assert';
import * as rx from 'rxjs';
import * as etl from '../../../../../lib/index.js';

describe('FTP endpoint', () => {
    let ftp: etl.filesystems.Ftp.Endpoint;
    before(() => {
        ftp = new etl.filesystems.Ftp.Endpoint({host: "localhost", user: process.env.DOCKER_FTP_USER, password: process.env.DOCKER_FTP_PASSWORD});
    })

    after(async () => {
        if (ftp) await ftp.releaseEndpoint();
    })

    test('select()', async () => {
        const ROOT_FOLDER = 'ftp/dir-1';

        const folder = ftp.getFolder(ROOT_FOLDER);

        const values = await folder.select();
        const res: any[] = values.map(v => v.name);

        assert.deepStrictEqual(res, ['child-dir-1', 'file-1.txt', 'file-2.txt']);
    });

    test('selectGen()', async () => {
        const ROOT_FOLDER = 'ftp/dir-1';

        const folder = ftp.getFolder(ROOT_FOLDER);

        const res: any[] = [];

        for await (const v of folder.selectGen()) res.push(v.name);

        assert.deepStrictEqual(res, ['child-dir-1', 'file-1.txt', 'file-2.txt']);
    });

    test('selectIx()', async () => {
        const ROOT_FOLDER = 'ftp/dir-1';

        const folder = ftp.getFolder(ROOT_FOLDER);

        const res: any[] = [];

        await folder.selectIx().forEach(v => {
            res.push(v.name);
        })

        assert.deepStrictEqual(res, ['child-dir-1', 'file-1.txt', 'file-2.txt']);
    });

    test('selectRx()', async () => {
        const ROOT_FOLDER = 'ftp/dir-1';

        const folder = ftp.getFolder(ROOT_FOLDER);

        const res: any[] = [];

        let stream$ = folder.selectRx().pipe(
            rx.tap(v => res.push(v.name))
        );
        await etl.run(stream$);

        assert.deepStrictEqual(res, ['child-dir-1', 'file-1.txt', 'file-2.txt']);
    });

    test('selectStream()', async () => {
        const ROOT_FOLDER = 'ftp/dir-1';

        const folder = ftp.getFolder(ROOT_FOLDER);

        const res: any[] = [];

        const reader = folder.selectStream().getReader();
        let v;
        while(!(v = await reader.read()).done) res.push(v.value.name);

        assert.deepStrictEqual(res, ['child-dir-1', 'file-1.txt', 'file-2.txt']);
    });
});