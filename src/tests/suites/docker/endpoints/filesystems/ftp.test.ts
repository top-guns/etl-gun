import { describe, test, before, after } from 'node:test';
import assert from 'node:assert';
import * as rx from 'rxjs';
import * as etl from '../../../../../lib/index.js';

describe('FTP endpoint', () => {
    let ftp: etl.filesystems.Ftp.Endpoint;
    before(() => {
        ftp = new etl.filesystems.Ftp.Endpoint({host: process.env.FTP_HOST, user: process.env.FTP_USER_NAME, password: process.env.FTP_USER_PASS});
    })

    after(async () => {
        if (ftp) await ftp.releaseEndpoint();
    })
    
    test('list folder', async () => {
        const ROOT_FOLDER = 'ftp/dir-1';

        const folder = ftp.getFolder(ROOT_FOLDER);

        const res: any[] = [];

        let stream$ = folder.select().pipe(
            rx.tap(v => res.push(v.name))
        );
        await etl.run(stream$);

        assert.deepStrictEqual(res, ['child-dir-1', 'file-1.txt', 'file-2.txt']);
    });
});