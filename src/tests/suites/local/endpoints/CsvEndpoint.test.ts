import { describe, test } from 'node:test';
import { should, expect, assert } from "chai";
import * as rx from 'rxjs';
import * as etl from '../../../../lib/index.js';
import { deleteFileIfExists, getTempPath, loadFileContent } from '../../../../utils/filesystem.js';
import { Endpoint as CsvEndpoint } from '../../../../lib/endpoints/csv.js'

describe('CsvEndpoint', () => {
    test('push method', async () => {
        const OUT_FILE_NAME = getTempPath('test_output.csv');
        try {
            deleteFileIfExists(OUT_FILE_NAME);

            const endpoint = new CsvEndpoint();
            const src = endpoint.getFile(OUT_FILE_NAME);
            await src.insert([1, '1']);
            await src.insert([' a\\b/c;', '11,22']);

            const res = loadFileContent(OUT_FILE_NAME);
            assert.strictEqual(res, '"1","1"\n" a\\b/c;","11,22"\n');
        }
        finally {
            deleteFileIfExists(OUT_FILE_NAME);
        }
    });

    test('clear method', async () => {
        const OUT_FILE_NAME = getTempPath('test_output.csv');
        try {
            deleteFileIfExists(OUT_FILE_NAME);

            const endpoint = new CsvEndpoint();
            const src = endpoint.getFile(OUT_FILE_NAME);
            await src.insert([1, '1']);
            await src.insert([' a\\b/c;', '11,22']);

            await src.delete();

            assert.strictEqual(loadFileContent(OUT_FILE_NAME), '');
        }
        finally {
            deleteFileIfExists(OUT_FILE_NAME);
        }
    });

    test('read method', async () => {
        const OUT_FILE_NAME = getTempPath('test_output.csv');
        try {
            deleteFileIfExists(OUT_FILE_NAME);

            const endpoint = new CsvEndpoint();
            const src = endpoint.getFile(OUT_FILE_NAME);
            await src.insert([10, 'abc']);
            await src.insert(['11', ' a\\b/c;']);
            await src.insert(['33', '66,55']);

            const res: any[][] = [];
            let stream$ = src.selectRx().pipe(
                rx.tap(v => res.push(v))
            );
            await etl.run(stream$);

            assert.deepStrictEqual(res, [ [ '10', 'abc' ], [ '11', ' a\\b/c;' ], [ '33', '66,55' ] ]);
        }
        finally {
            deleteFileIfExists(OUT_FILE_NAME);
        }
    });
});