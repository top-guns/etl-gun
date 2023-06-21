import { describe, test } from 'node:test';
import { should, expect, assert } from "chai";
import path from 'path';
import * as fs from "fs";
import * as rx from 'rxjs';
import * as etl from '../../../../lib/index.js';
import { deleteFileIfExists, loadFileContent } from '../../../../utils/filesystem.js';

const TEMP_FOLDER = "./src/tests/tmp/";

describe('Etl.filesystems.Local.Endpoint', () => {
    test('push method with simple parameters to create file', async () => {
        const OUT_FILE_NAME = 'test_output.tmp';
        const ROOT_FOLDER = TEMP_FOLDER;
        const OUT_FILE_FULL_PATH = path.join(ROOT_FOLDER, OUT_FILE_NAME);
        try {
            deleteFileIfExists(OUT_FILE_FULL_PATH);

            const ep = new etl.filesystems.Local.Endpoint(ROOT_FOLDER);
            const src = ep.getFolder('.');
            await src.insert(OUT_FILE_NAME, 'test');

            const res = loadFileContent(OUT_FILE_FULL_PATH);
            assert.strictEqual(res, 'test');
        }
        finally {
            deleteFileIfExists(OUT_FILE_FULL_PATH);
        }
    });

    test('push method with simple parameters to create folder', async () => {
        const OUT_FOLDER_NAME = 'test_output';
        const ROOT_FOLDER = TEMP_FOLDER;
        const OUT_FOLDER_FULL_PATH = path.join(ROOT_FOLDER, OUT_FOLDER_NAME);
        try {
            deleteFileIfExists(OUT_FOLDER_FULL_PATH);

            const ep = new etl.filesystems.Local.Endpoint(ROOT_FOLDER);
            const src = ep.getFolder('.');
            await src.insert(OUT_FOLDER_NAME);

            const res = fs.existsSync(OUT_FOLDER_FULL_PATH);
            assert.strictEqual(res, true);
        }
        finally {
            deleteFileIfExists(OUT_FOLDER_FULL_PATH);
        }
    });

    test('clear method', async () => {
        const ROOT_FOLDER = TEMP_FOLDER + "temp_out_dir";
        const OUT_FILE_NAME = 'test_output.tmp';
        const OUT_FILE_FULL_PATH = path.join(ROOT_FOLDER, OUT_FILE_NAME);
        try {
            deleteFileIfExists(ROOT_FOLDER);

            const ep = new etl.filesystems.Local.Endpoint(ROOT_FOLDER);
            const src = ep.getFolder('.');
            await src.insert(OUT_FILE_NAME, 'test');

            let res = fs.existsSync(OUT_FILE_FULL_PATH);
            assert.strictEqual(res, true, "Output file was not created!");
            
            await src.delete();

            res = fs.existsSync(OUT_FILE_FULL_PATH);
            assert.strictEqual(res, false, "Output file exists!");

            //await src.delete('*');

            //res = fs.existsSync(ROOT_FOLDER);
            //assert.strictEqual(res, false, "Root folder exists!");
        }
        finally {
            deleteFileIfExists(ROOT_FOLDER);
        }
    });

    test('select() method', async () => {
        const ROOT_FOLDER = TEMP_FOLDER + "temp_out_dir";
        const OUT_FILE_NAME1 = 'test_output1.tmp';
        const OUT_FILE_NAME2 = 'test_output2.tmp';
        try {
            deleteFileIfExists(ROOT_FOLDER);

            const ep = new etl.filesystems.Local.Endpoint(ROOT_FOLDER);
            const src = ep.getFolder('.');
            await src.insert(OUT_FILE_NAME1, 'test1');
            await src.insert(OUT_FILE_NAME2, 'test2');

            const items = await src.select();
            const res: string[] = items.map(item => {
                return item.name;
            });

            res.sort((a, b) => (a > b) ? 1 : -1)
            assert.deepStrictEqual(res, [ OUT_FILE_NAME1, OUT_FILE_NAME2 ]);
        }
        finally {
            deleteFileIfExists(ROOT_FOLDER);
        }
    });

    test('selectGen() method', async () => {
        const ROOT_FOLDER = TEMP_FOLDER + "temp_out_dir";
        const OUT_FILE_NAME1 = 'test_output1.tmp';
        const OUT_FILE_NAME2 = 'test_output2.tmp';
        try {
            deleteFileIfExists(ROOT_FOLDER);

            const ep = new etl.filesystems.Local.Endpoint(ROOT_FOLDER);
            const src = ep.getFolder('.');
            await src.insert(OUT_FILE_NAME1, 'test1');
            await src.insert(OUT_FILE_NAME2, 'test2');

            const res: string[] = [];
            for await (const item of src.selectGen()) res.push(item.name)

            res.sort((a, b) => (a > b) ? 1 : -1)
            assert.deepStrictEqual(res, [ OUT_FILE_NAME1, OUT_FILE_NAME2 ]);
        }
        finally {
            deleteFileIfExists(ROOT_FOLDER);
        }
    });

    test('selectIx() method', async () => {
        const ROOT_FOLDER = TEMP_FOLDER + "temp_out_dir";
        const OUT_FILE_NAME1 = 'test_output1.tmp';
        const OUT_FILE_NAME2 = 'test_output2.tmp';
        try {
            deleteFileIfExists(ROOT_FOLDER);

            const ep = new etl.filesystems.Local.Endpoint(ROOT_FOLDER);
            const src = ep.getFolder('.');
            await src.insert(OUT_FILE_NAME1, 'test1');
            await src.insert(OUT_FILE_NAME2, 'test2');

            const res: string[] = [];
            await src.selectIx().forEach(v => {
                res.push(v.name);
            })

            res.sort((a, b) => (a > b) ? 1 : -1)
            assert.deepStrictEqual(res, [ OUT_FILE_NAME1, OUT_FILE_NAME2 ]);
        }
        finally {
            deleteFileIfExists(ROOT_FOLDER);
        }
    });

    test('selectRx() method', async () => {
        const ROOT_FOLDER = TEMP_FOLDER + "temp_out_dir";
        const OUT_FILE_NAME1 = 'test_output1.tmp';
        const OUT_FILE_NAME2 = 'test_output2.tmp';
        try {
            deleteFileIfExists(ROOT_FOLDER);

            const ep = new etl.filesystems.Local.Endpoint(ROOT_FOLDER);
            const src = ep.getFolder('.');
            await src.insert(OUT_FILE_NAME1, 'test1');
            await src.insert(OUT_FILE_NAME2, 'test2');

            const res: string[] = [];
            let stream$ = src.selectRx().pipe(
                rx.tap(v => res.push(v.name))
            );
            await etl.run(stream$);

            res.sort((a, b) => (a > b) ? 1 : -1)
            assert.deepStrictEqual(res, [ OUT_FILE_NAME1, OUT_FILE_NAME2 ]);
        }
        finally {
            deleteFileIfExists(ROOT_FOLDER);
        }
    });

    test('selectStream() method', async () => {
        const ROOT_FOLDER = TEMP_FOLDER + "temp_out_dir";
        const OUT_FILE_NAME1 = 'test_output1.tmp';
        const OUT_FILE_NAME2 = 'test_output2.tmp';
        try {
            deleteFileIfExists(ROOT_FOLDER);

            const ep = new etl.filesystems.Local.Endpoint(ROOT_FOLDER);
            const src = ep.getFolder('.');
            await src.insert(OUT_FILE_NAME1, 'test1');
            await src.insert(OUT_FILE_NAME2, 'test2');

            const res: string[] = [];
            const reader = src.selectStream().getReader();
            let v;
            while(!(v = await reader.read()).done) res.push(v.value.name);

            res.sort((a, b) => (a > b) ? 1 : -1)
            assert.deepStrictEqual(res, [ OUT_FILE_NAME1, OUT_FILE_NAME2 ]);
        }
        finally {
            deleteFileIfExists(ROOT_FOLDER);
        }
    });
});