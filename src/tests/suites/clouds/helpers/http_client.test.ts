import { after, before, describe, test } from 'node:test';
import { should, expect, assert } from "chai";
import * as rx from 'rxjs';
import * as etl from '../../../../lib/index.js';
import http from 'node:http';
import https from 'node:https';
import constants from 'constants';


describe('HttpClientHelper with clouds', () => {
    test('Fetch from clouds https', async () => {
        const helper = new etl.HttpClientHelper(process.env.CLOUDS_HTTPS_URL);
        const res = await helper.fetch();
        assert.strictEqual(res.status, 200);
    });
});