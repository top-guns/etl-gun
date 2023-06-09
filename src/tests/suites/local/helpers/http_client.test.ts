import { after, before, describe, test } from 'node:test';
import assert from 'node:assert';
import * as rx from 'rxjs';
import * as etl from '../../../../lib/index.js';
import http from 'node:http';
import https from 'node:https';
import * as fs from "fs";

const HTTP_PORT: number = parseInt(process.env.LOCAL_HTTP_PORT);
const HTTPS_PORT: number = parseInt(process.env.LOCAL_HTTPS_PORT);
const HTTP_URL: string = `http://localhost:${HTTP_PORT}`;
const HTTPS_URL: string = `https://localhost:${HTTPS_PORT}`;


describe('HttpClientHelper', () => {
    let httpServer: http.Server;
    let httpsServer: https.Server;

    // before(async () => {
    //     // HTTP SERVER
    //     httpServer = http.createServer((req,res)=>{
    //         // Handling Request and Response 
    //         res.statusCode = 200;
    //         res.setHeader('Content-Type', 'text/plain');
    //         res.end("ok");
    //     });
    //     httpServer.listen(HTTP_PORT, 'localhost', ()=> {
    //         console.log(`Test http server is running on port ${HTTP_PORT}`);
    //     });

    //     // HTTPS SERVER
    //     httpsServer = https.createServer({
    //         //secureOptions: constants.SSL_OP_NO_SSLv2 | constants.SSL_OP_NO_SSLv3 | constants.SSL_OP_NO_TLSv1 | constants.SSL_OP_NO_TLSv1_1,
    //         //maxVersion:'TLSv1.2',
    //         //ciphers: 'ALL',
    //         //secureProtocol: 'TLSv1_1_method',
    //         //ecdhCurve: 'auto' // <-- Does the trick
    //         key: fs.readFileSync(
    //             process.env.LOCAL_HTTPS_KEY_PATH
    //         ),
    //         cert: fs.readFileSync(
    //             process.env.LOCAL_HTTPS_CERT_PATH
    //         )
    //     }, 
    //     (req,res)=> {
    //         // Handling Request and Response 
    //         res.statusCode = 200;
    //         res.setHeader('Content-Type', 'text/plain');
    //         res.end("ok");
    //     });
    //     httpsServer.listen(HTTPS_PORT, 'localhost', ()=> {
    //         console.log(`Test https server is running on port ${HTTPS_PORT}`);
    //     });
    // });

    // after(async () => {
    //     if (httpServer) httpServer.close();
    //     httpServer = undefined;
    //     if (httpsServer) httpsServer.close();
    //     httpsServer = undefined;
    // })

    // test('Fetch from local http', async () => {
    //     const helper = new etl.HttpClientHelper(HTTP_URL, {}, { timeout: 2000 });
    //     const res = await helper.fetch();
    //     assert.strictEqual(res.status, 200);
    // });

    // test('Fetch from local https', async () => {
    //     const helper = new etl.HttpClientHelper(HTTPS_URL, {}, { dontRejectUnauthorized: true, timeout: 2000 });
    //     const res = await helper.fetch();
    //     assert.strictEqual(res.status, 200);
    // });
});