import { afterEach, beforeEach, describe, test } from 'node:test';
import * as rx from 'rxjs';
import * as etl from '../../../../../lib/index.js';
import { strictNotNullish, strictTruthy } from '../../../../utils.js';


const ENDPOINT_URL = process.env.CLOUDS_MAGENTO_URL!;
const ENDPOINT_LOGIN = process.env.CLOUDS_MAGENTO_LOGIN!;
const ENDPOINT_PASSWORD = process.env.CLOUDS_MAGENTO_PASSWORD!;

function checkResultType(value: any) {
    strictNotNullish(value);
    strictTruthy(value.id);
}


describe('Magento categories', () => {
    let endpoint: etl.endpoints.Magento.Endpoint= null;
    let collection: etl.endpoints.Magento.CategoryCollection= null;

    beforeEach(async () => {
        endpoint = new etl.endpoints.Magento.Endpoint(ENDPOINT_URL, ENDPOINT_LOGIN, ENDPOINT_PASSWORD);
        collection = endpoint.getCategories();
    })

    afterEach(async () => {
        if (collection) {
            endpoint.releaseCategories();
            collection = null;
        }
        if (endpoint) {
            endpoint.releaseEndpoint();
            endpoint = null;
        }
    })

    test('selectOne() method', async () => {
        const items = await collection.select();
        const item = await collection.selectOne(items[0].id);
        checkResultType(item);
    });

    test('select() method', async () => {
        const items = await collection.select();
        strictNotNullish(items);
        strictTruthy(items.length);
        checkResultType(items[0]);
    });

    test('selectGen() method', async () => {
        const generator = await collection.selectGen();
        strictNotNullish(generator);

        const first = await generator.next();
        strictNotNullish(first);

        checkResultType(first.value);
    });

    test('selectIx() method', async () => {
        const iterable = await collection.selectIx();
        strictNotNullish(iterable);
        const value = await iterable.first();
        checkResultType(value);
    });

    test('selectRx() method', async () => {
        const observable = collection.selectRx();
        strictNotNullish(observable);

        let value: any;
        await etl.operators.run(observable.pipe(
            rx.take(1),
            rx.tap(v => value = v)
        ));

        checkResultType(value);
    });

    test('selectStream() method', async () => {
        const stream = collection.selectStream();
        strictNotNullish(stream);

        const reader = stream.getReader();
        strictNotNullish(reader);

        let first = await reader.read();
        strictNotNullish(first);

        checkResultType(first.value);
    });
});