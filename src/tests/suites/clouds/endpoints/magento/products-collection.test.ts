import { afterEach, beforeEach, describe, test } from 'node:test';
import * as rx from 'rxjs';
import * as etl from '../../../../../lib/index.js';
import { isNotNullish, isTruthy } from '../../../../utils.js';


const ENDPOINT_URL = process.env.CLOUDS_MAGENTO_URL;
const ENDPOINT_LOGIN = process.env.CLOUDS_MAGENTO_LOGIN;
const ENDPOINT_PASSWORD = process.env.CLOUDS_MAGENTO_PASSWORD;

function checkResultType(value: any) {
    isNotNullish(value);
    isTruthy(value.id);
}


describe('Magento products', () => {
    let endpoint: etl.Magento.Endpoint = null;
    let collection: etl.Magento.ProductCollection = null;

    beforeEach(async () => {
        endpoint = new etl.Magento.Endpoint(ENDPOINT_URL, ENDPOINT_LOGIN, ENDPOINT_PASSWORD);
        collection = endpoint.getProducts();
    })

    afterEach(async () => {
        if (collection) {
            endpoint.releaseProducts();
            collection = null;
        }
        if (endpoint) {
            endpoint.releaseEndpoint();
            endpoint = null;
        }
    })

    test('selectOne() method', async () => {
        const items = await collection.select();
        const item = await collection.selectOne(items[0].sku);
        checkResultType(item);
    });

    test('select() method', async () => {
        const items = await collection.select();
        isNotNullish(items);
        isTruthy(items.length);
        checkResultType(items[0]);
    });

    test('selectGen() method', async () => {
        const generator = await collection.selectGen();
        isNotNullish(generator);

        const first = await generator.next();
        isNotNullish(first);

        checkResultType(first.value);
    });

    test('selectIx() method', async () => {
        const iterable = await collection.selectIx();
        isNotNullish(iterable);
        const value = await iterable.first();
        checkResultType(value);
    });

    test('selectRx() method', async () => {
        const observable = collection.selectRx();
        isNotNullish(observable);

        let value: any;
        await etl.run(observable.pipe(
            rx.take(1),
            rx.tap(v => value = v)
        ));

        checkResultType(value);
    });

    test('selectStream() method', async () => {
        const stream = collection.selectStream();
        isNotNullish(stream);

        const reader = stream.getReader();
        isNotNullish(reader);

        let first = await reader.read();
        isNotNullish(first);

        checkResultType(first.value);
    });
});