import * as rx from "rxjs";
import * as ix from 'ix';
import * as internal from "node:stream";
import otag from "observable-to-async-generator";
import { BaseObservable } from "../core/observable.js";
import { run } from "../operators/run.js";
import { BaseCollection } from "../core/base_collection.js";
import { mapAsync } from "../operators/mapAsync.js";

// --------------------------------------------------------------------------------------------------------------
// Wrappers

export type DisableEvents = 'disable_events';

export async function wrapPromise<T>(promise: Promise<T[]>, collection: BaseCollection<T>, disableEvents?: DisableEvents): Promise<T[]> {
    try {
        const values = await promise; 
        if (collection && !disableEvents) collection.sendSelectEvent(values);
        return values;
    }
    catch(err) {
        if (collection && !disableEvents) collection.sendErrorEvent(err);
        throw err;
    }
}

export async function* wrapGenerator<T>(generator: AsyncGenerator<T, void, void>, collection: BaseCollection<T>, disableEvents?: DisableEvents): AsyncGenerator<T, void, void> {
    try {
        if (!disableEvents) collection.sendStartEvent();
        for await (const value of generator) {
            await collection.waitWhilePaused();
            if (!disableEvents) collection.sendReciveEvent(value);
            yield value;
        }
        if (!disableEvents) collection.sendEndEvent();
    }
    catch(err) {
        if (!disableEvents) collection.sendErrorEvent(err);
        throw err;
    }
}

export function wrapObservable<T>(observable: rx.Observable<T>, collection: BaseCollection<T>, disableEvents?: DisableEvents): BaseObservable<T> {
    const result = new BaseObservable<T>(collection, (subscriber: rx.Subscriber<T>) => {
        (async () => {
            try {
                const pipeline$ = observable.pipe(
                    mapAsync(async value => {
                        await collection.waitWhilePaused();
                        if (subscriber.closed) return;
                        if (!disableEvents) collection.sendReciveEvent(value);
                        subscriber.next(value);
                    }),
                    rx.finalize(() => {
                        if (collection && !disableEvents) collection.sendEndEvent();
                        if (!subscriber.closed) subscriber.complete();
                    })
                )

                if (!disableEvents) collection.sendStartEvent();
                subscriber.add(pipeline$.subscribe());
            }
            catch(err) {
                if (!disableEvents) collection.sendErrorEvent(err);
                if (!subscriber.closed) subscriber.error(err);
            }
        })();
    });
    return result;
}

export function wrapIterable<T>(iterable: ix.AsyncIterable<T>, collection: BaseCollection<T>, disableEvents?: DisableEvents): ix.AsyncIterable<T> {
    const generator = iterable2Generator(iterable);
    return generator2Iterable(wrapGenerator(generator, collection, disableEvents));
}

export function wrapStream<T>(stream: ReadableStream<T>, collection: BaseCollection<T>, disableEvents?: DisableEvents): ReadableStream<T> {
    const observable = stream2Observable(stream);
    return observable2Stream(wrapObservable(observable, collection, disableEvents));
}

// --------------------------------------------------------------------------------------------------------------
// Promise

export async function* promise2Generator<T>(promise: Promise<T[]>): AsyncGenerator<T, void, void> {
    const values = await promise; 
    for (const value of values) yield value;
}

export function promise2Observable<T>(promise: Promise<T[]>): rx.Observable<T> {
    // return rx.from(values);
    return new rx.Observable<T>((subscriber: rx.Subscriber<T>) => {
        (async () => {
            try {
                const values = await promise; 
                for (const value of values) {
                    if (subscriber.closed) break;
                    subscriber.next(value);
                }
                if (!subscriber.closed) subscriber.complete();
            }
            catch(err) {
                if (!subscriber.closed) subscriber.error(err);
            }
        })();
    });
}

export function promise2Iterable<T>(promise: Promise<T[]>): ix.AsyncIterable<T> {
    return ix.AsyncIterable.from(promise2Generator(promise));
}

export function promise2Stream<T>(promise: Promise<T[]>): ReadableStream<T> {
    let stop = false;

    return new ReadableStream<T>({
        start: async (controller) => {
            try {
                const values = await promise; 
                for (const value of values) {
                    if (stop) break;
                    controller.enqueue(value);
                }
                controller.close();
            }
            catch(err) {
                throw err;
            }
        },
        pull(controller) {
            // We don't really need a pull
        },
        cancel: () => {
            stop = true;
        }
    });
}

export function promise2Readable<T>(promise: Promise<T[]>): internal.Readable {
    return internal.Readable.from(promise2Iterable(promise));
}

// --------------------------------------------------------------------------------------------------------------
// Generator

export async function generator2Promise<T>(generator: AsyncGenerator<T, void, void>): Promise<T[]> {
    const values: T[] = []; 
    for await (const value of generator) values.push(value);
    return values;
}

export function generator2Observable<T>(generator: AsyncGenerator<T, void, void>): rx.Observable<T> {
    return rx.from(generator);
}

export function generator2Iterable<T>(generator: AsyncGenerator<T, void, void>): ix.AsyncIterable<T> {
    return ix.AsyncIterable.from(generator);
}

export function generator2Stream<T>(generator: AsyncGenerator<T, void, void>, collection?: BaseCollection<T>, disableEvents?: DisableEvents): ReadableStream<T> {
    let stop = false;

    return new ReadableStream<T>({
        start: async (controller) => {
            for await (const value of generator) {
                if (stop) break;
                controller.enqueue(value);
            }
            controller.close();
        },
        pull(controller) {
            // We don't really need a pull
        },
        cancel: () => {
            stop = true;
        }
    });
}

export function generator2Readable<T>(generator: AsyncGenerator<T, void, void>): internal.Readable {
    return internal.Readable.from(generator2Iterable(generator));
}

// --------------------------------------------------------------------------------------------------------------
// RxJs

export async function observable2Promise<T>(observable: rx.Observable<T>): Promise<T[]> {
    const values: T[] = []; 
    const pipeline$ = observable.pipe(
        rx.tap(value => values.push(value))
    )
    await run(pipeline$);
    return values;
}

// XXX check it!
// TODO: handle the stop event
export async function* observable2Generator<T>(observable: rx.Observable<T>): AsyncGenerator<T, void, void> {
    const generator = otag(observable);
    for await (const value of generator) yield value;
}

// XXX check it!
// TODO: handle the stop event
export function observable2Iterable<T>(observable: rx.Observable<T>): ix.AsyncIterable<T> {
    return ix.AsyncIterable.from(observable);
}

export function observable2Stream<T>(observable: rx.Observable<T>): ReadableStream<T> {
    let subscribtion: rx.Subscription | null = null;

    return new ReadableStream<T>({
        start: (controller) => {
            const pipeline$ = observable.pipe(
                rx.tap(value => {
                    controller.enqueue(value);
                }),
                rx.finalize(() => {
                    controller.close();
                })
            )
            subscribtion = pipeline$.subscribe();
        },
        pull(controller) {
            // We don't really need a pull
        },
        cancel: () => {
            if (subscribtion) subscribtion.unsubscribe();
            subscribtion = null;
        }
    });
}

export function observable2Readable<T>(observable: BaseObservable<T>): internal.Readable {
    return internal.Readable.from(observable2Iterable(observable));
}

// --------------------------------------------------------------------------------------------------------------
// IxJs

export async function iterable2Promise<T>(iterable: ix.AsyncIterable<T>): Promise<T[]> {
    return generator2Promise(iterable2Generator(iterable));
}

export async function* iterable2Generator<T>(iterable: ix.AsyncIterable<T>): AsyncGenerator<T, void, void> {
    for await (const value of iterable) yield value;
}

export function iterable2Observable<T>(iterable: ix.AsyncIterable<T>): rx.Observable<T> {
    return rx.from(iterable);
}

export function iterable2Stream<T>(iterable: ix.AsyncIterable<T>): ReadableStream<T> {
    return observable2Stream(iterable2Observable(iterable));
}

export function ix2Readable<T>(iterable: ix.AsyncIterable<T>): internal.Readable {
    return internal.Readable.from(iterable);
}

// --------------------------------------------------------------------------------------------------------------
// ReadableStream

export async function stream2Promise<T>(stream: ReadableStream<T>): Promise<T[]> {
    const values: T[] = []; 
    const reader = stream.getReader();
    
    // Result objects contain two properties:
    // done  - true if the stream has already given you all its data.
    // value - some data. Always undefined when done is true.
    let res: ReadableStreamReadResult<T>;
    while (!(res = await reader.read()).done) values.push(res.value);

    return values;
}

export async function* stream2Generator<T>(stream: ReadableStream<T>): AsyncGenerator<T, void, void> {
    const reader = stream.getReader();
    let res: ReadableStreamReadResult<T>;
    while (!(res = await reader.read()).done) yield res.value;
}

export function stream2Observable<T>(stream: ReadableStream<T>): rx.Observable<T> {
    const observable = new rx.Observable<T>((subscriber) => {
        (async () => {
            try {
                const reader = stream.getReader();
                let res: ReadableStreamReadResult<T>;
                while (!subscriber.closed && !(res = await reader.read()).done) subscriber.next(res.value);
                subscriber.complete();
            }
            catch(err) {
                if (!subscriber.closed) subscriber.error(err);
            }
        })();
    });
    return observable;
}

export function stream2Iterable<T>(stream: ReadableStream<T>): ix.AsyncIterable<T> {
    return observable2Iterable(stream2Observable(stream));
}

export function stream2Readable<T>(stream: ReadableStream<T>): internal.Readable {
    return internal.Readable.from(stream2Iterable(stream));
}

// --------------------------------------------------------------------------------------------------------------
// internal.Readable

// export async function readable2Promise<T>(readable: internal.Readable): Promise<T[]> {
//     const values = []; 
//     for await (const chunk of readable) values.push(chunk);
//     return values;
// }

// export async function* readable2Generator<T>(readable: internal.Readable): AsyncGenerator<T, void, void> {
//     for await (const chunk of readable) {
//         yield chunk;
//     }
// }

// export function readable2Rx<T>(readable: internal.Readable): BaseObservable<T> {
//     const observable = new BaseObservable<T>(this, (subscriber) => {
//         (async () => {
//             try {
//                 for await (const chunk of readable) {
//                     if (subscriber.closed) break;
//                     await this.waitWhilePaused();
//                     subscriber.next(chunk);
//                 }
//                 subscriber.complete();
//             }
//             catch(err) {
//                 subscriber.error(err);
//             }
//         })();
//     });
//     return observable;
// }

// export function readable2Ix<T>(readable: internal.Readable): ix.AsyncIterable<T> {
//     return ix.AsyncIterable.from(readable2Generator(readable));
// }

// // XXX
// export function readable2Stream<T>(readable: internal.Readable): ReadableStream<T> {
//     return rx2Stream(readable2Rx(readable));
// }

// --------------------------------------------------------------------------------------------------------------
// NodeJS.ReadableStream (old stype streams, prior to Node.js 0.10)

export function old2Readable<T>(old: NodeJS.ReadableStream): internal.Readable {
    return new internal.Readable().wrap(old);
}

export function old2Iterable<T>(old: NodeJS.ReadableStream): ix.AsyncIterable<string | Buffer> {
    return ix.fromNodeStream(old);
}


// --------------------------------------------------------------------------------------------------------------
// selectOne

export async function selectOne_from_Promise<T>(data: Promise<T[]>): Promise<T | null> {
    const values: T[] = await data;
    if (!values || !values.length) return null;
    return values[0]; 
}

export async function selectOne_from_Generator<T>(generator: AsyncGenerator<T, void, void>): Promise<T | null> {
    const result = await generator.next();
    if (!result || result.done) return null;
    return result.value as T; 
}

export async function selectOne_from_Observable<T>(stream: BaseObservable<T>): Promise<T | null> {
    return new Promise<T | null>(async (resolve: (value: T | null | PromiseLike<T | null>) => void, reject: (reason?: any) => void) => {
        try {
            let result: T | null = null;
            const stream$ = stream.pipe(
                rx.take(1),
                rx.tap(value => result = value)
            )
            await run(stream$);
            resolve(result);
        }
        catch (err) {
            reject(err);
        }
    })
}

export async function selectOne_from_Iterable<T>(iterable: ix.AsyncIterable<T>): Promise<T | null> {
    const value = await iterable.first();
    return value ?? null; 
}

export async function selectOne_from_Stream<T>(stream: ReadableStream<T>): Promise<T | null> {
    const result = await stream.getReader().read()
    if (!result || result.done) return null;
    return result.value as T; 
}

// export async function selectOne_from_Readable<T>(readable: internal.Readable): Promise<T> {
// }

function it2Stream<T>(it: AsyncIterable<T>, collection?: BaseCollection<T>, disableEvents?: DisableEvents): ReadableStream<T> {
    let stop = false;

    const stream = new ReadableStream<T>({
        async start(controller) {
            try {
                if (collection && !disableEvents) collection.sendStartEvent();
                for await (const value of it) {
                    if (stop) break;
                    if (collection) await collection.waitWhilePaused();
                    if (stop) break;
                    if (collection && !disableEvents) collection.sendReciveEvent(value);
                    controller.enqueue(value);
                }
                if (collection && !disableEvents) collection.sendEndEvent();
                controller.close();
            }
            catch(err) {
                if (collection && !disableEvents) collection.sendErrorEvent(err);
                throw err;
            }
        },
        pull(controller) {
            // We don't really need a pull
        },
        cancel() {
            stop = true;
        }
    });
      
    return stream;
}