import { identity, MonoTypeOperatorFunction, Observable, OperatorFunction, Subscriber, tap, TeardownLogic, UnaryFunction } from "rxjs";
import { BaseCollection } from "./collection.js";

export class BaseObservable<T> extends Observable<T> {
  protected _collection: BaseCollection<T>;
  get collection(): BaseCollection<T> {
    return this._collection;
  }

  constructor(collection: BaseCollection<T>, subscribe?: (this: Observable<T>, subscriber: Subscriber<T>) => TeardownLogic) {
    super(subscribe);
    this._collection = collection;
  }

  pipe(): BaseObservable<T>;
  pipe<A>(op1: OperatorFunction<T, A>): BaseObservable<A>;
  pipe<A, B>(op1: OperatorFunction<T, A>, op2: OperatorFunction<A, B>): BaseObservable<B>;
  pipe<A, B, C>(op1: OperatorFunction<T, A>, op2: OperatorFunction<A, B>, op3: OperatorFunction<B, C>): BaseObservable<C>;
  pipe<A, B, C, D>(
    op1: OperatorFunction<T, A>,
    op2: OperatorFunction<A, B>,
    op3: OperatorFunction<B, C>,
    op4: OperatorFunction<C, D>
  ): BaseObservable<D>;
  pipe<A, B, C, D, E>(
    op1: OperatorFunction<T, A>,
    op2: OperatorFunction<A, B>,
    op3: OperatorFunction<B, C>,
    op4: OperatorFunction<C, D>,
    op5: OperatorFunction<D, E>
  ): BaseObservable<E>;
  pipe<A, B, C, D, E, F>(
    op1: OperatorFunction<T, A>,
    op2: OperatorFunction<A, B>,
    op3: OperatorFunction<B, C>,
    op4: OperatorFunction<C, D>,
    op5: OperatorFunction<D, E>,
    op6: OperatorFunction<E, F>
  ): BaseObservable<F>;
  pipe<A, B, C, D, E, F, G>(
    op1: OperatorFunction<T, A>,
    op2: OperatorFunction<A, B>,
    op3: OperatorFunction<B, C>,
    op4: OperatorFunction<C, D>,
    op5: OperatorFunction<D, E>,
    op6: OperatorFunction<E, F>,
    op7: OperatorFunction<F, G>
  ): BaseObservable<G>;
  pipe<A, B, C, D, E, F, G, H>(
    op1: OperatorFunction<T, A>,
    op2: OperatorFunction<A, B>,
    op3: OperatorFunction<B, C>,
    op4: OperatorFunction<C, D>,
    op5: OperatorFunction<D, E>,
    op6: OperatorFunction<E, F>,
    op7: OperatorFunction<F, G>,
    op8: OperatorFunction<G, H>
  ): BaseObservable<H>;
  pipe<A, B, C, D, E, F, G, H, I>(
    op1: OperatorFunction<T, A>,
    op2: OperatorFunction<A, B>,
    op3: OperatorFunction<B, C>,
    op4: OperatorFunction<C, D>,
    op5: OperatorFunction<D, E>,
    op6: OperatorFunction<E, F>,
    op7: OperatorFunction<F, G>,
    op8: OperatorFunction<G, H>,
    op9: OperatorFunction<H, I>
  ): BaseObservable<I>;
  pipe<A, B, C, D, E, F, G, H, I>(
    op1: OperatorFunction<T, A>,
    op2: OperatorFunction<A, B>,
    op3: OperatorFunction<B, C>,
    op4: OperatorFunction<C, D>,
    op5: OperatorFunction<D, E>,
    op6: OperatorFunction<E, F>,
    op7: OperatorFunction<F, G>,
    op8: OperatorFunction<G, H>,
    op9: OperatorFunction<H, I>,
    ...operations: OperatorFunction<any, any>[]
  ): BaseObservable<unknown>;

  pipe(...operations: OperatorFunction<any, any>[]): BaseObservable<T> {
    const oldObservable = pipeFromArray([startOperator(this._collection), ...operations, endOperator(this._collection)])(this);
    return oldObservable as BaseObservable<T>;
  }
}

function pipeFromArray<T, R>(fns: Array<UnaryFunction<T, R>>): UnaryFunction<T, R> {
  if (fns.length === 0) {
    return identity as UnaryFunction<any, any>;
  }

  if (fns.length === 1) {
    return fns[0];
  }

  return function piped(input: T): R {
    return fns.reduce((prev: any, fn: UnaryFunction<T, R>) => fn(prev), input as any);
  };
}


// function toEtlObservable<T>(source: Observable<T>) {
//   //let started = false;
//   const res = new EtlObservable<T>(subscriber => {
//     return source.subscribe(subscriber);
//     // return source.subscribe({
//     //   next(value) {
//     //     // if (!started) {
//     //     //   //res.sendStartEvent();
//     //     //   started = true;
//     //     // }
//     //     //res.sendDataEvent(value);
//     //     subscriber.next(value);
//     //   },
//     //   complete() {
//     //     subscriber.complete();
//     //     //res.sendEndEvent();
//     //   },
//     //   error(err) {
//     //     //res.sendErrorEvent(err);
//     //     subscriber.error(err);
//     //   }
//     // });
//   });
//   return res;
// }

function startOperator<T>(collection: BaseCollection<T>): MonoTypeOperatorFunction<T> {
  return tap<T>(v => collection.sendPipeStartEvent(v)); 
}

export interface EndOperatorFunction<T> extends UnaryFunction<Observable<T>, BaseObservable<T>> {}


function endOperator<T>(collection: BaseCollection<T>): EndOperatorFunction<T> {
  return (observable: Observable<T>): BaseObservable<T> => (
    new BaseObservable<T>(collection, (subscriber) => {
      // this function will be called each time this Observable is subscribed to.
      const subscription = observable.subscribe({
        next(value) {
          subscriber.next(value);
        },
        error(err) {
          subscriber.error(err);
        },
        complete() {
          subscriber.complete();
        },
      });
 
      // Return the finalization logic. This will be invoked when
      // the result errors, completes, or is unsubscribed.
      return () => {
        subscription.unsubscribe();
      };
    })
  )
}


