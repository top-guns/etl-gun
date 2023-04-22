import { lastValueFrom, Observable, forkJoin, first } from "rxjs";

export async function run(...observables: Observable<any>[]) {
    //const observable = forkJoin([...observables]);
    //await lastValueFrom<any>(observables[0]);

    //await observables[0].toPromise();

    //await observable.toPromise();

    const promises: Promise<any>[] = [];
    //for (const observable of observables) promises.push(lastValueFrom<any>(observable));
    // Should use depricated observable.toPromise() instead of lastValueFrom() to process streams without elements
    for (const observable of observables) promises.push(observable.toPromise());
    await Promise.all(promises);  
}