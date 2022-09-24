import { lastValueFrom, Observable, forkJoin, first } from "rxjs";

export async function run(...observables: Observable<any>[]) {
    //const observable = forkJoin([...observables]);
    //await lastValueFrom<any>(observable);

    //await observable.toPromise();

    const promises: Promise<any>[] = [];
    for (const observable of observables) promises.push(lastValueFrom<any>(observable));
    await Promise.all(promises);
    
}