import { lastValueFrom, Observable, forkJoin, first } from "rxjs";

export async function run(...observables: Observable<any>[]) {
    const observable = forkJoin([...observables]);
    await lastValueFrom<any>(observable);
    //await observable.toPromise();
}