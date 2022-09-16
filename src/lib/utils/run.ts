import { lastValueFrom, Observable, forkJoin } from "rxjs";

export async function run(...observables: Observable<any>[]) {
    const observable = forkJoin([...observables]);
    await lastValueFrom<any>(observable);
}