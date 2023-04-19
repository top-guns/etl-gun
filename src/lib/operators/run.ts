import { lastValueFrom, Observable, forkJoin, first } from "rxjs";
import { GuiManager } from "../index.js";

export async function run(...observables: Observable<any>[]) {
    //const observable = forkJoin([...observables]);
    //await lastValueFrom<any>(observables[0]);

    //await observables[0].toPromise();

    //await observable.toPromise();

    const promises: Promise<any>[] = [];
    for (const observable of observables) promises.push(lastValueFrom<any>(observable));
    // for (const observable of observables) promises.push(observable.toPromise());
    const result = await Promise.all(promises);  
    GuiManager.log(result[0])
}