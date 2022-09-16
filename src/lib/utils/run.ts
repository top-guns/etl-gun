import { lastValueFrom, Observable } from "rxjs";

export async function run<T>(observable: Observable<T>) {
    await lastValueFrom<T>(observable);
}