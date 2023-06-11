import assert from 'node:assert';

export function isNull(value: unknown, message?: string | Error) {
    assert.strictEqual(value, null, message ?? `Assertion isNull failed: '${value}' is not null`);
}

export function isNotNull(value: unknown, message?: string | Error) {
    assert.notStrictEqual(value, null, message ?? `Assertion isNotNull failed: '${value}' is null`);
}

export function isUndefined(value: unknown, message?: string | Error) {
    assert.strictEqual(value, undefined, message ?? `Assertion isUndefined failed: '${value}' is not undefined`);
}

export function isDefined(value: unknown, message?: string | Error) {
    assert.notStrictEqual(value, undefined, message ?? `Assertion isDefined failed: '${value}' is undefined`);
}

export function isNullish(value: unknown, message?: string | Error) {
    assert.equal(value, false, message ?? `Assertion isNullish failed: '${value}' is not nullish`);
}

export function isNotNullish(value: unknown, message?: string | Error) {
    assert.notEqual(value, false, message ?? `Assertion isNotNullish failed: '${value}' is nullish`);
}

export function isTruthy(value: unknown, message?: string | Error) {
    assert.ok(value, message ?? `Assertion isTruthy failed: '${value}' is not truthy`);
}

export function isFalsy(value: unknown, message?: string | Error) {
    assert.equal(value, false, message ?? `Assertion isFalsy failed: '${value}' is truthy`);
}