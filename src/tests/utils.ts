import assert from 'node:assert';

export function strictNull(value: unknown, message?: string | Error) {
    assert.strictEqual(value, null, message ?? `Assertion isNull failed: '${value}' is not null`);
}

export function strictNotNull(value: unknown, message?: string | Error) {
    assert.notStrictEqual(value, null, message ?? `Assertion isNotNull failed: '${value}' is null`);
}

export function strictUndefined(value: unknown, message?: string | Error) {
    assert.strictEqual(typeof value, 'undefined', message ?? `Assertion isUndefined failed: '${value}' is not undefined`);
}

export function strictDefined(value: unknown, message?: string | Error) {
    assert.notStrictEqual(typeof value, 'undefined', message ?? `Assertion isDefined failed: '${value}' is undefined`);
}

export function strictNullish(value: unknown, message?: string | Error) {
    assert.strictEqual((typeof value === 'undefined' || value === null), true, message ?? `Assertion isNullish failed: '${value}' is not nullish`);
}

export function strictNotNullish(value: unknown, message?: string | Error) {
    assert.strictEqual((typeof value === 'undefined' || value === null), false, message ?? `Assertion isNotNullish failed: '${value}' is nullish`);
}

export function strictTruthy(value: unknown, message?: string | Error) {
    assert.strictEqual(!!value, true, message ?? `Assertion isTruthy failed: '${value}' is not truthy`);
}

export function strictFalsy(value: unknown, message?: string | Error) {
    assert.strictEqual(!!value, false, message ?? `Assertion isFalsy failed: '${value}' is truthy`);
}