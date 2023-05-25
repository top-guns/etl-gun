export default {
  preset: 'ts-jest',
  testEnvironment: 'node',
  resolver: "ts-jest-resolver",
  globalSetup: "<rootDir>/src/tests/jest.setup.ts"
};
