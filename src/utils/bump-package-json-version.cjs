const fs = require('fs');

let pjson = require('../../package.json');

let version = pjson.version.split('.').map((v, i) => i == 2 ? parseInt(v) + 1 : v).reduce((p, c) => p ? p + '.' + c : c);
pjson.version = version;

const text = JSON.stringify(pjson, undefined, 2);
fs.writeFileSync('package.json', text);
console.log(text)
