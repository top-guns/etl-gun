import { JSONPath } from 'jsonpath-plus';

export { Header } from './header.js';

export function pathJoin(parts: string[], sep: string = '/') {
  return parts.reduce((prev, cur, i) => {
    if (!cur || !prev) return cur + prev;

    if (cur.startsWith(sep)) cur = cur.substring(1);
    if (prev.endsWith(sep)) prev = prev.substring(0, prev.length - 1);
    return prev + sep + cur;
  })
}

export function extractFileName(path: string): string {
  return path.substring(path.lastIndexOf('/') + 1);
}

export function extractParentFolderPath(path: string): string {
  return path.substring(0, path.lastIndexOf('/'));
}

export function getByJsonPath(obj: {}, jsonPath?: string): any;
export function getByJsonPath(obj: {}, jsonPaths?: string[]): any;
export function getByJsonPath(obj: {}, jsonPath: any = ''): any {
    let result: any = JSONPath({path: jsonPath, json: obj, wrap: true});
    return result;
}

export function getChildByPropVal(obj: {} | undefined | null, propName: string, propVal?: any): any;
export function getChildByPropVal(arr: [] | undefined | null, propName: string, propVal?: any): any;
export function getChildByPropVal(obj: any, propName: string, propVal: any = ''): any {
  if (typeof obj === 'undefined') return null;

  if (typeof obj.length === 'number' || obj === null) {
    for (let i = 0; i < obj.length; i++) {
      if (obj[i][propName] === propVal) return obj[i];
    }
    return null;
  }

  for (let key in obj) {
    if (!obj.hasOwnProperty(key)) continue;
    if (obj[key][propName] === propVal) return obj[key];
  }

  return null;
}

export function dumpObject(obj: any, deep: number = 1): string {
  switch (typeof obj) {
      case 'number': return '' + obj;
      case 'string': return `"${obj}"`;
      case 'boolean': return '' + obj;
      case 'function': return '()';
      case 'object': {
          if (deep <= 0) return 'object';

          let res = '';
          for (let key in obj) {
              if (obj.hasOwnProperty(key)) {
                  if (res) res += `, `;
                  res += `${key}: ${dumpObject(obj[key], deep - 1)}`;
              }
          }
          return `{${res}}`;
      }
      default: return '' + obj;
  }
}