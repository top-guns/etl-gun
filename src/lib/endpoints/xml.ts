import * as ix from 'ix';
import * as fs from "fs";
import { tap } from 'rxjs';
import * as XPath from 'xpath';
//import { DOMParserImpl, XMLSerializerImpl } from 'xmldom-ts';
import 'xmldom-ts';
import { BaseEndpoint} from "../core/endpoint.js";
import { pathJoin } from "../utils/index.js";
import { BaseObservable } from "../core/observable.js";
import { CollectionOptions } from "../core/base_collection.js";
import { BaseCollection_ID } from "../core/base_collection_id.js";
import { generator2Iterable, generator2Observable, generator2Stream, observable2Stream, promise2Generator, promise2Observable, wrapGenerator, wrapObservable } from "../utils/flows.js";


export class Endpoint extends BaseEndpoint {
    protected rootFolder: string | null = null;
    constructor(rootFolder: string | null = null) {
        super();
        this.rootFolder = rootFolder;
    }

    getFile(filename: string, autosave: boolean = true, autoload: boolean = false, encoding?: BufferEncoding, options: CollectionOptions<string[]> = {}): Collection {
        options.displayName ??= this.getName(filename);
        let path = filename;
        if (this.rootFolder) path = pathJoin([this.rootFolder, filename], '/');
        return this._addCollection(filename, new Collection(this, filename, path, autosave, autoload, encoding, options));
    }

    releaseFile(filename: string) {
        this._removeCollection(filename);
    }

    protected getName(filename: string) {
        return filename.substring(filename.lastIndexOf('/') + 1);
    }

    get displayName(): string {
        return this.rootFolder ? `XML (${this.rootFolder})` : `XML (${this.instanceNo})`;
    }
}

export function getEndpoint(rootFolder: string | null = null): Endpoint {
    return new Endpoint(rootFolder);
}

// Every node contains:
// .attributes, .parentNode and .childNodes, which forms nodes hierarchy
// .hasChildNodes, .firstChild and .lastChild
// .tagName and .nodeValue
// Text inside the tag (like <tag>TEXT</tag>) is child node too, the only child.
export class Collection extends BaseCollection_ID<any> {
    protected static instanceNo = 0;

    protected filename: string;
    protected encoding: BufferEncoding | undefined;
    protected xmlDocument: Document | undefined;
    protected autosave: boolean;
    protected autoload: boolean;

    constructor(endpoint: Endpoint, collectionName: string, filename: string, autosave: boolean = true, autoload: boolean = false, encoding?: BufferEncoding, options: CollectionOptions<any> = {}) {
        Collection.instanceNo++;
        super(endpoint, collectionName, options);
        this.filename = filename;
        this.encoding = encoding;
        this.autosave = autosave;
        this.autoload = autoload;
        this.load();
    }

    protected async _select(xpath: string = ''): Promise<Array<XPath.SelectedValue>> {
        if (this.autoload) this.load();
        let result: any = xpath ? XPath.select(xpath, this.xmlDocument) : this.xmlDocument;
        if (!Array.isArray(result)) result = [result];
        return result;
    }

    public async select(xpath: string = ''): Promise<Array<XPath.SelectedValue>> {
        const values = await this._select(xpath);
        this.sendSelectEvent(values);
        return values;
    }

    public async* selectGen(xpath: string = ''): AsyncGenerator<Node, void, void> {
        const values = this._select(xpath);
        const generator = wrapGenerator(promise2Generator(values), this);
        for await (const value of generator) yield value;
    }
    public selectIx(xpath: string = ''): ix.AsyncIterable<Node> {
        const generator = this.selectGen(xpath);
        return generator2Iterable(generator);
    }

    public selectStream(xpath: string = ''): ReadableStream<Node> {
        const generator = this.selectGen(xpath);
        return generator2Stream(generator);
    }

    public selectRx(xpath: string = ''): BaseObservable<Node> {
        const values = this._select(xpath);
        return wrapObservable(promise2Observable(values), this);
    }

    // Uses simple path syntax from lodash.get function
    // path example: '/store/book/author'
    // use xpath '' for the root object
    public async _selectOne(xpath: string = ''): Promise<XPath.SelectedValue> {
        if (this.autoload) this.load();
        let result: any = xpath ? XPath.select(xpath, this.xmlDocument!, true) : this.xmlDocument;
        return result;
    }

    public async selectOne(xpath: string = ''): Promise<XPath.SelectedValue> {
        const value = await this._selectOne(xpath);
        this.sendSelectOneEvent(value);
        return value;
    }

    // Pushes value to the array specified by xpath
    // or update attribute of object specified by xpath and attribute parameter
    protected async _insert(value: any, xpath: string = '', attribute: string = ''): Promise<void> {
        const selectedValue = await this._selectOne(xpath);
        let node: Node | undefined = (selectedValue as any).nodeType ? selectedValue as Node : undefined;
        if (!node) throw new Error('Unexpected result of xpath in push method. Should by Node, but we have: ' + selectedValue.toString());

        if (node.nodeType === node.TEXT_NODE) {
            node.nodeValue = value;
            return;
        }
        if (node.nodeType === node.ELEMENT_NODE) {
            let element: Element | undefined = (node as any).tagName ? node as Element : undefined;
            if (!element) throw new Error('Unexpected result of xpath in push method. Should by Node, but we have: ' + selectedValue.toString());

            if (attribute) element.setAttribute(attribute, value);
            else {
                element.appendChild(value);
            }
            return;
        }

        if (this.autosave) this.save();
    }

    public async delete(): Promise<boolean> {
        this.sendDeleteEvent();
        const exists: boolean = !!(this.xmlDocument || this.xmlDocument!.firstChild || this.xmlDocument!.body);
        this.xmlDocument = new DOMParser().parseFromString("", 'text/xml'); // ??? Test it !!!
        if (this.autosave) this.save();
        return exists;
    }

    public load() {
        const text = fs.readFileSync(this.filename).toString(this.encoding);
        this.xmlDocument = new DOMParser().parseFromString(text, 'text/xml');
    }

    public save() {
        const text = new XMLSerializer().serializeToString(this.xmlDocument!);
        fs.writeFile(this.filename, text, function(){});
    }

    public logNode() {
        return tap<any>(v => {
            let node: Node | undefined = v.nodeType ? v as Node : undefined;
            let element: Element | undefined = v.nodeType === node?.ELEMENT_NODE ? v as Element : undefined;

            if (!node) {
                console.log(v);
                return;
            }

            let printedObject: any = {
                nodeType: this.getNodeTypeNameByValue(node.nodeType),
            }
            
            if (v.nodeValue) printedObject.nodeValue = v.nodeValue;
            if (v.nodeName) printedObject.nodeName = v.nodeName;
            if (v.localName) printedObject.localName = v.localName;
            if (v.lineNumber) printedObject.lineNumber = v.lineNumber;
            if (v.columnNumber) printedObject.columnNumber = v.columnNumber;
            if (v.namespaceURI) printedObject.namespaceURI = v.namespaceURI;
            if (v.prefix) printedObject.prefix = v.prefix;
            if (v._data) printedObject._data = v._data;
            if (v.tagName) printedObject.tagName = v.tagName;
            
            if (node.hasChildNodes()) printedObject.childNodes = `[${node.childNodes.length}]`;
            if (element) {
                if (element.hasAttributes()) {
                    printedObject.attributes = [];
                    for (let i = 0; i < element.attributes.length; i++) {
                        const attr = element.attributes.item(i);
                        printedObject.attributes.push({
                            name: attr?.name,
                            textContent: attr?.textContent
                        });
                    }
                }
            }

            console.log(printedObject);
        });
    }

    protected getNodeTypeNameByValue(value: number) {
        const NodeTypeValues = {
            ELEMENT_NODE: 1,
            ATTRIBUTE_NODE: 2,
            TEXT_NODE: 3,
            CDATA_SECTION_NODE: 4,
            ENTITY_REFERENCE_NODE: 5,
            ENTITY_NODE: 6,
            PROCESSING_INSTRUCTION_NODE: 7,
            COMMENT_NODE: 8,
            DOCUMENT_NODE: 9,
            DOCUMENT_TYPE_NODE: 10,
            DOCUMENT_FRAGMENT_NODE: 11,
            NOTATION_NODE: 12
        }

        for (let key in NodeTypeValues) {
            if (NodeTypeValues.hasOwnProperty(key) && NodeTypeValues[key] == value) return key;
        }
    }
}


