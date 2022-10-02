import * as fs from "fs";
import { Observable, Subscriber, tap } from 'rxjs';
import * as XPath from 'xpath';
//import { DOMParserImpl, XMLSerializerImpl } from 'xmldom-ts';
import 'xmldom-ts';
import { Endpoint, EndpointGuiOptions, EndpointImpl } from "../core/endpoint";
import { EtlObservable } from "../core/observable";

export type ReadOptions = {
    // foundedOnly is default
    searchReturns?: 'foundedOnly' | 'foundedImmediateChildrenOnly' | 'foundedWithDescendants';
    addRelativePathAsAttribute?: string;
}

// Every node contains:
// .attributes, .parentNode and .childNodes, which forms nodes hierarchy
// .hasChildNodes, .firstChild and .lastChild
// .tagName and .nodeValue
// Text inside the tag (like <tag>TEXT</tag>) is child node too, the only child.
export class XmlEndpoint extends EndpointImpl<any> {
    protected static instanceNo = 0;
    protected filename: string;
    protected encoding: BufferEncoding;
    protected xmlDocument: Document;
    protected autosave: boolean;
    protected autoload: boolean;

    constructor(filename: string, autosave: boolean = true, autoload: boolean = false, encoding?: BufferEncoding, guiOptions: EndpointGuiOptions<any> = {}) {
        guiOptions.displayName = guiOptions.displayName ?? `XML ${++XmlEndpoint.instanceNo}(${filename.substring(filename.lastIndexOf('/') + 1)})`;
        super(guiOptions);
        this.filename = filename;
        this.encoding = encoding;
        this.autosave = autosave;
        this.autoload = autoload;
        this.load();
    }

    public read(xpath: string = '', options: ReadOptions = {}): EtlObservable<Node> {
        const observable = new EtlObservable<any>((subscriber) => {
            (async () => {
                try {
                    this.sendStartEvent();
                    let selectedValue: XPath.SelectedValue = this.get(xpath);
                    if (selectedValue) {
                        if (Array.isArray(selectedValue)) {
                            for (const value of selectedValue) await this.processOneSelectedValue(value, options, '', subscriber, observable);
                        }
                        else {
                            await this.processOneSelectedValue(selectedValue, options, '', subscriber, observable);
                        }
                    }
                    subscriber.complete();
                    this.sendEndEvent();
                }
                catch(err) {
                    this.sendErrorEvent(err);
                    subscriber.error(err);
                }
            })();
        });
        return observable;
    }

    protected async processOneSelectedValue(selectedValue: XPath.SelectedValue, options: ReadOptions, relativePath: string, subscriber: Subscriber<any>, observable: EtlObservable<any>) {
        const element = (selectedValue as Element).tagName ? selectedValue as Element : undefined;

        if (options.searchReturns == 'foundedOnly' || !options.searchReturns) {
            if (options.addRelativePathAsAttribute && element) element.setAttribute(options.addRelativePathAsAttribute, relativePath);
            await this.waitWhilePaused();
            this.sendDataEvent(selectedValue);
            subscriber.next(selectedValue);
            return;
        }

        if (options.searchReturns == 'foundedWithDescendants') {
            await this.sendElementWithChildren(selectedValue, subscriber, observable, options, relativePath);
            return;
        }

        if (options.searchReturns == 'foundedImmediateChildrenOnly' && element) {
            for (let i = 0; i < element.childNodes.length; i++) {
                const value = element.childNodes[i];
                const childElement = (value as Element).tagName ? value as Element : undefined;
                let childPath = '';
                if (childElement) {
                    childPath = relativePath ? relativePath + `/${element.tagName}[${i}]` : `${element.tagName}[${i}]`;
                } 
                if (options.addRelativePathAsAttribute && childElement) childElement.setAttribute(options.addRelativePathAsAttribute, childPath);
                await this.waitWhilePaused();
                this.sendDataEvent(value);
                subscriber.next(value);
            };
        }
    }

    protected async sendElementWithChildren(selectedValue: XPath.SelectedValue, subscriber: Subscriber<any>, observable: EtlObservable<any>, options: ReadOptions = {}, relativePath = '') {
        let element: Element = (selectedValue as any).tagName ? selectedValue as Element : undefined;
        if (options.addRelativePathAsAttribute && element) element.setAttribute(options.addRelativePathAsAttribute, relativePath);
        await this.waitWhilePaused();
        this.sendDataEvent(selectedValue);
        subscriber.next(selectedValue);

        if (element && element.hasChildNodes()) {
            let sendedDown = false;
            const tagIndexes: Record<string, number> = {};
            for (let i = 0; i < element.childNodes.length; i++) {
                const childNode = element.childNodes[i];
                let childElement = (childNode as any).tagName ? childNode as Element : undefined;
                if (childElement) {
                    if (!sendedDown) {
                        this.sendDownEvent();
                        sendedDown = true;
                    }

                    if (!tagIndexes[childElement.tagName]) tagIndexes[childElement.tagName] = 0;
                    let childPath = `${childElement.tagName}[${tagIndexes[childElement.tagName]}]`;
                    if (relativePath) childPath = relativePath + '/' + childPath;

                    await this.sendElementWithChildren(childElement, subscriber, observable, options, childPath);

                    tagIndexes[childElement.tagName]++;
                }
            }
            if (sendedDown) this.sendUpEvent();
        }
    }

    // Uses simple path syntax from lodash.get function
    // path example: '/store/book/author'
    // use xpath '' for the root object
    public get(xpath: string = ''): XPath.SelectedValue {
        if (this.autoload) this.load();
        let result: any = xpath ? XPath.select(xpath, this.xmlDocument) : this.xmlDocument;
        return result;
    }

    // Pushes value to the array specified by xpath
    // or update attribute of object specified by xpath and attribute parameter
    public async push(value: any, xpath: string = '', attribute: string = '') {
        super.push(value);
        
        const selectedValue = this.get(xpath);
        let node: Node = (selectedValue as any).nodeType ? selectedValue as Node : undefined;
        if (!node) throw new Error('Unexpected result of xpath in push method. Should by Node, but we have: ' + selectedValue.toString());

        if (node.nodeType === node.TEXT_NODE) {
            node.nodeValue = value;
            return;
        }
        if (node.nodeType === node.ELEMENT_NODE) {
            let element: Element = (node as any).tagName ? node as Element : undefined;
            if (!element) throw new Error('Unexpected result of xpath in push method. Should by Node, but we have: ' + selectedValue.toString());

            if (attribute) element.setAttribute(attribute, value);
            else {
                element.appendChild(value);
            }
            return;
        }

        if (this.autosave) this.save();
    }

    public async clear() {
        super.clear();
        this.xmlDocument = new DOMParser().parseFromString("", 'text/xml'); // ??? Test it !!!
        if (this.autosave) this.save();
    }

    public load() {
        const text = fs.readFileSync(this.filename).toString(this.encoding);
        this.xmlDocument = new DOMParser().parseFromString(text, 'text/xml');
    }

    public save() {
        const text = new XMLSerializer().serializeToString(this.xmlDocument);
        fs.writeFile(this.filename, text, function(){});
    }

    public logNode() {
        return tap<any>(v => {
            let node: Node = v.nodeType ? v as Node : undefined;
            let element: Element = v.nodeType === node.ELEMENT_NODE ? v as Element : undefined;

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
                            name: attr.name,
                            textContent: attr.textContent
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


