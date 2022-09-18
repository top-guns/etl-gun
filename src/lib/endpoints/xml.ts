import * as fs from "fs";
import { Observable } from 'rxjs';
import * as XPath from 'xpath';
import { DOMParser, XMLSerializer } from 'xmldom';
import { Endpoint } from "../core/endpoint";

// Every node contains:
// .attributes, .parentNode and .childNodes, which forms nodes hierarchy
// .hasChildNodes, .firstChild and .lastChild
// .tagName and .nodeValue
// Text inside the tag (like <tag>TEXT</tag>) is child node too, the only child.
export class XmlEndpoint extends Endpoint<any> {
    protected filename: string;
    protected encoding: BufferEncoding;
    protected xmlDocument: any;
    protected autosave: boolean;
    protected autoload: boolean;

    constructor(filename: string, autosave: boolean = true, autoload: boolean = false, encoding?: BufferEncoding) {
        super();
        this.filename = filename;
        this.encoding = encoding;
        this.load();
        this.autosave = autosave;
        this.autoload = autoload;
    }

    public read(xpathToCollection: string = ''): Observable<any> {
        return new Observable<any>((subscriber) => {
            try {
                if (this.autoload) this.load();
                let nodes: any = this.get(xpathToCollection);
                if (nodes) {
                    if (Array.isArray(nodes)) {
                        nodes.forEach(value => {
                            subscriber.next(value);
                        });
                    }
                    else if (typeof nodes == 'object') {
                        for (let i = 0; i < nodes.childNodes.length; i++) {
                            subscriber.next(nodes.childNodes[i]);
                        }
                    }
                }
                subscriber.complete();
            }
            catch(err) {
                subscriber.error(err);
            }
        });
    }

    // Uses simple path syntax from lodash.get function
    // path example: '/store/book/author'
    // use xpath '' for the root object
    public get(xpath: string = ''): any {
        if (this.autoload) this.load();
        xpath = xpath.trim();
        let result: any = xpath ? XPath.select(xpath, this.xmlDocument) : this.xmlDocument;
        return result;
    }

    // Pushes value to the array specified by xpath
    // or update attribute of object specified by xpath and attribute parameter
    public async push(value: any, xpath: string = '', attribute: string = '') {
        const node = this.get(xpath);

        if (attribute) node.setAttribute(attribute, value);
        else node.appendChild(value);

        if (this.autosave) this.save();
    }

    public async clear() {
        this.xmlDocument = new DOMParser().parseFromString("");
        if (this.autosave) this.save();
    }

    public load() {
        const text = fs.readFileSync(this.filename).toString(this.encoding);
        this.xmlDocument = new DOMParser().parseFromString(text);
    }

    public save() {
        const text = XMLSerializer.serializeToString(this.xmlDocument);
        fs.writeFile(this.filename, text, function(){});
    }
}


