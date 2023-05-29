import nodemailer, { Transporter } from "nodemailer";
import SMTPTransport from "nodemailer/lib/smtp-transport/index.js";
import Imap from 'node-imap';
import { inspect } from 'util';
import { MessangerService, SendError } from './messanger-service.js';
import { BaseEndpoint } from '../../core/endpoint.js';
import { BaseCollection, CollectionOptions } from '../../core/base_collection.js';
import { BaseObservable } from '../../core/observable.js';
import { Subscriber } from "rxjs";



type EMail = {
    /** Comma separated list or an array of recipients e-mail addresses that will appear on the To: field */
    to: string[];
    subject: string;
    body: string;
    /** Comma separated list or an array of recipients e-mail addresses that will appear on the Cc: field */
    cc?: string[];
    /** Comma separated list or an array of recipients e-mail addresses that will appear on the Bcc: field */
    bcc?: string[];
    from?: string;
    date?: string;
}

export interface ConnectionOptions {
    /** Username for plain-text authentication. */
    user: string;
    /** Password for plain-text authentication. */
    password: string;

    /** Hostname or IP address of the IMAP server. Default: "localhost" */
    imapHost: string | undefined;
    /** Port number of the IMAP server. Default: 143 */
    imapPort?: number | undefined;

    /** The hostname or IP address to connect to. Default: "localhost" */
    smptHost: string | undefined;
    /** The port to connect to. Default: 25 or 465 */
    smtpPort?: number | undefined;
    /** Defines if the connection should use SSL (if true) or not (if false). Default: false */
    smtpSecure?: boolean | undefined;
    /** Service name */
    smtpService?: string | undefined;
}


export class Endpoint extends BaseEndpoint  implements MessangerService {
    protected connectionOptions: ConnectionOptions;

    protected _smtpConnection: Transporter<SMTPTransport.SentMessageInfo> = null;
    async getSmtpConnection(): Promise<Transporter<SMTPTransport.SentMessageInfo>> {
        return this._smtpConnection ||= await nodemailer.createTransport({
            service: this.connectionOptions.smtpService,
            host: this.connectionOptions.smptHost,
            port: this.connectionOptions.smtpPort || (this.connectionOptions.smtpSecure ? 465 : 25),
            secure: this.connectionOptions.smtpSecure || false,
            auth: {
                user: this.connectionOptions.user,
                pass: this.connectionOptions.password,
            }
        });
    }

    openImapConnection(): Imap {
        return new Imap({
            user: this.connectionOptions.user,
            password: this.connectionOptions.password,
            host: this.connectionOptions.imapHost,
            port: this.connectionOptions.imapPort,
            tls: true
        });
    }

    constructor(connectionOptions: ConnectionOptions) {
        super();
        this.connectionOptions = connectionOptions;
    }

    getInbox(options: CollectionOptions<EMail> = {}): Collection {
        options.displayName ??= 'INBOX';
        const collection: Collection = this.collections['INBOX'] as Collection;
        if (collection) return collection;
        return this._addCollection('INBOX', new Collection(this, 'INBOX', 'INBOX', options));
    }
    releaseInbox() {
        this._removeCollection('INBOX');
    }

    getMailbox(mailBox: string, options: CollectionOptions<EMail> = {}): Collection {
        options.displayName ??= mailBox;
        const collection: Collection = this.collections[mailBox] as Collection;
        if (collection) return collection;
        return this._addCollection(mailBox, new Collection(this, mailBox, mailBox, options));
    }
    releaseMailbox(mailBox: string) {
        this._removeCollection(mailBox);
    }

    async releaseEndpoint() {
        if (this._smtpConnection) this._smtpConnection.close();
        
        for (let key in this.collections) {
            if (!this.collections.hasOwnProperty(key)) continue;
            (this.collections[key] as Collection).closeConnection();
        }

        await super.releaseEndpoint();
    }

    public async send(to: string[] | string, subject: string, body: string, cc?: string[] | string, bcc?: string[] | string, from?: string): Promise<SendError | undefined>;
    public async send(value: EMail): Promise<SendError | undefined>;
    public async send(value: EMail | string[] | string, subject?: string, body?: string, cc?: string[] | string, bcc?: string[] | string, from?: string): Promise<SendError | undefined> {
        if (Array.isArray(value) || typeof value === 'string') {
            value = {
                to: typeof value === 'string' ? [value] : value,
                cc: typeof cc === 'string' ? [cc] : cc,
                bcc: typeof bcc === 'string' ? [bcc] : bcc,
                subject,
                body,
                from
            }
        }
        const smtp = await this.getSmtpConnection();
        let info = await smtp.sendMail({...value, text: value.body});
        if (info.accepted) return undefined;
        return info.response;
    }

    get displayName(): string {
        return `Email (${this.connectionOptions.user} on ${this.connectionOptions.smptHost})`;
    }    
}


export type SearchOptions = {
    seen?: boolean;
    answared?: boolean;
    fromContains?: string;
    subjectContains?: string;
    bodyContains?: string;
    beforeDate?: Date;
    sinceDate?: Date;
    onDate?: Date;
    sizeLager?: number;
    sizeSmaller?: number;
}

/**
 * Searches the currently open mailbox for messages using given criteria.
 * criteria is a list describing what you want to find.
 * For criteria types that require arguments, use an array instead of just the string criteria type name (e.g. ['FROM', 'foo@bar.com']).
 * Prefix criteria types with an "!" to negate.
 * The following message flags are valid types that do not have arguments:
 * ALL:            void;    // All messages.
 * ANSWERED:       void;    // Messages with the Answered flag set.
 * DELETED:        void;    // Messages with the Deleted flag set.
 * DRAFT:          void;    // Messages with the Draft flag set.
 * FLAGGED:        void;    // Messages with the Flagged flag set.
 * NEW:            void;    // Messages that have the Recent flag set but not the Seen flag.
 * SEEN:           void;    // Messages that have the Seen flag set.
 * RECENT:         void;    // Messages that have the Recent flag set.
 * OLD:            void;    // Messages that do not have the Recent flag set. This is functionally equivalent to "!RECENT" (as opposed to "!NEW").
 * UNANSWERED:     void;    // Messages that do not have the Answered flag set.
 * UNDELETED:      void;    // Messages that do not have the Deleted flag set.
 * UNDRAFT:        void;    // Messages that do not have the Draft flag set.
 * UNFLAGGED:      void;    // Messages that do not have the Flagged flag set.
 * UNSEEN:         void;    // Messages that do not have the Seen flag set.
 * The following are valid types that require string value(s):
 * BCC:            any;    // Messages that contain the specified string in the BCC field.
 * CC:             any;    // Messages that contain the specified string in the CC field.
 * FROM:           any;    // Messages that contain the specified string in the FROM field.
 * SUBJECT:        any;    // Messages that contain the specified string in the SUBJECT field.
 * TO:             any;    // Messages that contain the specified string in the TO field.
 * BODY:           any;    // Messages that contain the specified string in the message body.
 * TEXT:           any;    // Messages that contain the specified string in the header OR the message body.
 * KEYWORD:        any;    // Messages with the specified keyword set.
 * HEADER:         any;    // Requires two string values, with the first being the header name and the second being the value to search for.
 * If this second string is empty, all messages that contain the given header name will be returned.
 * The following are valid types that require a string parseable by JavaScripts Date object OR a Date instance:
 * BEFORE:         any;    // Messages whose internal date (disregarding time and timezone) is earlier than the specified date.
 * ON:             any;    // Messages whose internal date (disregarding time and timezone) is within the specified date.
 * SINCE:          any;    // Messages whose internal date (disregarding time and timezone) is within or later than the specified date.
 * SENTBEFORE:     any;    // Messages whose Date header (disregarding time and timezone) is earlier than the specified date.
 * SENTON:         any;    // Messages whose Date header (disregarding time and timezone) is within the specified date.
 * SENTSINCE:      any;    // Messages whose Date header (disregarding time and timezone) is within or later than the specified date.
 * The following are valid types that require one Integer value:
 * LARGER:         number;    // Messages with a size larger than the specified number of bytes.
 * SMALLER:        number;    // Messages with a size smaller than the specified number of bytes.
 * The following are valid criterion that require one or more Integer values:
 * UID:            any;    // Messages with UIDs corresponding to the specified UID set. Ranges are permitted (e.g. '2504:2507' or '*' or '2504:*').
 */

export class Collection extends BaseCollection<EMail> {
    protected static instanceCount = 0;

    protected mailBox: string;
    protected connection: Imap = null;

    constructor(endpoint: BaseEndpoint, collectionName: string, mailBox: string, options: CollectionOptions<EMail> = {}) {
        Collection.instanceCount++;
        super(endpoint, collectionName, options);
        this.mailBox = mailBox;
    }

    // TODO: make it stopable with with await this.waitWhilePaused()
    public select(): BaseObservable<EMail>;
    public select(searchOptions: SearchOptions, markSeen?: boolean): BaseObservable<EMail>;
    public select(range: string, markSeen?: boolean): BaseObservable<EMail>;
    public select(searchCriteria: any[], markSeen?: boolean ): BaseObservable<EMail>;
    public select(searchCriteria?: any[] | string | SearchOptions, markSeen: boolean = false): BaseObservable<EMail> {
        const observable = new BaseObservable<EMail>(this, (subscriber) => {
                try {
                    //if (this.connection) this.closeConnection();
                    if (!this.connection) this.connection = this.endpoint.openImapConnection();

                    const openInbox = (cb) => {
                        this.connection.openBox(this.mailBox, true, cb);
                    }

                    this.connection.once('ready', () => {
                        openInbox((err, box) => {
                            if (err) throw err;

                            this.sendStartEvent();

                            const options = { 
                                bodies: ['HEADER.FIELDS (FROM TO CC BCC SUBJECT DATE)', 'TEXT'], 
                                markSeen 
                            };

                            const selectAllRecords = () => {
                                const f = this.connection.seq.fetch(box.messages.total + ':*', options);
                                this._select(f, subscriber);
                            }

                            switch (typeof searchCriteria) {
                                case 'undefined': {
                                    return selectAllRecords();
                                }
                                case 'string': {
                                    if (!searchCriteria) return selectAllRecords();
                                    const f = this.connection.seq.fetch(searchCriteria, options);
                                    this._select(f, subscriber);
                                    return;
                                }
                                case 'object': {
                                    if (Array.isArray(searchCriteria)) {
                                        if (!searchCriteria.length) return selectAllRecords();
                                        this.connection.search(searchCriteria, (err, results) => {
                                            if (err) throw err;
                                            const f = this.connection.fetch(results, options);
                                            this._select(f, subscriber);
                                        })
                                    }
                                    else {
                                        const searchCriteriaArr: any[] = [];
                                        if (typeof searchCriteria.seen !== 'undefined') searchCriteriaArr.push(searchCriteria.seen ? 'SEEN' : 'UNSEEN');
                                        if (typeof searchCriteria.answared !== 'undefined') searchCriteriaArr.push(searchCriteria.answared ? 'ANSWERED' : 'UNANSWERED');
                                        if (searchCriteria.sinceDate) searchCriteriaArr.push(['SINCE', searchCriteria.sinceDate.toDateString()]);
                                        if (searchCriteria.beforeDate) searchCriteriaArr.push(['BEFORE', searchCriteria.beforeDate.toDateString()]);
                                        if (searchCriteria.onDate) searchCriteriaArr.push(['ON', searchCriteria.onDate.toDateString()]);
                                        if (searchCriteria.fromContains) searchCriteriaArr.push(['FROM', searchCriteria.fromContains]);
                                        if (searchCriteria.bodyContains) searchCriteriaArr.push(['BODY', searchCriteria.bodyContains]);
                                        if (searchCriteria.subjectContains) searchCriteriaArr.push(['SUBJECT', searchCriteria.subjectContains]);
                                        if (searchCriteria.sizeLager) searchCriteriaArr.push(['LARGER', searchCriteria.sizeLager]);
                                        if (searchCriteria.sizeSmaller) searchCriteriaArr.push(['SMALLER', searchCriteria.sizeSmaller]);

                                        if (!searchCriteriaArr.length) return selectAllRecords();

                                        this.connection.search(searchCriteriaArr, (err, uids: number[]) => {
                                            if (err) throw err;
                                            const f = this.connection.fetch(uids, options);
                                            this._select(f, subscriber);
                                        })
                                    }
                                }
                            }
                        });
                    });
                        
                    this.connection.once('error', (err) => {
                        this.sendErrorEvent(err);
                        if (!subscriber.closed) subscriber.error(err);
                        //console.log(err);
                    });
                        
                    this.connection.once('end', () => {
                        //console.log('Connection ended');
                    });
                        
                    this.connection.connect();
                }
                catch(err) {
                    this.sendErrorEvent(err);
                    if (!subscriber.closed) subscriber.error(err);
                }

        });
        return observable;
    }

    // TODO: make it stopable with with await this.waitWhilePaused()
    protected _select(f: Imap.ImapFetch, subscriber: Subscriber<EMail>) {
        f.on('message', (msg, seqno) => {
            //console.log('Message #%d', seqno);

            //let prefix = '(#' + seqno + ') ';
            
            const email: EMail = {
                to: [],
                cc: [],
                bcc: [],
                from: '',
                subject: '',
                body: '',
                date: ''
            };

            msg.on('body', (stream, info) => {
                let buffer: string = '', count = 0;

                if (info.which === 'TEXT') {
                    //console.log(prefix + 'Body [%s] found, %d total bytes', inspect(info.which), info.size);
                }

                stream.on('data', (chunk) => {
                    count += chunk.length;
                    buffer += chunk.toString('utf8');
                    if (info.which === 'TEXT') {
                        //console.log(prefix + 'Body [%s] (%d/%d)', inspect(info.which), count, info.size);
                    }
                });

                stream.once('end', () => {
                    if (info.which === 'TEXT') {
                        email.body = buffer;
                        //console.log(prefix + 'Body [%s] Finished', inspect(info.which));
                    }
                    else {
                        const header = Imap.parseHeader(buffer);
                        email.from = header.from && header.from[0] || undefined;
                        email.to = header.to;
                        email.cc = header.cc;
                        email.bcc = header.bcc;
                        email.subject = header.subject && header.subject[0] || '';
                        email.date = header.date && header.date[0] || undefined;
                        //console.log(prefix + 'Parsed header: %s', inspect(Imap.parseHeader(buffer)));
                    }  
                });
            });

            // msg.once('attributes', (attrs) => {
            //     console.log(prefix + 'Attributes: %s', inspect(attrs, false, 8));
            // });

            msg.once('end', () => {
                //console.log(prefix + 'Finished');
                if (subscriber.closed) return;
                //await this.waitWhilePaused();
                //if (subscriber.closed) return;

                this.sendReciveEvent(email);
                subscriber.next(email);
            });
        });

        f.once('error', (err) => {
            //console.log('Fetch error: ' + err);
            this.sendErrorEvent(err);
            if (!subscriber.closed) subscriber.error(err);
        });

        f.once('end', () => {
            //console.log('Done fetching all messages!');
            if (!subscriber.closed) subscriber.complete();
            this.sendEndEvent();
            this.closeConnection();
        });
    }

    public async get(UID: string | number, markSeen: boolean = false): Promise<EMail> {
        return new Promise<EMail>((resolve: (value: EMail | PromiseLike<EMail>) => void, reject: (reason?: any) => void) => {
            if (!this.connection) this.connection = this.endpoint.openImapConnection();

            const openInbox = (cb) => {
                this.connection.openBox(this.mailBox, true, cb);
            }

            this.connection.once('ready', () => {
                openInbox((err, box) => {
                    if (err) throw err;

                    const options = { 
                        bodies: ['HEADER.FIELDS (FROM TO CC BCC SUBJECT DATE)', 'TEXT'], 
                        markSeen 
                    };

                    const searchCriteria = [[ 'UID', [UID] ]];

                    this.connection.search(searchCriteria, (err, results) => {
                        if (err) throw err;
                        const f = this.connection.fetch(results, options);
                        this._get(f, resolve, reject);
                    })
                });
            });
                
            this.connection.once('error', (err) => {
                reject(err);
            });
                
            this.connection.connect();
        });
    }

    // TODO: make it stopable with with await this.waitWhilePaused()
    protected _get(f: Imap.ImapFetch, resolve: ((value: EMail | PromiseLike<EMail>) => void), reject: ((reason?: any) => void)) {
        f.on('message', (msg, seqno) => {
            const email: EMail = {
                to: [],
                cc: [],
                bcc: [],
                from: '',
                subject: '',
                body: '',
                date: ''
            };

            msg.on('body', (stream, info) => {
                let buffer: string = '', count = 0;

                stream.on('data', (chunk) => {
                    count += chunk.length;
                    buffer += chunk.toString('utf8');
                });

                stream.once('end', () => {
                    if (info.which === 'TEXT') {
                        email.body = buffer;
                    }
                    else {
                        const header = Imap.parseHeader(buffer);
                        email.from = header.from && header.from[0] || undefined;
                        email.to = header.to;
                        email.cc = header.cc;
                        email.bcc = header.bcc;
                        email.subject = header.subject && header.subject[0] || '';
                        email.date = header.date && header.date[0] || undefined;
                    }  
                });
            });

            msg.once('end', () => {
                resolve(email);
            });
        });

        f.once('error', (err) => {
            reject(err);
        });

        f.once('end', () => {
            this.closeConnection();
        });
    }

    closeConnection() {
        // if (this.connection) this.connection.closeBox(err => {
        //     throw err;
        // });
        this.connection.end();
        this.connection = null;
    }

    get endpoint(): Endpoint {
        return super.endpoint as Endpoint;
    }    
}

