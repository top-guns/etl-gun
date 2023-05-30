import { Subscriber } from "rxjs";
import { BaseEndpoint} from "../../core/endpoint.js";
import TelegramBot, { InlineKeyboardButton, InlineKeyboardMarkup } from 'node-telegram-bot-api';
import { BaseObservable } from '../../core/observable.js';
import { BaseCollection_I } from '../../core/base_collection_i.js';
import { CollectionOptions } from '../../core/base_collection.js';

export type InputMessage = {
    chatId: string;
    message: string;
}

export class Endpoint extends BaseEndpoint {
    protected static _instance: Endpoint;
    static get instance(): Endpoint {
        return Endpoint._instance ||= new Endpoint();
    }

    protected constructor() {
        super();
    }

    startBot(collectionName: string, token: string, keyboard?: any, options: CollectionOptions<InputMessage> = {}): Collection {
        options.displayName ??= collectionName;
        return this._addCollection(collectionName, new Collection(this, collectionName, token, keyboard, options));
    }

    releaseBot(collectionName: string) {
        this.collections[collectionName].stop();
        this._removeCollection(collectionName);
    }

    get displayName(): string {
        return `Telegram (${this.instanceNo})`;
    }
}

export function getEndpoint(): Endpoint {
    return Endpoint.instance;
}

export class Collection extends BaseCollection_I<InputMessage> {
    protected static instanceNo = 0;

    protected token: string;
    protected keyboard: any;
    protected subscriber: Subscriber<InputMessage>;
    protected bot: TelegramBot;

    constructor(endpoint: Endpoint, collectionName: string, token: string, keyboard?: any, options: CollectionOptions<InputMessage> = {}) {
        Collection.instanceNo++;
        super(endpoint, collectionName, options);
        this.token = token;
        this.keyboard = keyboard;
    }

    public select(): BaseObservable<InputMessage> {
        const observable = new BaseObservable<InputMessage>(this, (subscriber) => {
            try {
                this.sendStartEvent();

                this.bot = new TelegramBot(this.token, { polling: true });

                this.subscriber = subscriber;

                this.bot.onText(/(.+)/, (msg: any, match: any) => {
                    (async () => {
                        const message: InputMessage = {
                            chatId: msg.chat.id,
                            message: match[1]
                        }

                        if (subscriber.closed) {
                            await this.stop();
                            return;
                        }
                        await this.waitWhilePaused();
                        this.sendReciveEvent(message);
                        subscriber.next(message);

                        if (this.keyboard) this.installKeyboard(message.chatId, "", this.keyboard);
                    })();
                })
            }
            catch(err) {
                this.sendErrorEvent(err);
                subscriber.error(err);
            }
        });
        return observable;
    }

    public async stop() {
        await this.bot.close();
        this.subscriber.complete();
        this.sendEndEvent();
        this.subscriber = undefined;
        this.bot = undefined;
    }

    public async insert(value: InputMessage): Promise<void>;
    public async insert(chatId: string, message: string): Promise<void>;
    public async insert(valueOrChartId: InputMessage | string, message?: string): Promise<void> {
        if (!this.bot) throw new Error("Cannot use push() while telegram bot is not active. Please, call list() before.");
        const value = typeof valueOrChartId === 'string' ? {chatId: valueOrChartId, message} : valueOrChartId;
        this.sendInsertEvent(value);
        this.bot.sendMessage(value.chatId, value.message);
    }

    public setKeyboard(keyboard: any) {
        this.keyboard = keyboard;
    }

    protected installKeyboard(chatid: string, message: string, keyboard: any) {
        this.bot.sendMessage(chatid, message, {
            "reply_markup": {
                "keyboard": keyboard
            }
        })
    }

    async installKeyboard2(chatid: number, message: string) {
        //if (!force && chats[chatid]) return;

        //chats[chatid] = 'ok';

        // bot.sendMessage(chatid, "Welcome", {
        //     "reply_markup": {
        //         "keyboard": [["one", "two"],   ["3"], ["4"]]
        //     }
        // });

        let inlineKeyboardButton: InlineKeyboardButton = {text: "/задача", callback_data: 'Button \"задача\" has been pressed'};
        let inlineKeyboardMarkup: InlineKeyboardMarkup = {inline_keyboard: [[inlineKeyboardButton]]};

        await this.bot.sendMessage(chatid, message, {reply_markup: inlineKeyboardMarkup});
    }

    async removeKeyboard(chatid: number) {
        await this.bot.sendMessage(chatid, 'Keyboard has closed', {
            "reply_markup": {
                "remove_keyboard": true
            }
        })
    }

}