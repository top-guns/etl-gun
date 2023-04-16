import * as pg from 'pg'
import { Observable, Subscriber } from "rxjs";
import { Endpoint} from "../core/endpoint.js";
import { Collection, CollectionGuiOptions, CollectionImpl } from "../core/collection.js";
import { EtlObservable } from '../core/observable.js';
import TelegramBot, { InlineKeyboardButton, InlineKeyboardMarkup } from 'node-telegram-bot-api';

export type TelegramInputMessage = {
    chatId: string;
    message: string;
}

export class TelegramEndpoint extends Endpoint {
    constructor() {
        super();
    }

    startBot(collectionName: string, token: string, keyboard?: any, guiOptions: CollectionGuiOptions<TelegramInputMessage> = {}): MessageCollection {
        guiOptions.displayName ??= collectionName;
        return this._addCollection(collectionName, new MessageCollection(this, token, keyboard, guiOptions));
    }

    releaseBot(collectionName: string) {
        this.collections[collectionName].stop();
        this._removeCollection(collectionName);
    }

    get displayName(): string {
        return `Telegram (${this.instanceNo})`;
    }
}

export class MessageCollection extends CollectionImpl<TelegramInputMessage> {
    protected static instanceNo = 0;
    protected token: string;
    protected keyboard: any;
    protected subscriber: Subscriber<TelegramInputMessage>;
    protected bot: TelegramBot;

    constructor(endpoint: TelegramEndpoint, token: string, keyboard?: any, guiOptions: CollectionGuiOptions<TelegramInputMessage> = {}) {
        MessageCollection.instanceNo++;
        super(endpoint, guiOptions);
        this.token = token;
        this.keyboard = keyboard;
    }

    public list(): EtlObservable<TelegramInputMessage> {
        const observable = new EtlObservable<TelegramInputMessage>((subscriber) => {
            try {
                this.sendStartEvent();

                this.bot = new TelegramBot(this.token, { polling: true });

                this.subscriber = subscriber;

                this.bot.onText(/(.+)/, (msg: any, match: any) => {
                    (async () => {
                        const message: TelegramInputMessage = {
                            chatId: msg.chat.id,
                            message: match[1]
                        }

                        await this.waitWhilePaused();
                        this.sendDataEvent(message);
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

    public async push(value: TelegramInputMessage);
    public async push(chatId: string, message: string);
    public async push(valueOrChartId: TelegramInputMessage | string, message?: string) {
        if (!this.bot) throw new Error("Cannot use push() while telegram bot is not active. Please, call list() before.");
        const value = typeof valueOrChartId === 'string' ? {chatId: valueOrChartId, message} : valueOrChartId;
        super.push(value);
        this.bot.sendMessage(value.chatId, value.message);
    }

    public async clear() {
        //super.clear();
        throw new Error("Method not implemented.");
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
