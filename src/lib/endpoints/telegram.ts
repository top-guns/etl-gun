import * as pg from 'pg'
import { Observable, Subscriber } from "rxjs";
import { Endpoint } from '../core';
import { EndpointImpl } from '../core/endpoint';
import { EtlObservable } from '../core/observable';
import TelegramBot from 'node-telegram-bot-api';

export type TelegramInputMessage = {
    chatId: string;
    message: string;
}

export class TelegramEndpoint extends EndpointImpl<TelegramInputMessage> {
    protected token: string;
    protected keyboard: any;
    protected subscriber: Subscriber<TelegramInputMessage>;
    protected bot: TelegramBot;

    constructor(token: string, keyboard?: any, displayName: string = '') {
        super(displayName ? displayName : `Telegram`);
        this.token = token;
        this.keyboard = keyboard;
    }

    public read(): EtlObservable<TelegramInputMessage> {
        const observable = new EtlObservable<TelegramInputMessage>((subscriber) => {
            try {
                this.sendStartEvent();

                this.bot = new TelegramBot(this.token, { polling: true });

                this.subscriber = subscriber;

                this.bot.onText(/(.+)/, (msg: any, match: any) => {
                    const message: TelegramInputMessage = {
                        chatId: msg.chat.id,
                        message: match[1]
                    }

                    this.sendDataEvent(message);
                    subscriber.next(message);
            
                    if (this.keyboard) this.installKeyboard(message.chatId, "", this.keyboard);
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
        if (!this.bot) throw new Error("Cannot use push() while telegram bot is not active. Please, call read() before.");
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
        });
    }

}
