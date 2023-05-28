export type SmsError = {
    code: string;
    message: string;
}

export interface SmsService {
    sendSms(message: string, toPhone: string, from?: string): Promise<SmsError | undefined>;
}