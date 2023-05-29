export type SendError = string;

export interface MessangerService {
    send(message: string | any, to: string | any, from?: string): Promise<SendError | undefined>;
}