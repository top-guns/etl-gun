export type SendError = string;

export interface MessangerService {
    send(message: string, to: string | string[], from?: string): Promise<SendError | undefined>;
}