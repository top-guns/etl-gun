import * as email from './email.js';

/*
You should to get the application password from Gmail service to use this endpoint. 

Follow this steps to get it:
 - 1. Login to you Gmail account
 - 2. Open this link https://myaccount.google.com/security
 - 3. Enable 2 factor authentication
 - 4. Go to https://myaccount.google.com/apppasswords
 - 5. From Select App options select Other and write your app name (it could be any name like mycustomapp)
 - 6. It will generate you the password - copy the password from the popup
 - Use that copied password in the application password parameter in the Gmail endpoint constructor.
*/

export class Endpoint extends email.Endpoint {
    protected userEmail: string;

    constructor(userEmail: string, appPassword: string) {
        super({
            user: userEmail,
            password: appPassword,

            imapHost: 'imap.gmail.com',
            imapPort: 993,

            smptHost: 'smtp.gmail.com',
            smtpPort: 465,
            smtpSecure: true,
            smtpService: 'gmail'
        });
        this.userEmail = userEmail;
    }

    get displayName(): string {
        return `Gmail (${this.userEmail})`;
    }
}