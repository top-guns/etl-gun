import http from "http";
import open, {openApp, apps} from 'open';

type Credentials = {
    access_token: string
    expires_in: number
    refresh_token: string
    scope: 'identify' | string
    token_type: 'Bearer' | string
}

export class DiscordHelper {
    protected DISCORD_CLIENT_ID: string;
    protected DISCORD_CLIENT_SECRET: string;
    protected DISCORD_REDIRECT_URI: string;
    protected PORT: number;

    protected credentials: Credentials = null;

    constructor() {
        this.DISCORD_CLIENT_ID = process.env.DISCORD_CLIENT_ID!;
        this.DISCORD_CLIENT_SECRET = process.env.DISCORD_CLIENT_SECRET!;
        this.DISCORD_REDIRECT_URI = process.env.DISCORD_REDIRECT_URI!;
        this.PORT = 3000;
    }

    async loginViaBrowser() {
        return await new Promise<Credentials>((resolve, reject) => {
            const server = http.createServer(async (req, res) => {
                const url = new URL(req.url);
                const code = url.searchParams.get('code');

                const credentials = await this.getCredentialsByCode(code);

                var body = 'Log in successful';
                var content_length = body.length;
                res.writeHead(200, {
                    'Content-Length': content_length,
                    'Content-Type': 'text/plain'
                });

                res.end(body);

                server.close();
                resolve(credentials);
            })

            server.listen(this.PORT);
            open(this.DISCORD_REDIRECT_URI);
        })
    }

    protected async getCredentialsByCode(code: string) {
        const response = await fetch('https://discord.com/api/v10/oauth2/token', {
            method: 'POST',
            body: new URLSearchParams({
                client_id: this.DISCORD_CLIENT_ID!,
                client_secret: this.DISCORD_CLIENT_SECRET!,
                code,
                grant_type: 'authorization_code',
                redirect_uri: this.DISCORD_REDIRECT_URI!,
            }).toString(),
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
        })

        return await response.json();
    }
}