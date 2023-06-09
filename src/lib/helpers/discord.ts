import http from "http";
import open from "open";

type Credentials = {
    access_token: string
    expires_in: number
    refresh_token: string
    scope: 'identify' | string
    token_type: 'Bearer' | string
}

export class DiscordHelper {
    async loginViaBrowser() {
        return await new Promise<Credentials>((resolve, reject) => {
            const server = http.createServer(async (req, res) => {
                const url = new URL(`http://localhost:${process.env.DISCORD_REDIRECT_PORT}${req.url}`);
                const code = url.searchParams.get('code');

                if (!code) {
                    this.sendResponse(res, 400, 'Authorisation error');
                    server.close();
                    reject('Authorisation error');
                    return;
                }

                const credentials = await this.getCredentialsByCode(code);
                this.sendResponse(res, 200, 'Log in successful');
                server.close();
                resolve(credentials);
            })

            server.listen(process.env.DISCORD_REDIRECT_PORT);
            open(process.env.DISCORD_LOGIN_URL!);
        })
    }

    protected async getCredentialsByCode(code: string) {
        const response = await fetch('https://discord.com/api/v10/oauth2/token', {
            method: 'POST',
            body: new URLSearchParams({
                client_id: process.env.DISCORD_CLIENT_ID!,
                client_secret: process.env.DISCORD_CLIENT_SECRET!,
                code,
                grant_type: 'authorization_code',
                redirect_uri: process.env.DISCORD_REDIRECT_URI!,
            }).toString(),
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
        })

        return await response.json();
    }

    protected sendResponse(resp: any, status: number, text: string) {
        resp.writeHead(status, {
            'Content-Length': text.length,
            'Content-Type': 'text/plain'
        });
        resp.end(text);
    }
}