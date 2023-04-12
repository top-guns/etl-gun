import { ForegroundColor } from 'chalk';
import { ConsoleManager, OptionPopup, InputPopup, PageBuilder, ButtonPopup, ConfirmPopup } from 'console-gui-tools-cjs';
import { SimplifiedStyledElement } from 'console-gui-tools-cjs/dist/components/PageBuilder';
import { Endpoint, EndpointGuiOptions } from './endpoint';

type EndpointDesc = {
    endpoint: Endpoint<any>;
    displayName: string;
    status: 'waiting' | 'running' | 'finished' | 'error' | 'pushed' | 'cleared';
    value: any;
    guiOptions: EndpointGuiOptions<any>;
}

export class GuiManager {
    protected static _instance: GuiManager = null;

    protected title = 'RxJs-ETL-Kit'; // Title of the console
    public processStatus: 'paused' | 'started' | 'finished' = 'started';
    public makeStepForward: boolean = false;
    protected consoleManager: ConsoleManager;
    protected endpoints: EndpointDesc[] = [];
    protected popup: ConfirmPopup = null;

    public static startGui(title = '', startPaused = false, logPageSize = 8) {
        if (GuiManager._instance) throw new Error("GuiManager: gui manager allready started. You cannot use more then one gui manager.");
        GuiManager._instance = new GuiManager(title, startPaused, logPageSize);
    }

    public static stopGui() {
        if (GuiManager._instance) {
            //GuiManager._instance.consoleManager.removeListener("keypressed", GuiManager._instance.keypressListener);
            //GuiManager._instance.consoleManager.removeAllListeners();
            //console.clear();
            if (GuiManager._instance.popup) GuiManager._instance.popup.hide();
            GuiManager._instance.processStatus = 'finished';
            GuiManager._instance.updateConsole();
            process.stdout.cursorTo(0, 14 + GuiManager._instance.consoleManager.getLogPageSize());
            delete GuiManager._instance.consoleManager;
            delete GuiManager._instance;
            process.stdin.setRawMode(false);
        }
        GuiManager._instance = null;
    }

    public static isGuiStarted() {
        return !!GuiManager._instance;
    }

    public static get instance() {
        //if (!GuiManager._instance) throw new Error("GuiManager: gui is not started.");
        return GuiManager._instance;
    }

    protected constructor(title = '', startPaused = false, logPageSize = 8) {
        this.consoleManager = new ConsoleManager({
            title: this.title,
            //enableMouse: true,
            logPageSize,            // Number of lines to show in logs page
            showLogKey: 'ctrl+l',   // Change layout with ctrl+l to switch to the logs page
        })
        
        this.title = title;
        this.processStatus = startPaused ? 'paused' : 'started';

        for(let i = 0; i < logPageSize; i++) this.consoleManager.log('');

        // this.consoleManager.on("exit", () => {
        //     this.exit();
        // })

        // And manage the keypress event from the library
        this.consoleManager.on("keypressed", this.keypressListener);
    }

    public static quitApp() {
        GuiManager.stopGui();
        process.exit();
    }

    protected keypressListener = (key) => {
        switch (key.name) {
            case 'space':
                this.processStatus = this.processStatus == 'finished' ? 'finished' : this.processStatus == 'paused' ? 'started' : 'paused';
                this.updateConsole();
                break
            case 'return':
                this.makeStepForward = this.processStatus == 'paused';
                break
            case 'escape':
                this.popup = new ConfirmPopup("popupQuit", "Exit application", "Are you sure want to exit?").show().on("confirm", () => GuiManager.quitApp())
                break
            default:
                break
        }
    }

    // Creating a main page updater:
    protected updateConsole(){
        const p: PageBuilder = new PageBuilder();

        p.addRow({ text: " Process:  ", color: 'white' }, { 
            text: ` ${this.processStatus} `, 
            color: this.processStatus == 'paused' ? 'black' : 'white', 
            bold: true, 
            bg: this.processStatus == 'paused' ? 'bgYellowBright' : this.processStatus == 'started' ? 'bgGreen' : 'bgRedBright' 
        });

        p.addSpacer();

        p.addRow({ text: " Endpoints:", color: 'white', bg: 'bgBlack' });
        this.endpoints.forEach(desc => {
            let color: ForegroundColor;
            switch (desc.status) {
                case 'running':
                    color = 'blueBright';
                    break;
                case 'finished':
                    color = 'green';
                    break;
                case 'error':
                    color = 'red';
                    break;
                case 'pushed':
                    color = 'magenta';
                    break;
                case 'cleared':
                    color = 'yellow';
                    break;
                case 'waiting':
                    color = 'white';
                    break;
                default:
                    color = 'white';
            }
            p.addRow({ text: `  ` }, 
                { text: `${this.getEndpointDisplayName(desc)}`, color: 'blueBright' }, 
                { text: `  ${desc.status.padEnd(8, ' ')}`, color }, 
                { text: `  ${desc.value ? (desc.guiOptions.watch && desc.status !== 'error' ? desc.guiOptions.watch(desc.value) : desc.value) : ''}`, color: 'white' });
        })

        // Spacer
        p.addSpacer();

        // if (lastErr.length > 0) {
        //     p.addRow({ text: lastErr, color: 'red' })
        //     p.addSpacer(2)
        // }

        p.addRow({ text: " Commands:", color: 'white', bg: 'bgBlack' });
        p.addRow({ text: `  'space'`, color: 'gray', bold: true },  { text: `   - Pause/resume process`, color: 'white', italic: true });
        p.addRow({ text: `  'enter'`, color: 'gray', bold: true },  { text: `   - Make one step in paused mode`, color: 'white', italic: true });
        p.addRow({ text: `  'esc'`, color: 'gray', bold: true },    { text: `     - Quit`, color: 'white', italic: true });

        this.consoleManager.setPage(p)
    }

    public log(obj: {}, before?: string);
    public log(message: string, before?: string);
    public log(obj: any, before: string = '') {
        if (typeof obj === 'string') this.consoleManager.stdOut.addRow({text: before, color: 'white'}, { text: '' + obj, color: "white" });
        else this.consoleManager.stdOut.addRow({text: before, color: 'white'}, ...this.dumpObject(obj));
        this.updateConsole();
    }
    public warn(message: string) {
        this.consoleManager.warn(message);
    }
    public error(message: string) {
        this.consoleManager.error(message);
    }
    public info(message: string) {
        this.consoleManager.info(message);
    }

    public registerEndpoint(endpoint: Endpoint<any>, guiOptions: EndpointGuiOptions<any> = {}) {
        const displayName = guiOptions.displayName ? guiOptions.displayName : `Endpoint ${this.endpoints.length}`;
        const desc: EndpointDesc = {endpoint, displayName, status: 'waiting', value: '', guiOptions};
        this.endpoints.push(desc);

        endpoint.on('read.start', () => { desc.status = 'running'; this.updateConsole(); });
        endpoint.on('read.end', () => { desc.status = 'finished'; this.updateConsole(); });
        endpoint.on('read.data', v => { desc.status = 'running'; desc.value = v; this.updateConsole(); });

        endpoint.on('read.error', v => { desc.status = 'error'; desc.value = v; this.updateConsole(); });
        endpoint.on('push', v => { desc.status = 'pushed'; desc.value = v; this.updateConsole(); });
        endpoint.on('clear', v => { desc.status = 'cleared'; desc.value = v; this.updateConsole(); });

        this.updateConsole();
    }

    protected getEndpointNameLength(): number {
        const maxName = this.endpoints.reduce((p, c) => p.displayName > c.displayName ? p : c);
        return maxName.displayName.length;
    }

    protected getEndpointDisplayName(desc: EndpointDesc): string {
        return desc.displayName.padEnd(this.getEndpointNameLength() + 4, ' ');
    }

    protected deleteCurrentLine() {
        process.stdout.write("\x1B[1A\x1B[K");
    }

    protected dumpObject(obj: any, deep: number = 1): SimplifiedStyledElement[] {
        let res: SimplifiedStyledElement[] = [];
        for (let key in obj) {
            if (obj.hasOwnProperty(key)) {
                if (res.length) res.push({ text: `, `, color: "cyanBright" });
                if (!obj.length) res.push({ text: `${key}: `, color: "cyanBright" });

                switch (typeof obj[key]) {
                    case 'number': res.push({text: '' + obj[key], color: "blueBright"}); break;
                    case 'string': res.push({text: `"${obj[key]}"`, color: "yellowBright"}); break; 
                    case 'boolean': res.push({text: '' + obj[key], color: "greenBright"}); break; 
                    case 'function': res.push({text: '()', color: "white"}); break; 
                    case 'object': {
                        if (obj[key].length) res.push({text: '[]', color: "white"}); 
                        else res.push({text: '{}', color: "white"}); 
                        break;
                    }
                    default: res.push({text: '' + obj[key], color: "white"}); break; 
                }
            }
        }
        if (obj.length) return [{ text: `[`, color: "cyanBright" }, ...res, { text: `]`, color: "cyanBright" }];
        return [{ text: `{`, color: "cyanBright" }, ...res, { text: `}`, color: "cyanBright" }];
    }
}
