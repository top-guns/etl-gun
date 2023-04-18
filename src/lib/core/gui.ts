import { ForegroundColor } from 'chalk';
import { ConsoleManager, OptionPopup, InputPopup, PageBuilder, ButtonPopup, ConfirmPopup } from 'console-gui-tools';
import { SimplifiedStyledElement } from 'console-gui-tools';
import { BaseCollection, CollectionGuiOptions } from './collection.js';
import { BaseEndpoint } from './endpoint.js';

type EndpointDesc = {
    endpoint: BaseEndpoint;
    collections: CollectionDesc[];
}

type CollectionDesc = {
    collection: BaseCollection<any>;
    displayName: string;
    status: 'waiting' | 'running' | 'finished' | 'error' | 'inserted' | 'deleted' | 'recived' | 'updated' | 'upserted';
    value: any;
    guiOptions: CollectionGuiOptions<any>;
}

export class GuiManager {
    protected static _instance: GuiManager = null;

    protected title = 'RxJs-ETL-Kit'; // Title of the console
    public processStatus: 'paused' | 'started' | 'finished' = 'started';
    public stepByStepMode: boolean = false;
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
            GuiManager._instance.setCursorAfterWindow();
            //delete GuiManager._instance.consoleManager;
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
        this.stepByStepMode = startPaused;
        this.makeStepForward = false;
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
                switch (this.processStatus) {
                    case 'finished': break;
                    case 'paused': 
                        this.processStatus = 'started'; 
                        this.stepByStepMode = false; 
                        break;
                    case 'started': 
                        this.processStatus = 'paused'; 
                        this.stepByStepMode = true; 
                        this.makeStepForward = false; 
                        break;
                }
                this.updateConsole();
                break
            case 'return':
                if (this.processStatus == 'finished') break;
                this.stepByStepMode = true; 
                if (this.processStatus == 'paused') {
                    this.processStatus = 'started'; 
                    this.makeStepForward = true;
                }
                else {
                    this.processStatus = 'paused'; 
                    this.makeStepForward = false;
                }
                this.updateConsole();
                break;
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

        p.addRow({ text: " Process:  ", color: 'white', bold: true }, { 
            text: ` ${this.processStatus} `, 
            color: this.processStatus == 'paused' ? 'black' : 'white', 
            bold: true, 
            bg: this.processStatus == 'paused' ? 'bgYellowBright' : this.processStatus == 'started' ? 'bgGreen' : 'bgRedBright' 
        });

        p.addSpacer();

        p.addRow({ text: " Collections:", color: 'white', bg: 'bgBlack', bold: true });

        this.endpoints.forEach(epDesc => {
            const parsedDisplayName = epDesc.endpoint.displayName.match(/^(.*[^\s])\s*\((.*)\)$/);
            if (parsedDisplayName.length > 2) {
                p.addRow({ text: `  ` }, { text: `${parsedDisplayName[1]}`, color: 'white', underline: true }, { text: ` ${parsedDisplayName[2]}`, color: 'whiteBright' } );
            }
            else {
                p.addRow({ text: `  ` }, { text: `${epDesc.endpoint.displayName}`, color: 'white', underline: true } );
            }
            

            epDesc.collections.forEach(desc => {
                let color: ForegroundColor;
                switch (desc.status) {
                    case 'running':
                        color = 'blueBright';
                        break;
                    case 'recived':
                        color = 'magentaBright';
                        break;
                    case 'finished':
                        color = 'green';
                        break;
                    case 'error':
                        color = 'red';
                        break;
                    case 'inserted':
                        color = 'greenBright';
                        break;
                    case 'updated':
                        color = 'cyanBright';
                        break;
                    case 'upserted':
                        color = 'yellowBright';
                        break;
                    case 'deleted':
                        color = 'redBright';
                        break;
                    case 'waiting':
                        color = 'white';
                        break;
                    default:
                        color = 'white';
                }
                p.addRow({ text: `    ` }, 
                    { text: `${this.getCollectionDisplayName(desc)}`, color: 'white' }, 
                    { text: `  ${desc.status.padEnd(8, ' ')}    `, color }, 
                    ...(
                        desc.status == 'error' ? [{ text: `${desc.value}`, color: 'red' } as SimplifiedStyledElement] :  
                        !desc.value ? [{ text: ``, color: 'white' } as SimplifiedStyledElement] : 
                        this.dumpObject(desc.guiOptions.watch ? desc.guiOptions.watch(desc.value) : desc.value)
                    )
                )
            })
        })

        // Spacer
        p.addSpacer();

        // if (lastErr.length > 0) {
        //     p.addRow({ text: lastErr, color: 'red' })
        //     p.addSpacer(2)
        // }

        p.addRow({ text: " Commands:", color: 'white', bg: 'bgBlack', bold: true });
        p.addRow({ text: `  'space'`, color: 'gray', bold: true },      { text: `    - Pause/resume process`, color: 'white', italic: true });
        p.addRow({ text: `  'enter'`, color: 'gray', bold: true },      { text: `    - Make one step in paused mode`, color: 'white', italic: true });
        p.addRow({ text: `  'esc'`, color: 'gray', bold: true },        { text: `      - Quit`, color: 'white', italic: true });
        p.addRow({ text: `  'ctrl+l'`, color: 'gray', bold: true },     { text: `   - Switch to log`, color: 'white', italic: true });
        p.addRow({ text: `  'up/down'`, color: 'gray', bold: true },    { text: `  - Scroll log`, color: 'white', italic: true });

        this.consoleManager.setPage(p);
        this.setCursorAfterWindow();
    }

    public static log(message?: any, ...optionalParams: any[]) {
        if (GuiManager.isGuiStarted()) GuiManager.instance.log(optionalParams.length ? optionalParams[0] : message, optionalParams.length ? message : undefined);
        else console.log(message, ...optionalParams);
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

    public registerEndpoint(endpoint: BaseEndpoint) {
        const desc: EndpointDesc = {endpoint, collections: []};
        this.endpoints.push(desc);

        this.updateConsole();
    }

    protected getEndpointDesc(endpoint: BaseEndpoint) {
        for (let desc of this.endpoints) {
            if (desc.endpoint.displayName == endpoint.displayName) return desc;
        }
        return null;
    }

    public registerCollection(collection: BaseCollection<any>, guiOptions: CollectionGuiOptions<any> = {}) {
        const endpoint = collection.endpoint;
        const endpointdesc = this.getEndpointDesc(endpoint);
        if (!endpointdesc) throw new Error(`Endpoint ${endpoint.displayName} is not registered in the GuiManager`);

        const displayName = guiOptions.displayName ? guiOptions.displayName : `Collection ${endpointdesc.collections.length}`;
        const desc: CollectionDesc = {collection, displayName, status: 'waiting', value: '', guiOptions};
        endpointdesc.collections.push(desc);

        collection.on('select.start', () => { desc.status = 'running'; this.updateConsole(); });
        collection.on('select.end', () => { desc.status = 'finished'; this.updateConsole(); });
        collection.on('select.recive', v => { desc.status = 'recived'; desc.value = v; this.updateConsole(); });

        collection.on('select.error', v => { desc.status = 'error'; desc.value = v; this.updateConsole(); });
        collection.on('insert', v => { desc.status = 'inserted'; desc.value = v; this.updateConsole(); });
        collection.on('update', v => { desc.status = 'updated'; desc.value = v; this.updateConsole(); });
        collection.on('upsert', v => { desc.status = 'upserted'; desc.value = v; this.updateConsole(); }); // ???
        collection.on('delete', v => { desc.status = 'deleted'; desc.value = v; this.updateConsole(); });

        this.updateConsole();
    }

    protected getCollectionNameLength(): number {
        let maxName = '';
        for (let epDesc of this.endpoints) {
            for (let colDesc of epDesc.collections) {
                if (colDesc.displayName > maxName) maxName = colDesc.displayName;
            }
        }
        return maxName.length;
    }

    protected getCollectionDisplayName(desc: CollectionDesc): string {
        return desc.displayName.padEnd(this.getCollectionNameLength() + 4, ' ');
    }

    protected deleteCurrentLine() {
        process.stdout.write("\x1B[1A\x1B[K");
    }

    protected setCursorAfterWindow() {
        let colCount = 0;
        for (let epDesc of this.endpoints) colCount += epDesc.collections.length + 1;
        process.stdout.cursorTo(0, 14 + colCount + GuiManager._instance.consoleManager.getLogPageSize());
    }

    protected dumpObject(obj: any, deep: number = 1): SimplifiedStyledElement[] {
        switch (typeof obj) {
            case 'number': return [{text: '' + obj, color: "blueBright"}];
            case 'string': return [{text: `"${obj}"`, color: "yellowBright"}];
            case 'boolean': return [{text: '' + obj, color: "greenBright"}];
            case 'function': return [{text: '()', color: "white"}];
        }

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
