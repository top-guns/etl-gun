/*********************************************************************************************************************************** 
This file contains temporary code to test library while development process. 
It is not contains any usefull code, is not a part of library and is not an example of library using.
************************************************************************************************************************************/

import * as rx from "rxjs";
import * as etl from "./lib/index.js";
import { deleteFileIfExists } from "./utils/filesystem.js";
//import { DiscordHelper } from "./lib/index.js";


const START = new Date;
//GuiManager.startGui(true, 20);
console.log("START", START);

const res: number[] = [];

const mem = etl.Memory.getEndpoint();
const src = mem.getBuffer<number>('bufer1', [1, 2, 3]);
let stream$ = src.selectRx().pipe(
    //rx.tap(v => res.push(v))
    etl.log()
);
//src.selectRx().subscribe(v => console.log(v))
await etl.run(stream$);

//const vals = await src.select()
//console.log(vals)

//for await (const val of src.selectGen()) console.log(val)

//src.selectIx().forEach(v => console.log(v));

// const reader = src.selectStream().getReader();
// let v = await reader.read();
// while(!v.done) {
//     console.log(v)
//     v = await reader.read();
// }


// import path from 'path';
// const TEMP_FOLDER = "./tests/tmp/";
// const OUT_FILE_NAME = 'test_output.tmp';
// const ROOT_FOLDER = TEMP_FOLDER;
// const OUT_FILE_FULL_PATH = path.join(ROOT_FOLDER, OUT_FILE_NAME);
// try {
//     deleteFileIfExists(OUT_FILE_FULL_PATH);

//     console.log(111111)

//     const memory = new Memory.Endpoint();

//     //const ep = new etl.filesystems.Local.Endpoint(ROOT_FOLDER);
//     console.log(222222)
//     // const src = ep.getFolder('.');
//     // console.log(333333)
//     // await src.insert(OUT_FILE_NAME, 'test');
//     // console.log(444444)

//     // const res = loadFileContent(OUT_FILE_FULL_PATH);
//     // assert.strictEqual(res, 'test');
// }
// finally {
//     //deleteFileIfExists(OUT_FILE_FULL_PATH);
// }




//mysql.releaseEndpoint();
//if (etl.GuiManager.isGuiStarted()) etl.GuiManager.stopGui();
console.log("END");
console.log('start', START);
console.log('end', new Date());
//etl.GuiManager.quitApp();
