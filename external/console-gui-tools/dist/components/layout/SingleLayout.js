"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.SingleLayout = void 0;
const ConsoleGui_1 = require("../../ConsoleGui");
/**
 * @class SingleLayout
 * @description This class is a layout that has two pages.
 *
 * ![single layout](https://user-images.githubusercontent.com/14907987/170997567-b1260996-cc7e-4c26-8389-39519313f3f6.png)
 *
 * @param {PageBuilder} page The first page.
 * @param {boolean} options Layout options.
 * @example const layout = new SingleLayout(page1, page2, true, 0)
 */
class SingleLayout {
    constructor(page, options) {
        /** @const {ConsoleManager} CM the instance of ConsoleManager (singleton) */
        this.CM = new ConsoleGui_1.ConsoleManager();
        this.options = options;
        this.page = page;
        this.boxBold = this.options.boxStyle === "bold" ? true : false;
        /** @const {string} pageTitle The application title. */
        this.pageTitle = this.options.pageTitle || this.CM.applicationTitle;
    }
    /**
     * @description This function is used to overwrite the page content.
     * @param {PageBuilder} page the page to be added
     * @memberof SingleLayout
     */
    setPage(page) { this.page = page; }
    /**
     * @description This function is used to set the title of the layout.
     * @param {string} title the title to be set
     * @memberof SingleLayout
     * @returns {void}
     * @example layout.setTitle("My Title")
     */
    setTitle(title) { this.pageTitle = title; }
    /**
     * @description This function is used to enable or disable the layout border.
     * @param {boolean} border enable or disable the border
     * @memberof SingleLayout
     */
    setBorder(border) { this.options.boxed = border; }
    /**
     * @description This function is used to draw a single line of the layout to the screen. It also trim the line if it is too long.
     * @param {Array<StyledElement>} line the line to be drawn
     * @memberof SingleLayout
     * @returns {void}
     */
    drawLine(line) {
        const bsize = this.options.boxed ? 2 : 0;
        let unformattedLine = "";
        let newLine = [...line];
        line.forEach((element) => {
            unformattedLine += element.text;
        });
        if (unformattedLine.length > this.CM.Screen.width - bsize) {
            if (unformattedLine.length > this.CM.Screen.width - bsize) { // Need to truncate
                const offset = 2;
                newLine = [...JSON.parse(JSON.stringify(line))]; // Shallow copy because I just want to modify the values but not the original
                let diff = unformattedLine.length - this.CM.Screen.width + 1;
                // remove truncated text
                for (let j = newLine.length - 1; j >= 0; j--) {
                    if (newLine[j].text.length > diff + offset) {
                        newLine[j].text = this.CM.truncate(newLine[j].text, (newLine[j].text.length - diff) - offset, true);
                        break;
                    }
                    else {
                        diff -= newLine[j].text.length;
                        newLine.splice(j, 1);
                    }
                }
                // Update unformatted line
                unformattedLine = newLine.map((element) => element.text).join("");
            }
        }
        if (this.options.boxed)
            newLine.unshift({ text: "│", style: { color: this.options.boxColor, bold: this.boxBold } });
        if (unformattedLine.length <= this.CM.Screen.width - bsize) {
            newLine.push({ text: `${" ".repeat((this.CM.Screen.width - unformattedLine.length) - bsize)}`, style: { color: "" } });
        }
        if (this.options.boxed)
            newLine.push({ text: "│", style: { color: this.options.boxColor, bold: this.boxBold } });
        this.CM.Screen.write(...newLine);
    }
    /**
     * @description This function is used to draw the layout to the screen.
     * @memberof SingleLayout
     * @returns {void}
     * @example layout.draw()
     */
    draw() {
        this.isOdd = this.CM.Screen.width % 2 === 1;
        const trimmedTitle = this.CM.truncate(this.pageTitle, this.CM.Screen.width - 2, false);
        if (this.options.boxed) { // Draw pages with borders
            if (this.options.showTitle) {
                this.CM.Screen.write({ text: `┌─${trimmedTitle}${"─".repeat(this.CM.Screen.width - trimmedTitle.length - 3)}┐`, style: { color: this.options.boxColor, bold: this.boxBold } });
            }
            else {
                this.CM.Screen.write({ text: `┌─${"─".repeat(this.CM.Screen.width - 3)}┐`, style: { color: this.options.boxColor, bold: this.boxBold } });
            }
            this.page.getContent().forEach((line) => {
                this.drawLine(line);
            });
            this.CM.Screen.write({ text: `└${"─".repeat(this.CM.Screen.width - 2)}┘`, style: { color: this.options.boxColor, bold: this.boxBold } });
        }
        else { // Draw pages without borders
            if (this.options.showTitle) {
                this.CM.Screen.write({ text: `${trimmedTitle}`, style: { color: this.options.boxColor, bold: this.boxBold } });
            }
            this.page.getContent().forEach((line) => {
                this.drawLine(line);
            });
        }
    }
}
exports.SingleLayout = SingleLayout;
exports.default = SingleLayout;
//# sourceMappingURL=SingleLayout.js.map