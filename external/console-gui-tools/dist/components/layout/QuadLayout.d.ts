import { ForegroundColor } from "chalk";
import { ConsoleManager, PageBuilder } from "../../ConsoleGui";
/**
 * @description The type containing all the possible options for the QuadLayout.
 * @typedef {Object} QuadLayoutOptions
 * @prop {boolean} [showTitle] - If the title should be shown.
 * @prop {boolean} [boxed] - If the layout should be boxed.
 * @prop {ForegroundColor | ""} [boxColor] - The color of the box taken from the chalk library.
 * @prop {"bold"} [boxStyle] - If the border of the box should be bold.
 * @prop {string} [changeFocusKey] - The key that should be pressed to change the focus.
 * @prop {string} [page1Title] - The title of the first page.
 * @prop {string} [page2Title] - The title of the second page.
 * @prop {string} [page3Title] - The title of the third page.
 * @prop {string} [page4Title] - The title of the fourth page.
 * @prop {[number, number] | [[number, number]]} [pageRatio] - The ratio of the pages.
 *
 * @export
 * @interface DoubleLayoutOptions
 */
export interface QuadLayoutOptions {
    showTitle?: boolean;
    boxed?: boolean;
    boxColor?: typeof ForegroundColor | "";
    boxStyle?: "bold";
    changeFocusKey: string;
    page1Title?: string;
    page2Title?: string;
    page3Title?: string;
    page4Title?: string;
    pageRatio?: [[number, number], [number, number]];
}
/**
 * @class QuadLayout
 * @description This class is a layout that has two pages.
 *
 * ![quad layout](https://user-images.githubusercontent.com/14907987/170998201-59880c90-7b1a-491a-8a45-6610e5c33de9.png)
 *
 * @param {PageBuilder} page1 The first page.
 * @param {PageBuilder} page2 The second page.
 * @param {PageBuilder} page3 The third page.
 * @param {PageBuilder} page4 The fourth page.
 * @param {boolean} options Layout options.
 * @param {number} selected The selected page.
 * @example const layout = new QuadLayout(page1, page2, true, 0)
 */
export declare class QuadLayout {
    CM: ConsoleManager;
    options: QuadLayoutOptions;
    selected: 0 | 1 | 2 | 3;
    page1: PageBuilder;
    page2: PageBuilder;
    page3: PageBuilder;
    page4: PageBuilder;
    boxBold: boolean;
    proportions: [[number, number], [number, number]];
    page1Title: string;
    page2Title: string;
    page3Title: string;
    page4Title: string;
    realWidth: [[number, number], [number, number]];
    isOdd: boolean | undefined;
    constructor(page1: PageBuilder, page2: PageBuilder, page3: PageBuilder, page4: PageBuilder, options: QuadLayoutOptions, selected?: 0 | 1 | 2 | 3);
    /**
     * @description This function is used to overwrite the page content.
     * @param {PageBuilder} page the page to be added
     * @memberof QuadLayout
     */
    setPage(page: PageBuilder, index: number): void;
    /**
     * @description This function is used to overwrite the first page content.
     * @param {PageBuilder} page the page to be added
     * @memberof QuadLayout
     */
    setPage1(page: PageBuilder): void;
    /**
     * @description This function is used to overwrite the second page content.
     * @param {PageBuilder} page the page to be added
     * @memberof QuadLayout
     */
    setPage2(page: PageBuilder): void;
    /**
     * @description This function is used to overwrite the third page content.
     * @param {PageBuilder} page the page to be added
     * @memberof QuadLayout
     */
    setPage3(page: PageBuilder): void;
    /**
     * @description This function is used to overwrite the forth page content.
     * @param {PageBuilder} page the page to be added
     * @memberof QuadLayout
     */
    setPage4(page: PageBuilder): void;
    /**
     * @description This function is used to set the page titles.
     * @param {string[]} titles the titles of the pages
     * @memberof QuadLayout
     * @example layout.setTitles(["Page 1", "Page 2", "Page 3", "Page 4"])
     */
    setTitles(titles: string[]): void;
    /**
     * @description This function is used to set the page title at the given index.
     * @param {string} title the title of the page
     * @param {number} index the index of the page
     * @memberof QuadLayout
     * @example layout.setTitle("Page 1", 0)
     */
    setTitle(title: string, index: number): void;
    /**
     * @description This function is used to enable or disable the layout border.
     * @param {boolean} border enable or disable the border
     * @memberof QuadLayout
     */
    setBorder(border: boolean): void;
    /**
     * @description This function is used to choose the page to be highlighted.
     * @param {number} selected 0 for page1, 1 for page2
     * @memberof QuadLayout
     */
    setSelected(selected: 0 | 1 | 2 | 3): void;
    /**
     * @description This function is used to get the selected page.
     * @returns {number} 0 for page1, 1 for page2
     * @memberof QuadLayout
     */
    getSelected(): number;
    /**
     * @description This function is used to get switch the selected page.
     * @returns {void}
     * @memberof QuadLayout
     */
    changeLayout(): void;
    /**
     * @description This function is used to change the page ratio.
     * @param {Array<Array<number>>} ratio the ratio of pages
     * @memberof QuadLayout
     * @example layout.setRatio([[0.4, 0.6], [0.5, 0.5]])
     */
    setRatio(ratio: [[number, number], [number, number]]): void;
    /**
     * @description This function is used to increase the page ratio of the selected row by the given ratio to add.
     * @param {number} quantity the ratio to add
     * @memberof QuadLayout
     * @example layout.increaseRatio(0.01)
     */
    increaseRatio(quantity: number): void;
    /**
     * @description This function is used to decrease the page ratio of the selected row by the given ratio to add.
     * @param {number} quantity the ratio to subtract
     * @memberof QuadLayout
     * @example layout.decreaseRatio(0.01)
     */
    decreaseRatio(quantity: number): void;
    /**
     * @description This function is used to draw a single line of the layout to the screen. It also trim the line if it is too long.
     * @param {Array<StyledElement>} line the line to be drawn
     * @param {Array<StyledElement>} secondLine the line to be drawn
     * @param {number} row the row of the quad grid to be drawn
     * @memberof QuadLayout
     * @returns {void}
     */
    private drawLine;
    /**
     * @description This function is used to draw the layout to the screen.
     * @memberof QuadLayout
     * @returns {void}
     * @example layout.draw()
     */
    draw(): void;
}
export default QuadLayout;
