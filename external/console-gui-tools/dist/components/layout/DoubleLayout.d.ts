import { ForegroundColor } from "chalk";
import { ConsoleManager, PageBuilder } from "../../ConsoleGui";
/**
 * @description The type containing all the possible options for the DoubleLayout.
 * @typedef {Object} DoubleLayoutOptions
 * @prop {boolean} [showTitle] - If the title should be shown.
 * @prop {boolean} [boxed] - If the layout should be boxed.
 * @prop {ForegroundColor | ""} [boxColor] - The color of the box taken from the chalk library.
 * @prop {"bold"} [boxStyle] - If the border of the box should be bold.
 * @prop {string} [changeFocusKey] - The key that should be pressed to change the focus.
 * @prop {"horizontal" | "vertical"} [direction] - The direction of the layout.
 * @prop {string} [page1Title] - The title of the first page.
 * @prop {string} [page2Title] - The title of the second page.
 * @prop {[number, number]} [pageRatio] - The ratio of the pages. (in horizontal direction)
 *
 * @export
 * @interface DoubleLayoutOptions
 */
export interface DoubleLayoutOptions {
    showTitle?: boolean;
    boxed?: boolean;
    boxColor?: typeof ForegroundColor | "";
    boxStyle?: "bold";
    changeFocusKey: string;
    direction?: "horizontal" | "vertical";
    page1Title?: string;
    page2Title?: string;
    pageRatio?: [number, number];
}
/**
 * @class DoubleLayout
 * @description This class is a layout that has two pages.
 *
 * ![double layout](https://user-images.githubusercontent.com/14907987/170996957-cb28414b-7be2-4aa0-938b-f6d1724cfa4c.png)
 *
 * @param {PageBuilder} page1 The first page.
 * @param {PageBuilder} page2 The second page.
 * @param {boolean} options Layout options.
 * @param {number} selected The selected page.
 * @example const layout = new DoubleLayout(page1, page2, true, 0)
 */
export declare class DoubleLayout {
    CM: ConsoleManager;
    options: DoubleLayoutOptions;
    selected: 0 | 1;
    page1: PageBuilder;
    page2: PageBuilder;
    boxBold: boolean;
    proportions: [number, number];
    page2Title: string;
    page1Title: string;
    realWidth: number | [number, number];
    isOdd: boolean | undefined;
    constructor(page1: PageBuilder, page2: PageBuilder, options: DoubleLayoutOptions, selected?: 0 | 1);
    /**
     * @description This function is used to overwrite the page content.
     * @param {PageBuilder} page the page to be added
     * @memberof DoubleLayout
     */
    setPage(page: PageBuilder, index: number): void;
    /**
     * @description This function is used to overwrite the page content.
     * @param {PageBuilder} page the page to be added
     * @memberof DoubleLayout
     */
    setPage1(page: PageBuilder): void;
    /**
     * @description This function is used to overwrite the page content.
     * @param {PageBuilder} page the page to be added
     * @memberof DoubleLayout
     */
    setPage2(page: PageBuilder): void;
    /**
     * @description This function is used to set the page titles.
     * @param {string[]} titles the titles of the pages
     * @memberof DoubleLayout
     * @example layout.setTitles(["Page 1", "Page 2"])
     */
    setTitles(titles: string[]): void;
    /**
     * @description This function is used to set the page title at the given index.
     * @param {string} title the title of the page
     * @param {number} index the index of the page
     * @memberof DoubleLayout
     * @example layout.setTitle("Page 1", 0)
     */
    setTitle(title: string, index: number): void;
    /**
     * @description This function is used to enable or disable the layout border.
     * @param {boolean} border enable or disable the border
     * @memberof DoubleLayout
     */
    setBorder(border: boolean): void;
    /**
     * @description This function is used to choose the page to be highlighted.
     * @param {number} selected 0 for page1, 1 for page2
     * @memberof DoubleLayout
     */
    setSelected(selected: 0 | 1): void;
    /**
     * @description This function is used to get the selected page.
     * @returns {number} 0 for page1, 1 for page2
     * @memberof DoubleLayout
     */
    getSelected(): number;
    /**
     * @description This function is used to get switch the selected page.
     * @returns {void}
     * @memberof DoubleLayout
     */
    changeLayout(): void;
    /**
     * @description This function is used to change the page ratio.
     * @param {Array<number>} ratio the ratio of pages
     * @memberof QuadLayout
     * @example layout.setRatio([0.4, 0.6])
     */
    setRatio(ratio: [number, number]): void;
    /**
     * @description This function is used to increase the page ratio by the given ratio to add. (Only works if the direction is horizontal)
     * @param {number} quantity the ratio to add
     * @memberof QuadLayout
     * @example layout.increaseRatio(0.01)
     */
    increaseRatio(quantity: number): void;
    /**
     * @description This function is used to decrease the page ratio by the given ratio to subtract. (Only works if the direction is horizontal).
     * @param {number} quantity the ratio to subtract
     * @memberof QuadLayout
     * @example layout.decreaseRatio(0.01)
     */
    decreaseRatio(quantity: number): void;
    /**
     * @description This function is used to draw a single line of the layout to the screen. It also trim the line if it is too long.
     * @param {Array<StyledElement>} line the line to be drawn
     * @param {number} lineIndex the index of the selected line
     * @memberof DoubleLayout
     * @returns {void}
     */
    private drawLine;
    /**
     * @description This function is used to draw the layout to the screen.
     * @memberof DoubleLayout
     * @returns {void}
     * @example layout.draw()
     */
    draw(): void;
}
export default DoubleLayout;
