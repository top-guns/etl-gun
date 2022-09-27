import { ForegroundColor } from "chalk";
import { ConsoleManager, PageBuilder } from "../../ConsoleGui";
/**
 * @description The type containing all the possible options for the SingleLayout.
 * @typedef {Object} SingleLayoutOptions
 * @prop {boolean} [showTitle] - If the title should be shown.
 * @prop {boolean} [boxed] - If the layout should be boxed.
 * @prop {ForegroundColor | ""} [boxColor] - The color of the box taken from the chalk library.
 * @prop {"bold"} [boxStyle] - If the border of the box should be bold.
 * @prop {string} [pageTitle] - The title of the first page.
 *
 * @export
 * @interface SingleLayoutOptions
 */
export interface SingleLayoutOptions {
    showTitle?: boolean;
    boxed?: boolean;
    boxColor?: typeof ForegroundColor | "";
    boxStyle?: "bold";
    pageTitle?: string;
}
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
export declare class SingleLayout {
    CM: ConsoleManager;
    options: SingleLayoutOptions;
    page: PageBuilder;
    boxBold: boolean;
    pageTitle: string;
    isOdd: boolean | undefined;
    constructor(page: PageBuilder, options: SingleLayoutOptions);
    /**
     * @description This function is used to overwrite the page content.
     * @param {PageBuilder} page the page to be added
     * @memberof SingleLayout
     */
    setPage(page: PageBuilder): void;
    /**
     * @description This function is used to set the title of the layout.
     * @param {string} title the title to be set
     * @memberof SingleLayout
     * @returns {void}
     * @example layout.setTitle("My Title")
     */
    setTitle(title: string): void;
    /**
     * @description This function is used to enable or disable the layout border.
     * @param {boolean} border enable or disable the border
     * @memberof SingleLayout
     */
    setBorder(border: boolean): void;
    /**
     * @description This function is used to draw a single line of the layout to the screen. It also trim the line if it is too long.
     * @param {Array<StyledElement>} line the line to be drawn
     * @memberof SingleLayout
     * @returns {void}
     */
    private drawLine;
    /**
     * @description This function is used to draw the layout to the screen.
     * @memberof SingleLayout
     * @returns {void}
     * @example layout.draw()
     */
    draw(): void;
}
export default SingleLayout;
