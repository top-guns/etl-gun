import { ForegroundColor } from "chalk";
import PageBuilder from "../PageBuilder";
import DoubleLayout from "./DoubleLayout";
import QuadLayout from "./QuadLayout";
import SingleLayout from "./SingleLayout";
/**
 * @description The type containing all the possible options for the layout.
 * @typedef {Object} LayoutOptions
 * @prop {boolean} [showTitle] - If the title should be shown.
 * @prop {boolean} [boxed] - If the layout should be boxed.
 * @prop {ForegroundColor | ""} [boxColor] - The color of the box taken from the chalk library.
 * @prop {"bold"} [boxStyle] - If the border of the box should be bold.
 * @prop {"single" | "double" | "triple" | "quad"} [type] - The type of the layout.
 * @prop {string} [changeFocusKey] - The key that should be pressed to change the focus.
 * @prop {"horizontal" | "vertical"} [direction] - The direction of the layout.
 * @prop {string[]} [pageTitles] - The title of the first page.
 * @prop {[number, number] | [[number, number]]} [pageRatio] - The ratio of the pages. (in horizontal direction)
 *
 * @export
 * @interface LayoutOptions
 */
export interface LayoutOptions {
    showTitle?: boolean;
    boxed?: boolean;
    boxColor?: typeof ForegroundColor | "";
    boxStyle?: "bold";
    changeFocusKey: string;
    type: "single" | "double" | "triple" | "quad";
    direction?: "horizontal" | "vertical";
    pageTitles?: string[];
    pageRatio?: [number, number] | [[number, number]];
}
/**
 * @class LayoutManager
 * @description This class is a layout that has two pages.
 *
 * ![change ratio](https://user-images.githubusercontent.com/14907987/170999347-868eac7b-6bdf-4147-bcb0-b7465282ed5f.gif)
 *
 * @param {PageBuilder[]} pages The pages that should be shown.
 * @param {boolean} options Layout options.
 * @example const layout = new LayoutManager([page1, page2], pageOptions);
 */
export declare class LayoutManager {
    private CM;
    private options;
    private optionsRelative;
    pages: {
        [key: number]: PageBuilder;
    };
    private pageTitles;
    layout: SingleLayout | DoubleLayout | QuadLayout;
    private instance;
    constructor(pages: PageBuilder[], options: LayoutOptions);
    /**
     * @description This function is used to check if the layout is a single layout by checking the type of the instance.
     * @param {unknown} x - The instance of the layout.
     * @returns {boolean} - If the layout is a single layout.
     * @memberof LayoutManager
     * @example const isSingleLayout = this.isSingleLayout(layout)
     */
    private isSingleLayout;
    /**
     * @description This function is used to update the layout pages.
     * @param {PageBuilder[]} pages The pages that should be shown.
     * @memberof LayoutManager
     * @example layout.updatePages([page1, page2])
     * @example layout.updatePages([page1, page2, page3])
     */
    setPages(pages: PageBuilder[]): void;
    /**
     * @description This function is used to overwrite the page content.
     * @param {PageBuilder} page the page to be added
     * @param {number} index the index of the page
     * @memberof LayoutManager
     */
    setPage(page: PageBuilder, index: number): void;
    /**
     * @description This function is used to update the page title.
     * @param {string} title The title of the page.
     * @param {number} index The index of the page.
     * @memberof LayoutManager
     * @example layout.setTitle("Page Title", 1)
     */
    setTitle(title: string, index: number): void;
    /**
     * @description This function is used to update the page titles.
     * @param {string[]} titles The titles of the pages.
     * @memberof LayoutManager
     * @example layout.setTitles(["Page Title 1", "Page Title 2"])
     */
    setTitles(titles: string[]): void;
    /**
     * @description This function is used to enable or disable the layout border.
     * @param {boolean} border enable or disable the border
     * @memberof LayoutManager
     */
    setBorder(border: boolean): void;
    /**
     * @description This function is used to choose the page to be highlighted.
     * @param {0 | 1 | 2 | 3} selected 0 for page1, 1 for page2
     * @memberof LayoutManager
     */
    setSelected(selected: 0 | 1 | 2 | 3): void;
    /**
      * @description This function is used to get the selected page.
      * @returns {0 | 1 | 2 | 3} 0 for page1, 1 for page2, 2 for page3, 3 for page4
      * @memberof LayoutManager
      */
    getSelected(): number;
    /**
      * @description This function is used to get switch the selected page. If the layout is a single layout, it will do nothing.
      * @returns {void}
      * @memberof LayoutManager
      */
    changeLayout(): void;
    /**
     * @description This function is used to decrease the row ratio between the pages in the selected row. This is propagated to the layout instance.
     * @param {quantity} quantity The amount of aspect ratio to be decreased.
     * @memberof LayoutManager
     * @example layout.decreaseRowRatio(0.01)
     */
    decreaseRatio(quantity: number): void;
    /**
     * @description This function is used to increase the row ratio between the pages in the selected row. This is propagated to the layout instance.
     * @param {quantity} quantity The amount of aspect ratio to be increased.
     * @memberof LayoutManager
     * @example layout.increaseRowRatio(0.01)
     */
    increaseRatio(quantity: number): void;
    /**
     * @description This function is used to draw the layout to the screen.
     * @memberof LayoutManager
     * @returns {void}
     * @example layout.draw()
     */
    draw(): void;
}
export default LayoutManager;
