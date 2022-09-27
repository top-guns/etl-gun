"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.LayoutManager = void 0;
const ConsoleGui_1 = require("../../ConsoleGui");
const DoubleLayout_1 = __importDefault(require("./DoubleLayout"));
const QuadLayout_1 = __importDefault(require("./QuadLayout"));
const SingleLayout_1 = __importDefault(require("./SingleLayout"));
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
class LayoutManager {
    constructor(pages, options) {
        this.pages = {};
        this.pageTitles = [];
        /**
         * @description This function is used to check if the layout is a single layout by checking the type of the instance.
         * @param {unknown} x - The instance of the layout.
         * @returns {boolean} - If the layout is a single layout.
         * @memberof LayoutManager
         * @example const isSingleLayout = this.isSingleLayout(layout)
         */
        this.isSingleLayout = (x) => {
            return x instanceof SingleLayout_1.default;
        };
        if (this.instance) {
            return this.instance;
        }
        else {
            this.instance = this;
            /** @const {ConsoleManager} CM the instance of ConsoleManager (singleton) */
            this.CM = new ConsoleGui_1.ConsoleManager();
            this.options = options;
            pages.forEach((page, index) => {
                this.pages[index] = page;
            });
            /** @const {string} pageTitle The application title. */
            this.pageTitles = this.options.pageTitles || [this.CM.applicationTitle];
            switch (this.options.type) {
                case "single":
                    this.optionsRelative = {
                        showTitle: this.options.showTitle,
                        boxed: this.options.boxed,
                        boxColor: this.options.boxColor,
                        boxStyle: this.options.boxStyle,
                        pageTitle: this.pageTitles ? this.pageTitles[0] : "",
                    };
                    this.layout = new SingleLayout_1.default(this.pages[0], this.optionsRelative);
                    break;
                case "double":
                    this.optionsRelative = {
                        showTitle: this.options.showTitle,
                        boxed: this.options.boxed,
                        boxColor: this.options.boxColor,
                        boxStyle: this.options.boxStyle,
                        changeFocusKey: this.options.changeFocusKey,
                        direction: this.options.direction,
                        page1Title: this.pageTitles ? this.pageTitles[0] : "",
                        page2Title: this.pageTitles ? this.pageTitles[1] : "",
                        pageRatio: this.options.pageRatio,
                    };
                    this.layout = new DoubleLayout_1.default(this.pages[0], this.pages[1], this.optionsRelative);
                    break;
                case "triple":
                    break;
                case "quad":
                    this.optionsRelative = {
                        showTitle: this.options.showTitle,
                        boxed: this.options.boxed,
                        boxColor: this.options.boxColor,
                        boxStyle: this.options.boxStyle,
                        changeFocusKey: this.options.changeFocusKey,
                        direction: this.options.direction,
                        page1Title: this.pageTitles ? this.pageTitles[0] : "",
                        page2Title: this.pageTitles ? this.pageTitles[1] : "",
                        page3Title: this.pageTitles ? this.pageTitles[2] : "",
                        page4Title: this.pageTitles ? this.pageTitles[3] : "",
                        pageRatio: this.options.pageRatio,
                    };
                    this.layout = new QuadLayout_1.default(this.pages[0], this.pages[1], this.pages[2], this.pages[3], this.optionsRelative);
                    break;
                default:
                    break;
            }
        }
    }
    /**
     * @description This function is used to update the layout pages.
     * @param {PageBuilder[]} pages The pages that should be shown.
     * @memberof LayoutManager
     * @example layout.updatePages([page1, page2])
     * @example layout.updatePages([page1, page2, page3])
     */
    setPages(pages) {
        pages.forEach((page, index) => {
            this.pages[index] = page;
            if (this.isSingleLayout(this.layout)) {
                this.layout.setPage(page);
            }
            else {
                this.layout.setPage(page, index);
            }
        });
    }
    /**
     * @description This function is used to overwrite the page content.
     * @param {PageBuilder} page the page to be added
     * @param {number} index the index of the page
     * @memberof LayoutManager
     */
    setPage(page, index) {
        this.pages[index] = page;
        if (this.isSingleLayout(this.layout)) {
            this.layout.setPage(page);
        }
        else {
            this.layout.setPage(page, index);
        }
    }
    /**
     * @description This function is used to update the page title.
     * @param {string} title The title of the page.
     * @param {number} index The index of the page.
     * @memberof LayoutManager
     * @example layout.setTitle("Page Title", 1)
     */
    setTitle(title, index) {
        this.pageTitles[index] = title;
        if (this.isSingleLayout(this.layout)) {
            this.layout.setTitle(title);
        }
        else {
            this.layout.setTitle(title, index);
        }
    }
    /**
     * @description This function is used to update the page titles.
     * @param {string[]} titles The titles of the pages.
     * @memberof LayoutManager
     * @example layout.setTitles(["Page Title 1", "Page Title 2"])
     */
    setTitles(titles) {
        this.pageTitles = titles;
        if (this.isSingleLayout(this.layout)) {
            this.layout.setTitle(titles[0]);
        }
        else {
            this.layout.setTitles(titles);
        }
    }
    /**
     * @description This function is used to enable or disable the layout border.
     * @param {boolean} border enable or disable the border
     * @memberof LayoutManager
     */
    setBorder(border) { this.options.boxed = border; }
    /**
     * @description This function is used to choose the page to be highlighted.
     * @param {0 | 1 | 2 | 3} selected 0 for page1, 1 for page2
     * @memberof LayoutManager
     */
    setSelected(selected) {
        if (!this.isSingleLayout(this.layout)) {
            this.layout.setSelected(selected);
        }
    }
    /**
      * @description This function is used to get the selected page.
      * @returns {0 | 1 | 2 | 3} 0 for page1, 1 for page2, 2 for page3, 3 for page4
      * @memberof LayoutManager
      */
    getSelected() {
        if (!this.isSingleLayout(this.layout)) {
            return this.layout.selected;
        }
        return 0;
    }
    /**
      * @description This function is used to get switch the selected page. If the layout is a single layout, it will do nothing.
      * @returns {void}
      * @memberof LayoutManager
      */
    changeLayout() {
        if (!this.isSingleLayout(this.layout)) {
            this.layout.changeLayout();
        }
    }
    /**
     * @description This function is used to decrease the row ratio between the pages in the selected row. This is propagated to the layout instance.
     * @param {quantity} quantity The amount of aspect ratio to be decreased.
     * @memberof LayoutManager
     * @example layout.decreaseRowRatio(0.01)
     */
    decreaseRatio(quantity) {
        if (!this.isSingleLayout(this.layout)) {
            this.layout.decreaseRatio(quantity);
        }
    }
    /**
     * @description This function is used to increase the row ratio between the pages in the selected row. This is propagated to the layout instance.
     * @param {quantity} quantity The amount of aspect ratio to be increased.
     * @memberof LayoutManager
     * @example layout.increaseRowRatio(0.01)
     */
    increaseRatio(quantity) {
        if (!this.isSingleLayout(this.layout)) {
            this.layout.increaseRatio(quantity);
        }
    }
    /**
     * @description This function is used to draw the layout to the screen.
     * @memberof LayoutManager
     * @returns {void}
     * @example layout.draw()
     */
    draw() {
        this.layout.draw();
    }
}
exports.LayoutManager = LayoutManager;
exports.default = LayoutManager;
//# sourceMappingURL=LayoutManager.js.map