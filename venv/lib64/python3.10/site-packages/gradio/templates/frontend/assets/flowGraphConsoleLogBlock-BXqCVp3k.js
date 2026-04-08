import { b as FlowGraphExecutionBlockWithOutSignal } from './KHR_interactivity-B3gTvBRt.js';
import { R as RichTypeAny } from './declarationMapper-C9QH6ObI.js';
import { L as Logger, R as RegisterClass } from './index-BI1f02lL.js';
import './index-CDZuCcOm.js';
import './objectModelMapping-CCJRR4nC.js';

/**
 * Block that logs a message to the console.
 */
class FlowGraphConsoleLogBlock extends FlowGraphExecutionBlockWithOutSignal {
    constructor(config) {
        super(config);
        this.message = this.registerDataInput("message", RichTypeAny);
        this.logType = this.registerDataInput("logType", RichTypeAny, "log");
        if (config?.messageTemplate) {
            const matches = this._getTemplateMatches(config.messageTemplate);
            for (const match of matches) {
                this.registerDataInput(match, RichTypeAny);
            }
        }
    }
    /**
     * @internal
     */
    _execute(context) {
        const typeValue = this.logType.getValue(context);
        const messageValue = this._getMessageValue(context);
        if (typeValue === "warn") {
            Logger.Warn(messageValue);
        }
        else if (typeValue === "error") {
            Logger.Error(messageValue);
        }
        else {
            Logger.Log(messageValue);
        }
        // activate the output flow block
        this.out._activateSignal(context);
    }
    /**
     * @returns class name of the block.
     */
    getClassName() {
        return "FlowGraphConsoleLogBlock" /* FlowGraphBlockNames.ConsoleLog */;
    }
    _getMessageValue(context) {
        if (this.config?.messageTemplate) {
            let template = this.config.messageTemplate;
            const matches = this._getTemplateMatches(template);
            for (const match of matches) {
                const value = this.getDataInput(match)?.getValue(context);
                if (value !== undefined) {
                    // replace all
                    template = template.replace(new RegExp(`\\{${match}\\}`, "g"), value.toString());
                }
            }
            return template;
        }
        else {
            return this.message.getValue(context);
        }
    }
    _getTemplateMatches(template) {
        const regex = /\{([^}]+)\}/g;
        const matches = [];
        let match;
        while ((match = regex.exec(template)) !== null) {
            matches.push(match[1]);
        }
        return matches;
    }
}
RegisterClass("FlowGraphConsoleLogBlock" /* FlowGraphBlockNames.ConsoleLog */, FlowGraphConsoleLogBlock);

export { FlowGraphConsoleLogBlock };
//# sourceMappingURL=flowGraphConsoleLogBlock-BXqCVp3k.js.map
