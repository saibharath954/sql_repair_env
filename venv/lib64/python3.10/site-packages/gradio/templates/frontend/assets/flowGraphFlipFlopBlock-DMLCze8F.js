import { e as FlowGraphExecutionBlock } from './KHR_interactivity-B3gTvBRt.js';
import { c as RichTypeBoolean } from './declarationMapper-C9QH6ObI.js';
import { R as RegisterClass } from './index-BI1f02lL.js';
import './index-CDZuCcOm.js';
import './objectModelMapping-CCJRR4nC.js';

/**
 * This block flip flops between two outputs.
 */
class FlowGraphFlipFlopBlock extends FlowGraphExecutionBlock {
    constructor(config) {
        super(config);
        this.onOn = this._registerSignalOutput("onOn");
        this.onOff = this._registerSignalOutput("onOff");
        this.value = this.registerDataOutput("value", RichTypeBoolean);
    }
    _execute(context, _callingSignal) {
        let value = context._getExecutionVariable(this, "value", typeof this.config?.startValue === "boolean" ? !this.config.startValue : false);
        value = !value;
        context._setExecutionVariable(this, "value", value);
        this.value.setValue(value, context);
        if (value) {
            this.onOn._activateSignal(context);
        }
        else {
            this.onOff._activateSignal(context);
        }
    }
    /**
     * @returns class name of the block.
     */
    getClassName() {
        return "FlowGraphFlipFlopBlock" /* FlowGraphBlockNames.FlipFlop */;
    }
}
RegisterClass("FlowGraphFlipFlopBlock" /* FlowGraphBlockNames.FlipFlop */, FlowGraphFlipFlopBlock);

export { FlowGraphFlipFlopBlock };
//# sourceMappingURL=flowGraphFlipFlopBlock-DMLCze8F.js.map
