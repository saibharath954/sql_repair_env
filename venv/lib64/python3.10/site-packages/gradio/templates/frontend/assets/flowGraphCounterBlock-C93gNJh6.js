import { b as RichTypeNumber } from './declarationMapper-C9QH6ObI.js';
import { b as FlowGraphExecutionBlockWithOutSignal } from './KHR_interactivity-B3gTvBRt.js';
import { R as RegisterClass } from './index-BI1f02lL.js';
import './index-CDZuCcOm.js';
import './objectModelMapping-CCJRR4nC.js';

/**
 * A block that counts the number of times it has been called.
 * Afterwards it activates its out signal.
 */
class FlowGraphCallCounterBlock extends FlowGraphExecutionBlockWithOutSignal {
    constructor(config) {
        super(config);
        this.count = this.registerDataOutput("count", RichTypeNumber);
        this.reset = this._registerSignalInput("reset");
    }
    _execute(context, callingSignal) {
        if (callingSignal === this.reset) {
            context._setExecutionVariable(this, "count", 0);
            this.count.setValue(0, context);
            return;
        }
        const countValue = context._getExecutionVariable(this, "count", 0) + 1;
        context._setExecutionVariable(this, "count", countValue);
        this.count.setValue(countValue, context);
        this.out._activateSignal(context);
    }
    /**
     * @returns class name of the block.
     */
    getClassName() {
        return "FlowGraphCallCounterBlock" /* FlowGraphBlockNames.CallCounter */;
    }
}
RegisterClass("FlowGraphCallCounterBlock" /* FlowGraphBlockNames.CallCounter */, FlowGraphCallCounterBlock);

export { FlowGraphCallCounterBlock };
//# sourceMappingURL=flowGraphCounterBlock-C93gNJh6.js.map
