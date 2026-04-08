import { R as RegisterClass } from './index-BI1f02lL.js';
import { e as FlowGraphExecutionBlock } from './KHR_interactivity-B3gTvBRt.js';
import './index-CDZuCcOm.js';
import './declarationMapper-C9QH6ObI.js';
import './objectModelMapping-CCJRR4nC.js';

/**
 * A block that executes its output flows in sequence.
 */
class FlowGraphSequenceBlock extends FlowGraphExecutionBlock {
    constructor(
    /**
     * the configuration of the block
     */
    config) {
        super(config);
        this.config = config;
        /**
         * The output flows.
         */
        this.executionSignals = [];
        this.setNumberOfOutputSignals(this.config.outputSignalCount);
    }
    _execute(context) {
        for (let i = 0; i < this.executionSignals.length; i++) {
            this.executionSignals[i]._activateSignal(context);
        }
    }
    /**
     * Sets the block's output flows. Would usually be passed from the constructor but can be changed afterwards.
     * @param outputSignalCount the number of output flows
     */
    setNumberOfOutputSignals(outputSignalCount = 1) {
        // check the size of the outFlow Array, see if it is not larger than needed
        while (this.executionSignals.length > outputSignalCount) {
            const flow = this.executionSignals.pop();
            if (flow) {
                flow.disconnectFromAll();
                this._unregisterSignalOutput(flow.name);
            }
        }
        while (this.executionSignals.length < outputSignalCount) {
            this.executionSignals.push(this._registerSignalOutput(`out_${this.executionSignals.length}`));
        }
    }
    /**
     * @returns class name of the block.
     */
    getClassName() {
        return "FlowGraphSequenceBlock" /* FlowGraphBlockNames.Sequence */;
    }
}
RegisterClass("FlowGraphSequenceBlock" /* FlowGraphBlockNames.Sequence */, FlowGraphSequenceBlock);

export { FlowGraphSequenceBlock };
//# sourceMappingURL=flowGraphSequenceBlock-fYCnRT_5.js.map
