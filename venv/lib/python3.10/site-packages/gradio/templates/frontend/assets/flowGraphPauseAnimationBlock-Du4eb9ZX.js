import { b as FlowGraphExecutionBlockWithOutSignal } from './KHR_interactivity-B3gTvBRt.js';
import { R as RichTypeAny } from './declarationMapper-C9QH6ObI.js';
import { R as RegisterClass } from './index-BI1f02lL.js';
import './index-CDZuCcOm.js';
import './objectModelMapping-CCJRR4nC.js';

/**
 * @experimental
 * Block that pauses a running animation
 */
class FlowGraphPauseAnimationBlock extends FlowGraphExecutionBlockWithOutSignal {
    constructor(config) {
        super(config);
        this.animationToPause = this.registerDataInput("animationToPause", RichTypeAny);
    }
    _execute(context) {
        const animationToPauseValue = this.animationToPause.getValue(context);
        animationToPauseValue.pause();
        this.out._activateSignal(context);
    }
    /**
     * @returns class name of the block.
     */
    getClassName() {
        return "FlowGraphPauseAnimationBlock" /* FlowGraphBlockNames.PauseAnimation */;
    }
}
RegisterClass("FlowGraphPauseAnimationBlock" /* FlowGraphBlockNames.PauseAnimation */, FlowGraphPauseAnimationBlock);

export { FlowGraphPauseAnimationBlock };
//# sourceMappingURL=flowGraphPauseAnimationBlock-Du4eb9ZX.js.map
