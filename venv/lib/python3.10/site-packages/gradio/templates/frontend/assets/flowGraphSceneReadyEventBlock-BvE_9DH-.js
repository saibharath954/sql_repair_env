import { c as FlowGraphEventBlock } from './KHR_interactivity-B3gTvBRt.js';
import { R as RegisterClass } from './index-BI1f02lL.js';
import './index-CDZuCcOm.js';
import './declarationMapper-C9QH6ObI.js';
import './objectModelMapping-CCJRR4nC.js';

/**
 * Block that triggers when a scene is ready.
 */
class FlowGraphSceneReadyEventBlock extends FlowGraphEventBlock {
    constructor() {
        super(...arguments);
        this.initPriority = -1;
        this.type = "SceneReady" /* FlowGraphEventType.SceneReady */;
    }
    _executeEvent(context, _payload) {
        this._execute(context);
        return true;
    }
    _preparePendingTasks(context) {
        // no-op
    }
    _cancelPendingTasks(context) {
        // no-op
    }
    /**
     * @returns class name of the block.
     */
    getClassName() {
        return "FlowGraphSceneReadyEventBlock" /* FlowGraphBlockNames.SceneReadyEvent */;
    }
}
RegisterClass("FlowGraphSceneReadyEventBlock" /* FlowGraphBlockNames.SceneReadyEvent */, FlowGraphSceneReadyEventBlock);

export { FlowGraphSceneReadyEventBlock };
//# sourceMappingURL=flowGraphSceneReadyEventBlock-BvE_9DH-.js.map
