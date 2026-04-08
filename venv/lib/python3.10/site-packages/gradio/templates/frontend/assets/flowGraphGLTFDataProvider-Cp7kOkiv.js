import { F as FlowGraphBlock } from './KHR_interactivity-B3gTvBRt.js';
import { R as RichTypeAny } from './declarationMapper-C9QH6ObI.js';
import './index-CDZuCcOm.js';
import './index-BI1f02lL.js';
import './objectModelMapping-CCJRR4nC.js';

/**
 * a glTF-based FlowGraph block that provides arrays with babylon object, based on the glTF tree
 * Can be used, for example, to get animation index from a glTF animation
 */
class FlowGraphGLTFDataProvider extends FlowGraphBlock {
    constructor(config) {
        super();
        const glTF = config.glTF;
        const animationGroups = glTF.animations?.map((a) => a._babylonAnimationGroup) || [];
        this.animationGroups = this.registerDataOutput("animationGroups", RichTypeAny, animationGroups);
        const nodes = glTF.nodes?.map((n) => n._babylonTransformNode) || [];
        this.nodes = this.registerDataOutput("nodes", RichTypeAny, nodes);
    }
    getClassName() {
        return "FlowGraphGLTFDataProvider";
    }
}

export { FlowGraphGLTFDataProvider };
//# sourceMappingURL=flowGraphGLTFDataProvider-Cp7kOkiv.js.map
