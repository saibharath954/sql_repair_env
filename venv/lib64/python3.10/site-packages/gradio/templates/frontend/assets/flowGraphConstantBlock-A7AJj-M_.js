import { F as FlowGraphBlock, f as defaultValueSerializationFunction } from './KHR_interactivity-B3gTvBRt.js';
import { j as getRichTypeFromValue } from './declarationMapper-C9QH6ObI.js';
import { R as RegisterClass } from './index-BI1f02lL.js';
import './index-CDZuCcOm.js';
import './objectModelMapping-CCJRR4nC.js';

/**
 * Block that returns a constant value.
 */
class FlowGraphConstantBlock extends FlowGraphBlock {
    constructor(
    /**
     * the configuration of the block
     */
    config) {
        super(config);
        this.config = config;
        this.output = this.registerDataOutput("output", getRichTypeFromValue(config.value));
    }
    _updateOutputs(context) {
        this.output.setValue(this.config.value, context);
    }
    /**
     * Gets the class name of this block
     * @returns the class name
     */
    getClassName() {
        return "FlowGraphConstantBlock" /* FlowGraphBlockNames.Constant */;
    }
    /**
     * Serializes this block
     * @param serializationObject the object to serialize to
     * @param valueSerializeFunction the function to use to serialize the value
     */
    serialize(serializationObject = {}, valueSerializeFunction = defaultValueSerializationFunction) {
        super.serialize(serializationObject);
        valueSerializeFunction("value", this.config.value, serializationObject.config);
    }
}
RegisterClass("FlowGraphConstantBlock" /* FlowGraphBlockNames.Constant */, FlowGraphConstantBlock);

export { FlowGraphConstantBlock };
//# sourceMappingURL=flowGraphConstantBlock-A7AJj-M_.js.map
