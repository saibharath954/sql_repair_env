import { F as FlowGraphBlock, g as getNumericValue, i as isNumeric } from './KHR_interactivity-B3gTvBRt.js';
import { R as RichTypeAny } from './declarationMapper-C9QH6ObI.js';
import { R as RegisterClass } from './index-BI1f02lL.js';
import './index-CDZuCcOm.js';
import './objectModelMapping-CCJRR4nC.js';

/**
 * This block conditionally outputs one of its inputs, based on a condition and a list of cases.
 *
 * This of it as a passive (data) version of the switch statement in programming languages.
 */
class FlowGraphDataSwitchBlock extends FlowGraphBlock {
    constructor(
    /**
     * the configuration of the block
     */
    config) {
        super(config);
        this.config = config;
        this._inputCases = new Map();
        this.case = this.registerDataInput("case", RichTypeAny, NaN);
        this.default = this.registerDataInput("default", RichTypeAny);
        this.value = this.registerDataOutput("value", RichTypeAny);
        // iterate the set not using for of
        const array = this.config.cases || [];
        for (let caseValue of array) {
            // if treat as integers, make sure not to set it again if it exists
            caseValue = getNumericValue(caseValue);
            if (this.config.treatCasesAsIntegers) {
                caseValue = caseValue | 0;
                if (this._inputCases.has(caseValue)) {
                    return;
                }
            }
            this._inputCases.set(caseValue, this.registerDataInput(`in_${caseValue}`, RichTypeAny));
        }
    }
    _updateOutputs(context) {
        const selectionValue = this.case.getValue(context);
        let outputValue;
        if (isNumeric(selectionValue)) {
            outputValue = this._getOutputValueForCase(getNumericValue(selectionValue), context);
        }
        if (!outputValue) {
            outputValue = this.default.getValue(context);
        }
        this.value.setValue(outputValue, context);
    }
    _getOutputValueForCase(caseValue, context) {
        return this._inputCases.get(caseValue)?.getValue(context);
    }
    getClassName() {
        return "FlowGraphDataSwitchBlock" /* FlowGraphBlockNames.DataSwitch */;
    }
}
RegisterClass("FlowGraphDataSwitchBlock" /* FlowGraphBlockNames.DataSwitch */, FlowGraphDataSwitchBlock);

export { FlowGraphDataSwitchBlock };
//# sourceMappingURL=flowGraphDataSwitchBlock-CYVp9cjl.js.map
