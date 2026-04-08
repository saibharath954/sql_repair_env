import { S as ShaderStore } from './index-BI1f02lL.js';
import './helperFunctions-DBi8xAe_.js';
import './index-CDZuCcOm.js';

// Do not edit.
const name = "rgbdEncodePixelShader";
const shader = `varying vUV: vec2f;var textureSamplerSampler: sampler;var textureSampler: texture_2d<f32>;
#include<helperFunctions>
#define CUSTOM_FRAGMENT_DEFINITIONS
@fragment
fn main(input: FragmentInputs)->FragmentOutputs {fragmentOutputs.color=toRGBD(textureSample(textureSampler,textureSamplerSampler,input.vUV).rgb);}`;
// Sideeffect
if (!ShaderStore.ShadersStoreWGSL[name]) {
    ShaderStore.ShadersStoreWGSL[name] = shader;
}
/** @internal */
const rgbdEncodePixelShaderWGSL = { name, shader };

export { rgbdEncodePixelShaderWGSL };
//# sourceMappingURL=rgbdEncode.fragment-CYMiKa4k.js.map
