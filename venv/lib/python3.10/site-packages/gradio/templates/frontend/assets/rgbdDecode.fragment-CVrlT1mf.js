import { S as ShaderStore } from './index-BI1f02lL.js';
import './helperFunctions-DBi8xAe_.js';
import './index-CDZuCcOm.js';

// Do not edit.
const name = "rgbdDecodePixelShader";
const shader = `varying vUV: vec2f;var textureSamplerSampler: sampler;var textureSampler: texture_2d<f32>;
#include<helperFunctions>
#define CUSTOM_FRAGMENT_DEFINITIONS
@fragment
fn main(input: FragmentInputs)->FragmentOutputs {fragmentOutputs.color=vec4f(fromRGBD(textureSample(textureSampler,textureSamplerSampler,input.vUV)),1.0);}`;
// Sideeffect
if (!ShaderStore.ShadersStoreWGSL[name]) {
    ShaderStore.ShadersStoreWGSL[name] = shader;
}
/** @internal */
const rgbdDecodePixelShaderWGSL = { name, shader };

export { rgbdDecodePixelShaderWGSL };
//# sourceMappingURL=rgbdDecode.fragment-CVrlT1mf.js.map
