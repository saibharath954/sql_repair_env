import { S as ShaderStore } from './index-BI1f02lL.js';
import './index-CDZuCcOm.js';

// Do not edit.
const name = "oitFinalSimpleBlendPixelShader";
const shader = `var uFrontColor: texture_2d<f32>;@fragment
fn main(input: FragmentInputs)->FragmentOutputs {var fragCoord: vec2i=vec2i(fragmentInputs.position.xy);var frontColor: vec4f=textureLoad(uFrontColor,fragCoord,0);fragmentOutputs.color=frontColor;}
`;
// Sideeffect
if (!ShaderStore.ShadersStoreWGSL[name]) {
    ShaderStore.ShadersStoreWGSL[name] = shader;
}
/** @internal */
const oitFinalSimpleBlendPixelShaderWGSL = { name, shader };

export { oitFinalSimpleBlendPixelShaderWGSL };
//# sourceMappingURL=oitFinalSimpleBlend.fragment-Crp-oYhz.js.map
