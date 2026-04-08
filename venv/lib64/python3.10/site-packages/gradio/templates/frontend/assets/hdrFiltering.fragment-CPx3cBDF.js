import { S as ShaderStore } from './index-BI1f02lL.js';
import './helperFunctions-DBi8xAe_.js';
import './hdrFilteringFunctions-150YjpL6.js';
import './pbrBRDFFunctions-ZWfLNehP.js';
import './index-CDZuCcOm.js';

// Do not edit.
const name = "hdrFilteringPixelShader";
const shader = `#include<helperFunctions>
#include<importanceSampling>
#include<pbrBRDFFunctions>
#include<hdrFilteringFunctions>
uniform alphaG: f32;var inputTextureSampler: sampler;var inputTexture: texture_cube<f32>;uniform vFilteringInfo: vec2f;uniform hdrScale: f32;varying direction: vec3f;@fragment
fn main(input: FragmentInputs)->FragmentOutputs {var color: vec3f=radiance(uniforms.alphaG,inputTexture,inputTextureSampler,input.direction,uniforms.vFilteringInfo);fragmentOutputs.color= vec4f(color*uniforms.hdrScale,1.0);}`;
// Sideeffect
if (!ShaderStore.ShadersStoreWGSL[name]) {
    ShaderStore.ShadersStoreWGSL[name] = shader;
}
/** @internal */
const hdrFilteringPixelShaderWGSL = { name, shader };

export { hdrFilteringPixelShaderWGSL };
//# sourceMappingURL=hdrFiltering.fragment-CPx3cBDF.js.map
