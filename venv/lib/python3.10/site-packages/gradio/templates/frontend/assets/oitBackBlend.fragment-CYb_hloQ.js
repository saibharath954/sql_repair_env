import { S as ShaderStore } from './index-BI1f02lL.js';
import './index-CDZuCcOm.js';

// Do not edit.
const name = "oitBackBlendPixelShader";
const shader = `precision highp float;uniform sampler2D uBackColor;void main() {glFragColor=texelFetch(uBackColor,ivec2(gl_FragCoord.xy),0);if (glFragColor.a==0.0) { 
discard;}}`;
// Sideeffect
if (!ShaderStore.ShadersStore[name]) {
    ShaderStore.ShadersStore[name] = shader;
}
/** @internal */
const oitBackBlendPixelShader = { name, shader };

export { oitBackBlendPixelShader };
//# sourceMappingURL=oitBackBlend.fragment-CYb_hloQ.js.map
