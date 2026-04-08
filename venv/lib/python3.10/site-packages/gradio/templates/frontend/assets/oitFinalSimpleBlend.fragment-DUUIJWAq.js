import { S as ShaderStore } from './index-BI1f02lL.js';
import './index-CDZuCcOm.js';

// Do not edit.
const name = "oitFinalSimpleBlendPixelShader";
const shader = `precision highp float;uniform sampler2D uFrontColor;void main() {ivec2 fragCoord=ivec2(gl_FragCoord.xy);vec4 frontColor=texelFetch(uFrontColor,fragCoord,0);glFragColor=frontColor;}
`;
// Sideeffect
if (!ShaderStore.ShadersStore[name]) {
    ShaderStore.ShadersStore[name] = shader;
}
/** @internal */
const oitFinalSimpleBlendPixelShader = { name, shader };

export { oitFinalSimpleBlendPixelShader };
//# sourceMappingURL=oitFinalSimpleBlend.fragment-DUUIJWAq.js.map
