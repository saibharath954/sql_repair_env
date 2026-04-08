import { S as ShaderStore } from './index-BI1f02lL.js';
import './helperFunctions-fmJYxuab.js';
import './index-CDZuCcOm.js';

// Do not edit.
const name = "rgbdDecodePixelShader";
const shader = `varying vec2 vUV;uniform sampler2D textureSampler;
#include<helperFunctions>
#define CUSTOM_FRAGMENT_DEFINITIONS
void main(void) 
{gl_FragColor=vec4(fromRGBD(texture2D(textureSampler,vUV)),1.0);}`;
// Sideeffect
if (!ShaderStore.ShadersStore[name]) {
    ShaderStore.ShadersStore[name] = shader;
}
/** @internal */
const rgbdDecodePixelShader = { name, shader };

export { rgbdDecodePixelShader };
//# sourceMappingURL=rgbdDecode.fragment-DPM-wA6U.js.map
