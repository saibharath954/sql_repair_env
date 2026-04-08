import { S as ShaderStore } from './index-BI1f02lL.js';
import './helperFunctions-fmJYxuab.js';
import './index-CDZuCcOm.js';

// Do not edit.
const name = "rgbdEncodePixelShader";
const shader = `varying vec2 vUV;uniform sampler2D textureSampler;
#include<helperFunctions>
#define CUSTOM_FRAGMENT_DEFINITIONS
void main(void) 
{gl_FragColor=toRGBD(texture2D(textureSampler,vUV).rgb);}`;
// Sideeffect
if (!ShaderStore.ShadersStore[name]) {
    ShaderStore.ShadersStore[name] = shader;
}
/** @internal */
const rgbdEncodePixelShader = { name, shader };

export { rgbdEncodePixelShader };
//# sourceMappingURL=rgbdEncode.fragment-svcaYv5-.js.map
