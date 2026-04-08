import { S as ShaderStore } from './index-BI1f02lL.js';
import './imageProcessingFunctions-EWGnB6ap.js';
import './helperFunctions-fmJYxuab.js';
import './index-CDZuCcOm.js';

// Do not edit.
const name = "imageProcessingPixelShader";
const shader = `varying vec2 vUV;uniform sampler2D textureSampler;
#include<imageProcessingDeclaration>
#include<helperFunctions>
#include<imageProcessingFunctions>
#define CUSTOM_FRAGMENT_DEFINITIONS
void main(void)
{vec4 result=texture2D(textureSampler,vUV);result.rgb=max(result.rgb,vec3(0.));
#ifdef IMAGEPROCESSING
#ifndef FROMLINEARSPACE
result.rgb=toLinearSpace(result.rgb);
#endif
result=applyImageProcessing(result);
#else
#ifdef FROMLINEARSPACE
result=applyImageProcessing(result);
#endif
#endif
gl_FragColor=result;}`;
// Sideeffect
if (!ShaderStore.ShadersStore[name]) {
    ShaderStore.ShadersStore[name] = shader;
}
/** @internal */
const imageProcessingPixelShader = { name, shader };

export { imageProcessingPixelShader };
//# sourceMappingURL=imageProcessing.fragment-C_vIZos0.js.map
