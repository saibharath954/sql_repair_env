import { S as ShaderStore } from './index-BI1f02lL.js';
import './helperFunctions-fmJYxuab.js';
import './index-CDZuCcOm.js';

// Do not edit.
const name = "iblScaledLuminancePixelShader";
const shader = `precision highp sampler2D;precision highp samplerCube;
#include<helperFunctions>
varying vec2 vUV;
#ifdef IBL_USE_CUBE_MAP
uniform samplerCube iblSource;
#else
uniform sampler2D iblSource;
#endif
uniform int iblWidth;uniform int iblHeight;float fetchLuminance(vec2 coords) {
#ifdef IBL_USE_CUBE_MAP
vec3 direction=equirectangularToCubemapDirection(coords);vec3 color=textureCubeLodEXT(iblSource,direction,0.0).rgb;
#else
vec3 color=textureLod(iblSource,coords,0.0).rgb;
#endif
return dot(color,LuminanceEncodeApprox);}
void main(void) {float deform=sin(vUV.y*PI);float luminance=fetchLuminance(vUV);gl_FragColor=vec4(vec3(deform*luminance),1.0);}`;
// Sideeffect
if (!ShaderStore.ShadersStore[name]) {
    ShaderStore.ShadersStore[name] = shader;
}
/** @internal */
const iblScaledLuminancePixelShader = { name, shader };

export { iblScaledLuminancePixelShader };
//# sourceMappingURL=iblScaledLuminance.fragment-L2cpV9ed.js.map
