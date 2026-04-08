import { S as ShaderStore } from './index-BI1f02lL.js';
import './kernelBlurVaryingDeclaration-B7AaBPTM.js';
import './index-CDZuCcOm.js';

// Do not edit.
const name$1 = "kernelBlurVertex";
const shader$1 = `sampleCoord{X}=sampleCenter+delta*KERNEL_OFFSET{X};`;
// Sideeffect
if (!ShaderStore.IncludesShadersStore[name$1]) {
    ShaderStore.IncludesShadersStore[name$1] = shader$1;
}

// Do not edit.
const name = "kernelBlurVertexShader";
const shader = `attribute vec2 position;uniform vec2 delta;varying vec2 sampleCenter;
#include<kernelBlurVaryingDeclaration>[0..varyingCount]
const vec2 madd=vec2(0.5,0.5);
#define CUSTOM_VERTEX_DEFINITIONS
void main(void) {
#define CUSTOM_VERTEX_MAIN_BEGIN
sampleCenter=(position*madd+madd);
#include<kernelBlurVertex>[0..varyingCount]
gl_Position=vec4(position,0.0,1.0);
#define CUSTOM_VERTEX_MAIN_END
}`;
// Sideeffect
if (!ShaderStore.ShadersStore[name]) {
    ShaderStore.ShadersStore[name] = shader;
}
/** @internal */
const kernelBlurVertexShader = { name, shader };

export { kernelBlurVertexShader };
//# sourceMappingURL=kernelBlur.vertex-DXqZkozG.js.map
