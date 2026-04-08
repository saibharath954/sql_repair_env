import { S as ShaderStore } from './index-BI1f02lL.js';
import './helperFunctions-fmJYxuab.js';
import './hdrFilteringFunctions-B8UdcXau.js';
import './pbrBRDFFunctions-gS-vvtBC.js';
import './index-CDZuCcOm.js';

// Do not edit.
const name = "hdrFilteringPixelShader";
const shader = `#include<helperFunctions>
#include<importanceSampling>
#include<pbrBRDFFunctions>
#include<hdrFilteringFunctions>
uniform float alphaG;uniform samplerCube inputTexture;uniform vec2 vFilteringInfo;uniform float hdrScale;varying vec3 direction;void main() {vec3 color=radiance(alphaG,inputTexture,direction,vFilteringInfo);gl_FragColor=vec4(color*hdrScale,1.0);}`;
// Sideeffect
if (!ShaderStore.ShadersStore[name]) {
    ShaderStore.ShadersStore[name] = shader;
}
/** @internal */
const hdrFilteringPixelShader = { name, shader };

export { hdrFilteringPixelShader };
//# sourceMappingURL=hdrFiltering.fragment-CxSBv9iE.js.map
