import { S as ShaderStore } from './index-BI1f02lL.js';
import './helperFunctions-fmJYxuab.js';
import './hdrFilteringFunctions-B8UdcXau.js';
import './pbrBRDFFunctions-gS-vvtBC.js';
import './index-CDZuCcOm.js';

// Do not edit.
const name = "hdrIrradianceFilteringPixelShader";
const shader = `#include<helperFunctions>
#include<importanceSampling>
#include<pbrBRDFFunctions>
#include<hdrFilteringFunctions>
uniform samplerCube inputTexture;
#ifdef IBL_CDF_FILTERING
uniform sampler2D icdfTexture;
#endif
uniform vec2 vFilteringInfo;uniform float hdrScale;varying vec3 direction;void main() {vec3 color=irradiance(inputTexture,direction,vFilteringInfo,0.0,vec3(1.0),direction
#ifdef IBL_CDF_FILTERING
,icdfTexture
#endif
);gl_FragColor=vec4(color*hdrScale,1.0);}`;
// Sideeffect
if (!ShaderStore.ShadersStore[name]) {
    ShaderStore.ShadersStore[name] = shader;
}
/** @internal */
const hdrIrradianceFilteringPixelShader = { name, shader };

export { hdrIrradianceFilteringPixelShader };
//# sourceMappingURL=hdrIrradianceFiltering.fragment-CjrMNRZz.js.map
