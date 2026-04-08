import { S as ShaderStore } from './index-BI1f02lL.js';
import './index-CDZuCcOm.js';

// Do not edit.
const name = "shadowMapFragmentSoftTransparentShadow";
const shader = `#if SM_SOFTTRANSPARENTSHADOW==1
if ((bayerDither8(floor(((fragmentInputs.position.xy)%(8.0)))))/64.0>=uniforms.softTransparentShadowSM.x*alpha) {discard;}
#endif
`;
// Sideeffect
if (!ShaderStore.IncludesShadersStoreWGSL[name]) {
    ShaderStore.IncludesShadersStoreWGSL[name] = shader;
}
/** @internal */
const shadowMapFragmentSoftTransparentShadowWGSL = { name, shader };

export { shadowMapFragmentSoftTransparentShadowWGSL };
//# sourceMappingURL=shadowMapFragmentSoftTransparentShadow-Cv5BgDmm.js.map
