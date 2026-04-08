import { S as ShaderStore } from './index-BI1f02lL.js';
import './index-CDZuCcOm.js';

// Do not edit.
const name = "shadowMapFragmentSoftTransparentShadow";
const shader = `#if SM_SOFTTRANSPARENTSHADOW==1
if ((bayerDither8(floor(mod(gl_FragCoord.xy,8.0))))/64.0>=softTransparentShadowSM.x*alpha) discard;
#endif
`;
// Sideeffect
if (!ShaderStore.IncludesShadersStore[name]) {
    ShaderStore.IncludesShadersStore[name] = shader;
}
/** @internal */
const shadowMapFragmentSoftTransparentShadow = { name, shader };

export { shadowMapFragmentSoftTransparentShadow };
//# sourceMappingURL=shadowMapFragmentSoftTransparentShadow-BtUJmJ2m.js.map
