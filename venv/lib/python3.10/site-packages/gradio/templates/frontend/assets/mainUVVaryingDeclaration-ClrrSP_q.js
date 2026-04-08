import { S as ShaderStore } from './index-BI1f02lL.js';

// Do not edit.
const name = "mainUVVaryingDeclaration";
const shader = `#ifdef MAINUV{X}
varying vec2 vMainUV{X};
#endif
`;
// Sideeffect
if (!ShaderStore.IncludesShadersStore[name]) {
    ShaderStore.IncludesShadersStore[name] = shader;
}
//# sourceMappingURL=mainUVVaryingDeclaration-ClrrSP_q.js.map
