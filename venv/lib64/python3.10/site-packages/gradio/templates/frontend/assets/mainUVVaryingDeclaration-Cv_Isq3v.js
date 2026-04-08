import { S as ShaderStore } from './index-BI1f02lL.js';

// Do not edit.
const name = "mainUVVaryingDeclaration";
const shader = `#ifdef MAINUV{X}
varying vMainUV{X}: vec2f;
#endif
`;
// Sideeffect
if (!ShaderStore.IncludesShadersStoreWGSL[name]) {
    ShaderStore.IncludesShadersStoreWGSL[name] = shader;
}
//# sourceMappingURL=mainUVVaryingDeclaration-Cv_Isq3v.js.map
