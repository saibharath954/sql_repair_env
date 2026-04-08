import { S as ShaderStore } from './index-BI1f02lL.js';

// Do not edit.
const name = "logDepthDeclaration";
const shader = `#ifdef LOGARITHMICDEPTH
uniform logarithmicDepthConstant: f32;varying vFragmentDepth: f32;
#endif
`;
// Sideeffect
if (!ShaderStore.IncludesShadersStoreWGSL[name]) {
    ShaderStore.IncludesShadersStoreWGSL[name] = shader;
}
//# sourceMappingURL=logDepthDeclaration-C6VLMPiR.js.map
