import { S as ShaderStore } from './index-BI1f02lL.js';

// Do not edit.
const name = "meshUboDeclaration";
const shader = `#ifdef WEBGL2
uniform mat4 world;uniform float visibility;
#else
layout(std140,column_major) uniform;uniform Mesh
{mat4 world;float visibility;};
#endif
#define WORLD_UBO
`;
// Sideeffect
if (!ShaderStore.IncludesShadersStore[name]) {
    ShaderStore.IncludesShadersStore[name] = shader;
}
//# sourceMappingURL=meshUboDeclaration-DnEqXuAx.js.map
