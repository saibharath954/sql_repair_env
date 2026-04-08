import { S as ShaderStore } from './index-BI1f02lL.js';
import './bakedVertexAnimation-C2fdGo3z.js';
import './instancesDeclaration-DYQMqn2t.js';
import './morphTargetsVertex-u3k3wtlK.js';
import './index-CDZuCcOm.js';

// Do not edit.
const name = "iblVoxelGridVertexShader";
const shader = `attribute vec3 position;varying vec3 vNormalizedPosition;
#include<bonesDeclaration>
#include<bakedVertexAnimationDeclaration>
#include<instancesDeclaration>
#include<morphTargetsVertexGlobalDeclaration>
#include<morphTargetsVertexDeclaration>[0..maxSimultaneousMorphTargets]
uniform mat4 invWorldScale;uniform mat4 viewMatrix;void main(void) {vec3 positionUpdated=position;
#include<morphTargetsVertexGlobal>
#include<morphTargetsVertex>[0..maxSimultaneousMorphTargets]
#include<instancesVertex>
#include<bonesVertex>
#include<bakedVertexAnimation>
vec4 worldPos=finalWorld*vec4(positionUpdated,1.0);gl_Position=viewMatrix*invWorldScale*worldPos;vNormalizedPosition.xyz=gl_Position.xyz*0.5+0.5;
#ifdef IS_NDC_HALF_ZRANGE
gl_Position.z=gl_Position.z*0.5+0.5;
#endif
}`;
// Sideeffect
if (!ShaderStore.ShadersStore[name]) {
    ShaderStore.ShadersStore[name] = shader;
}
/** @internal */
const iblVoxelGridVertexShader = { name, shader };

export { iblVoxelGridVertexShader };
//# sourceMappingURL=iblVoxelGrid.vertex-lC9m1bIG.js.map
