import { GLTFLoader } from './glTFLoader-BUcHb6gN.js';
import { aG as unregisterGLTFExtension, aH as registerGLTFExtension } from './index-BI1f02lL.js';
import './index-CDZuCcOm.js';
import './bone-Ba-ia2sG.js';
import './skeleton-DVci1tzV.js';
import './rawTexture-B2_UZ8dA.js';
import './assetContainer-DVDIKsIk.js';
import './objectModelMapping-CCJRR4nC.js';

const NAME = "MSFT_minecraftMesh";
/** @internal */
// eslint-disable-next-line @typescript-eslint/naming-convention
class MSFT_minecraftMesh {
    /** @internal */
    constructor(loader) {
        /** @internal */
        this.name = NAME;
        this._loader = loader;
        this.enabled = this._loader.isExtensionUsed(NAME);
    }
    /** @internal */
    dispose() {
        this._loader = null;
    }
    /** @internal */
    // eslint-disable-next-line no-restricted-syntax
    loadMaterialPropertiesAsync(context, material, babylonMaterial) {
        return GLTFLoader.LoadExtraAsync(context, material, this.name, async (extraContext, extra) => {
            if (extra) {
                if (!this._loader._pbrMaterialImpl) {
                    throw new Error(`${extraContext}: Material type not supported`);
                }
                const promise = this._loader.loadMaterialPropertiesAsync(context, material, babylonMaterial);
                if (babylonMaterial.needAlphaBlending()) {
                    babylonMaterial.forceDepthWrite = true;
                    babylonMaterial.separateCullingPass = true;
                }
                babylonMaterial.backFaceCulling = babylonMaterial.forceDepthWrite;
                babylonMaterial.twoSidedLighting = true;
                return await promise;
            }
        });
    }
}
unregisterGLTFExtension(NAME);
registerGLTFExtension(NAME, true, (loader) => new MSFT_minecraftMesh(loader));

export { MSFT_minecraftMesh };
//# sourceMappingURL=MSFT_minecraftMesh-B7UDeOn9.js.map
