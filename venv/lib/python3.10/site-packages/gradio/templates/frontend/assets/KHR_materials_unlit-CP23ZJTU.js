import { C as Color3, aG as unregisterGLTFExtension, aH as registerGLTFExtension } from './index-BI1f02lL.js';
import { GLTFLoader } from './glTFLoader-BUcHb6gN.js';
import './index-CDZuCcOm.js';
import './bone-Ba-ia2sG.js';
import './skeleton-DVci1tzV.js';
import './rawTexture-B2_UZ8dA.js';
import './assetContainer-DVDIKsIk.js';
import './objectModelMapping-CCJRR4nC.js';

const NAME = "KHR_materials_unlit";
/**
 * [Specification](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_unlit/README.md)
 */
// eslint-disable-next-line @typescript-eslint/naming-convention
class KHR_materials_unlit {
    /**
     * @internal
     */
    constructor(loader) {
        /**
         * The name of this extension.
         */
        this.name = NAME;
        /**
         * Defines a number that determines the order the extensions are applied.
         */
        this.order = 210;
        this._loader = loader;
        this.enabled = this._loader.isExtensionUsed(NAME);
    }
    /** @internal */
    dispose() {
        this._loader = null;
    }
    /**
     * @internal
     */
    // eslint-disable-next-line no-restricted-syntax
    loadMaterialPropertiesAsync(context, material, babylonMaterial) {
        return GLTFLoader.LoadExtensionAsync(context, material, this.name, async () => {
            return await this._loadUnlitPropertiesAsync(context, material, babylonMaterial);
        });
    }
    // eslint-disable-next-line @typescript-eslint/promise-function-async, no-restricted-syntax
    _loadUnlitPropertiesAsync(context, material, babylonMaterial) {
        const adapter = this._loader._getOrCreateMaterialAdapter(babylonMaterial);
        const promises = new Array();
        const properties = material.pbrMetallicRoughness;
        if (properties) {
            if (properties.baseColorFactor) {
                adapter.baseColor = Color3.FromArray(properties.baseColorFactor);
                adapter.geometryOpacity = properties.baseColorFactor[3];
            }
            if (properties.baseColorTexture) {
                promises.push(this._loader.loadTextureInfoAsync(`${context}/baseColorTexture`, properties.baseColorTexture, (texture) => {
                    texture.name = `${babylonMaterial.name} (Base Color)`;
                    adapter.baseColorTexture = texture;
                }));
            }
        }
        adapter.isUnlit = true;
        if (material.doubleSided) {
            adapter.backFaceCulling = false;
            adapter.twoSidedLighting = true;
        }
        this._loader.loadMaterialAlphaProperties(context, material, babylonMaterial);
        // eslint-disable-next-line github/no-then
        return Promise.all(promises).then(() => { });
    }
}
unregisterGLTFExtension(NAME);
registerGLTFExtension(NAME, true, (loader) => new KHR_materials_unlit(loader));

export { KHR_materials_unlit };
//# sourceMappingURL=KHR_materials_unlit-CP23ZJTU.js.map
