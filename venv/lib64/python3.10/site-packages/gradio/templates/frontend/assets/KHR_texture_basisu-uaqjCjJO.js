import { GLTFLoader, ArrayItem } from './glTFLoader-BUcHb6gN.js';
import { aG as unregisterGLTFExtension, aH as registerGLTFExtension } from './index-BI1f02lL.js';
import './index-CDZuCcOm.js';
import './bone-Ba-ia2sG.js';
import './skeleton-DVci1tzV.js';
import './rawTexture-B2_UZ8dA.js';
import './assetContainer-DVDIKsIk.js';
import './objectModelMapping-CCJRR4nC.js';

const NAME = "KHR_texture_basisu";
/**
 * [Specification](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_texture_basisu/README.md)
 */
// eslint-disable-next-line @typescript-eslint/naming-convention
class KHR_texture_basisu {
    /**
     * @internal
     */
    constructor(loader) {
        /** The name of this extension. */
        this.name = NAME;
        this._loader = loader;
        this.enabled = loader.isExtensionUsed(NAME);
    }
    /** @internal */
    dispose() {
        this._loader = null;
    }
    /**
     * @internal
     */
    // eslint-disable-next-line no-restricted-syntax
    _loadTextureAsync(context, texture, assign) {
        return GLTFLoader.LoadExtensionAsync(context, texture, this.name, async (extensionContext, extension) => {
            const sampler = texture.sampler == undefined ? GLTFLoader.DefaultSampler : ArrayItem.Get(`${context}/sampler`, this._loader.gltf.samplers, texture.sampler);
            const image = ArrayItem.Get(`${extensionContext}/source`, this._loader.gltf.images, extension.source);
            return await this._loader._createTextureAsync(context, sampler, image, (babylonTexture) => {
                assign(babylonTexture);
            }, texture._textureInfo.nonColorData ? { useRGBAIfASTCBC7NotAvailableWhenUASTC: true } : undefined, !texture._textureInfo.nonColorData);
        });
    }
}
unregisterGLTFExtension(NAME);
registerGLTFExtension(NAME, true, (loader) => new KHR_texture_basisu(loader));

export { KHR_texture_basisu };
//# sourceMappingURL=KHR_texture_basisu-uaqjCjJO.js.map
