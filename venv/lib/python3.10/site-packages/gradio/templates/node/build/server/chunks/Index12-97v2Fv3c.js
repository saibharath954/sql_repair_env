import './async-D55cHugf.js';
import { s as slot } from './index-K3l_dLem.js';
import './2-DKaY_6dX.js';
import { G as Gradio } from './utils.svelte-D1m0ck_w.js';
import { B as BaseForm } from './BaseForm-meBrG-oF.js';
import './escaping-CBnpiEl5.js';
import './context-DF4-UEpk.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';

function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { $$slots, $$events, ...props } = $$props;
    const gradio = new Gradio(props);
    BaseForm($$renderer2, {
      visible: gradio.shared.visible,
      scale: gradio.shared.scale,
      min_width: gradio.shared.min_width,
      children: ($$renderer3) => {
        $$renderer3.push(`<!--[-->`);
        slot($$renderer3, $$props, "default", {}, null);
        $$renderer3.push(`<!--]-->`);
      },
      $$slots: { default: true }
    });
  });
}

export { BaseForm, Index as default };
//# sourceMappingURL=Index12-97v2Fv3c.js.map
