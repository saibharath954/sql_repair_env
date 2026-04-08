import './async-D55cHugf.js';
import { a as attr, f as attr_class, g as attr_style, i as stringify, s as slot } from './index-K3l_dLem.js';
import './2-DKaY_6dX.js';
import { G as Gradio } from './utils.svelte-D1m0ck_w.js';
import './escaping-CBnpiEl5.js';
import './context-DF4-UEpk.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';

function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const { $$slots, $$events, ...props } = $$props;
    new Gradio(props);
    const elem_id = props.elem_id || "";
    const elem_classes = props.elem_classes || [];
    const visible = props.visible === void 0 ? true : props.visible;
    $$renderer2.push(`<div${attr("id", elem_id)}${attr_class(`gr-group ${stringify(elem_classes.join(" "))}`, "svelte-1p9262q", { "hide": !visible })}><div class="styler svelte-1p9262q"${attr_style("", {
      "--block-radius": "0px",
      "--block-border-width": "0px",
      "--layout-gap": "1px",
      "--form-gap-width": "1px",
      "--button-border-width": "0px",
      "--button-large-radius": "0px",
      "--button-small-radius": "0px"
    })}><!--[-->`);
    slot($$renderer2, $$props, "default", {}, null);
    $$renderer2.push(`<!--]--></div></div>`);
  });
}

export { Index as default };
//# sourceMappingURL=Index27-BZz1ZFAc.js.map
