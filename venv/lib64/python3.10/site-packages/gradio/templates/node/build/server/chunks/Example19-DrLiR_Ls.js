import { f as fallback } from './async-D55cHugf.js';
import { f as attr_class, d as bind_props } from './index-K3l_dLem.js';
import './2-DKaY_6dX.js';
import './MarkdownCode.svelte_svelte_type_style_lang-B2xYMNIW.js';
import { I as Image } from './Image-CZw3rP1w.js';
import './escaping-CBnpiEl5.js';
import './context-DF4-UEpk.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './prism-python-CNqfI2Ql.js';

/* empty css                                            */
/* empty css                                     */
/* empty css                                       */
function Example($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let value = $$props["value"];
    let type = $$props["type"];
    let selected = fallback($$props["selected"], false);
    $$renderer2.push(`<div${attr_class("container svelte-ous74z", void 0, {
      "table": type === "table",
      "gallery": type === "gallery",
      "selected": selected
    })}>`);
    Image($$renderer2, { src: value.composite?.url || value.background?.url });
    $$renderer2.push(`<!----></div>`);
    bind_props($$props, { value, type, selected });
  });
}

export { Example as default };
//# sourceMappingURL=Example19-DrLiR_Ls.js.map
