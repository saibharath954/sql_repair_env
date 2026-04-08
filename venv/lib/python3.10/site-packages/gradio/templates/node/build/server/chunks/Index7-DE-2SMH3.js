import { f as fallback } from './async-D55cHugf.js';
import { d as bind_props, s as slot } from './index-K3l_dLem.js';
import { B as Block } from './Block-qDbnR9DW.js';
import './MarkdownCode.svelte_svelte_type_style_lang-B2xYMNIW.js';
import './2-DKaY_6dX.js';
import './escaping-CBnpiEl5.js';
import './context-DF4-UEpk.js';
import './prism-python-CNqfI2Ql.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';

function Index($$renderer, $$props) {
  let elem_id = $$props["elem_id"];
  let elem_classes = $$props["elem_classes"];
  let visible = fallback($$props["visible"], true);
  Block($$renderer, {
    elem_id,
    elem_classes,
    visible,
    explicit_call: true,
    children: ($$renderer2) => {
      $$renderer2.push(`<!--[-->`);
      slot($$renderer2, $$props, "default", {}, null);
      $$renderer2.push(`<!--]-->`);
    },
    $$slots: { default: true }
  });
  bind_props($$props, { elem_id, elem_classes, visible });
}

export { Index as default };
//# sourceMappingURL=Index7-DE-2SMH3.js.map
