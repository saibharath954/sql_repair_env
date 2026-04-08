import './async-D55cHugf.js';
import { f as attr_class } from './index-K3l_dLem.js';
import { J as JSON_1 } from './JSON-DQANennA.js';
import './escaping-CBnpiEl5.js';
import './context-DF4-UEpk.js';
import './Check-B-uwlXei.js';
import './Copy-lixG99xU.js';
import './MarkdownCode.svelte_svelte_type_style_lang-B2xYMNIW.js';
import './prism-python-CNqfI2Ql.js';
import './2-DKaY_6dX.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './IconButton-BOK4HpdV.js';
import './Empty-Dg8eJz4H.js';
import './IconButtonWrapper-BSVqsNEI.js';

function Example($$renderer, $$props) {
  let { value, theme_mode = "system", type, selected = false } = $$props;
  let show_indices = false;
  let label_height = 0;
  $$renderer.push(`<div${attr_class("container svelte-19cq9h3", void 0, {
    "table": type === "table",
    "gallery": type === "gallery",
    "selected": selected,
    "border": value
  })}>`);
  if (value) {
    $$renderer.push("<!--[-->");
    JSON_1($$renderer, {
      value,
      open: true,
      theme_mode,
      show_indices,
      label_height,
      interactive: false,
      show_copy_button: false
    });
  } else {
    $$renderer.push("<!--[!-->");
  }
  $$renderer.push(`<!--]--></div>`);
}

export { Example as default };
//# sourceMappingURL=Example21-BRPvJBEu.js.map
