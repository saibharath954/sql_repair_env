import { f as fallback } from './async-D55cHugf.js';
import { f as attr_class, d as bind_props } from './index-K3l_dLem.js';
import { h as html } from './html-CfyvkLET.js';
import './escaping-CBnpiEl5.js';
import './context-DF4-UEpk.js';

function Example($$renderer, $$props) {
  let value = $$props["value"];
  let type = $$props["type"];
  let selected = fallback($$props["selected"], false);
  $$renderer.push(`<div${attr_class("prose svelte-s7j0w2", void 0, {
    "table": type === "table",
    "gallery": type === "gallery",
    "selected": selected
  })}>${html(value)}</div>`);
  bind_props($$props, { value, type, selected });
}

export { Example as default };
//# sourceMappingURL=Example18-CKy701I9.js.map
