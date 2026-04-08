import { f as fallback } from './async-D55cHugf.js';
import { d as bind_props } from './index-K3l_dLem.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import './context-DF4-UEpk.js';

function Example($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let value = fallback($$props["value"], () => ({ visible: true, home_page_title: "Home" }), true);
    $$renderer2.push(`<div style="display: none;">Navbar config: visible=${escape_html(value.visible)}, home_page_title="${escape_html(value.home_page_title)}"</div>`);
    bind_props($$props, { value });
  });
}

export { Example as default };
//# sourceMappingURL=Example9-Di0cEFmI.js.map
