import './async-D55cHugf.js';
import { f as attr_class } from './index-K3l_dLem.js';

/* empty css                                         */
function ScrollFade($$renderer, $$props) {
  let { visible = false, position = "sticky" } = $$props;
  if (visible) {
    $$renderer.push("<!--[-->");
    $$renderer.push(`<div${attr_class("scroll-fade svelte-kmbucf", void 0, { "absolute": position === "absolute" })}></div>`);
  } else {
    $$renderer.push("<!--[!-->");
  }
  $$renderer.push(`<!--]-->`);
}

export { ScrollFade as S };
//# sourceMappingURL=ScrollFade-74-kimoc.js.map
