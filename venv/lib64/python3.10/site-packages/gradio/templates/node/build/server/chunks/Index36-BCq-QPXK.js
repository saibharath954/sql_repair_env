import './async-D55cHugf.js';
import { c as spread_props, f as attr_class, g as attr_style, s as slot, d as bind_props } from './index-K3l_dLem.js';
import { B as Block } from './Block-qDbnR9DW.js';
import './MarkdownCode.svelte_svelte_type_style_lang-B2xYMNIW.js';
import './2-DKaY_6dX.js';
import { G as Gradio } from './utils.svelte-D1m0ck_w.js';
import { S as Static } from './index3-C2SvQ33H.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import { B as BaseColumn } from './Index.svelte_svelte_type_style_lang-Cv_JxLo5.js';
import './context-DF4-UEpk.js';
import './prism-python-CNqfI2Ql.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';
import './IconButton-BOK4HpdV.js';
import './Clear-DH-TDCgr.js';

function Accordion($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { open = true, label = "", onexpand, oncollapse } = $$props;
    $$renderer2.push(`<button${attr_class("label-wrap svelte-e5lyqv", void 0, { "open": open })}><span class="svelte-e5lyqv">${escape_html(label)}</span> <span class="icon svelte-e5lyqv"${attr_style("", { transform: open ? "rotate(0)" : "rotate(90deg)" })}>▼</span></button> <div${attr_style("", { display: open ? "block" : "none" })}><!--[-->`);
    slot($$renderer2, $$props, "default", {}, null);
    $$renderer2.push(`<!--]--></div>`);
    bind_props($$props, { open });
  });
}
function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { $$slots, $$events, ...props } = $$props;
    class AccordionGradio extends Gradio {
      set_data(data) {
        if ("open" in data && data.open) {
          this.dispatch("gradio_expand");
        }
        super.set_data(data);
        this.shared.loading_status.status = "complete";
      }
    }
    const gradio = new AccordionGradio(props);
    let label = gradio.shared.label || "";
    let visibility = gradio.shared.visible === true ? true : "hidden";
    Block($$renderer2, {
      elem_id: gradio.shared.elem_id,
      elem_classes: gradio.shared.elem_classes,
      visible: visibility,
      children: ($$renderer3) => {
        if (gradio.shared.loading_status) {
          $$renderer3.push("<!--[-->");
          Static($$renderer3, spread_props([
            { autoscroll: gradio.shared.autoscroll, i18n: gradio.i18n },
            gradio.shared.loading_status
          ]));
        } else {
          $$renderer3.push("<!--[!-->");
        }
        $$renderer3.push(`<!--]--> `);
        Accordion($$renderer3, {
          label,
          open: gradio.props.open,
          onexpand: () => {
            gradio.dispatch("expand");
            gradio.dispatch("gradio_expand");
          },
          oncollapse: () => gradio.dispatch("collapse"),
          children: ($$renderer4) => {
            BaseColumn($$renderer4, {
              children: ($$renderer5) => {
                $$renderer5.push(`<!--[-->`);
                slot($$renderer5, $$props, "default", {}, null);
                $$renderer5.push(`<!--]-->`);
              },
              $$slots: { default: true }
            });
          },
          $$slots: { default: true }
        });
        $$renderer3.push(`<!---->`);
      },
      $$slots: { default: true }
    });
  });
}

export { Index as default };
//# sourceMappingURL=Index36-BCq-QPXK.js.map
