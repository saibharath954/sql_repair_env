import './async-D55cHugf.js';
import { c as spread_props } from './index-K3l_dLem.js';
import './2-DKaY_6dX.js';
import { G as Gradio } from './utils.svelte-D1m0ck_w.js';
import { a as JSON$1, J as JSON_1 } from './JSON-DQANennA.js';
import { B as Block } from './Block-qDbnR9DW.js';
import './MarkdownCode.svelte_svelte_type_style_lang-B2xYMNIW.js';
import { B as BlockLabel } from './BlockLabel-C-NWYVSw.js';
import { S as Static } from './index3-C2SvQ33H.js';
import './escaping-CBnpiEl5.js';
import './context-DF4-UEpk.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';
import './Check-B-uwlXei.js';
import './Copy-lixG99xU.js';
import './IconButton-BOK4HpdV.js';
import './Empty-Dg8eJz4H.js';
import './IconButtonWrapper-BSVqsNEI.js';
import './prism-python-CNqfI2Ql.js';
import './Clear-DH-TDCgr.js';

function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const { $$slots, $$events, ...props } = $$props;
    const gradio = new Gradio(props);
    gradio.props.value;
    let label_height = 0;
    Block($$renderer2, {
      visible: gradio.shared.visible,
      test_id: "json",
      elem_id: gradio.shared.elem_id,
      elem_classes: gradio.shared.elem_classes,
      container: gradio.shared.container,
      scale: gradio.shared.scale,
      min_width: gradio.shared.min_width,
      padding: false,
      allow_overflow: true,
      overflow_behavior: "auto",
      height: gradio.props.height,
      min_height: gradio.props.min_height,
      max_height: gradio.props.max_height,
      children: ($$renderer3) => {
        $$renderer3.push(`<div>`);
        if (gradio.shared.label) {
          $$renderer3.push("<!--[-->");
          BlockLabel($$renderer3, {
            Icon: JSON$1,
            show_label: gradio.shared.show_label,
            label: gradio.shared.label,
            float: false,
            disable: gradio.shared.container === false
          });
        } else {
          $$renderer3.push("<!--[!-->");
        }
        $$renderer3.push(`<!--]--></div> `);
        Static($$renderer3, spread_props([
          { autoscroll: gradio.shared.autoscroll, i18n: gradio.i18n },
          gradio.shared.loading_status,
          {
            on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
          }
        ]));
        $$renderer3.push(`<!----> `);
        JSON_1($$renderer3, {
          value: gradio.props.value,
          open: gradio.props.open,
          theme_mode: gradio.props.theme_mode,
          show_indices: gradio.props.show_indices,
          show_copy_button: gradio.props.buttons == null ? true : gradio.props.buttons.some((btn) => typeof btn === "string" && btn === "copy"),
          buttons: gradio.props.buttons,
          on_custom_button_click: (id) => {
            gradio.dispatch("custom_button_click", { id });
          },
          label_height
        });
        $$renderer3.push(`<!---->`);
      },
      $$slots: { default: true }
    });
  });
}

export { JSON_1 as BaseJSON, Index as default };
//# sourceMappingURL=Index13-C6Z5VP9w.js.map
