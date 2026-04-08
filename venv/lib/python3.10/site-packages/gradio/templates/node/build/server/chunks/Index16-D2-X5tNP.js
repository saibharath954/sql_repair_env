import './async-D55cHugf.js';
import { c as spread_props } from './index-K3l_dLem.js';
import './2-DKaY_6dX.js';
import { G as Gradio } from './utils.svelte-D1m0ck_w.js';
import { a as Plot, P as Plot$2 } from './Plot-PT1z5jhi.js';
import { B as Block } from './Block-qDbnR9DW.js';
import './MarkdownCode.svelte_svelte_type_style_lang-B2xYMNIW.js';
import { B as BlockLabel } from './BlockLabel-C-NWYVSw.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-BSVqsNEI.js';
import { F as FullscreenButton } from './FullscreenButton-D3sdKON5.js';
import { S as Static } from './index3-C2SvQ33H.js';
import './escaping-CBnpiEl5.js';
import './context-DF4-UEpk.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';
import './Empty-Dg8eJz4H.js';
import './prism-python-CNqfI2Ql.js';
import './IconButton-BOK4HpdV.js';
import './Maximize-B77VDSzq.js';
import './Clear-DH-TDCgr.js';

function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { $$slots, $$events, ...props } = $$props;
    const gradio = new Gradio(props);
    let fullscreen = false;
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      Block($$renderer3, {
        padding: false,
        elem_id: gradio.shared.elem_id,
        elem_classes: gradio.shared.elem_classes,
        visible: gradio.shared.visible,
        container: gradio.shared.container,
        scale: gradio.shared.scale,
        min_width: gradio.shared.min_width,
        allow_overflow: false,
        get fullscreen() {
          return fullscreen;
        },
        set fullscreen($$value) {
          fullscreen = $$value;
          $$settled = false;
        },
        children: ($$renderer4) => {
          BlockLabel($$renderer4, {
            show_label: gradio.shared.show_label,
            label: gradio.shared.label || gradio.i18n("plot.plot"),
            Icon: Plot$2
          });
          $$renderer4.push(`<!----> `);
          if (gradio.props.buttons && gradio.props.buttons.length > 0 || gradio.props.show_fullscreen_button) {
            $$renderer4.push("<!--[-->");
            IconButtonWrapper($$renderer4, {
              buttons: gradio.props.buttons ?? [],
              on_custom_button_click: (id) => {
                gradio.dispatch("custom_button_click", { id });
              },
              children: ($$renderer5) => {
                if (gradio.props.show_fullscreen_button) {
                  $$renderer5.push("<!--[-->");
                  FullscreenButton($$renderer5, { fullscreen });
                } else {
                  $$renderer5.push("<!--[!-->");
                }
                $$renderer5.push(`<!--]-->`);
              }
            });
          } else {
            $$renderer4.push("<!--[!-->");
          }
          $$renderer4.push(`<!--]--> `);
          Static($$renderer4, spread_props([
            { autoscroll: gradio.shared.autoscroll, i18n: gradio.i18n },
            gradio.shared.loading_status,
            {
              on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
            }
          ]));
          $$renderer4.push(`<!----> `);
          Plot($$renderer4, {
            value: gradio.props.value,
            theme_mode: gradio.props.theme_mode,
            show_label: gradio.shared.show_label,
            caption: gradio.props.caption,
            bokeh_version: gradio.props.bokeh_version,
            show_actions_button: gradio.props.show_actions_button,
            _selectable: gradio.props._selectable,
            x_lim: gradio.props.x_lim,
            show_fullscreen_button: gradio.props.show_fullscreen_button,
            on_change: () => gradio.dispatch("change")
          });
          $$renderer4.push(`<!---->`);
        },
        $$slots: { default: true }
      });
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
  });
}

export { Plot as BasePlot, Index as default };
//# sourceMappingURL=Index16-D2-X5tNP.js.map
