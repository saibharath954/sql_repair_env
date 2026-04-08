import './async-D55cHugf.js';
import { c as spread_props, f as attr_class, g as attr_style } from './index-K3l_dLem.js';
import { a as prepare_files } from './2-DKaY_6dX.js';
import { G as Gradio, c as css_units } from './utils.svelte-D1m0ck_w.js';
import HTML from './HTML-DrgcWH8K.js';
import { S as Static } from './index3-C2SvQ33H.js';
import { C as Code } from './Code-DcA0iOIn.js';
import { B as Block } from './Block-qDbnR9DW.js';
import './MarkdownCode.svelte_svelte_type_style_lang-B2xYMNIW.js';
import { B as BlockLabel } from './BlockLabel-C-NWYVSw.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-BSVqsNEI.js';
export { default as BaseExample } from './Example18-CKy701I9.js';
import './escaping-CBnpiEl5.js';
import './context-DF4-UEpk.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';
import './_commonjs-dynamic-modules-DvJQ8VpC.js';
import 'fs';
import './IconButton-BOK4HpdV.js';
import './Clear-DH-TDCgr.js';
import './prism-python-CNqfI2Ql.js';
import './html-CfyvkLET.js';

function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { $$slots, $$events, ...props } = $$props;
    let children = props.children;
    const gradio = new Gradio(props);
    let _props = {
      value: gradio.props.value ?? "",
      label: gradio.shared.label,
      visible: gradio.shared.visible,
      ...gradio.props.props
    };
    gradio.props.value;
    let watch_entries = [];
    function watch(propOrProps, callback) {
      const prop_list = Array.isArray(propOrProps) ? propOrProps : [propOrProps];
      watch_entries.push({ props: prop_list, callback });
    }
    function fire_watchers(changed_keys) {
      const seen = /* @__PURE__ */ new Set();
      for (const entry of watch_entries) {
        if (entry.props.some((k) => changed_keys.includes(k))) {
          seen.add(entry);
        }
      }
      for (const entry of seen) {
        try {
          entry.callback();
        } catch (e) {
          console.error("Error in watch callback:", e);
        }
      }
    }
    async function upload(file) {
      try {
        const file_data = await prepare_files([file]);
        const result = await gradio.shared.client.upload(file_data, gradio.shared.root, void 0, gradio.shared.max_file_size ?? void 0);
        if (result && result[0]) {
          return { path: result[0].path, url: result[0].url };
        }
        throw new Error("Upload failed");
      } catch (e) {
        gradio.dispatch("error", e instanceof Error ? e.message : String(e));
        throw e;
      }
    }
    Block($$renderer2, {
      visible: gradio.shared.visible,
      elem_id: gradio.shared.elem_id,
      elem_classes: gradio.shared.elem_classes,
      container: gradio.shared.container,
      padding: gradio.props.padding !== false,
      overflow_behavior: "visible",
      children: ($$renderer3) => {
        if (gradio.shared.show_label && gradio.props.buttons && gradio.props.buttons.length > 0) {
          $$renderer3.push("<!--[-->");
          IconButtonWrapper($$renderer3, {
            buttons: gradio.props.buttons,
            on_custom_button_click: (id) => {
              gradio.dispatch("custom_button_click", { id });
            }
          });
        } else {
          $$renderer3.push("<!--[!-->");
        }
        $$renderer3.push(`<!--]--> `);
        if (gradio.shared.show_label) {
          $$renderer3.push("<!--[-->");
          BlockLabel($$renderer3, {
            Icon: Code,
            show_label: gradio.shared.show_label,
            label: gradio.shared.label,
            float: true
          });
        } else {
          $$renderer3.push("<!--[!-->");
        }
        $$renderer3.push(`<!--]--> `);
        Static($$renderer3, spread_props([
          { autoscroll: gradio.shared.autoscroll, i18n: gradio.i18n },
          gradio.shared.loading_status,
          {
            variant: "center",
            on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
          }
        ]));
        $$renderer3.push(`<!----> <div${attr_class("html-container svelte-1jts93g", void 0, {
          "pending": gradio.shared.loading_status?.status === "pending" && gradio.shared.loading_status?.show_progress !== "hidden",
          "label-padding": gradio.shared.show_label ?? void 0
        })}${attr_style("", {
          "min-height": gradio.props.min_height && gradio.shared.loading_status?.status !== "pending" ? css_units(gradio.props.min_height) : void 0,
          "max-height": gradio.props.max_height ? css_units(gradio.props.max_height) : void 0,
          "overflow-y": gradio.props.max_height ? "auto" : void 0
        })}>`);
        HTML($$renderer3, {
          props: _props,
          html_template: gradio.props.html_template,
          css_template: gradio.props.css_template,
          js_on_load: gradio.props.js_on_load,
          elem_classes: gradio.shared.elem_classes,
          visible: gradio.shared.visible,
          autoscroll: gradio.shared.autoscroll,
          apply_default_css: gradio.props.apply_default_css,
          head: gradio.props.head,
          component_class_name: gradio.props.component_class_name,
          upload,
          server: gradio.shared.server,
          watch_fn: watch,
          fire_watchers,
          children: ($$renderer4) => {
            children?.($$renderer4);
          }
        });
        $$renderer3.push(`<!----></div>`);
      },
      $$slots: { default: true }
    });
  });
}

export { HTML as BaseHTML, Index as default };
//# sourceMappingURL=Index28-98aMjBbh.js.map
