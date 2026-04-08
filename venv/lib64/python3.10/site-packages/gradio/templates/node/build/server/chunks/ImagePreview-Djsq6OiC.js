import { f as fallback } from './async-D55cHugf.js';
import { f as attr_class, d as bind_props } from './index-K3l_dLem.js';
import './2-DKaY_6dX.js';
import { u as uploadToHuggingFace } from './utils.svelte-D1m0ck_w.js';
import './MarkdownCode.svelte_svelte_type_style_lang-B2xYMNIW.js';
import { B as BlockLabel } from './BlockLabel-C-NWYVSw.js';
import { D as DownloadLink } from './DownloadLink-CR_zSSrd.js';
import { I as IconButton } from './IconButton-BOK4HpdV.js';
import { E as Empty } from './Empty-Dg8eJz4H.js';
import { S as ShareButton } from './ShareButton-lm5teuLR.js';
import { D as Download } from './Download-ByiErn53.js';
import { I as Image$1 } from './Image2-BLbvQKZw.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-BSVqsNEI.js';
import { F as FullscreenButton } from './FullscreenButton-D3sdKON5.js';
import { I as Image } from './Image-CZw3rP1w.js';
import './escaping-CBnpiEl5.js';
import './context-DF4-UEpk.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';
import './prism-python-CNqfI2Ql.js';
import './index-server-BzRj6e_1.js';
import './Maximize-B77VDSzq.js';

/* empty css                                           */
function ImagePreview($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let value = $$props["value"];
    let label = fallback($$props["label"], void 0);
    let show_label = $$props["show_label"];
    let buttons = fallback($$props["buttons"], () => [], true);
    let on_custom_button_click = fallback($$props["on_custom_button_click"], null);
    let selectable = fallback($$props["selectable"], false);
    let i18n = $$props["i18n"];
    let display_icon_button_wrapper_top_corner = fallback($$props["display_icon_button_wrapper_top_corner"], false);
    let fullscreen = fallback($$props["fullscreen"], false);
    let show_button_background = fallback($$props["show_button_background"], true);
    BlockLabel($$renderer2, {
      show_label,
      Icon: Image$1,
      label: !show_label ? "" : label || i18n("image.image")
    });
    $$renderer2.push(`<!----> `);
    if (value == null || !value?.url) {
      $$renderer2.push("<!--[-->");
      Empty($$renderer2, {
        unpadded_box: true,
        size: "large",
        children: ($$renderer3) => {
          Image$1($$renderer3);
        },
        $$slots: { default: true }
      });
    } else {
      $$renderer2.push("<!--[!-->");
      $$renderer2.push(`<div class="image-container svelte-12vrxzd">`);
      IconButtonWrapper($$renderer2, {
        display_top_corner: display_icon_button_wrapper_top_corner,
        show_background: show_button_background,
        buttons,
        on_custom_button_click,
        children: ($$renderer3) => {
          if (buttons.some((btn) => typeof btn === "string" && btn === "fullscreen")) {
            $$renderer3.push("<!--[-->");
            FullscreenButton($$renderer3, {
              fullscreen,
              onclick: (is_fullscreen) => {
                fullscreen = is_fullscreen;
              }
            });
          } else {
            $$renderer3.push("<!--[!-->");
          }
          $$renderer3.push(`<!--]--> `);
          if (buttons.some((btn) => typeof btn === "string" && btn === "download")) {
            $$renderer3.push("<!--[-->");
            DownloadLink($$renderer3, {
              href: value.url,
              download: value.orig_name || "image",
              children: ($$renderer4) => {
                IconButton($$renderer4, { Icon: Download, label: i18n("common.download") });
              },
              $$slots: { default: true }
            });
          } else {
            $$renderer3.push("<!--[!-->");
          }
          $$renderer3.push(`<!--]--> `);
          if (buttons.some((btn) => typeof btn === "string" && btn === "share")) {
            $$renderer3.push("<!--[-->");
            ShareButton($$renderer3, {
              i18n,
              formatter: async (value2) => {
                if (!value2) return "";
                let url = await uploadToHuggingFace(value2);
                return `<img src="${url}" />`;
              },
              value
            });
          } else {
            $$renderer3.push("<!--[!-->");
          }
          $$renderer3.push(`<!--]-->`);
        }
      });
      $$renderer2.push(`<!----> <button class="svelte-12vrxzd"><div${attr_class("image-frame svelte-12vrxzd", void 0, { "selectable": selectable })}>`);
      Image($$renderer2, { src: value.url, restProps: { loading: "lazy", alt: "" } });
      $$renderer2.push(`<!----></div></button></div>`);
    }
    $$renderer2.push(`<!--]-->`);
    bind_props($$props, {
      value,
      label,
      show_label,
      buttons,
      on_custom_button_click,
      selectable,
      i18n,
      display_icon_button_wrapper_top_corner,
      fullscreen,
      show_button_background
    });
  });
}

export { ImagePreview as default };
//# sourceMappingURL=ImagePreview-Djsq6OiC.js.map
