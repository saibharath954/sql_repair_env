import './async-D55cHugf.js';
import { c as spread_props, f as attr_class, j as clsx } from './index-K3l_dLem.js';
import { t as tick } from './index-server-BzRj6e_1.js';
import { B as Block } from './Block-qDbnR9DW.js';
import './MarkdownCode.svelte_svelte_type_style_lang-B2xYMNIW.js';
import './2-DKaY_6dX.js';
import { G as Gradio } from './utils.svelte-D1m0ck_w.js';
import { U as UploadText } from './UploadText-DJMtFVv8.js';
import { S as SelectSource } from './SelectSource-DraGihvu.js';
import { G as Gallery, h as handle_save } from './Gallery-__TE_eFm.js';
import { S as Static } from './index3-C2SvQ33H.js';
import { a as FileUpload } from './FileUpload-Bseag68o.js';
import { W as Webcam } from './Webcam2-CSs6lJlq.js';
export { default as BaseExample } from './Example17-CiehcRUM.js';
import './escaping-CBnpiEl5.js';
import './context-DF4-UEpk.js';
import './prism-python-CNqfI2Ql.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';
import './Upload-CIQ-D6yx.js';
import './Microphone-CCAKTpuQ.js';
import './Video-_1zl9-Cr.js';
import './Webcam-iV1BaoQQ.js';
import './BlockLabel-C-NWYVSw.js';
import './IconButton-BOK4HpdV.js';
import './Empty-Dg8eJz4H.js';
import './ShareButton-lm5teuLR.js';
import './Clear-DH-TDCgr.js';
import './Download-ByiErn53.js';
import './Image2-BLbvQKZw.js';
import './Play-BlZWjD1q.js';
import './IconButtonWrapper-BSVqsNEI.js';
import './FullscreenButton-D3sdKON5.js';
import './Maximize-B77VDSzq.js';
import './Upload2-COmifmPq.js';
import './ModifyUpload-DbaqJZ53.js';
import './DownloadLink-CR_zSSrd.js';
import './Edit-W_0aHh0i.js';
import './Undo-Col2BcUY.js';
import './Image-CZw3rP1w.js';
import './Video2-DiC5rUZt.js';
import './File-2S6P7zIO.js';
import './html-CfyvkLET.js';
import './StreamingBar-C51_uM3L.js';

function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let upload_promise = void 0;
    class GalleryGradio extends Gradio {
      async get_data() {
        if (upload_promise) {
          await upload_promise;
          await tick();
        }
        const data = await super.get_data();
        return data;
      }
    }
    const { $$slots, $$events, ...props } = $$props;
    const gradio = new GalleryGradio(props, { selected_index: null, file_types: ["image", "video"] });
    let fullscreen = false;
    function handle_delete(event) {
      if (!gradio.props.value) return;
      const { index } = event.detail;
      gradio.dispatch("delete", event.detail);
      gradio.props.value = gradio.props.value.filter((_, i) => i !== index);
      gradio.dispatch("change", gradio.props.value);
    }
    async function process_upload_files(files) {
      const processed_files = await Promise.all(files.map(async (x) => {
        if (x.path?.toLowerCase().endsWith(".svg") && x.url) {
          const response = await fetch(x.url);
          const svgContent = await response.text();
          return {
            ...x,
            url: `data:image/svg+xml,${encodeURIComponent(svgContent)}`
          };
        }
        return x;
      }));
      return processed_files.map((x) => x.mime_type?.includes("video") ? { video: x, caption: null } : { image: x, caption: null });
    }
    let active_source = gradio.props.sources ? gradio.props.sources[0] : "upload";
    let no_value = gradio.props.value === null ? true : gradio.props.value.length === 0;
    let sources = (() => {
      if (gradio.props.file_types?.includes("video") && gradio.props.sources.includes("webcam")) {
        return gradio.props.sources.concat(["webcam-video"]);
      } else {
        return gradio.props.sources;
      }
    })();
    async function paste_clipboard() {
      navigator.clipboard.read().then(async (items) => {
        let file = null;
        for (let i = 0; i < items.length; i++) {
          const type = items[i].types.find((t) => (gradio.props.file_types || ["image"]).some((ft) => t.startsWith(ft + "/")));
          if (type) {
            const blob = await items[i].getType(type);
            file = new File([blob], `clipboard.${type.replace("image/", "")}`);
            break;
          }
        }
        if (file) {
          const f = await handle_save(file, (f2) => gradio.shared.client.upload(f2, gradio.shared.root), "clipboard_upload");
          const processed_files = await process_upload_files(f);
          gradio.props.value?.push(...processed_files);
          gradio.dispatch("change", gradio.props.value);
          active_source = null;
        } else {
          gradio.dispatch("warning", "No image or video found in clipboard");
        }
      });
    }
    async function handle_select_source(source) {
      switch (source) {
        case "clipboard":
          await paste_clipboard();
          break;
      }
    }
    async function onsource_change(source) {
      await tick();
      if (source === "clipboard") {
        await paste_clipboard();
      } else {
        active_source = source;
        no_value = true;
      }
    }
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      Block($$renderer3, {
        visible: gradio.shared.visible,
        variant: "solid",
        padding: false,
        elem_id: gradio.shared.elem_id,
        elem_classes: gradio.shared.elem_classes,
        container: gradio.shared.container,
        scale: gradio.shared.scale,
        min_width: gradio.shared.min_width,
        allow_overflow: false,
        height: gradio.props.height || void 0,
        get fullscreen() {
          return fullscreen;
        },
        set fullscreen($$value) {
          fullscreen = $$value;
          $$settled = false;
        },
        children: ($$renderer4) => {
          Static($$renderer4, spread_props([
            { autoscroll: gradio.shared.autoscroll, i18n: gradio.i18n },
            gradio.shared.loading_status,
            {
              on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
            }
          ]));
          $$renderer4.push(`<!----> `);
          if (gradio.shared.interactive && no_value) {
            $$renderer4.push("<!--[-->");
            $$renderer4.push(`<div${attr_class(clsx(!gradio.props.value || active_source && active_source.includes("webcam") ? "hidden-upload-input" : "upload-wrapper"), "svelte-vrqwbn")}>`);
            FileUpload($$renderer4, {
              value: null,
              root: gradio.shared.root,
              label: gradio.shared.label,
              max_file_size: gradio.shared.max_file_size,
              file_count: "multiple",
              file_types: gradio.props.file_types,
              i18n: gradio.i18n,
              upload: (...args) => gradio.shared.client.upload(...args),
              stream_handler: (...args) => gradio.shared.client.stream(...args),
              onupload: async (e) => {
                const files = Array.isArray(e) ? e : [e];
                gradio.props.value = await process_upload_files(files);
                active_source = null;
                gradio.dispatch("upload", gradio.props.value);
                gradio.dispatch("change", gradio.props.value);
              },
              onerror: ({ detail }) => {
                gradio.shared.loading_status = gradio.shared.loading_status || {};
                gradio.shared.loading_status.status = "error";
                gradio.dispatch("error", detail);
              },
              get upload_promise() {
                return upload_promise;
              },
              set upload_promise($$value) {
                upload_promise = $$value;
                $$settled = false;
              },
              children: ($$renderer5) => {
                UploadText($$renderer5, { i18n: gradio.i18n, type: "gallery" });
              },
              $$slots: { default: true }
            });
            $$renderer4.push(`<!----></div> `);
            if (active_source === "webcam") {
              $$renderer4.push("<!--[-->");
              Webcam($$renderer4, {
                root: gradio.shared.root,
                value: null,
                mirror_webcam: true,
                streaming: false,
                mode: "image",
                include_audio: false,
                i18n: gradio.i18n,
                upload: (...args) => gradio.shared.client.upload(...args)
              });
            } else {
              $$renderer4.push("<!--[!-->");
              if (active_source === "webcam-video") {
                $$renderer4.push("<!--[-->");
                Webcam($$renderer4, {
                  root: gradio.shared.root,
                  value: null,
                  mirror_webcam: true,
                  streaming: false,
                  mode: "video",
                  include_audio: false,
                  i18n: gradio.i18n,
                  upload: (...args) => gradio.shared.client.upload(...args)
                });
              } else {
                $$renderer4.push("<!--[!-->");
              }
              $$renderer4.push(`<!--]-->`);
            }
            $$renderer4.push(`<!--]--> `);
            if (sources.length > 1 || sources.includes("clipboard")) {
              $$renderer4.push("<!--[-->");
              SelectSource($$renderer4, {
                sources,
                handle_clear: () => gradio.dispatch("clear"),
                handle_select: handle_select_source,
                get active_source() {
                  return active_source;
                },
                set active_source($$value) {
                  active_source = $$value;
                  $$settled = false;
                }
              });
            } else {
              $$renderer4.push("<!--[!-->");
            }
            $$renderer4.push(`<!--]-->`);
          } else {
            $$renderer4.push("<!--[!-->");
            Gallery($$renderer4, {
              onchange: () => gradio.dispatch("change"),
              onclear: () => gradio.dispatch("change"),
              onselect: (e) => gradio.dispatch("select", e),
              onshare: (e) => gradio.dispatch("share", e.detail),
              onerror: (e) => gradio.dispatch("error", e.detail),
              onpreview_open: () => {
                gradio.dispatch("preview_open");
              },
              onpreview_close: () => gradio.dispatch("preview_close"),
              onfullscreen: ({ detail }) => {
                fullscreen = detail;
              },
              ondelete: handle_delete,
              onupload: async (e) => {
                const files = Array.isArray(e) ? e : [e];
                const new_value = await process_upload_files(files);
                gradio.props.value = gradio.props.value ? [...gradio.props.value, ...new_value] : new_value;
                gradio.dispatch("upload", new_value);
                gradio.dispatch("change", gradio.props.value);
              },
              sources,
              onsource_change,
              label: gradio.shared.label,
              show_label: gradio.shared.show_label,
              columns: gradio.props.columns,
              rows: gradio.props.rows,
              height: gradio.props.height,
              preview: gradio.props.preview,
              object_fit: gradio.props.object_fit,
              interactive: gradio.shared.interactive,
              allow_preview: gradio.props.allow_preview,
              show_share_button: gradio.props.buttons.some((btn) => typeof btn === "string" && btn === "share"),
              show_download_button: gradio.props.buttons.some((btn) => typeof btn === "string" && btn === "download"),
              fit_columns: gradio.props.fit_columns,
              i18n: gradio.i18n,
              _fetch: (...args) => gradio.shared.client.fetch(...args),
              show_fullscreen_button: gradio.props.buttons.some((btn) => typeof btn === "string" && btn === "fullscreen"),
              buttons: gradio.props.buttons,
              on_custom_button_click: (id) => {
                gradio.dispatch("custom_button_click", { id });
              },
              fullscreen,
              root: gradio.shared.root,
              file_types: gradio.props.file_types,
              max_file_size: gradio.shared.max_file_size,
              upload: (...args) => gradio.shared.client.upload(...args),
              stream_handler: (...args) => gradio.shared.client.stream(...args),
              get selected_index() {
                return gradio.props.selected_index;
              },
              set selected_index($$value) {
                gradio.props.selected_index = $$value;
                $$settled = false;
              },
              get value() {
                return gradio.props.value;
              },
              set value($$value) {
                gradio.props.value = $$value;
                $$settled = false;
              }
            });
          }
          $$renderer4.push(`<!--]-->`);
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

export { Gallery as BaseGallery, Index as default };
//# sourceMappingURL=Index26-w1rs7k8w.js.map
