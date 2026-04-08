import './async-D55cHugf.js';
import './MarkdownCode.svelte_svelte_type_style_lang-B2xYMNIW.js';
import { D as DownloadLink } from './DownloadLink-CR_zSSrd.js';
import { I as IconButton } from './IconButton-BOK4HpdV.js';
import { C as Clear } from './Clear-DH-TDCgr.js';
import { D as Download } from './Download-ByiErn53.js';
import { E as Edit } from './Edit-W_0aHh0i.js';
import { U as Undo } from './Undo-Col2BcUY.js';
import './2-DKaY_6dX.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-BSVqsNEI.js';

function ModifyUpload($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      editable = false,
      undoable = false,
      download = null,
      i18n,
      onedit,
      onclear,
      onundo,
      children
    } = $$props;
    IconButtonWrapper($$renderer2, {
      children: ($$renderer3) => {
        if (editable) {
          $$renderer3.push("<!--[-->");
          IconButton($$renderer3, {
            Icon: Edit,
            label: i18n("common.edit"),
            onclick: () => onedit?.()
          });
        } else {
          $$renderer3.push("<!--[!-->");
        }
        $$renderer3.push(`<!--]--> `);
        if (undoable) {
          $$renderer3.push("<!--[-->");
          IconButton($$renderer3, {
            Icon: Undo,
            label: i18n("common.undo"),
            onclick: () => onundo?.()
          });
        } else {
          $$renderer3.push("<!--[!-->");
        }
        $$renderer3.push(`<!--]--> `);
        if (download) {
          $$renderer3.push("<!--[-->");
          DownloadLink($$renderer3, {
            href: download,
            download: true,
            children: ($$renderer4) => {
              IconButton($$renderer4, { Icon: Download, label: i18n("common.download") });
            },
            $$slots: { default: true }
          });
        } else {
          $$renderer3.push("<!--[!-->");
        }
        $$renderer3.push(`<!--]--> `);
        if (children) {
          $$renderer3.push("<!--[-->");
          children($$renderer3);
          $$renderer3.push(`<!---->`);
        } else {
          $$renderer3.push("<!--[!-->");
        }
        $$renderer3.push(`<!--]--> `);
        IconButton($$renderer3, {
          Icon: Clear,
          label: i18n("common.clear"),
          onclick: (event) => {
            onclear?.();
            event.stopPropagation();
          }
        });
        $$renderer3.push(`<!---->`);
      }
    });
  });
}

export { ModifyUpload as M };
//# sourceMappingURL=ModifyUpload-DbaqJZ53.js.map
