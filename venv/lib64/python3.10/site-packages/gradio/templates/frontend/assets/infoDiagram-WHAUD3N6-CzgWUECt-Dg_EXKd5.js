import { _ as __name, l as log, H as selectSvgElement, e as configureSvgSize, I as package_default } from './mermaid.core-vMMZVCDT-CT2eqvo9.js';
import { p as parse } from './treemap-KMMF4GRG-BLLAjc28-C9ZroOed.js';
import './index-CDZuCcOm.js';
import './Index-eKU_FSUW.js';
import './i18n-dpAHICcw.js';
import './utils.svelte-CyWLYi-B.js';
import './clone-dZfS06Ds.js';
import './index-DnoGeqVF.js';
import './dsv-BhAd467f.js';
import './props-BwDqDG8n.js';
import './misc-C2MjMwBX.js';
import './index-By61_kAe.js';
import './Upload-BlCAnBIo.js';
import './snippet-DVkMfmSq.js';
import './actions-BTh6ZJJ8.js';
import './ScrollFade.svelte_svelte_type_style_lang-DUvCSfRk.js';
import './MarkdownCode.svelte_svelte_type_style_lang-GeNUGzep.js';
import './prism-python-C_fanlsZ.js';
import './html-h_YSgefI.js';
import './input-UUW65DyE.js';
import './event-modifiers-DanhKw3_.js';
import './MarkdownCode-Q694H4-C.js';
import './StreamingBar.svelte_svelte_type_style_lang-BxBb9ZZb.js';
import './Checkbox-DC-W5bhf.js';
import './size-CuuZBRle.js';
import './Check-4kogBHUX.js';
import './DropdownArrow-BRSpwupS.js';
import './Copy-C8W4pNlO.js';
import './FullscreenButton-wNz2x9hr.js';
import './Maximize-CNFXHhlb.js';
import './Example-CrpSmVPg.js';
import './min-BP3TZd4l-DOaP9HiD.js';
import './_baseUniq-CrYdfo_J-C1cWc0s3.js';

var parser = {
  parse: /* @__PURE__ */ __name(async (input) => {
    const ast = await parse("info", input);
    log.debug(ast);
  }, "parse")
};
var DEFAULT_INFO_DB = {
  version: package_default.version + ""
};
var getVersion = /* @__PURE__ */ __name(() => DEFAULT_INFO_DB.version, "getVersion");
var db = {
  getVersion
};
var draw = /* @__PURE__ */ __name((text, id, version) => {
  log.debug("rendering info diagram\n" + text);
  const svg = selectSvgElement(id);
  configureSvgSize(svg, 100, 400, true);
  const group = svg.append("g");
  group.append("text").attr("x", 100).attr("y", 40).attr("class", "version").attr("font-size", 32).style("text-anchor", "middle").text(`v${version}`);
}, "draw");
var renderer = { draw };
var diagram = {
  parser,
  db,
  renderer
};

export { diagram };
//# sourceMappingURL=infoDiagram-WHAUD3N6-CzgWUECt-Dg_EXKd5.js.map
