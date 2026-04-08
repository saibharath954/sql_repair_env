import { s as styles_default, c as classRenderer_v3_unified_default, a as classDiagram_default, C as ClassDB } from './chunk-B4BG7PRW-DRga82o6.js';
import { _ as __name } from './mermaid.core-J5-DjKr9.js';
import './chunk-FMBD7UC4-Bw2Dv4HH.js';
import './chunk-55IACEB6-CaCx72i-.js';
import './select-k8gDf_61.js';
import './chunk-QN33PNHL-Bg2EmnyT.js';
import './index-CDZuCcOm.js';
import './i18n-dpAHICcw.js';
import './step-TZOpqHBK.js';
import './dispatch-tQmgj1It.js';

// src/diagrams/class/classDiagram-v2.ts
var diagram = {
  parser: classDiagram_default,
  get db() {
    return new ClassDB();
  },
  renderer: classRenderer_v3_unified_default,
  styles: styles_default,
  init: /* @__PURE__ */ __name((cnf) => {
    if (!cnf.class) {
      cnf.class = {};
    }
    cnf.class.arrowMarkerAbsolute = cnf.arrowMarkerAbsolute;
  }, "init")
};

export { diagram };
//# sourceMappingURL=classDiagram-v2-WZHVMYZB-_XA78A9a.js.map
