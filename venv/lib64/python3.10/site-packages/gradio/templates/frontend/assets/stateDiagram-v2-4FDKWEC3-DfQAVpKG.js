import { s as styles_default, b as stateRenderer_v3_unified_default, a as stateDiagram_default, S as StateDB } from './chunk-DI55MBZ5-CXSYKZ-n.js';
import { _ as __name } from './mermaid.core-J5-DjKr9.js';
import './chunk-55IACEB6-CaCx72i-.js';
import './select-k8gDf_61.js';
import './chunk-QN33PNHL-Bg2EmnyT.js';
import './index-CDZuCcOm.js';
import './i18n-dpAHICcw.js';
import './step-TZOpqHBK.js';
import './dispatch-tQmgj1It.js';

// src/diagrams/state/stateDiagram-v2.ts
var diagram = {
  parser: stateDiagram_default,
  get db() {
    return new StateDB(2);
  },
  renderer: stateRenderer_v3_unified_default,
  styles: styles_default,
  init: /* @__PURE__ */ __name((cnf) => {
    if (!cnf.state) {
      cnf.state = {};
    }
    cnf.state.arrowMarkerAbsolute = cnf.arrowMarkerAbsolute;
  }, "init")
};

export { diagram };
//# sourceMappingURL=stateDiagram-v2-4FDKWEC3-DfQAVpKG.js.map
