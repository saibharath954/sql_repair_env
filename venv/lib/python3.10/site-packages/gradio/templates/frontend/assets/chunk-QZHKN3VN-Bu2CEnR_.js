import { _ as __name } from './mermaid.core-J5-DjKr9.js';

// src/utils/imperativeState.ts
var ImperativeState = class {
  /**
   * @param init - Function that creates the default state.
   */
  constructor(init) {
    this.init = init;
    this.records = this.init();
  }
  static {
    __name(this, "ImperativeState");
  }
  reset() {
    this.records = this.init();
  }
};

export { ImperativeState as I };
//# sourceMappingURL=chunk-QZHKN3VN-Bu2CEnR_.js.map
