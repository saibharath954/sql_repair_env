import { _ as __name, D as cleanAndMerge, H as selectSvgElement, e as configureSvgSize, l as log, b as setAccTitle, a as getAccTitle, p as setDiagramTitle, q as getDiagramTitle, g as getAccDescription, s as setAccDescription, F as defaultConfig_default, E as getConfig, y as clear$2 } from './mermaid.core-vMMZVCDT-CT2eqvo9.js';
import { p as populateCommonDb } from './chunk-4BX2VUAB-DiOwmOep-BPS7paKb.js';
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

var _a;
var DEFAULT_PACKET_CONFIG = defaultConfig_default.packet;
var PacketDB = (_a = class {
  constructor() {
    this.packet = [];
    this.setAccTitle = setAccTitle;
    this.getAccTitle = getAccTitle;
    this.setDiagramTitle = setDiagramTitle;
    this.getDiagramTitle = getDiagramTitle;
    this.getAccDescription = getAccDescription;
    this.setAccDescription = setAccDescription;
  }
  getConfig() {
    const config = cleanAndMerge({
      ...DEFAULT_PACKET_CONFIG,
      ...getConfig().packet
    });
    if (config.showBits) {
      config.paddingY += 10;
    }
    return config;
  }
  getPacket() {
    return this.packet;
  }
  pushWord(word) {
    if (word.length > 0) {
      this.packet.push(word);
    }
  }
  clear() {
    clear$2();
    this.packet = [];
  }
}, __name(_a, "PacketDB"), _a);
var maxPacketSize = 1e4;
var populate = /* @__PURE__ */ __name((ast, db) => {
  populateCommonDb(ast, db);
  let lastBit = -1;
  let word = [];
  let row = 1;
  const { bitsPerRow } = db.getConfig();
  for (let { start, end, bits, label } of ast.blocks) {
    if (start !== void 0 && end !== void 0 && end < start) {
      throw new Error(`Packet block ${start} - ${end} is invalid. End must be greater than start.`);
    }
    start ?? (start = lastBit + 1);
    if (start !== lastBit + 1) {
      throw new Error(
        `Packet block ${start} - ${end ?? start} is not contiguous. It should start from ${lastBit + 1}.`
      );
    }
    if (bits === 0) {
      throw new Error(`Packet block ${start} is invalid. Cannot have a zero bit field.`);
    }
    end ?? (end = start + (bits ?? 1) - 1);
    bits ?? (bits = end - start + 1);
    lastBit = end;
    log.debug(`Packet block ${start} - ${lastBit} with label ${label}`);
    while (word.length <= bitsPerRow + 1 && db.getPacket().length < maxPacketSize) {
      const [block, nextBlock] = getNextFittingBlock({ start, end, bits, label }, row, bitsPerRow);
      word.push(block);
      if (block.end + 1 === row * bitsPerRow) {
        db.pushWord(word);
        word = [];
        row++;
      }
      if (!nextBlock) {
        break;
      }
      ({ start, end, bits, label } = nextBlock);
    }
  }
  db.pushWord(word);
}, "populate");
var getNextFittingBlock = /* @__PURE__ */ __name((block, row, bitsPerRow) => {
  if (block.start === void 0) {
    throw new Error("start should have been set during first phase");
  }
  if (block.end === void 0) {
    throw new Error("end should have been set during first phase");
  }
  if (block.start > block.end) {
    throw new Error(`Block start ${block.start} is greater than block end ${block.end}.`);
  }
  if (block.end + 1 <= row * bitsPerRow) {
    return [block, void 0];
  }
  const rowEnd = row * bitsPerRow - 1;
  const rowStart = row * bitsPerRow;
  return [
    {
      start: block.start,
      end: rowEnd,
      label: block.label,
      bits: rowEnd - block.start
    },
    {
      start: rowStart,
      end: block.end,
      label: block.label,
      bits: block.end - rowStart
    }
  ];
}, "getNextFittingBlock");
var parser = {
  // @ts-expect-error - PacketDB is not assignable to DiagramDB
  parser: { yy: void 0 },
  parse: /* @__PURE__ */ __name(async (input) => {
    var _a2;
    const ast = await parse("packet", input);
    const db = (_a2 = parser.parser) == null ? void 0 : _a2.yy;
    if (!(db instanceof PacketDB)) {
      throw new Error(
        "parser.parser?.yy was not a PacketDB. This is due to a bug within Mermaid, please report this issue at https://github.com/mermaid-js/mermaid/issues."
      );
    }
    log.debug(ast);
    populate(ast, db);
  }, "parse")
};
var draw = /* @__PURE__ */ __name((_text, id, _version, diagram2) => {
  const db = diagram2.db;
  const config = db.getConfig();
  const { rowHeight, paddingY, bitWidth, bitsPerRow } = config;
  const words = db.getPacket();
  const title = db.getDiagramTitle();
  const totalRowHeight = rowHeight + paddingY;
  const svgHeight = totalRowHeight * (words.length + 1) - (title ? 0 : rowHeight);
  const svgWidth = bitWidth * bitsPerRow + 2;
  const svg = selectSvgElement(id);
  svg.attr("viewbox", `0 0 ${svgWidth} ${svgHeight}`);
  configureSvgSize(svg, svgHeight, svgWidth, config.useMaxWidth);
  for (const [word, packet] of words.entries()) {
    drawWord(svg, packet, word, config);
  }
  svg.append("text").text(title).attr("x", svgWidth / 2).attr("y", svgHeight - totalRowHeight / 2).attr("dominant-baseline", "middle").attr("text-anchor", "middle").attr("class", "packetTitle");
}, "draw");
var drawWord = /* @__PURE__ */ __name((svg, word, rowNumber, { rowHeight, paddingX, paddingY, bitWidth, bitsPerRow, showBits }) => {
  const group = svg.append("g");
  const wordY = rowNumber * (rowHeight + paddingY) + paddingY;
  for (const block of word) {
    const blockX = block.start % bitsPerRow * bitWidth + 1;
    const width = (block.end - block.start + 1) * bitWidth - paddingX;
    group.append("rect").attr("x", blockX).attr("y", wordY).attr("width", width).attr("height", rowHeight).attr("class", "packetBlock");
    group.append("text").attr("x", blockX + width / 2).attr("y", wordY + rowHeight / 2).attr("class", "packetLabel").attr("dominant-baseline", "middle").attr("text-anchor", "middle").text(block.label);
    if (!showBits) {
      continue;
    }
    const isSingleBlock = block.end === block.start;
    const bitNumberY = wordY - 2;
    group.append("text").attr("x", blockX + (isSingleBlock ? width / 2 : 0)).attr("y", bitNumberY).attr("class", "packetByte start").attr("dominant-baseline", "auto").attr("text-anchor", isSingleBlock ? "middle" : "start").text(block.start);
    if (!isSingleBlock) {
      group.append("text").attr("x", blockX + width).attr("y", bitNumberY).attr("class", "packetByte end").attr("dominant-baseline", "auto").attr("text-anchor", "end").text(block.end);
    }
  }
}, "drawWord");
var renderer = { draw };
var defaultPacketStyleOptions = {
  byteFontSize: "10px",
  startByteColor: "black",
  endByteColor: "black",
  labelColor: "black",
  labelFontSize: "12px",
  titleColor: "black",
  titleFontSize: "14px",
  blockStrokeColor: "black",
  blockStrokeWidth: "1",
  blockFillColor: "#efefef"
};
var styles = /* @__PURE__ */ __name(({ packet } = {}) => {
  const options = cleanAndMerge(defaultPacketStyleOptions, packet);
  return `
	.packetByte {
		font-size: ${options.byteFontSize};
	}
	.packetByte.start {
		fill: ${options.startByteColor};
	}
	.packetByte.end {
		fill: ${options.endByteColor};
	}
	.packetLabel {
		fill: ${options.labelColor};
		font-size: ${options.labelFontSize};
	}
	.packetTitle {
		fill: ${options.titleColor};
		font-size: ${options.titleFontSize};
	}
	.packetBlock {
		stroke: ${options.blockStrokeColor};
		stroke-width: ${options.blockStrokeWidth};
		fill: ${options.blockFillColor};
	}
	`;
}, "styles");
var diagram = {
  parser,
  get db() {
    return new PacketDB();
  },
  renderer,
  styles
};

export { diagram };
//# sourceMappingURL=diagram-S2PKOQOG-D3Ak1y1h-DY7lUTrx.js.map
