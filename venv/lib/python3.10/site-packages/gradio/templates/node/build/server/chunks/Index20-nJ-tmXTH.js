import { f as fallback } from './async-D55cHugf.js';
import { b as store_get, u as unsubscribe_stores, d as bind_props, f as attr_class, e as ensure_array_like, a as attr, g as attr_style, i as stringify, s as slot, j as clsx } from './index-K3l_dLem.js';
import './2-DKaY_6dX.js';
import { G as Gradio } from './utils.svelte-D1m0ck_w.js';
import { d as dequal$1 } from './index6-OFtK7MnQ.js';
import { c as createEventDispatcher, o as onDestroy, t as tick } from './index-server-BzRj6e_1.js';
import { d as dequal } from './index9-CM0pa6Ue.js';
import { U as Upload } from './Upload2-COmifmPq.js';
import './MarkdownCode.svelte_svelte_type_style_lang-B2xYMNIW.js';
import { M as MarkdownCode } from './MarkdownCode-ucE6Lq0M.js';
import { w as writable, g as get } from './index-Cg-Pg6j3.js';
import { C as Checkbox } from './Checkbox-gakH17MX.js';
import { C as Check } from './Check-B-uwlXei.js';
import { C as Copy } from './Copy-lixG99xU.js';
import { I as IconButton } from './IconButton-BOK4HpdV.js';
import { F as FullscreenButton } from './FullscreenButton-D3sdKON5.js';
import Example from './Example14-G-qLdLjt.js';
import { s as setContext } from './context-DF4-UEpk.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import { h as html$1 } from './html-CfyvkLET.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './clone-Yk88IHKV.js';
import './prism-python-CNqfI2Ql.js';
import './index35-BGR9YwH8.js';
import 'path';
import 'url';
import 'fs';
import './Maximize-B77VDSzq.js';

var __defProp = Object.defineProperty;
var __typeError = (msg) => {
  throw TypeError(msg);
};
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
var __accessCheck = (obj, member, msg) => member.has(obj) || __typeError("Cannot " + msg);
var __privateGet = (obj, member, getter) => (__accessCheck(obj, member, "read from private field"), member.get(obj));
var __privateAdd = (obj, member, value) => member.has(obj) ? __typeError("Cannot add the same private member more than once") : member instanceof WeakSet ? member.add(obj) : member.set(obj, value);
var __privateSet = (obj, member, value, setter) => (__accessCheck(obj, member, "write to private field"), member.set(obj, value), value);
var __privateMethod = (obj, member, method) => (__accessCheck(obj, member, "access private method"), method);
var _Marked_instances, parseMarkdown_fn, onError_fn, _currentLine;
const PUBLIC_VERSION = "4";
if (typeof window !== "undefined")
  (window.__svelte || (window.__svelte = { v: /* @__PURE__ */ new Set() })).v.add(PUBLIC_VERSION);
function _getDefaults() {
  return {
    async: false,
    breaks: false,
    extensions: null,
    gfm: true,
    hooks: null,
    pedantic: false,
    renderer: null,
    silent: false,
    tokenizer: null,
    walkTokens: null
  };
}
let _defaults = _getDefaults();
function changeDefaults(newDefaults) {
  _defaults = newDefaults;
}
const escapeTest$1 = /[&<>"']/;
const escapeReplace$1 = new RegExp(escapeTest$1.source, "g");
const escapeTestNoEncode$1 = /[<>"']|&(?!(#\d{1,7}|#[Xx][a-fA-F0-9]{1,6}|\w+);)/;
const escapeReplaceNoEncode$1 = new RegExp(escapeTestNoEncode$1.source, "g");
const escapeReplacements$1 = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;"
};
const getEscapeReplacement$1 = (ch) => escapeReplacements$1[ch];
function escape$1$1(html2, encode) {
  if (encode) {
    if (escapeTest$1.test(html2)) {
      return html2.replace(escapeReplace$1, getEscapeReplacement$1);
    }
  } else {
    if (escapeTestNoEncode$1.test(html2)) {
      return html2.replace(escapeReplaceNoEncode$1, getEscapeReplacement$1);
    }
  }
  return html2;
}
const unescapeTest = /&(#(?:\d+)|(?:#x[0-9A-Fa-f]+)|(?:\w+));?/ig;
function unescape(html2) {
  return html2.replace(unescapeTest, (_, n) => {
    n = n.toLowerCase();
    if (n === "colon")
      return ":";
    if (n.charAt(0) === "#") {
      return n.charAt(1) === "x" ? String.fromCharCode(parseInt(n.substring(2), 16)) : String.fromCharCode(+n.substring(1));
    }
    return "";
  });
}
const caret = /(^|[^\[])\^/g;
function edit(regex2, opt) {
  let source = typeof regex2 === "string" ? regex2 : regex2.source;
  opt = opt || "";
  const obj = {
    replace: (name, val) => {
      let valSource = typeof val === "string" ? val : val.source;
      valSource = valSource.replace(caret, "$1");
      source = source.replace(name, valSource);
      return obj;
    },
    getRegex: () => {
      return new RegExp(source, opt);
    }
  };
  return obj;
}
function cleanUrl(href) {
  try {
    href = encodeURI(href).replace(/%25/g, "%");
  } catch (e) {
    return null;
  }
  return href;
}
const noopTest = { exec: () => null };
function splitCells(tableRow, count) {
  const row = tableRow.replace(/\|/g, (match, offset, str) => {
    let escaped = false;
    let curr = offset;
    while (--curr >= 0 && str[curr] === "\\")
      escaped = !escaped;
    if (escaped) {
      return "|";
    } else {
      return " |";
    }
  }), cells = row.split(/ \|/);
  let i = 0;
  if (!cells[0].trim()) {
    cells.shift();
  }
  if (cells.length > 0 && !cells[cells.length - 1].trim()) {
    cells.pop();
  }
  if (count) {
    if (cells.length > count) {
      cells.splice(count);
    } else {
      while (cells.length < count)
        cells.push("");
    }
  }
  for (; i < cells.length; i++) {
    cells[i] = cells[i].trim().replace(/\\\|/g, "|");
  }
  return cells;
}
function rtrim(str, c, invert) {
  const l = str.length;
  if (l === 0) {
    return "";
  }
  let suffLen = 0;
  while (suffLen < l) {
    const currChar = str.charAt(l - suffLen - 1);
    if (currChar === c && true) {
      suffLen++;
    } else {
      break;
    }
  }
  return str.slice(0, l - suffLen);
}
function findClosingBracket(str, b) {
  if (str.indexOf(b[1]) === -1) {
    return -1;
  }
  let level = 0;
  for (let i = 0; i < str.length; i++) {
    if (str[i] === "\\") {
      i++;
    } else if (str[i] === b[0]) {
      level++;
    } else if (str[i] === b[1]) {
      level--;
      if (level < 0) {
        return i;
      }
    }
  }
  return -1;
}
function outputLink(cap, link2, raw, lexer) {
  const href = link2.href;
  const title = link2.title ? escape$1$1(link2.title) : null;
  const text2 = cap[1].replace(/\\([\[\]])/g, "$1");
  if (cap[0].charAt(0) !== "!") {
    lexer.state.inLink = true;
    const token = {
      type: "link",
      raw,
      href,
      title,
      text: text2,
      tokens: lexer.inlineTokens(text2)
    };
    lexer.state.inLink = false;
    return token;
  }
  return {
    type: "image",
    raw,
    href,
    title,
    text: escape$1$1(text2)
  };
}
function indentCodeCompensation(raw, text2) {
  const matchIndentToCode = raw.match(/^(\s+)(?:```)/);
  if (matchIndentToCode === null) {
    return text2;
  }
  const indentToCode = matchIndentToCode[1];
  return text2.split("\n").map((node) => {
    const matchIndentInNode = node.match(/^\s+/);
    if (matchIndentInNode === null) {
      return node;
    }
    const [indentInNode] = matchIndentInNode;
    if (indentInNode.length >= indentToCode.length) {
      return node.slice(indentToCode.length);
    }
    return node;
  }).join("\n");
}
class _Tokenizer {
  // set by the lexer
  constructor(options) {
    __publicField(this, "options");
    __publicField(this, "rules");
    __publicField(this, "lexer");
    this.options = options || _defaults;
  }
  space(src) {
    const cap = this.rules.block.newline.exec(src);
    if (cap && cap[0].length > 0) {
      return {
        type: "space",
        raw: cap[0]
      };
    }
  }
  code(src) {
    const cap = this.rules.block.code.exec(src);
    if (cap) {
      const text2 = cap[0].replace(/^ {1,4}/gm, "");
      return {
        type: "code",
        raw: cap[0],
        codeBlockStyle: "indented",
        text: !this.options.pedantic ? rtrim(text2, "\n") : text2
      };
    }
  }
  fences(src) {
    const cap = this.rules.block.fences.exec(src);
    if (cap) {
      const raw = cap[0];
      const text2 = indentCodeCompensation(raw, cap[3] || "");
      return {
        type: "code",
        raw,
        lang: cap[2] ? cap[2].trim().replace(this.rules.inline.anyPunctuation, "$1") : cap[2],
        text: text2
      };
    }
  }
  heading(src) {
    const cap = this.rules.block.heading.exec(src);
    if (cap) {
      let text2 = cap[2].trim();
      if (/#$/.test(text2)) {
        const trimmed = rtrim(text2, "#");
        if (this.options.pedantic) {
          text2 = trimmed.trim();
        } else if (!trimmed || / $/.test(trimmed)) {
          text2 = trimmed.trim();
        }
      }
      return {
        type: "heading",
        raw: cap[0],
        depth: cap[1].length,
        text: text2,
        tokens: this.lexer.inline(text2)
      };
    }
  }
  hr(src) {
    const cap = this.rules.block.hr.exec(src);
    if (cap) {
      return {
        type: "hr",
        raw: cap[0]
      };
    }
  }
  blockquote(src) {
    const cap = this.rules.block.blockquote.exec(src);
    if (cap) {
      let text2 = cap[0].replace(/\n {0,3}((?:=+|-+) *)(?=\n|$)/g, "\n    $1");
      text2 = rtrim(text2.replace(/^ *>[ \t]?/gm, ""), "\n");
      const top = this.lexer.state.top;
      this.lexer.state.top = true;
      const tokens = this.lexer.blockTokens(text2);
      this.lexer.state.top = top;
      return {
        type: "blockquote",
        raw: cap[0],
        tokens,
        text: text2
      };
    }
  }
  list(src) {
    let cap = this.rules.block.list.exec(src);
    if (cap) {
      let bull = cap[1].trim();
      const isordered = bull.length > 1;
      const list2 = {
        type: "list",
        raw: "",
        ordered: isordered,
        start: isordered ? +bull.slice(0, -1) : "",
        loose: false,
        items: []
      };
      bull = isordered ? `\\d{1,9}\\${bull.slice(-1)}` : `\\${bull}`;
      if (this.options.pedantic) {
        bull = isordered ? bull : "[*+-]";
      }
      const itemRegex = new RegExp(`^( {0,3}${bull})((?:[	 ][^\\n]*)?(?:\\n|$))`);
      let raw = "";
      let itemContents = "";
      let endsWithBlankLine = false;
      while (src) {
        let endEarly = false;
        if (!(cap = itemRegex.exec(src))) {
          break;
        }
        if (this.rules.block.hr.test(src)) {
          break;
        }
        raw = cap[0];
        src = src.substring(raw.length);
        let line = cap[2].split("\n", 1)[0].replace(/^\t+/, (t) => " ".repeat(3 * t.length));
        let nextLine = src.split("\n", 1)[0];
        let indent = 0;
        if (this.options.pedantic) {
          indent = 2;
          itemContents = line.trimStart();
        } else {
          indent = cap[2].search(/[^ ]/);
          indent = indent > 4 ? 1 : indent;
          itemContents = line.slice(indent);
          indent += cap[1].length;
        }
        let blankLine = false;
        if (!line && /^ *$/.test(nextLine)) {
          raw += nextLine + "\n";
          src = src.substring(nextLine.length + 1);
          endEarly = true;
        }
        if (!endEarly) {
          const nextBulletRegex = new RegExp(`^ {0,${Math.min(3, indent - 1)}}(?:[*+-]|\\d{1,9}[.)])((?:[ 	][^\\n]*)?(?:\\n|$))`);
          const hrRegex = new RegExp(`^ {0,${Math.min(3, indent - 1)}}((?:- *){3,}|(?:_ *){3,}|(?:\\* *){3,})(?:\\n+|$)`);
          const fencesBeginRegex = new RegExp(`^ {0,${Math.min(3, indent - 1)}}(?:\`\`\`|~~~)`);
          const headingBeginRegex = new RegExp(`^ {0,${Math.min(3, indent - 1)}}#`);
          while (src) {
            const rawLine = src.split("\n", 1)[0];
            nextLine = rawLine;
            if (this.options.pedantic) {
              nextLine = nextLine.replace(/^ {1,4}(?=( {4})*[^ ])/g, "  ");
            }
            if (fencesBeginRegex.test(nextLine)) {
              break;
            }
            if (headingBeginRegex.test(nextLine)) {
              break;
            }
            if (nextBulletRegex.test(nextLine)) {
              break;
            }
            if (hrRegex.test(src)) {
              break;
            }
            if (nextLine.search(/[^ ]/) >= indent || !nextLine.trim()) {
              itemContents += "\n" + nextLine.slice(indent);
            } else {
              if (blankLine) {
                break;
              }
              if (line.search(/[^ ]/) >= 4) {
                break;
              }
              if (fencesBeginRegex.test(line)) {
                break;
              }
              if (headingBeginRegex.test(line)) {
                break;
              }
              if (hrRegex.test(line)) {
                break;
              }
              itemContents += "\n" + nextLine;
            }
            if (!blankLine && !nextLine.trim()) {
              blankLine = true;
            }
            raw += rawLine + "\n";
            src = src.substring(rawLine.length + 1);
            line = nextLine.slice(indent);
          }
        }
        if (!list2.loose) {
          if (endsWithBlankLine) {
            list2.loose = true;
          } else if (/\n *\n *$/.test(raw)) {
            endsWithBlankLine = true;
          }
        }
        let istask = null;
        let ischecked;
        if (this.options.gfm) {
          istask = /^\[[ xX]\] /.exec(itemContents);
          if (istask) {
            ischecked = istask[0] !== "[ ] ";
            itemContents = itemContents.replace(/^\[[ xX]\] +/, "");
          }
        }
        list2.items.push({
          type: "list_item",
          raw,
          task: !!istask,
          checked: ischecked,
          loose: false,
          text: itemContents,
          tokens: []
        });
        list2.raw += raw;
      }
      list2.items[list2.items.length - 1].raw = raw.trimEnd();
      list2.items[list2.items.length - 1].text = itemContents.trimEnd();
      list2.raw = list2.raw.trimEnd();
      for (let i = 0; i < list2.items.length; i++) {
        this.lexer.state.top = false;
        list2.items[i].tokens = this.lexer.blockTokens(list2.items[i].text, []);
        if (!list2.loose) {
          const spacers = list2.items[i].tokens.filter((t) => t.type === "space");
          const hasMultipleLineBreaks = spacers.length > 0 && spacers.some((t) => /\n.*\n/.test(t.raw));
          list2.loose = hasMultipleLineBreaks;
        }
      }
      if (list2.loose) {
        for (let i = 0; i < list2.items.length; i++) {
          list2.items[i].loose = true;
        }
      }
      return list2;
    }
  }
  html(src) {
    const cap = this.rules.block.html.exec(src);
    if (cap) {
      const token = {
        type: "html",
        block: true,
        raw: cap[0],
        pre: cap[1] === "pre" || cap[1] === "script" || cap[1] === "style",
        text: cap[0]
      };
      return token;
    }
  }
  def(src) {
    const cap = this.rules.block.def.exec(src);
    if (cap) {
      const tag2 = cap[1].toLowerCase().replace(/\s+/g, " ");
      const href = cap[2] ? cap[2].replace(/^<(.*)>$/, "$1").replace(this.rules.inline.anyPunctuation, "$1") : "";
      const title = cap[3] ? cap[3].substring(1, cap[3].length - 1).replace(this.rules.inline.anyPunctuation, "$1") : cap[3];
      return {
        type: "def",
        tag: tag2,
        raw: cap[0],
        href,
        title
      };
    }
  }
  table(src) {
    const cap = this.rules.block.table.exec(src);
    if (!cap) {
      return;
    }
    if (!/[:|]/.test(cap[2])) {
      return;
    }
    const headers = splitCells(cap[1]);
    const aligns = cap[2].replace(/^\||\| *$/g, "").split("|");
    const rows = cap[3] && cap[3].trim() ? cap[3].replace(/\n[ \t]*$/, "").split("\n") : [];
    const item = {
      type: "table",
      raw: cap[0],
      header: [],
      align: [],
      rows: []
    };
    if (headers.length !== aligns.length) {
      return;
    }
    for (const align of aligns) {
      if (/^ *-+: *$/.test(align)) {
        item.align.push("right");
      } else if (/^ *:-+: *$/.test(align)) {
        item.align.push("center");
      } else if (/^ *:-+ *$/.test(align)) {
        item.align.push("left");
      } else {
        item.align.push(null);
      }
    }
    for (const header of headers) {
      item.header.push({
        text: header,
        tokens: this.lexer.inline(header)
      });
    }
    for (const row of rows) {
      item.rows.push(splitCells(row, item.header.length).map((cell) => {
        return {
          text: cell,
          tokens: this.lexer.inline(cell)
        };
      }));
    }
    return item;
  }
  lheading(src) {
    const cap = this.rules.block.lheading.exec(src);
    if (cap) {
      return {
        type: "heading",
        raw: cap[0],
        depth: cap[2].charAt(0) === "=" ? 1 : 2,
        text: cap[1],
        tokens: this.lexer.inline(cap[1])
      };
    }
  }
  paragraph(src) {
    const cap = this.rules.block.paragraph.exec(src);
    if (cap) {
      const text2 = cap[1].charAt(cap[1].length - 1) === "\n" ? cap[1].slice(0, -1) : cap[1];
      return {
        type: "paragraph",
        raw: cap[0],
        text: text2,
        tokens: this.lexer.inline(text2)
      };
    }
  }
  text(src) {
    const cap = this.rules.block.text.exec(src);
    if (cap) {
      return {
        type: "text",
        raw: cap[0],
        text: cap[0],
        tokens: this.lexer.inline(cap[0])
      };
    }
  }
  escape(src) {
    const cap = this.rules.inline.escape.exec(src);
    if (cap) {
      return {
        type: "escape",
        raw: cap[0],
        text: escape$1$1(cap[1])
      };
    }
  }
  tag(src) {
    const cap = this.rules.inline.tag.exec(src);
    if (cap) {
      if (!this.lexer.state.inLink && /^<a /i.test(cap[0])) {
        this.lexer.state.inLink = true;
      } else if (this.lexer.state.inLink && /^<\/a>/i.test(cap[0])) {
        this.lexer.state.inLink = false;
      }
      if (!this.lexer.state.inRawBlock && /^<(pre|code|kbd|script)(\s|>)/i.test(cap[0])) {
        this.lexer.state.inRawBlock = true;
      } else if (this.lexer.state.inRawBlock && /^<\/(pre|code|kbd|script)(\s|>)/i.test(cap[0])) {
        this.lexer.state.inRawBlock = false;
      }
      return {
        type: "html",
        raw: cap[0],
        inLink: this.lexer.state.inLink,
        inRawBlock: this.lexer.state.inRawBlock,
        block: false,
        text: cap[0]
      };
    }
  }
  link(src) {
    const cap = this.rules.inline.link.exec(src);
    if (cap) {
      const trimmedUrl = cap[2].trim();
      if (!this.options.pedantic && /^</.test(trimmedUrl)) {
        if (!/>$/.test(trimmedUrl)) {
          return;
        }
        const rtrimSlash = rtrim(trimmedUrl.slice(0, -1), "\\");
        if ((trimmedUrl.length - rtrimSlash.length) % 2 === 0) {
          return;
        }
      } else {
        const lastParenIndex = findClosingBracket(cap[2], "()");
        if (lastParenIndex > -1) {
          const start = cap[0].indexOf("!") === 0 ? 5 : 4;
          const linkLen = start + cap[1].length + lastParenIndex;
          cap[2] = cap[2].substring(0, lastParenIndex);
          cap[0] = cap[0].substring(0, linkLen).trim();
          cap[3] = "";
        }
      }
      let href = cap[2];
      let title = "";
      if (this.options.pedantic) {
        const link2 = /^([^'"]*[^\s])\s+(['"])(.*)\2/.exec(href);
        if (link2) {
          href = link2[1];
          title = link2[3];
        }
      } else {
        title = cap[3] ? cap[3].slice(1, -1) : "";
      }
      href = href.trim();
      if (/^</.test(href)) {
        if (this.options.pedantic && !/>$/.test(trimmedUrl)) {
          href = href.slice(1);
        } else {
          href = href.slice(1, -1);
        }
      }
      return outputLink(cap, {
        href: href ? href.replace(this.rules.inline.anyPunctuation, "$1") : href,
        title: title ? title.replace(this.rules.inline.anyPunctuation, "$1") : title
      }, cap[0], this.lexer);
    }
  }
  reflink(src, links) {
    let cap;
    if ((cap = this.rules.inline.reflink.exec(src)) || (cap = this.rules.inline.nolink.exec(src))) {
      const linkString = (cap[2] || cap[1]).replace(/\s+/g, " ");
      const link2 = links[linkString.toLowerCase()];
      if (!link2) {
        const text2 = cap[0].charAt(0);
        return {
          type: "text",
          raw: text2,
          text: text2
        };
      }
      return outputLink(cap, link2, cap[0], this.lexer);
    }
  }
  emStrong(src, maskedSrc, prevChar = "") {
    let match = this.rules.inline.emStrongLDelim.exec(src);
    if (!match)
      return;
    if (match[3] && prevChar.match(/[\p{L}\p{N}]/u))
      return;
    const nextChar = match[1] || match[2] || "";
    if (!nextChar || !prevChar || this.rules.inline.punctuation.exec(prevChar)) {
      const lLength = [...match[0]].length - 1;
      let rDelim, rLength, delimTotal = lLength, midDelimTotal = 0;
      const endReg = match[0][0] === "*" ? this.rules.inline.emStrongRDelimAst : this.rules.inline.emStrongRDelimUnd;
      endReg.lastIndex = 0;
      maskedSrc = maskedSrc.slice(-1 * src.length + lLength);
      while ((match = endReg.exec(maskedSrc)) != null) {
        rDelim = match[1] || match[2] || match[3] || match[4] || match[5] || match[6];
        if (!rDelim)
          continue;
        rLength = [...rDelim].length;
        if (match[3] || match[4]) {
          delimTotal += rLength;
          continue;
        } else if (match[5] || match[6]) {
          if (lLength % 3 && !((lLength + rLength) % 3)) {
            midDelimTotal += rLength;
            continue;
          }
        }
        delimTotal -= rLength;
        if (delimTotal > 0)
          continue;
        rLength = Math.min(rLength, rLength + delimTotal + midDelimTotal);
        const lastCharLength = [...match[0]][0].length;
        const raw = src.slice(0, lLength + match.index + lastCharLength + rLength);
        if (Math.min(lLength, rLength) % 2) {
          const text3 = raw.slice(1, -1);
          return {
            type: "em",
            raw,
            text: text3,
            tokens: this.lexer.inlineTokens(text3)
          };
        }
        const text2 = raw.slice(2, -2);
        return {
          type: "strong",
          raw,
          text: text2,
          tokens: this.lexer.inlineTokens(text2)
        };
      }
    }
  }
  codespan(src) {
    const cap = this.rules.inline.code.exec(src);
    if (cap) {
      let text2 = cap[2].replace(/\n/g, " ");
      const hasNonSpaceChars = /[^ ]/.test(text2);
      const hasSpaceCharsOnBothEnds = /^ /.test(text2) && / $/.test(text2);
      if (hasNonSpaceChars && hasSpaceCharsOnBothEnds) {
        text2 = text2.substring(1, text2.length - 1);
      }
      text2 = escape$1$1(text2, true);
      return {
        type: "codespan",
        raw: cap[0],
        text: text2
      };
    }
  }
  br(src) {
    const cap = this.rules.inline.br.exec(src);
    if (cap) {
      return {
        type: "br",
        raw: cap[0]
      };
    }
  }
  del(src) {
    const cap = this.rules.inline.del.exec(src);
    if (cap) {
      return {
        type: "del",
        raw: cap[0],
        text: cap[2],
        tokens: this.lexer.inlineTokens(cap[2])
      };
    }
  }
  autolink(src) {
    const cap = this.rules.inline.autolink.exec(src);
    if (cap) {
      let text2, href;
      if (cap[2] === "@") {
        text2 = escape$1$1(cap[1]);
        href = "mailto:" + text2;
      } else {
        text2 = escape$1$1(cap[1]);
        href = text2;
      }
      return {
        type: "link",
        raw: cap[0],
        text: text2,
        href,
        tokens: [
          {
            type: "text",
            raw: text2,
            text: text2
          }
        ]
      };
    }
  }
  url(src) {
    var _a;
    let cap;
    if (cap = this.rules.inline.url.exec(src)) {
      let text2, href;
      if (cap[2] === "@") {
        text2 = escape$1$1(cap[0]);
        href = "mailto:" + text2;
      } else {
        let prevCapZero;
        do {
          prevCapZero = cap[0];
          cap[0] = ((_a = this.rules.inline._backpedal.exec(cap[0])) == null ? void 0 : _a[0]) ?? "";
        } while (prevCapZero !== cap[0]);
        text2 = escape$1$1(cap[0]);
        if (cap[1] === "www.") {
          href = "http://" + cap[0];
        } else {
          href = cap[0];
        }
      }
      return {
        type: "link",
        raw: cap[0],
        text: text2,
        href,
        tokens: [
          {
            type: "text",
            raw: text2,
            text: text2
          }
        ]
      };
    }
  }
  inlineText(src) {
    const cap = this.rules.inline.text.exec(src);
    if (cap) {
      let text2;
      if (this.lexer.state.inRawBlock) {
        text2 = cap[0];
      } else {
        text2 = escape$1$1(cap[0]);
      }
      return {
        type: "text",
        raw: cap[0],
        text: text2
      };
    }
  }
}
const newline = /^(?: *(?:\n|$))+/;
const blockCode = /^( {4}[^\n]+(?:\n(?: *(?:\n|$))*)?)+/;
const fences = /^ {0,3}(`{3,}(?=[^`\n]*(?:\n|$))|~{3,})([^\n]*)(?:\n|$)(?:|([\s\S]*?)(?:\n|$))(?: {0,3}\1[~`]* *(?=\n|$)|$)/;
const hr = /^ {0,3}((?:-[\t ]*){3,}|(?:_[ \t]*){3,}|(?:\*[ \t]*){3,})(?:\n+|$)/;
const heading = /^ {0,3}(#{1,6})(?=\s|$)(.*)(?:\n+|$)/;
const bullet = /(?:[*+-]|\d{1,9}[.)])/;
const lheading = edit(/^(?!bull |blockCode|fences|blockquote|heading|html)((?:.|\n(?!\s*?\n|bull |blockCode|fences|blockquote|heading|html))+?)\n {0,3}(=+|-+) *(?:\n+|$)/).replace(/bull/g, bullet).replace(/blockCode/g, / {4}/).replace(/fences/g, / {0,3}(?:`{3,}|~{3,})/).replace(/blockquote/g, / {0,3}>/).replace(/heading/g, / {0,3}#{1,6}/).replace(/html/g, / {0,3}<[^\n>]+>\n/).getRegex();
const _paragraph = /^([^\n]+(?:\n(?!hr|heading|lheading|blockquote|fences|list|html|table| +\n)[^\n]+)*)/;
const blockText = /^[^\n]+/;
const _blockLabel = /(?!\s*\])(?:\\.|[^\[\]\\])+/;
const def = edit(/^ {0,3}\[(label)\]: *(?:\n *)?([^<\s][^\s]*|<.*?>)(?:(?: +(?:\n *)?| *\n *)(title))? *(?:\n+|$)/).replace("label", _blockLabel).replace("title", /(?:"(?:\\"?|[^"\\])*"|'[^'\n]*(?:\n[^'\n]+)*\n?'|\([^()]*\))/).getRegex();
const list = edit(/^( {0,3}bull)([ \t][^\n]+?)?(?:\n|$)/).replace(/bull/g, bullet).getRegex();
const _tag = "address|article|aside|base|basefont|blockquote|body|caption|center|col|colgroup|dd|details|dialog|dir|div|dl|dt|fieldset|figcaption|figure|footer|form|frame|frameset|h[1-6]|head|header|hr|html|iframe|legend|li|link|main|menu|menuitem|meta|nav|noframes|ol|optgroup|option|p|param|search|section|summary|table|tbody|td|tfoot|th|thead|title|tr|track|ul";
const _comment = /<!--(?:-?>|[\s\S]*?(?:-->|$))/;
const html = edit("^ {0,3}(?:<(script|pre|style|textarea)[\\s>][\\s\\S]*?(?:</\\1>[^\\n]*\\n+|$)|comment[^\\n]*(\\n+|$)|<\\?[\\s\\S]*?(?:\\?>\\n*|$)|<![A-Z][\\s\\S]*?(?:>\\n*|$)|<!\\[CDATA\\[[\\s\\S]*?(?:\\]\\]>\\n*|$)|</?(tag)(?: +|\\n|/?>)[\\s\\S]*?(?:(?:\\n *)+\\n|$)|<(?!script|pre|style|textarea)([a-z][\\w-]*)(?:attribute)*? */?>(?=[ \\t]*(?:\\n|$))[\\s\\S]*?(?:(?:\\n *)+\\n|$)|</(?!script|pre|style|textarea)[a-z][\\w-]*\\s*>(?=[ \\t]*(?:\\n|$))[\\s\\S]*?(?:(?:\\n *)+\\n|$))", "i").replace("comment", _comment).replace("tag", _tag).replace("attribute", / +[a-zA-Z:_][\w.:-]*(?: *= *"[^"\n]*"| *= *'[^'\n]*'| *= *[^\s"'=<>`]+)?/).getRegex();
const paragraph = edit(_paragraph).replace("hr", hr).replace("heading", " {0,3}#{1,6}(?:\\s|$)").replace("|lheading", "").replace("|table", "").replace("blockquote", " {0,3}>").replace("fences", " {0,3}(?:`{3,}(?=[^`\\n]*\\n)|~{3,})[^\\n]*\\n").replace("list", " {0,3}(?:[*+-]|1[.)]) ").replace("html", "</?(?:tag)(?: +|\\n|/?>)|<(?:script|pre|style|textarea|!--)").replace("tag", _tag).getRegex();
const blockquote = edit(/^( {0,3}> ?(paragraph|[^\n]*)(?:\n|$))+/).replace("paragraph", paragraph).getRegex();
const blockNormal = {
  blockquote,
  code: blockCode,
  def,
  fences,
  heading,
  hr,
  html,
  lheading,
  list,
  newline,
  paragraph,
  table: noopTest,
  text: blockText
};
const gfmTable = edit("^ *([^\\n ].*)\\n {0,3}((?:\\| *)?:?-+:? *(?:\\| *:?-+:? *)*(?:\\| *)?)(?:\\n((?:(?! *\\n|hr|heading|blockquote|code|fences|list|html).*(?:\\n|$))*)\\n*|$)").replace("hr", hr).replace("heading", " {0,3}#{1,6}(?:\\s|$)").replace("blockquote", " {0,3}>").replace("code", " {4}[^\\n]").replace("fences", " {0,3}(?:`{3,}(?=[^`\\n]*\\n)|~{3,})[^\\n]*\\n").replace("list", " {0,3}(?:[*+-]|1[.)]) ").replace("html", "</?(?:tag)(?: +|\\n|/?>)|<(?:script|pre|style|textarea|!--)").replace("tag", _tag).getRegex();
const blockGfm = {
  ...blockNormal,
  table: gfmTable,
  paragraph: edit(_paragraph).replace("hr", hr).replace("heading", " {0,3}#{1,6}(?:\\s|$)").replace("|lheading", "").replace("table", gfmTable).replace("blockquote", " {0,3}>").replace("fences", " {0,3}(?:`{3,}(?=[^`\\n]*\\n)|~{3,})[^\\n]*\\n").replace("list", " {0,3}(?:[*+-]|1[.)]) ").replace("html", "</?(?:tag)(?: +|\\n|/?>)|<(?:script|pre|style|textarea|!--)").replace("tag", _tag).getRegex()
};
const blockPedantic = {
  ...blockNormal,
  html: edit(`^ *(?:comment *(?:\\n|\\s*$)|<(tag)[\\s\\S]+?</\\1> *(?:\\n{2,}|\\s*$)|<tag(?:"[^"]*"|'[^']*'|\\s[^'"/>\\s]*)*?/?> *(?:\\n{2,}|\\s*$))`).replace("comment", _comment).replace(/tag/g, "(?!(?:a|em|strong|small|s|cite|q|dfn|abbr|data|time|code|var|samp|kbd|sub|sup|i|b|u|mark|ruby|rt|rp|bdi|bdo|span|br|wbr|ins|del|img)\\b)\\w+(?!:|[^\\w\\s@]*@)\\b").getRegex(),
  def: /^ *\[([^\]]+)\]: *<?([^\s>]+)>?(?: +(["(][^\n]+[")]))? *(?:\n+|$)/,
  heading: /^(#{1,6})(.*)(?:\n+|$)/,
  fences: noopTest,
  // fences not supported
  lheading: /^(.+?)\n {0,3}(=+|-+) *(?:\n+|$)/,
  paragraph: edit(_paragraph).replace("hr", hr).replace("heading", " *#{1,6} *[^\n]").replace("lheading", lheading).replace("|table", "").replace("blockquote", " {0,3}>").replace("|fences", "").replace("|list", "").replace("|html", "").replace("|tag", "").getRegex()
};
const escape$2 = /^\\([!"#$%&'()*+,\-./:;<=>?@\[\]\\^_`{|}~])/;
const inlineCode = /^(`+)([^`]|[^`][\s\S]*?[^`])\1(?!`)/;
const br = /^( {2,}|\\)\n(?!\s*$)/;
const inlineText = /^(`+|[^`])(?:(?= {2,}\n)|[\s\S]*?(?:(?=[\\<!\[`*_]|\b_|$)|[^ ](?= {2,}\n)))/;
const _punctuation = "\\p{P}\\p{S}";
const punctuation = edit(/^((?![*_])[\spunctuation])/, "u").replace(/punctuation/g, _punctuation).getRegex();
const blockSkip = /\[[^[\]]*?\]\([^\(\)]*?\)|`[^`]*?`|<[^<>]*?>/g;
const emStrongLDelim = edit(/^(?:\*+(?:((?!\*)[punct])|[^\s*]))|^_+(?:((?!_)[punct])|([^\s_]))/, "u").replace(/punct/g, _punctuation).getRegex();
const emStrongRDelimAst = edit("^[^_*]*?__[^_*]*?\\*[^_*]*?(?=__)|[^*]+(?=[^*])|(?!\\*)[punct](\\*+)(?=[\\s]|$)|[^punct\\s](\\*+)(?!\\*)(?=[punct\\s]|$)|(?!\\*)[punct\\s](\\*+)(?=[^punct\\s])|[\\s](\\*+)(?!\\*)(?=[punct])|(?!\\*)[punct](\\*+)(?!\\*)(?=[punct])|[^punct\\s](\\*+)(?=[^punct\\s])", "gu").replace(/punct/g, _punctuation).getRegex();
const emStrongRDelimUnd = edit("^[^_*]*?\\*\\*[^_*]*?_[^_*]*?(?=\\*\\*)|[^_]+(?=[^_])|(?!_)[punct](_+)(?=[\\s]|$)|[^punct\\s](_+)(?!_)(?=[punct\\s]|$)|(?!_)[punct\\s](_+)(?=[^punct\\s])|[\\s](_+)(?!_)(?=[punct])|(?!_)[punct](_+)(?!_)(?=[punct])", "gu").replace(/punct/g, _punctuation).getRegex();
const anyPunctuation = edit(/\\([punct])/, "gu").replace(/punct/g, _punctuation).getRegex();
const autolink = edit(/^<(scheme:[^\s\x00-\x1f<>]*|email)>/).replace("scheme", /[a-zA-Z][a-zA-Z0-9+.-]{1,31}/).replace("email", /[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+(@)[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+(?![-_])/).getRegex();
const _inlineComment = edit(_comment).replace("(?:-->|$)", "-->").getRegex();
const tag = edit("^comment|^</[a-zA-Z][\\w:-]*\\s*>|^<[a-zA-Z][\\w-]*(?:attribute)*?\\s*/?>|^<\\?[\\s\\S]*?\\?>|^<![a-zA-Z]+\\s[\\s\\S]*?>|^<!\\[CDATA\\[[\\s\\S]*?\\]\\]>").replace("comment", _inlineComment).replace("attribute", /\s+[a-zA-Z:_][\w.:-]*(?:\s*=\s*"[^"]*"|\s*=\s*'[^']*'|\s*=\s*[^\s"'=<>`]+)?/).getRegex();
const _inlineLabel = /(?:\[(?:\\.|[^\[\]\\])*\]|\\.|`[^`]*`|[^\[\]\\`])*?/;
const link = edit(/^!?\[(label)\]\(\s*(href)(?:\s+(title))?\s*\)/).replace("label", _inlineLabel).replace("href", /<(?:\\.|[^\n<>\\])+>|[^\s\x00-\x1f]*/).replace("title", /"(?:\\"?|[^"\\])*"|'(?:\\'?|[^'\\])*'|\((?:\\\)?|[^)\\])*\)/).getRegex();
const reflink = edit(/^!?\[(label)\]\[(ref)\]/).replace("label", _inlineLabel).replace("ref", _blockLabel).getRegex();
const nolink = edit(/^!?\[(ref)\](?:\[\])?/).replace("ref", _blockLabel).getRegex();
const reflinkSearch = edit("reflink|nolink(?!\\()", "g").replace("reflink", reflink).replace("nolink", nolink).getRegex();
const inlineNormal = {
  _backpedal: noopTest,
  // only used for GFM url
  anyPunctuation,
  autolink,
  blockSkip,
  br,
  code: inlineCode,
  del: noopTest,
  emStrongLDelim,
  emStrongRDelimAst,
  emStrongRDelimUnd,
  escape: escape$2,
  link,
  nolink,
  punctuation,
  reflink,
  reflinkSearch,
  tag,
  text: inlineText,
  url: noopTest
};
const inlinePedantic = {
  ...inlineNormal,
  link: edit(/^!?\[(label)\]\((.*?)\)/).replace("label", _inlineLabel).getRegex(),
  reflink: edit(/^!?\[(label)\]\s*\[([^\]]*)\]/).replace("label", _inlineLabel).getRegex()
};
const inlineGfm = {
  ...inlineNormal,
  escape: edit(escape$2).replace("])", "~|])").getRegex(),
  url: edit(/^((?:ftp|https?):\/\/|www\.)(?:[a-zA-Z0-9\-]+\.?)+[^\s<]*|^email/, "i").replace("email", /[A-Za-z0-9._+-]+(@)[a-zA-Z0-9-_]+(?:\.[a-zA-Z0-9-_]*[a-zA-Z0-9])+(?![-_])/).getRegex(),
  _backpedal: /(?:[^?!.,:;*_'"~()&]+|\([^)]*\)|&(?![a-zA-Z0-9]+;$)|[?!.,:;*_'"~)]+(?!$))+/,
  del: /^(~~?)(?=[^\s~])([\s\S]*?[^\s~])\1(?=[^~]|$)/,
  text: /^([`~]+|[^`~])(?:(?= {2,}\n)|(?=[a-zA-Z0-9.!#$%&'*+\/=?_`{\|}~-]+@)|[\s\S]*?(?:(?=[\\<!\[`*~_]|\b_|https?:\/\/|ftp:\/\/|www\.|$)|[^ ](?= {2,}\n)|[^a-zA-Z0-9.!#$%&'*+\/=?_`{\|}~-](?=[a-zA-Z0-9.!#$%&'*+\/=?_`{\|}~-]+@)))/
};
const inlineBreaks = {
  ...inlineGfm,
  br: edit(br).replace("{2,}", "*").getRegex(),
  text: edit(inlineGfm.text).replace("\\b_", "\\b_| {2,}\\n").replace(/\{2,\}/g, "*").getRegex()
};
const block = {
  normal: blockNormal,
  gfm: blockGfm,
  pedantic: blockPedantic
};
const inline = {
  normal: inlineNormal,
  gfm: inlineGfm,
  breaks: inlineBreaks,
  pedantic: inlinePedantic
};
class _Lexer {
  constructor(options) {
    __publicField(this, "tokens");
    __publicField(this, "options");
    __publicField(this, "state");
    __publicField(this, "tokenizer");
    __publicField(this, "inlineQueue");
    this.tokens = [];
    this.tokens.links = /* @__PURE__ */ Object.create(null);
    this.options = options || _defaults;
    this.options.tokenizer = this.options.tokenizer || new _Tokenizer();
    this.tokenizer = this.options.tokenizer;
    this.tokenizer.options = this.options;
    this.tokenizer.lexer = this;
    this.inlineQueue = [];
    this.state = {
      inLink: false,
      inRawBlock: false,
      top: true
    };
    const rules = {
      block: block.normal,
      inline: inline.normal
    };
    if (this.options.pedantic) {
      rules.block = block.pedantic;
      rules.inline = inline.pedantic;
    } else if (this.options.gfm) {
      rules.block = block.gfm;
      if (this.options.breaks) {
        rules.inline = inline.breaks;
      } else {
        rules.inline = inline.gfm;
      }
    }
    this.tokenizer.rules = rules;
  }
  /**
   * Expose Rules
   */
  static get rules() {
    return {
      block,
      inline
    };
  }
  /**
   * Static Lex Method
   */
  static lex(src, options) {
    const lexer = new _Lexer(options);
    return lexer.lex(src);
  }
  /**
   * Static Lex Inline Method
   */
  static lexInline(src, options) {
    const lexer = new _Lexer(options);
    return lexer.inlineTokens(src);
  }
  /**
   * Preprocessing
   */
  lex(src) {
    src = src.replace(/\r\n|\r/g, "\n");
    this.blockTokens(src, this.tokens);
    for (let i = 0; i < this.inlineQueue.length; i++) {
      const next = this.inlineQueue[i];
      this.inlineTokens(next.src, next.tokens);
    }
    this.inlineQueue = [];
    return this.tokens;
  }
  blockTokens(src, tokens = []) {
    if (this.options.pedantic) {
      src = src.replace(/\t/g, "    ").replace(/^ +$/gm, "");
    } else {
      src = src.replace(/^( *)(\t+)/gm, (_, leading, tabs) => {
        return leading + "    ".repeat(tabs.length);
      });
    }
    let token;
    let lastToken;
    let cutSrc;
    let lastParagraphClipped;
    while (src) {
      if (this.options.extensions && this.options.extensions.block && this.options.extensions.block.some((extTokenizer) => {
        if (token = extTokenizer.call({ lexer: this }, src, tokens)) {
          src = src.substring(token.raw.length);
          tokens.push(token);
          return true;
        }
        return false;
      })) {
        continue;
      }
      if (token = this.tokenizer.space(src)) {
        src = src.substring(token.raw.length);
        if (token.raw.length === 1 && tokens.length > 0) {
          tokens[tokens.length - 1].raw += "\n";
        } else {
          tokens.push(token);
        }
        continue;
      }
      if (token = this.tokenizer.code(src)) {
        src = src.substring(token.raw.length);
        lastToken = tokens[tokens.length - 1];
        if (lastToken && (lastToken.type === "paragraph" || lastToken.type === "text")) {
          lastToken.raw += "\n" + token.raw;
          lastToken.text += "\n" + token.text;
          this.inlineQueue[this.inlineQueue.length - 1].src = lastToken.text;
        } else {
          tokens.push(token);
        }
        continue;
      }
      if (token = this.tokenizer.fences(src)) {
        src = src.substring(token.raw.length);
        tokens.push(token);
        continue;
      }
      if (token = this.tokenizer.heading(src)) {
        src = src.substring(token.raw.length);
        tokens.push(token);
        continue;
      }
      if (token = this.tokenizer.hr(src)) {
        src = src.substring(token.raw.length);
        tokens.push(token);
        continue;
      }
      if (token = this.tokenizer.blockquote(src)) {
        src = src.substring(token.raw.length);
        tokens.push(token);
        continue;
      }
      if (token = this.tokenizer.list(src)) {
        src = src.substring(token.raw.length);
        tokens.push(token);
        continue;
      }
      if (token = this.tokenizer.html(src)) {
        src = src.substring(token.raw.length);
        tokens.push(token);
        continue;
      }
      if (token = this.tokenizer.def(src)) {
        src = src.substring(token.raw.length);
        lastToken = tokens[tokens.length - 1];
        if (lastToken && (lastToken.type === "paragraph" || lastToken.type === "text")) {
          lastToken.raw += "\n" + token.raw;
          lastToken.text += "\n" + token.raw;
          this.inlineQueue[this.inlineQueue.length - 1].src = lastToken.text;
        } else if (!this.tokens.links[token.tag]) {
          this.tokens.links[token.tag] = {
            href: token.href,
            title: token.title
          };
        }
        continue;
      }
      if (token = this.tokenizer.table(src)) {
        src = src.substring(token.raw.length);
        tokens.push(token);
        continue;
      }
      if (token = this.tokenizer.lheading(src)) {
        src = src.substring(token.raw.length);
        tokens.push(token);
        continue;
      }
      cutSrc = src;
      if (this.options.extensions && this.options.extensions.startBlock) {
        let startIndex = Infinity;
        const tempSrc = src.slice(1);
        let tempStart;
        this.options.extensions.startBlock.forEach((getStartIndex) => {
          tempStart = getStartIndex.call({ lexer: this }, tempSrc);
          if (typeof tempStart === "number" && tempStart >= 0) {
            startIndex = Math.min(startIndex, tempStart);
          }
        });
        if (startIndex < Infinity && startIndex >= 0) {
          cutSrc = src.substring(0, startIndex + 1);
        }
      }
      if (this.state.top && (token = this.tokenizer.paragraph(cutSrc))) {
        lastToken = tokens[tokens.length - 1];
        if (lastParagraphClipped && lastToken.type === "paragraph") {
          lastToken.raw += "\n" + token.raw;
          lastToken.text += "\n" + token.text;
          this.inlineQueue.pop();
          this.inlineQueue[this.inlineQueue.length - 1].src = lastToken.text;
        } else {
          tokens.push(token);
        }
        lastParagraphClipped = cutSrc.length !== src.length;
        src = src.substring(token.raw.length);
        continue;
      }
      if (token = this.tokenizer.text(src)) {
        src = src.substring(token.raw.length);
        lastToken = tokens[tokens.length - 1];
        if (lastToken && lastToken.type === "text") {
          lastToken.raw += "\n" + token.raw;
          lastToken.text += "\n" + token.text;
          this.inlineQueue.pop();
          this.inlineQueue[this.inlineQueue.length - 1].src = lastToken.text;
        } else {
          tokens.push(token);
        }
        continue;
      }
      if (src) {
        const errMsg = "Infinite loop on byte: " + src.charCodeAt(0);
        if (this.options.silent) {
          console.error(errMsg);
          break;
        } else {
          throw new Error(errMsg);
        }
      }
    }
    this.state.top = true;
    return tokens;
  }
  inline(src, tokens = []) {
    this.inlineQueue.push({ src, tokens });
    return tokens;
  }
  /**
   * Lexing/Compiling
   */
  inlineTokens(src, tokens = []) {
    let token, lastToken, cutSrc;
    let maskedSrc = src;
    let match;
    let keepPrevChar, prevChar;
    if (this.tokens.links) {
      const links = Object.keys(this.tokens.links);
      if (links.length > 0) {
        while ((match = this.tokenizer.rules.inline.reflinkSearch.exec(maskedSrc)) != null) {
          if (links.includes(match[0].slice(match[0].lastIndexOf("[") + 1, -1))) {
            maskedSrc = maskedSrc.slice(0, match.index) + "[" + "a".repeat(match[0].length - 2) + "]" + maskedSrc.slice(this.tokenizer.rules.inline.reflinkSearch.lastIndex);
          }
        }
      }
    }
    while ((match = this.tokenizer.rules.inline.blockSkip.exec(maskedSrc)) != null) {
      maskedSrc = maskedSrc.slice(0, match.index) + "[" + "a".repeat(match[0].length - 2) + "]" + maskedSrc.slice(this.tokenizer.rules.inline.blockSkip.lastIndex);
    }
    while ((match = this.tokenizer.rules.inline.anyPunctuation.exec(maskedSrc)) != null) {
      maskedSrc = maskedSrc.slice(0, match.index) + "++" + maskedSrc.slice(this.tokenizer.rules.inline.anyPunctuation.lastIndex);
    }
    while (src) {
      if (!keepPrevChar) {
        prevChar = "";
      }
      keepPrevChar = false;
      if (this.options.extensions && this.options.extensions.inline && this.options.extensions.inline.some((extTokenizer) => {
        if (token = extTokenizer.call({ lexer: this }, src, tokens)) {
          src = src.substring(token.raw.length);
          tokens.push(token);
          return true;
        }
        return false;
      })) {
        continue;
      }
      if (token = this.tokenizer.escape(src)) {
        src = src.substring(token.raw.length);
        tokens.push(token);
        continue;
      }
      if (token = this.tokenizer.tag(src)) {
        src = src.substring(token.raw.length);
        lastToken = tokens[tokens.length - 1];
        if (lastToken && token.type === "text" && lastToken.type === "text") {
          lastToken.raw += token.raw;
          lastToken.text += token.text;
        } else {
          tokens.push(token);
        }
        continue;
      }
      if (token = this.tokenizer.link(src)) {
        src = src.substring(token.raw.length);
        tokens.push(token);
        continue;
      }
      if (token = this.tokenizer.reflink(src, this.tokens.links)) {
        src = src.substring(token.raw.length);
        lastToken = tokens[tokens.length - 1];
        if (lastToken && token.type === "text" && lastToken.type === "text") {
          lastToken.raw += token.raw;
          lastToken.text += token.text;
        } else {
          tokens.push(token);
        }
        continue;
      }
      if (token = this.tokenizer.emStrong(src, maskedSrc, prevChar)) {
        src = src.substring(token.raw.length);
        tokens.push(token);
        continue;
      }
      if (token = this.tokenizer.codespan(src)) {
        src = src.substring(token.raw.length);
        tokens.push(token);
        continue;
      }
      if (token = this.tokenizer.br(src)) {
        src = src.substring(token.raw.length);
        tokens.push(token);
        continue;
      }
      if (token = this.tokenizer.del(src)) {
        src = src.substring(token.raw.length);
        tokens.push(token);
        continue;
      }
      if (token = this.tokenizer.autolink(src)) {
        src = src.substring(token.raw.length);
        tokens.push(token);
        continue;
      }
      if (!this.state.inLink && (token = this.tokenizer.url(src))) {
        src = src.substring(token.raw.length);
        tokens.push(token);
        continue;
      }
      cutSrc = src;
      if (this.options.extensions && this.options.extensions.startInline) {
        let startIndex = Infinity;
        const tempSrc = src.slice(1);
        let tempStart;
        this.options.extensions.startInline.forEach((getStartIndex) => {
          tempStart = getStartIndex.call({ lexer: this }, tempSrc);
          if (typeof tempStart === "number" && tempStart >= 0) {
            startIndex = Math.min(startIndex, tempStart);
          }
        });
        if (startIndex < Infinity && startIndex >= 0) {
          cutSrc = src.substring(0, startIndex + 1);
        }
      }
      if (token = this.tokenizer.inlineText(cutSrc)) {
        src = src.substring(token.raw.length);
        if (token.raw.slice(-1) !== "_") {
          prevChar = token.raw.slice(-1);
        }
        keepPrevChar = true;
        lastToken = tokens[tokens.length - 1];
        if (lastToken && lastToken.type === "text") {
          lastToken.raw += token.raw;
          lastToken.text += token.text;
        } else {
          tokens.push(token);
        }
        continue;
      }
      if (src) {
        const errMsg = "Infinite loop on byte: " + src.charCodeAt(0);
        if (this.options.silent) {
          console.error(errMsg);
          break;
        } else {
          throw new Error(errMsg);
        }
      }
    }
    return tokens;
  }
}
class _Renderer {
  constructor(options) {
    __publicField(this, "options");
    this.options = options || _defaults;
  }
  code(code, infostring, escaped) {
    var _a;
    const lang = (_a = (infostring || "").match(/^\S*/)) == null ? void 0 : _a[0];
    code = code.replace(/\n$/, "") + "\n";
    if (!lang) {
      return "<pre><code>" + (escaped ? code : escape$1$1(code, true)) + "</code></pre>\n";
    }
    return '<pre><code class="language-' + escape$1$1(lang) + '">' + (escaped ? code : escape$1$1(code, true)) + "</code></pre>\n";
  }
  blockquote(quote) {
    return `<blockquote>
${quote}</blockquote>
`;
  }
  html(html2, block2) {
    return html2;
  }
  heading(text2, level, raw) {
    return `<h${level}>${text2}</h${level}>
`;
  }
  hr() {
    return "<hr>\n";
  }
  list(body, ordered, start) {
    const type = ordered ? "ol" : "ul";
    const startatt = ordered && start !== 1 ? ' start="' + start + '"' : "";
    return "<" + type + startatt + ">\n" + body + "</" + type + ">\n";
  }
  listitem(text2, task, checked) {
    return `<li>${text2}</li>
`;
  }
  checkbox(checked) {
    return "<input " + (checked ? 'checked="" ' : "") + 'disabled="" type="checkbox">';
  }
  paragraph(text2) {
    return `<p>${text2}</p>
`;
  }
  table(header, body) {
    if (body)
      body = `<tbody>${body}</tbody>`;
    return "<table>\n<thead>\n" + header + "</thead>\n" + body + "</table>\n";
  }
  tablerow(content) {
    return `<tr>
${content}</tr>
`;
  }
  tablecell(content, flags) {
    const type = flags.header ? "th" : "td";
    const tag2 = flags.align ? `<${type} align="${flags.align}">` : `<${type}>`;
    return tag2 + content + `</${type}>
`;
  }
  /**
   * span level renderer
   */
  strong(text2) {
    return `<strong>${text2}</strong>`;
  }
  em(text2) {
    return `<em>${text2}</em>`;
  }
  codespan(text2) {
    return `<code>${text2}</code>`;
  }
  br() {
    return "<br>";
  }
  del(text2) {
    return `<del>${text2}</del>`;
  }
  link(href, title, text2) {
    const cleanHref = cleanUrl(href);
    if (cleanHref === null) {
      return text2;
    }
    href = cleanHref;
    let out = '<a href="' + href + '"';
    if (title) {
      out += ' title="' + title + '"';
    }
    out += ">" + text2 + "</a>";
    return out;
  }
  image(href, title, text2) {
    const cleanHref = cleanUrl(href);
    if (cleanHref === null) {
      return text2;
    }
    href = cleanHref;
    let out = `<img src="${href}" alt="${text2}"`;
    if (title) {
      out += ` title="${title}"`;
    }
    out += ">";
    return out;
  }
  text(text2) {
    return text2;
  }
}
class _TextRenderer {
  // no need for block level renderers
  strong(text2) {
    return text2;
  }
  em(text2) {
    return text2;
  }
  codespan(text2) {
    return text2;
  }
  del(text2) {
    return text2;
  }
  html(text2) {
    return text2;
  }
  text(text2) {
    return text2;
  }
  link(href, title, text2) {
    return "" + text2;
  }
  image(href, title, text2) {
    return "" + text2;
  }
  br() {
    return "";
  }
}
class _Parser {
  constructor(options) {
    __publicField(this, "options");
    __publicField(this, "renderer");
    __publicField(this, "textRenderer");
    this.options = options || _defaults;
    this.options.renderer = this.options.renderer || new _Renderer();
    this.renderer = this.options.renderer;
    this.renderer.options = this.options;
    this.textRenderer = new _TextRenderer();
  }
  /**
   * Static Parse Method
   */
  static parse(tokens, options) {
    const parser = new _Parser(options);
    return parser.parse(tokens);
  }
  /**
   * Static Parse Inline Method
   */
  static parseInline(tokens, options) {
    const parser = new _Parser(options);
    return parser.parseInline(tokens);
  }
  /**
   * Parse Loop
   */
  parse(tokens, top = true) {
    let out = "";
    for (let i = 0; i < tokens.length; i++) {
      const token = tokens[i];
      if (this.options.extensions && this.options.extensions.renderers && this.options.extensions.renderers[token.type]) {
        const genericToken = token;
        const ret = this.options.extensions.renderers[genericToken.type].call({ parser: this }, genericToken);
        if (ret !== false || !["space", "hr", "heading", "code", "table", "blockquote", "list", "html", "paragraph", "text"].includes(genericToken.type)) {
          out += ret || "";
          continue;
        }
      }
      switch (token.type) {
        case "space": {
          continue;
        }
        case "hr": {
          out += this.renderer.hr();
          continue;
        }
        case "heading": {
          const headingToken = token;
          out += this.renderer.heading(this.parseInline(headingToken.tokens), headingToken.depth, unescape(this.parseInline(headingToken.tokens, this.textRenderer)));
          continue;
        }
        case "code": {
          const codeToken = token;
          out += this.renderer.code(codeToken.text, codeToken.lang, !!codeToken.escaped);
          continue;
        }
        case "table": {
          const tableToken = token;
          let header = "";
          let cell = "";
          for (let j = 0; j < tableToken.header.length; j++) {
            cell += this.renderer.tablecell(this.parseInline(tableToken.header[j].tokens), { header: true, align: tableToken.align[j] });
          }
          header += this.renderer.tablerow(cell);
          let body = "";
          for (let j = 0; j < tableToken.rows.length; j++) {
            const row = tableToken.rows[j];
            cell = "";
            for (let k = 0; k < row.length; k++) {
              cell += this.renderer.tablecell(this.parseInline(row[k].tokens), { header: false, align: tableToken.align[k] });
            }
            body += this.renderer.tablerow(cell);
          }
          out += this.renderer.table(header, body);
          continue;
        }
        case "blockquote": {
          const blockquoteToken = token;
          const body = this.parse(blockquoteToken.tokens);
          out += this.renderer.blockquote(body);
          continue;
        }
        case "list": {
          const listToken = token;
          const ordered = listToken.ordered;
          const start = listToken.start;
          const loose = listToken.loose;
          let body = "";
          for (let j = 0; j < listToken.items.length; j++) {
            const item = listToken.items[j];
            const checked = item.checked;
            const task = item.task;
            let itemBody = "";
            if (item.task) {
              const checkbox = this.renderer.checkbox(!!checked);
              if (loose) {
                if (item.tokens.length > 0 && item.tokens[0].type === "paragraph") {
                  item.tokens[0].text = checkbox + " " + item.tokens[0].text;
                  if (item.tokens[0].tokens && item.tokens[0].tokens.length > 0 && item.tokens[0].tokens[0].type === "text") {
                    item.tokens[0].tokens[0].text = checkbox + " " + item.tokens[0].tokens[0].text;
                  }
                } else {
                  item.tokens.unshift({
                    type: "text",
                    text: checkbox + " "
                  });
                }
              } else {
                itemBody += checkbox + " ";
              }
            }
            itemBody += this.parse(item.tokens, loose);
            body += this.renderer.listitem(itemBody, task, !!checked);
          }
          out += this.renderer.list(body, ordered, start);
          continue;
        }
        case "html": {
          const htmlToken = token;
          out += this.renderer.html(htmlToken.text, htmlToken.block);
          continue;
        }
        case "paragraph": {
          const paragraphToken = token;
          out += this.renderer.paragraph(this.parseInline(paragraphToken.tokens));
          continue;
        }
        case "text": {
          let textToken = token;
          let body = textToken.tokens ? this.parseInline(textToken.tokens) : textToken.text;
          while (i + 1 < tokens.length && tokens[i + 1].type === "text") {
            textToken = tokens[++i];
            body += "\n" + (textToken.tokens ? this.parseInline(textToken.tokens) : textToken.text);
          }
          out += top ? this.renderer.paragraph(body) : body;
          continue;
        }
        default: {
          const errMsg = 'Token with "' + token.type + '" type was not found.';
          if (this.options.silent) {
            console.error(errMsg);
            return "";
          } else {
            throw new Error(errMsg);
          }
        }
      }
    }
    return out;
  }
  /**
   * Parse Inline Tokens
   */
  parseInline(tokens, renderer2) {
    renderer2 = renderer2 || this.renderer;
    let out = "";
    for (let i = 0; i < tokens.length; i++) {
      const token = tokens[i];
      if (this.options.extensions && this.options.extensions.renderers && this.options.extensions.renderers[token.type]) {
        const ret = this.options.extensions.renderers[token.type].call({ parser: this }, token);
        if (ret !== false || !["escape", "html", "link", "image", "strong", "em", "codespan", "br", "del", "text"].includes(token.type)) {
          out += ret || "";
          continue;
        }
      }
      switch (token.type) {
        case "escape": {
          const escapeToken = token;
          out += renderer2.text(escapeToken.text);
          break;
        }
        case "html": {
          const tagToken = token;
          out += renderer2.html(tagToken.text);
          break;
        }
        case "link": {
          const linkToken = token;
          out += renderer2.link(linkToken.href, linkToken.title, this.parseInline(linkToken.tokens, renderer2));
          break;
        }
        case "image": {
          const imageToken = token;
          out += renderer2.image(imageToken.href, imageToken.title, imageToken.text);
          break;
        }
        case "strong": {
          const strongToken = token;
          out += renderer2.strong(this.parseInline(strongToken.tokens, renderer2));
          break;
        }
        case "em": {
          const emToken = token;
          out += renderer2.em(this.parseInline(emToken.tokens, renderer2));
          break;
        }
        case "codespan": {
          const codespanToken = token;
          out += renderer2.codespan(codespanToken.text);
          break;
        }
        case "br": {
          out += renderer2.br();
          break;
        }
        case "del": {
          const delToken = token;
          out += renderer2.del(this.parseInline(delToken.tokens, renderer2));
          break;
        }
        case "text": {
          const textToken = token;
          out += renderer2.text(textToken.text);
          break;
        }
        default: {
          const errMsg = 'Token with "' + token.type + '" type was not found.';
          if (this.options.silent) {
            console.error(errMsg);
            return "";
          } else {
            throw new Error(errMsg);
          }
        }
      }
    }
    return out;
  }
}
class _Hooks {
  constructor(options) {
    __publicField(this, "options");
    this.options = options || _defaults;
  }
  /**
   * Process markdown before marked
   */
  preprocess(markdown) {
    return markdown;
  }
  /**
   * Process HTML after marked is finished
   */
  postprocess(html2) {
    return html2;
  }
  /**
   * Process all tokens before walk tokens
   */
  processAllTokens(tokens) {
    return tokens;
  }
}
__publicField(_Hooks, "passThroughHooks", /* @__PURE__ */ new Set([
  "preprocess",
  "postprocess",
  "processAllTokens"
]));
class Marked {
  constructor(...args) {
    __privateAdd(this, _Marked_instances);
    __publicField(this, "defaults", _getDefaults());
    __publicField(this, "options", this.setOptions);
    __publicField(this, "parse", __privateMethod(this, _Marked_instances, parseMarkdown_fn).call(this, _Lexer.lex, _Parser.parse));
    __publicField(this, "parseInline", __privateMethod(this, _Marked_instances, parseMarkdown_fn).call(this, _Lexer.lexInline, _Parser.parseInline));
    __publicField(this, "Parser", _Parser);
    __publicField(this, "Renderer", _Renderer);
    __publicField(this, "TextRenderer", _TextRenderer);
    __publicField(this, "Lexer", _Lexer);
    __publicField(this, "Tokenizer", _Tokenizer);
    __publicField(this, "Hooks", _Hooks);
    this.use(...args);
  }
  /**
   * Run callback for every token
   */
  walkTokens(tokens, callback) {
    var _a, _b;
    let values = [];
    for (const token of tokens) {
      values = values.concat(callback.call(this, token));
      switch (token.type) {
        case "table": {
          const tableToken = token;
          for (const cell of tableToken.header) {
            values = values.concat(this.walkTokens(cell.tokens, callback));
          }
          for (const row of tableToken.rows) {
            for (const cell of row) {
              values = values.concat(this.walkTokens(cell.tokens, callback));
            }
          }
          break;
        }
        case "list": {
          const listToken = token;
          values = values.concat(this.walkTokens(listToken.items, callback));
          break;
        }
        default: {
          const genericToken = token;
          if ((_b = (_a = this.defaults.extensions) == null ? void 0 : _a.childTokens) == null ? void 0 : _b[genericToken.type]) {
            this.defaults.extensions.childTokens[genericToken.type].forEach((childTokens) => {
              const tokens2 = genericToken[childTokens].flat(Infinity);
              values = values.concat(this.walkTokens(tokens2, callback));
            });
          } else if (genericToken.tokens) {
            values = values.concat(this.walkTokens(genericToken.tokens, callback));
          }
        }
      }
    }
    return values;
  }
  use(...args) {
    const extensions = this.defaults.extensions || { renderers: {}, childTokens: {} };
    args.forEach((pack) => {
      const opts = { ...pack };
      opts.async = this.defaults.async || opts.async || false;
      if (pack.extensions) {
        pack.extensions.forEach((ext) => {
          if (!ext.name) {
            throw new Error("extension name required");
          }
          if ("renderer" in ext) {
            const prevRenderer = extensions.renderers[ext.name];
            if (prevRenderer) {
              extensions.renderers[ext.name] = function(...args2) {
                let ret = ext.renderer.apply(this, args2);
                if (ret === false) {
                  ret = prevRenderer.apply(this, args2);
                }
                return ret;
              };
            } else {
              extensions.renderers[ext.name] = ext.renderer;
            }
          }
          if ("tokenizer" in ext) {
            if (!ext.level || ext.level !== "block" && ext.level !== "inline") {
              throw new Error("extension level must be 'block' or 'inline'");
            }
            const extLevel = extensions[ext.level];
            if (extLevel) {
              extLevel.unshift(ext.tokenizer);
            } else {
              extensions[ext.level] = [ext.tokenizer];
            }
            if (ext.start) {
              if (ext.level === "block") {
                if (extensions.startBlock) {
                  extensions.startBlock.push(ext.start);
                } else {
                  extensions.startBlock = [ext.start];
                }
              } else if (ext.level === "inline") {
                if (extensions.startInline) {
                  extensions.startInline.push(ext.start);
                } else {
                  extensions.startInline = [ext.start];
                }
              }
            }
          }
          if ("childTokens" in ext && ext.childTokens) {
            extensions.childTokens[ext.name] = ext.childTokens;
          }
        });
        opts.extensions = extensions;
      }
      if (pack.renderer) {
        const renderer2 = this.defaults.renderer || new _Renderer(this.defaults);
        for (const prop in pack.renderer) {
          if (!(prop in renderer2)) {
            throw new Error(`renderer '${prop}' does not exist`);
          }
          if (prop === "options") {
            continue;
          }
          const rendererProp = prop;
          const rendererFunc = pack.renderer[rendererProp];
          const prevRenderer = renderer2[rendererProp];
          renderer2[rendererProp] = (...args2) => {
            let ret = rendererFunc.apply(renderer2, args2);
            if (ret === false) {
              ret = prevRenderer.apply(renderer2, args2);
            }
            return ret || "";
          };
        }
        opts.renderer = renderer2;
      }
      if (pack.tokenizer) {
        const tokenizer = this.defaults.tokenizer || new _Tokenizer(this.defaults);
        for (const prop in pack.tokenizer) {
          if (!(prop in tokenizer)) {
            throw new Error(`tokenizer '${prop}' does not exist`);
          }
          if (["options", "rules", "lexer"].includes(prop)) {
            continue;
          }
          const tokenizerProp = prop;
          const tokenizerFunc = pack.tokenizer[tokenizerProp];
          const prevTokenizer = tokenizer[tokenizerProp];
          tokenizer[tokenizerProp] = (...args2) => {
            let ret = tokenizerFunc.apply(tokenizer, args2);
            if (ret === false) {
              ret = prevTokenizer.apply(tokenizer, args2);
            }
            return ret;
          };
        }
        opts.tokenizer = tokenizer;
      }
      if (pack.hooks) {
        const hooks = this.defaults.hooks || new _Hooks();
        for (const prop in pack.hooks) {
          if (!(prop in hooks)) {
            throw new Error(`hook '${prop}' does not exist`);
          }
          if (prop === "options") {
            continue;
          }
          const hooksProp = prop;
          const hooksFunc = pack.hooks[hooksProp];
          const prevHook = hooks[hooksProp];
          if (_Hooks.passThroughHooks.has(prop)) {
            hooks[hooksProp] = (arg) => {
              if (this.defaults.async) {
                return Promise.resolve(hooksFunc.call(hooks, arg)).then((ret2) => {
                  return prevHook.call(hooks, ret2);
                });
              }
              const ret = hooksFunc.call(hooks, arg);
              return prevHook.call(hooks, ret);
            };
          } else {
            hooks[hooksProp] = (...args2) => {
              let ret = hooksFunc.apply(hooks, args2);
              if (ret === false) {
                ret = prevHook.apply(hooks, args2);
              }
              return ret;
            };
          }
        }
        opts.hooks = hooks;
      }
      if (pack.walkTokens) {
        const walkTokens = this.defaults.walkTokens;
        const packWalktokens = pack.walkTokens;
        opts.walkTokens = function(token) {
          let values = [];
          values.push(packWalktokens.call(this, token));
          if (walkTokens) {
            values = values.concat(walkTokens.call(this, token));
          }
          return values;
        };
      }
      this.defaults = { ...this.defaults, ...opts };
    });
    return this;
  }
  setOptions(opt) {
    this.defaults = { ...this.defaults, ...opt };
    return this;
  }
  lexer(src, options) {
    return _Lexer.lex(src, options ?? this.defaults);
  }
  parser(tokens, options) {
    return _Parser.parse(tokens, options ?? this.defaults);
  }
}
_Marked_instances = /* @__PURE__ */ new WeakSet();
parseMarkdown_fn = function(lexer, parser) {
  return (src, options) => {
    const origOpt = { ...options };
    const opt = { ...this.defaults, ...origOpt };
    if (this.defaults.async === true && origOpt.async === false) {
      if (!opt.silent) {
        console.warn("marked(): The async option was set to true by an extension. The async: false option sent to parse will be ignored.");
      }
      opt.async = true;
    }
    const throwError = __privateMethod(this, _Marked_instances, onError_fn).call(this, !!opt.silent, !!opt.async);
    if (typeof src === "undefined" || src === null) {
      return throwError(new Error("marked(): input parameter is undefined or null"));
    }
    if (typeof src !== "string") {
      return throwError(new Error("marked(): input parameter is of type " + Object.prototype.toString.call(src) + ", string expected"));
    }
    if (opt.hooks) {
      opt.hooks.options = opt;
    }
    if (opt.async) {
      return Promise.resolve(opt.hooks ? opt.hooks.preprocess(src) : src).then((src2) => lexer(src2, opt)).then((tokens) => opt.hooks ? opt.hooks.processAllTokens(tokens) : tokens).then((tokens) => opt.walkTokens ? Promise.all(this.walkTokens(tokens, opt.walkTokens)).then(() => tokens) : tokens).then((tokens) => parser(tokens, opt)).then((html2) => opt.hooks ? opt.hooks.postprocess(html2) : html2).catch(throwError);
    }
    try {
      if (opt.hooks) {
        src = opt.hooks.preprocess(src);
      }
      let tokens = lexer(src, opt);
      if (opt.hooks) {
        tokens = opt.hooks.processAllTokens(tokens);
      }
      if (opt.walkTokens) {
        this.walkTokens(tokens, opt.walkTokens);
      }
      let html2 = parser(tokens, opt);
      if (opt.hooks) {
        html2 = opt.hooks.postprocess(html2);
      }
      return html2;
    } catch (e) {
      return throwError(e);
    }
  };
};
onError_fn = function(silent, async) {
  return (e) => {
    e.message += "\nPlease report this to https://github.com/markedjs/marked.";
    if (silent) {
      const msg = "<p>An error occurred:</p><pre>" + escape$1$1(e.message + "", true) + "</pre>";
      if (async) {
        return Promise.resolve(msg);
      }
      return msg;
    }
    if (async) {
      return Promise.reject(e);
    }
    throw e;
  };
};
const markedInstance = new Marked();
function marked(src, opt) {
  return markedInstance.parse(src, opt);
}
marked.options = marked.setOptions = function(options) {
  markedInstance.setOptions(options);
  marked.defaults = markedInstance.defaults;
  changeDefaults(marked.defaults);
  return marked;
};
marked.getDefaults = _getDefaults;
marked.defaults = _defaults;
marked.use = function(...args) {
  markedInstance.use(...args);
  marked.defaults = markedInstance.defaults;
  changeDefaults(marked.defaults);
  return marked;
};
marked.walkTokens = function(tokens, callback) {
  return markedInstance.walkTokens(tokens, callback);
};
marked.parseInline = markedInstance.parseInline;
marked.Parser = _Parser;
marked.parser = _Parser.parse;
marked.Renderer = _Renderer;
marked.TextRenderer = _TextRenderer;
marked.Lexer = _Lexer;
marked.lexer = _Lexer.lex;
marked.Tokenizer = _Tokenizer;
marked.Hooks = _Hooks;
marked.parse = marked;
marked.options;
marked.setOptions;
marked.use;
marked.walkTokens;
marked.parseInline;
_Parser.parse;
_Lexer.lex;
const regex = /[\0-\x1F!-,\.\/:-@\[-\^`\{-\xA9\xAB-\xB4\xB6-\xB9\xBB-\xBF\xD7\xF7\u02C2-\u02C5\u02D2-\u02DF\u02E5-\u02EB\u02ED\u02EF-\u02FF\u0375\u0378\u0379\u037E\u0380-\u0385\u0387\u038B\u038D\u03A2\u03F6\u0482\u0530\u0557\u0558\u055A-\u055F\u0589-\u0590\u05BE\u05C0\u05C3\u05C6\u05C8-\u05CF\u05EB-\u05EE\u05F3-\u060F\u061B-\u061F\u066A-\u066D\u06D4\u06DD\u06DE\u06E9\u06FD\u06FE\u0700-\u070F\u074B\u074C\u07B2-\u07BF\u07F6-\u07F9\u07FB\u07FC\u07FE\u07FF\u082E-\u083F\u085C-\u085F\u086B-\u089F\u08B5\u08C8-\u08D2\u08E2\u0964\u0965\u0970\u0984\u098D\u098E\u0991\u0992\u09A9\u09B1\u09B3-\u09B5\u09BA\u09BB\u09C5\u09C6\u09C9\u09CA\u09CF-\u09D6\u09D8-\u09DB\u09DE\u09E4\u09E5\u09F2-\u09FB\u09FD\u09FF\u0A00\u0A04\u0A0B-\u0A0E\u0A11\u0A12\u0A29\u0A31\u0A34\u0A37\u0A3A\u0A3B\u0A3D\u0A43-\u0A46\u0A49\u0A4A\u0A4E-\u0A50\u0A52-\u0A58\u0A5D\u0A5F-\u0A65\u0A76-\u0A80\u0A84\u0A8E\u0A92\u0AA9\u0AB1\u0AB4\u0ABA\u0ABB\u0AC6\u0ACA\u0ACE\u0ACF\u0AD1-\u0ADF\u0AE4\u0AE5\u0AF0-\u0AF8\u0B00\u0B04\u0B0D\u0B0E\u0B11\u0B12\u0B29\u0B31\u0B34\u0B3A\u0B3B\u0B45\u0B46\u0B49\u0B4A\u0B4E-\u0B54\u0B58-\u0B5B\u0B5E\u0B64\u0B65\u0B70\u0B72-\u0B81\u0B84\u0B8B-\u0B8D\u0B91\u0B96-\u0B98\u0B9B\u0B9D\u0BA0-\u0BA2\u0BA5-\u0BA7\u0BAB-\u0BAD\u0BBA-\u0BBD\u0BC3-\u0BC5\u0BC9\u0BCE\u0BCF\u0BD1-\u0BD6\u0BD8-\u0BE5\u0BF0-\u0BFF\u0C0D\u0C11\u0C29\u0C3A-\u0C3C\u0C45\u0C49\u0C4E-\u0C54\u0C57\u0C5B-\u0C5F\u0C64\u0C65\u0C70-\u0C7F\u0C84\u0C8D\u0C91\u0CA9\u0CB4\u0CBA\u0CBB\u0CC5\u0CC9\u0CCE-\u0CD4\u0CD7-\u0CDD\u0CDF\u0CE4\u0CE5\u0CF0\u0CF3-\u0CFF\u0D0D\u0D11\u0D45\u0D49\u0D4F-\u0D53\u0D58-\u0D5E\u0D64\u0D65\u0D70-\u0D79\u0D80\u0D84\u0D97-\u0D99\u0DB2\u0DBC\u0DBE\u0DBF\u0DC7-\u0DC9\u0DCB-\u0DCE\u0DD5\u0DD7\u0DE0-\u0DE5\u0DF0\u0DF1\u0DF4-\u0E00\u0E3B-\u0E3F\u0E4F\u0E5A-\u0E80\u0E83\u0E85\u0E8B\u0EA4\u0EA6\u0EBE\u0EBF\u0EC5\u0EC7\u0ECE\u0ECF\u0EDA\u0EDB\u0EE0-\u0EFF\u0F01-\u0F17\u0F1A-\u0F1F\u0F2A-\u0F34\u0F36\u0F38\u0F3A-\u0F3D\u0F48\u0F6D-\u0F70\u0F85\u0F98\u0FBD-\u0FC5\u0FC7-\u0FFF\u104A-\u104F\u109E\u109F\u10C6\u10C8-\u10CC\u10CE\u10CF\u10FB\u1249\u124E\u124F\u1257\u1259\u125E\u125F\u1289\u128E\u128F\u12B1\u12B6\u12B7\u12BF\u12C1\u12C6\u12C7\u12D7\u1311\u1316\u1317\u135B\u135C\u1360-\u137F\u1390-\u139F\u13F6\u13F7\u13FE-\u1400\u166D\u166E\u1680\u169B-\u169F\u16EB-\u16ED\u16F9-\u16FF\u170D\u1715-\u171F\u1735-\u173F\u1754-\u175F\u176D\u1771\u1774-\u177F\u17D4-\u17D6\u17D8-\u17DB\u17DE\u17DF\u17EA-\u180A\u180E\u180F\u181A-\u181F\u1879-\u187F\u18AB-\u18AF\u18F6-\u18FF\u191F\u192C-\u192F\u193C-\u1945\u196E\u196F\u1975-\u197F\u19AC-\u19AF\u19CA-\u19CF\u19DA-\u19FF\u1A1C-\u1A1F\u1A5F\u1A7D\u1A7E\u1A8A-\u1A8F\u1A9A-\u1AA6\u1AA8-\u1AAF\u1AC1-\u1AFF\u1B4C-\u1B4F\u1B5A-\u1B6A\u1B74-\u1B7F\u1BF4-\u1BFF\u1C38-\u1C3F\u1C4A-\u1C4C\u1C7E\u1C7F\u1C89-\u1C8F\u1CBB\u1CBC\u1CC0-\u1CCF\u1CD3\u1CFB-\u1CFF\u1DFA\u1F16\u1F17\u1F1E\u1F1F\u1F46\u1F47\u1F4E\u1F4F\u1F58\u1F5A\u1F5C\u1F5E\u1F7E\u1F7F\u1FB5\u1FBD\u1FBF-\u1FC1\u1FC5\u1FCD-\u1FCF\u1FD4\u1FD5\u1FDC-\u1FDF\u1FED-\u1FF1\u1FF5\u1FFD-\u203E\u2041-\u2053\u2055-\u2070\u2072-\u207E\u2080-\u208F\u209D-\u20CF\u20F1-\u2101\u2103-\u2106\u2108\u2109\u2114\u2116-\u2118\u211E-\u2123\u2125\u2127\u2129\u212E\u213A\u213B\u2140-\u2144\u214A-\u214D\u214F-\u215F\u2189-\u24B5\u24EA-\u2BFF\u2C2F\u2C5F\u2CE5-\u2CEA\u2CF4-\u2CFF\u2D26\u2D28-\u2D2C\u2D2E\u2D2F\u2D68-\u2D6E\u2D70-\u2D7E\u2D97-\u2D9F\u2DA7\u2DAF\u2DB7\u2DBF\u2DC7\u2DCF\u2DD7\u2DDF\u2E00-\u2E2E\u2E30-\u3004\u3008-\u3020\u3030\u3036\u3037\u303D-\u3040\u3097\u3098\u309B\u309C\u30A0\u30FB\u3100-\u3104\u3130\u318F-\u319F\u31C0-\u31EF\u3200-\u33FF\u4DC0-\u4DFF\u9FFD-\u9FFF\uA48D-\uA4CF\uA4FE\uA4FF\uA60D-\uA60F\uA62C-\uA63F\uA673\uA67E\uA6F2-\uA716\uA720\uA721\uA789\uA78A\uA7C0\uA7C1\uA7CB-\uA7F4\uA828-\uA82B\uA82D-\uA83F\uA874-\uA87F\uA8C6-\uA8CF\uA8DA-\uA8DF\uA8F8-\uA8FA\uA8FC\uA92E\uA92F\uA954-\uA95F\uA97D-\uA97F\uA9C1-\uA9CE\uA9DA-\uA9DF\uA9FF\uAA37-\uAA3F\uAA4E\uAA4F\uAA5A-\uAA5F\uAA77-\uAA79\uAAC3-\uAADA\uAADE\uAADF\uAAF0\uAAF1\uAAF7-\uAB00\uAB07\uAB08\uAB0F\uAB10\uAB17-\uAB1F\uAB27\uAB2F\uAB5B\uAB6A-\uAB6F\uABEB\uABEE\uABEF\uABFA-\uABFF\uD7A4-\uD7AF\uD7C7-\uD7CA\uD7FC-\uD7FF\uE000-\uF8FF\uFA6E\uFA6F\uFADA-\uFAFF\uFB07-\uFB12\uFB18-\uFB1C\uFB29\uFB37\uFB3D\uFB3F\uFB42\uFB45\uFBB2-\uFBD2\uFD3E-\uFD4F\uFD90\uFD91\uFDC8-\uFDEF\uFDFC-\uFDFF\uFE10-\uFE1F\uFE30-\uFE32\uFE35-\uFE4C\uFE50-\uFE6F\uFE75\uFEFD-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF3E\uFF40\uFF5B-\uFF65\uFFBF-\uFFC1\uFFC8\uFFC9\uFFD0\uFFD1\uFFD8\uFFD9\uFFDD-\uFFFF]|\uD800[\uDC0C\uDC27\uDC3B\uDC3E\uDC4E\uDC4F\uDC5E-\uDC7F\uDCFB-\uDD3F\uDD75-\uDDFC\uDDFE-\uDE7F\uDE9D-\uDE9F\uDED1-\uDEDF\uDEE1-\uDEFF\uDF20-\uDF2C\uDF4B-\uDF4F\uDF7B-\uDF7F\uDF9E\uDF9F\uDFC4-\uDFC7\uDFD0\uDFD6-\uDFFF]|\uD801[\uDC9E\uDC9F\uDCAA-\uDCAF\uDCD4-\uDCD7\uDCFC-\uDCFF\uDD28-\uDD2F\uDD64-\uDDFF\uDF37-\uDF3F\uDF56-\uDF5F\uDF68-\uDFFF]|\uD802[\uDC06\uDC07\uDC09\uDC36\uDC39-\uDC3B\uDC3D\uDC3E\uDC56-\uDC5F\uDC77-\uDC7F\uDC9F-\uDCDF\uDCF3\uDCF6-\uDCFF\uDD16-\uDD1F\uDD3A-\uDD7F\uDDB8-\uDDBD\uDDC0-\uDDFF\uDE04\uDE07-\uDE0B\uDE14\uDE18\uDE36\uDE37\uDE3B-\uDE3E\uDE40-\uDE5F\uDE7D-\uDE7F\uDE9D-\uDEBF\uDEC8\uDEE7-\uDEFF\uDF36-\uDF3F\uDF56-\uDF5F\uDF73-\uDF7F\uDF92-\uDFFF]|\uD803[\uDC49-\uDC7F\uDCB3-\uDCBF\uDCF3-\uDCFF\uDD28-\uDD2F\uDD3A-\uDE7F\uDEAA\uDEAD-\uDEAF\uDEB2-\uDEFF\uDF1D-\uDF26\uDF28-\uDF2F\uDF51-\uDFAF\uDFC5-\uDFDF\uDFF7-\uDFFF]|\uD804[\uDC47-\uDC65\uDC70-\uDC7E\uDCBB-\uDCCF\uDCE9-\uDCEF\uDCFA-\uDCFF\uDD35\uDD40-\uDD43\uDD48-\uDD4F\uDD74\uDD75\uDD77-\uDD7F\uDDC5-\uDDC8\uDDCD\uDDDB\uDDDD-\uDDFF\uDE12\uDE38-\uDE3D\uDE3F-\uDE7F\uDE87\uDE89\uDE8E\uDE9E\uDEA9-\uDEAF\uDEEB-\uDEEF\uDEFA-\uDEFF\uDF04\uDF0D\uDF0E\uDF11\uDF12\uDF29\uDF31\uDF34\uDF3A\uDF45\uDF46\uDF49\uDF4A\uDF4E\uDF4F\uDF51-\uDF56\uDF58-\uDF5C\uDF64\uDF65\uDF6D-\uDF6F\uDF75-\uDFFF]|\uD805[\uDC4B-\uDC4F\uDC5A-\uDC5D\uDC62-\uDC7F\uDCC6\uDCC8-\uDCCF\uDCDA-\uDD7F\uDDB6\uDDB7\uDDC1-\uDDD7\uDDDE-\uDDFF\uDE41-\uDE43\uDE45-\uDE4F\uDE5A-\uDE7F\uDEB9-\uDEBF\uDECA-\uDEFF\uDF1B\uDF1C\uDF2C-\uDF2F\uDF3A-\uDFFF]|\uD806[\uDC3B-\uDC9F\uDCEA-\uDCFE\uDD07\uDD08\uDD0A\uDD0B\uDD14\uDD17\uDD36\uDD39\uDD3A\uDD44-\uDD4F\uDD5A-\uDD9F\uDDA8\uDDA9\uDDD8\uDDD9\uDDE2\uDDE5-\uDDFF\uDE3F-\uDE46\uDE48-\uDE4F\uDE9A-\uDE9C\uDE9E-\uDEBF\uDEF9-\uDFFF]|\uD807[\uDC09\uDC37\uDC41-\uDC4F\uDC5A-\uDC71\uDC90\uDC91\uDCA8\uDCB7-\uDCFF\uDD07\uDD0A\uDD37-\uDD39\uDD3B\uDD3E\uDD48-\uDD4F\uDD5A-\uDD5F\uDD66\uDD69\uDD8F\uDD92\uDD99-\uDD9F\uDDAA-\uDEDF\uDEF7-\uDFAF\uDFB1-\uDFFF]|\uD808[\uDF9A-\uDFFF]|\uD809[\uDC6F-\uDC7F\uDD44-\uDFFF]|[\uD80A\uD80B\uD80E-\uD810\uD812-\uD819\uD824-\uD82B\uD82D\uD82E\uD830-\uD833\uD837\uD839\uD83D\uD83F\uD87B-\uD87D\uD87F\uD885-\uDB3F\uDB41-\uDBFF][\uDC00-\uDFFF]|\uD80D[\uDC2F-\uDFFF]|\uD811[\uDE47-\uDFFF]|\uD81A[\uDE39-\uDE3F\uDE5F\uDE6A-\uDECF\uDEEE\uDEEF\uDEF5-\uDEFF\uDF37-\uDF3F\uDF44-\uDF4F\uDF5A-\uDF62\uDF78-\uDF7C\uDF90-\uDFFF]|\uD81B[\uDC00-\uDE3F\uDE80-\uDEFF\uDF4B-\uDF4E\uDF88-\uDF8E\uDFA0-\uDFDF\uDFE2\uDFE5-\uDFEF\uDFF2-\uDFFF]|\uD821[\uDFF8-\uDFFF]|\uD823[\uDCD6-\uDCFF\uDD09-\uDFFF]|\uD82C[\uDD1F-\uDD4F\uDD53-\uDD63\uDD68-\uDD6F\uDEFC-\uDFFF]|\uD82F[\uDC6B-\uDC6F\uDC7D-\uDC7F\uDC89-\uDC8F\uDC9A-\uDC9C\uDC9F-\uDFFF]|\uD834[\uDC00-\uDD64\uDD6A-\uDD6C\uDD73-\uDD7A\uDD83\uDD84\uDD8C-\uDDA9\uDDAE-\uDE41\uDE45-\uDFFF]|\uD835[\uDC55\uDC9D\uDCA0\uDCA1\uDCA3\uDCA4\uDCA7\uDCA8\uDCAD\uDCBA\uDCBC\uDCC4\uDD06\uDD0B\uDD0C\uDD15\uDD1D\uDD3A\uDD3F\uDD45\uDD47-\uDD49\uDD51\uDEA6\uDEA7\uDEC1\uDEDB\uDEFB\uDF15\uDF35\uDF4F\uDF6F\uDF89\uDFA9\uDFC3\uDFCC\uDFCD]|\uD836[\uDC00-\uDDFF\uDE37-\uDE3A\uDE6D-\uDE74\uDE76-\uDE83\uDE85-\uDE9A\uDEA0\uDEB0-\uDFFF]|\uD838[\uDC07\uDC19\uDC1A\uDC22\uDC25\uDC2B-\uDCFF\uDD2D-\uDD2F\uDD3E\uDD3F\uDD4A-\uDD4D\uDD4F-\uDEBF\uDEFA-\uDFFF]|\uD83A[\uDCC5-\uDCCF\uDCD7-\uDCFF\uDD4C-\uDD4F\uDD5A-\uDFFF]|\uD83B[\uDC00-\uDDFF\uDE04\uDE20\uDE23\uDE25\uDE26\uDE28\uDE33\uDE38\uDE3A\uDE3C-\uDE41\uDE43-\uDE46\uDE48\uDE4A\uDE4C\uDE50\uDE53\uDE55\uDE56\uDE58\uDE5A\uDE5C\uDE5E\uDE60\uDE63\uDE65\uDE66\uDE6B\uDE73\uDE78\uDE7D\uDE7F\uDE8A\uDE9C-\uDEA0\uDEA4\uDEAA\uDEBC-\uDFFF]|\uD83C[\uDC00-\uDD2F\uDD4A-\uDD4F\uDD6A-\uDD6F\uDD8A-\uDFFF]|\uD83E[\uDC00-\uDFEF\uDFFA-\uDFFF]|\uD869[\uDEDE-\uDEFF]|\uD86D[\uDF35-\uDF3F]|\uD86E[\uDC1E\uDC1F]|\uD873[\uDEA2-\uDEAF]|\uD87A[\uDFE1-\uDFFF]|\uD87E[\uDE1E-\uDFFF]|\uD884[\uDF4B-\uDFFF]|\uDB40[\uDC00-\uDCFF\uDDF0-\uDFFF]/g;
const own = Object.hasOwnProperty;
class BananaSlug {
  /**
   * Create a new slug class.
   */
  constructor() {
    this.occurrences;
    this.reset();
  }
  /**
   * Generate a unique slug.
  *
  * Tracks previously generated slugs: repeated calls with the same value
  * will result in different slugs.
  * Use the `slug` function to get same slugs.
   *
   * @param  {string} value
   *   String of text to slugify
   * @param  {boolean} [maintainCase=false]
   *   Keep the current case, otherwise make all lowercase
   * @return {string}
   *   A unique slug string
   */
  slug(value, maintainCase) {
    const self2 = this;
    let result = slug(value, maintainCase === true);
    const originalSlug = result;
    while (own.call(self2.occurrences, result)) {
      self2.occurrences[originalSlug]++;
      result = originalSlug + "-" + self2.occurrences[originalSlug];
    }
    self2.occurrences[result] = 0;
    return result;
  }
  /**
   * Reset - Forget all previous slugs
   *
   * @return void
   */
  reset() {
    this.occurrences = /* @__PURE__ */ Object.create(null);
  }
}
function slug(value, maintainCase) {
  if (typeof value !== "string") return "";
  if (!maintainCase) value = value.toLowerCase();
  return value.replace(regex, "").replace(/ /g, "-");
}
new BananaSlug();
var commonjsGlobal = typeof globalThis !== "undefined" ? globalThis : typeof window !== "undefined" ? window : typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : {};
var prism = { exports: {} };
(function(module) {
  var _self = typeof window !== "undefined" ? window : typeof WorkerGlobalScope !== "undefined" && self instanceof WorkerGlobalScope ? self : {};
  var Prism2 = (function(_self2) {
    var lang = /(?:^|\s)lang(?:uage)?-([\w-]+)(?=\s|$)/i;
    var uniqueId = 0;
    var plainTextGrammar = {};
    var _ = {
      /**
       * By default, Prism will attempt to highlight all code elements (by calling {@link Prism.highlightAll}) on the
       * current page after the page finished loading. This might be a problem if e.g. you wanted to asynchronously load
       * additional languages or plugins yourself.
       *
       * By setting this value to `true`, Prism will not automatically highlight all code elements on the page.
       *
       * You obviously have to change this value before the automatic highlighting started. To do this, you can add an
       * empty Prism object into the global scope before loading the Prism script like this:
       *
       * ```js
       * window.Prism = window.Prism || {};
       * Prism.manual = true;
       * // add a new <script> to load Prism's script
       * ```
       *
       * @default false
       * @type {boolean}
       * @memberof Prism
       * @public
       */
      manual: _self2.Prism && _self2.Prism.manual,
      /**
       * By default, if Prism is in a web worker, it assumes that it is in a worker it created itself, so it uses
       * `addEventListener` to communicate with its parent instance. However, if you're using Prism manually in your
       * own worker, you don't want it to do this.
       *
       * By setting this value to `true`, Prism will not add its own listeners to the worker.
       *
       * You obviously have to change this value before Prism executes. To do this, you can add an
       * empty Prism object into the global scope before loading the Prism script like this:
       *
       * ```js
       * window.Prism = window.Prism || {};
       * Prism.disableWorkerMessageHandler = true;
       * // Load Prism's script
       * ```
       *
       * @default false
       * @type {boolean}
       * @memberof Prism
       * @public
       */
      disableWorkerMessageHandler: _self2.Prism && _self2.Prism.disableWorkerMessageHandler,
      /**
       * A namespace for utility methods.
       *
       * All function in this namespace that are not explicitly marked as _public_ are for __internal use only__ and may
       * change or disappear at any time.
       *
       * @namespace
       * @memberof Prism
       */
      util: {
        encode: function encode(tokens) {
          if (tokens instanceof Token) {
            return new Token(tokens.type, encode(tokens.content), tokens.alias);
          } else if (Array.isArray(tokens)) {
            return tokens.map(encode);
          } else {
            return tokens.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/\u00a0/g, " ");
          }
        },
        /**
         * Returns the name of the type of the given value.
         *
         * @param {any} o
         * @returns {string}
         * @example
         * type(null)      === 'Null'
         * type(undefined) === 'Undefined'
         * type(123)       === 'Number'
         * type('foo')     === 'String'
         * type(true)      === 'Boolean'
         * type([1, 2])    === 'Array'
         * type({})        === 'Object'
         * type(String)    === 'Function'
         * type(/abc+/)    === 'RegExp'
         */
        type: function(o) {
          return Object.prototype.toString.call(o).slice(8, -1);
        },
        /**
         * Returns a unique number for the given object. Later calls will still return the same number.
         *
         * @param {Object} obj
         * @returns {number}
         */
        objId: function(obj) {
          if (!obj["__id"]) {
            Object.defineProperty(obj, "__id", { value: ++uniqueId });
          }
          return obj["__id"];
        },
        /**
         * Creates a deep clone of the given object.
         *
         * The main intended use of this function is to clone language definitions.
         *
         * @param {T} o
         * @param {Record<number, any>} [visited]
         * @returns {T}
         * @template T
         */
        clone: function deepClone(o, visited) {
          visited = visited || {};
          var clone;
          var id;
          switch (_.util.type(o)) {
            case "Object":
              id = _.util.objId(o);
              if (visited[id]) {
                return visited[id];
              }
              clone = /** @type {Record<string, any>} */
              {};
              visited[id] = clone;
              for (var key in o) {
                if (o.hasOwnProperty(key)) {
                  clone[key] = deepClone(o[key], visited);
                }
              }
              return (
                /** @type {any} */
                clone
              );
            case "Array":
              id = _.util.objId(o);
              if (visited[id]) {
                return visited[id];
              }
              clone = [];
              visited[id] = clone;
              o.forEach(function(v, i) {
                clone[i] = deepClone(v, visited);
              });
              return (
                /** @type {any} */
                clone
              );
            default:
              return o;
          }
        },
        /**
         * Returns the Prism language of the given element set by a `language-xxxx` or `lang-xxxx` class.
         *
         * If no language is set for the element or the element is `null` or `undefined`, `none` will be returned.
         *
         * @param {Element} element
         * @returns {string}
         */
        getLanguage: function(element2) {
          while (element2) {
            var m = lang.exec(element2.className);
            if (m) {
              return m[1].toLowerCase();
            }
            element2 = element2.parentElement;
          }
          return "none";
        },
        /**
         * Sets the Prism `language-xxxx` class of the given element.
         *
         * @param {Element} element
         * @param {string} language
         * @returns {void}
         */
        setLanguage: function(element2, language) {
          element2.className = element2.className.replace(RegExp(lang, "gi"), "");
          element2.classList.add("language-" + language);
        },
        /**
         * Returns the script element that is currently executing.
         *
         * This does __not__ work for line script element.
         *
         * @returns {HTMLScriptElement | null}
         */
        currentScript: function() {
          if (typeof document === "undefined") {
            return null;
          }
          if (document.currentScript && document.currentScript.tagName === "SCRIPT" && 1 < 2) {
            return (
              /** @type {any} */
              document.currentScript
            );
          }
          try {
            throw new Error();
          } catch (err) {
            var src = (/at [^(\r\n]*\((.*):[^:]+:[^:]+\)$/i.exec(err.stack) || [])[1];
            if (src) {
              var scripts = document.getElementsByTagName("script");
              for (var i in scripts) {
                if (scripts[i].src == src) {
                  return scripts[i];
                }
              }
            }
            return null;
          }
        },
        /**
         * Returns whether a given class is active for `element`.
         *
         * The class can be activated if `element` or one of its ancestors has the given class and it can be deactivated
         * if `element` or one of its ancestors has the negated version of the given class. The _negated version_ of the
         * given class is just the given class with a `no-` prefix.
         *
         * Whether the class is active is determined by the closest ancestor of `element` (where `element` itself is
         * closest ancestor) that has the given class or the negated version of it. If neither `element` nor any of its
         * ancestors have the given class or the negated version of it, then the default activation will be returned.
         *
         * In the paradoxical situation where the closest ancestor contains __both__ the given class and the negated
         * version of it, the class is considered active.
         *
         * @param {Element} element
         * @param {string} className
         * @param {boolean} [defaultActivation=false]
         * @returns {boolean}
         */
        isActive: function(element2, className, defaultActivation) {
          var no = "no-" + className;
          while (element2) {
            var classList = element2.classList;
            if (classList.contains(className)) {
              return true;
            }
            if (classList.contains(no)) {
              return false;
            }
            element2 = element2.parentElement;
          }
          return !!defaultActivation;
        }
      },
      /**
       * This namespace contains all currently loaded languages and the some helper functions to create and modify languages.
       *
       * @namespace
       * @memberof Prism
       * @public
       */
      languages: {
        /**
         * The grammar for plain, unformatted text.
         */
        plain: plainTextGrammar,
        plaintext: plainTextGrammar,
        text: plainTextGrammar,
        txt: plainTextGrammar,
        /**
         * Creates a deep copy of the language with the given id and appends the given tokens.
         *
         * If a token in `redef` also appears in the copied language, then the existing token in the copied language
         * will be overwritten at its original position.
         *
         * ## Best practices
         *
         * Since the position of overwriting tokens (token in `redef` that overwrite tokens in the copied language)
         * doesn't matter, they can technically be in any order. However, this can be confusing to others that trying to
         * understand the language definition because, normally, the order of tokens matters in Prism grammars.
         *
         * Therefore, it is encouraged to order overwriting tokens according to the positions of the overwritten tokens.
         * Furthermore, all non-overwriting tokens should be placed after the overwriting ones.
         *
         * @param {string} id The id of the language to extend. This has to be a key in `Prism.languages`.
         * @param {Grammar} redef The new tokens to append.
         * @returns {Grammar} The new language created.
         * @public
         * @example
         * Prism.languages['css-with-colors'] = Prism.languages.extend('css', {
         *     // Prism.languages.css already has a 'comment' token, so this token will overwrite CSS' 'comment' token
         *     // at its original position
         *     'comment': { ... },
         *     // CSS doesn't have a 'color' token, so this token will be appended
         *     'color': /\b(?:red|green|blue)\b/
         * });
         */
        extend: function(id, redef) {
          var lang2 = _.util.clone(_.languages[id]);
          for (var key in redef) {
            lang2[key] = redef[key];
          }
          return lang2;
        },
        /**
         * Inserts tokens _before_ another token in a language definition or any other grammar.
         *
         * ## Usage
         *
         * This helper method makes it easy to modify existing languages. For example, the CSS language definition
         * not only defines CSS highlighting for CSS documents, but also needs to define highlighting for CSS embedded
         * in HTML through `<style>` elements. To do this, it needs to modify `Prism.languages.markup` and add the
         * appropriate tokens. However, `Prism.languages.markup` is a regular JavaScript object literal, so if you do
         * this:
         *
         * ```js
         * Prism.languages.markup.style = {
         *     // token
         * };
         * ```
         *
         * then the `style` token will be added (and processed) at the end. `insertBefore` allows you to insert tokens
         * before existing tokens. For the CSS example above, you would use it like this:
         *
         * ```js
         * Prism.languages.insertBefore('markup', 'cdata', {
         *     'style': {
         *         // token
         *     }
         * });
         * ```
         *
         * ## Special cases
         *
         * If the grammars of `inside` and `insert` have tokens with the same name, the tokens in `inside`'s grammar
         * will be ignored.
         *
         * This behavior can be used to insert tokens after `before`:
         *
         * ```js
         * Prism.languages.insertBefore('markup', 'comment', {
         *     'comment': Prism.languages.markup.comment,
         *     // tokens after 'comment'
         * });
         * ```
         *
         * ## Limitations
         *
         * The main problem `insertBefore` has to solve is iteration order. Since ES2015, the iteration order for object
         * properties is guaranteed to be the insertion order (except for integer keys) but some browsers behave
         * differently when keys are deleted and re-inserted. So `insertBefore` can't be implemented by temporarily
         * deleting properties which is necessary to insert at arbitrary positions.
         *
         * To solve this problem, `insertBefore` doesn't actually insert the given tokens into the target object.
         * Instead, it will create a new object and replace all references to the target object with the new one. This
         * can be done without temporarily deleting properties, so the iteration order is well-defined.
         *
         * However, only references that can be reached from `Prism.languages` or `insert` will be replaced. I.e. if
         * you hold the target object in a variable, then the value of the variable will not change.
         *
         * ```js
         * var oldMarkup = Prism.languages.markup;
         * var newMarkup = Prism.languages.insertBefore('markup', 'comment', { ... });
         *
         * assert(oldMarkup !== Prism.languages.markup);
         * assert(newMarkup === Prism.languages.markup);
         * ```
         *
         * @param {string} inside The property of `root` (e.g. a language id in `Prism.languages`) that contains the
         * object to be modified.
         * @param {string} before The key to insert before.
         * @param {Grammar} insert An object containing the key-value pairs to be inserted.
         * @param {Object<string, any>} [root] The object containing `inside`, i.e. the object that contains the
         * object to be modified.
         *
         * Defaults to `Prism.languages`.
         * @returns {Grammar} The new grammar object.
         * @public
         */
        insertBefore: function(inside, before, insert2, root) {
          root = root || /** @type {any} */
          _.languages;
          var grammar = root[inside];
          var ret = {};
          for (var token in grammar) {
            if (grammar.hasOwnProperty(token)) {
              if (token == before) {
                for (var newToken in insert2) {
                  if (insert2.hasOwnProperty(newToken)) {
                    ret[newToken] = insert2[newToken];
                  }
                }
              }
              if (!insert2.hasOwnProperty(token)) {
                ret[token] = grammar[token];
              }
            }
          }
          var old = root[inside];
          root[inside] = ret;
          _.languages.DFS(_.languages, function(key, value) {
            if (value === old && key != inside) {
              this[key] = ret;
            }
          });
          return ret;
        },
        // Traverse a language definition with Depth First Search
        DFS: function DFS(o, callback, type, visited) {
          visited = visited || {};
          var objId = _.util.objId;
          for (var i in o) {
            if (o.hasOwnProperty(i)) {
              callback.call(o, i, o[i], type || i);
              var property = o[i];
              var propertyType = _.util.type(property);
              if (propertyType === "Object" && !visited[objId(property)]) {
                visited[objId(property)] = true;
                DFS(property, callback, null, visited);
              } else if (propertyType === "Array" && !visited[objId(property)]) {
                visited[objId(property)] = true;
                DFS(property, callback, i, visited);
              }
            }
          }
        }
      },
      plugins: {},
      /**
       * This is the most high-level function in Prism’s API.
       * It fetches all the elements that have a `.language-xxxx` class and then calls {@link Prism.highlightElement} on
       * each one of them.
       *
       * This is equivalent to `Prism.highlightAllUnder(document, async, callback)`.
       *
       * @param {boolean} [async=false] Same as in {@link Prism.highlightAllUnder}.
       * @param {HighlightCallback} [callback] Same as in {@link Prism.highlightAllUnder}.
       * @memberof Prism
       * @public
       */
      highlightAll: function(async, callback) {
        _.highlightAllUnder(document, async, callback);
      },
      /**
       * Fetches all the descendants of `container` that have a `.language-xxxx` class and then calls
       * {@link Prism.highlightElement} on each one of them.
       *
       * The following hooks will be run:
       * 1. `before-highlightall`
       * 2. `before-all-elements-highlight`
       * 3. All hooks of {@link Prism.highlightElement} for each element.
       *
       * @param {ParentNode} container The root element, whose descendants that have a `.language-xxxx` class will be highlighted.
       * @param {boolean} [async=false] Whether each element is to be highlighted asynchronously using Web Workers.
       * @param {HighlightCallback} [callback] An optional callback to be invoked on each element after its highlighting is done.
       * @memberof Prism
       * @public
       */
      highlightAllUnder: function(container, async, callback) {
        var env = {
          callback,
          container,
          selector: 'code[class*="language-"], [class*="language-"] code, code[class*="lang-"], [class*="lang-"] code'
        };
        _.hooks.run("before-highlightall", env);
        env.elements = Array.prototype.slice.apply(env.container.querySelectorAll(env.selector));
        _.hooks.run("before-all-elements-highlight", env);
        for (var i = 0, element2; element2 = env.elements[i++]; ) {
          _.highlightElement(element2, async === true, env.callback);
        }
      },
      /**
       * Highlights the code inside a single element.
       *
       * The following hooks will be run:
       * 1. `before-sanity-check`
       * 2. `before-highlight`
       * 3. All hooks of {@link Prism.highlight}. These hooks will be run by an asynchronous worker if `async` is `true`.
       * 4. `before-insert`
       * 5. `after-highlight`
       * 6. `complete`
       *
       * Some the above hooks will be skipped if the element doesn't contain any text or there is no grammar loaded for
       * the element's language.
       *
       * @param {Element} element The element containing the code.
       * It must have a class of `language-xxxx` to be processed, where `xxxx` is a valid language identifier.
       * @param {boolean} [async=false] Whether the element is to be highlighted asynchronously using Web Workers
       * to improve performance and avoid blocking the UI when highlighting very large chunks of code. This option is
       * [disabled by default](https://prismjs.com/faq.html#why-is-asynchronous-highlighting-disabled-by-default).
       *
       * Note: All language definitions required to highlight the code must be included in the main `prism.js` file for
       * asynchronous highlighting to work. You can build your own bundle on the
       * [Download page](https://prismjs.com/download.html).
       * @param {HighlightCallback} [callback] An optional callback to be invoked after the highlighting is done.
       * Mostly useful when `async` is `true`, since in that case, the highlighting is done asynchronously.
       * @memberof Prism
       * @public
       */
      highlightElement: function(element2, async, callback) {
        var language = _.util.getLanguage(element2);
        var grammar = _.languages[language];
        _.util.setLanguage(element2, language);
        var parent = element2.parentElement;
        if (parent && parent.nodeName.toLowerCase() === "pre") {
          _.util.setLanguage(parent, language);
        }
        var code = element2.textContent;
        var env = {
          element: element2,
          language,
          grammar,
          code
        };
        function insertHighlightedCode(highlightedCode) {
          env.highlightedCode = highlightedCode;
          _.hooks.run("before-insert", env);
          env.element.innerHTML = env.highlightedCode;
          _.hooks.run("after-highlight", env);
          _.hooks.run("complete", env);
          callback && callback.call(env.element);
        }
        _.hooks.run("before-sanity-check", env);
        parent = env.element.parentElement;
        if (parent && parent.nodeName.toLowerCase() === "pre" && !parent.hasAttribute("tabindex")) {
          parent.setAttribute("tabindex", "0");
        }
        if (!env.code) {
          _.hooks.run("complete", env);
          callback && callback.call(env.element);
          return;
        }
        _.hooks.run("before-highlight", env);
        if (!env.grammar) {
          insertHighlightedCode(_.util.encode(env.code));
          return;
        }
        if (async && _self2.Worker) {
          var worker = new Worker(_.filename);
          worker.onmessage = function(evt) {
            insertHighlightedCode(evt.data);
          };
          worker.postMessage(JSON.stringify({
            language: env.language,
            code: env.code,
            immediateClose: true
          }));
        } else {
          insertHighlightedCode(_.highlight(env.code, env.grammar, env.language));
        }
      },
      /**
       * Low-level function, only use if you know what you’re doing. It accepts a string of text as input
       * and the language definitions to use, and returns a string with the HTML produced.
       *
       * The following hooks will be run:
       * 1. `before-tokenize`
       * 2. `after-tokenize`
       * 3. `wrap`: On each {@link Token}.
       *
       * @param {string} text A string with the code to be highlighted.
       * @param {Grammar} grammar An object containing the tokens to use.
       *
       * Usually a language definition like `Prism.languages.markup`.
       * @param {string} language The name of the language definition passed to `grammar`.
       * @returns {string} The highlighted HTML.
       * @memberof Prism
       * @public
       * @example
       * Prism.highlight('var foo = true;', Prism.languages.javascript, 'javascript');
       */
      highlight: function(text2, grammar, language) {
        var env = {
          code: text2,
          grammar,
          language
        };
        _.hooks.run("before-tokenize", env);
        if (!env.grammar) {
          throw new Error('The language "' + env.language + '" has no grammar.');
        }
        env.tokens = _.tokenize(env.code, env.grammar);
        _.hooks.run("after-tokenize", env);
        return Token.stringify(_.util.encode(env.tokens), env.language);
      },
      /**
       * This is the heart of Prism, and the most low-level function you can use. It accepts a string of text as input
       * and the language definitions to use, and returns an array with the tokenized code.
       *
       * When the language definition includes nested tokens, the function is called recursively on each of these tokens.
       *
       * This method could be useful in other contexts as well, as a very crude parser.
       *
       * @param {string} text A string with the code to be highlighted.
       * @param {Grammar} grammar An object containing the tokens to use.
       *
       * Usually a language definition like `Prism.languages.markup`.
       * @returns {TokenStream} An array of strings and tokens, a token stream.
       * @memberof Prism
       * @public
       * @example
       * let code = `var foo = 0;`;
       * let tokens = Prism.tokenize(code, Prism.languages.javascript);
       * tokens.forEach(token => {
       *     if (token instanceof Prism.Token && token.type === 'number') {
       *         console.log(`Found numeric literal: ${token.content}`);
       *     }
       * });
       */
      tokenize: function(text2, grammar) {
        var rest = grammar.rest;
        if (rest) {
          for (var token in rest) {
            grammar[token] = rest[token];
          }
          delete grammar.rest;
        }
        var tokenList = new LinkedList();
        addAfter(tokenList, tokenList.head, text2);
        matchGrammar(text2, tokenList, grammar, tokenList.head, 0);
        return toArray(tokenList);
      },
      /**
       * @namespace
       * @memberof Prism
       * @public
       */
      hooks: {
        all: {},
        /**
         * Adds the given callback to the list of callbacks for the given hook.
         *
         * The callback will be invoked when the hook it is registered for is run.
         * Hooks are usually directly run by a highlight function but you can also run hooks yourself.
         *
         * One callback function can be registered to multiple hooks and the same hook multiple times.
         *
         * @param {string} name The name of the hook.
         * @param {HookCallback} callback The callback function which is given environment variables.
         * @public
         */
        add: function(name, callback) {
          var hooks = _.hooks.all;
          hooks[name] = hooks[name] || [];
          hooks[name].push(callback);
        },
        /**
         * Runs a hook invoking all registered callbacks with the given environment variables.
         *
         * Callbacks will be invoked synchronously and in the order in which they were registered.
         *
         * @param {string} name The name of the hook.
         * @param {Object<string, any>} env The environment variables of the hook passed to all callbacks registered.
         * @public
         */
        run: function(name, env) {
          var callbacks = _.hooks.all[name];
          if (!callbacks || !callbacks.length) {
            return;
          }
          for (var i = 0, callback; callback = callbacks[i++]; ) {
            callback(env);
          }
        }
      },
      Token
    };
    _self2.Prism = _;
    function Token(type, content, alias, matchedStr) {
      this.type = type;
      this.content = content;
      this.alias = alias;
      this.length = (matchedStr || "").length | 0;
    }
    Token.stringify = function stringify(o, language) {
      if (typeof o == "string") {
        return o;
      }
      if (Array.isArray(o)) {
        var s = "";
        o.forEach(function(e) {
          s += stringify(e, language);
        });
        return s;
      }
      var env = {
        type: o.type,
        content: stringify(o.content, language),
        tag: "span",
        classes: ["token", o.type],
        attributes: {},
        language
      };
      var aliases = o.alias;
      if (aliases) {
        if (Array.isArray(aliases)) {
          Array.prototype.push.apply(env.classes, aliases);
        } else {
          env.classes.push(aliases);
        }
      }
      _.hooks.run("wrap", env);
      var attributes = "";
      for (var name in env.attributes) {
        attributes += " " + name + '="' + (env.attributes[name] || "").replace(/"/g, "&quot;") + '"';
      }
      return "<" + env.tag + ' class="' + env.classes.join(" ") + '"' + attributes + ">" + env.content + "</" + env.tag + ">";
    };
    function matchPattern(pattern, pos, text2, lookbehind) {
      pattern.lastIndex = pos;
      var match = pattern.exec(text2);
      if (match && lookbehind && match[1]) {
        var lookbehindLength = match[1].length;
        match.index += lookbehindLength;
        match[0] = match[0].slice(lookbehindLength);
      }
      return match;
    }
    function matchGrammar(text2, tokenList, grammar, startNode, startPos, rematch) {
      for (var token in grammar) {
        if (!grammar.hasOwnProperty(token) || !grammar[token]) {
          continue;
        }
        var patterns = grammar[token];
        patterns = Array.isArray(patterns) ? patterns : [patterns];
        for (var j = 0; j < patterns.length; ++j) {
          if (rematch && rematch.cause == token + "," + j) {
            return;
          }
          var patternObj = patterns[j];
          var inside = patternObj.inside;
          var lookbehind = !!patternObj.lookbehind;
          var greedy = !!patternObj.greedy;
          var alias = patternObj.alias;
          if (greedy && !patternObj.pattern.global) {
            var flags = patternObj.pattern.toString().match(/[imsuy]*$/)[0];
            patternObj.pattern = RegExp(patternObj.pattern.source, flags + "g");
          }
          var pattern = patternObj.pattern || patternObj;
          for (var currentNode = startNode.next, pos = startPos; currentNode !== tokenList.tail; pos += currentNode.value.length, currentNode = currentNode.next) {
            if (rematch && pos >= rematch.reach) {
              break;
            }
            var str = currentNode.value;
            if (tokenList.length > text2.length) {
              return;
            }
            if (str instanceof Token) {
              continue;
            }
            var removeCount = 1;
            var match;
            if (greedy) {
              match = matchPattern(pattern, pos, text2, lookbehind);
              if (!match || match.index >= text2.length) {
                break;
              }
              var from = match.index;
              var to = match.index + match[0].length;
              var p = pos;
              p += currentNode.value.length;
              while (from >= p) {
                currentNode = currentNode.next;
                p += currentNode.value.length;
              }
              p -= currentNode.value.length;
              pos = p;
              if (currentNode.value instanceof Token) {
                continue;
              }
              for (var k = currentNode; k !== tokenList.tail && (p < to || typeof k.value === "string"); k = k.next) {
                removeCount++;
                p += k.value.length;
              }
              removeCount--;
              str = text2.slice(pos, p);
              match.index -= pos;
            } else {
              match = matchPattern(pattern, 0, str, lookbehind);
              if (!match) {
                continue;
              }
            }
            var from = match.index;
            var matchStr = match[0];
            var before = str.slice(0, from);
            var after = str.slice(from + matchStr.length);
            var reach = pos + str.length;
            if (rematch && reach > rematch.reach) {
              rematch.reach = reach;
            }
            var removeFrom = currentNode.prev;
            if (before) {
              removeFrom = addAfter(tokenList, removeFrom, before);
              pos += before.length;
            }
            removeRange(tokenList, removeFrom, removeCount);
            var wrapped = new Token(token, inside ? _.tokenize(matchStr, inside) : matchStr, alias, matchStr);
            currentNode = addAfter(tokenList, removeFrom, wrapped);
            if (after) {
              addAfter(tokenList, currentNode, after);
            }
            if (removeCount > 1) {
              var nestedRematch = {
                cause: token + "," + j,
                reach
              };
              matchGrammar(text2, tokenList, grammar, currentNode.prev, pos, nestedRematch);
              if (rematch && nestedRematch.reach > rematch.reach) {
                rematch.reach = nestedRematch.reach;
              }
            }
          }
        }
      }
    }
    function LinkedList() {
      var head = { value: null, prev: null, next: null };
      var tail = { value: null, prev: head, next: null };
      head.next = tail;
      this.head = head;
      this.tail = tail;
      this.length = 0;
    }
    function addAfter(list2, node, value) {
      var next = node.next;
      var newNode = { value, prev: node, next };
      node.next = newNode;
      next.prev = newNode;
      list2.length++;
      return newNode;
    }
    function removeRange(list2, node, count) {
      var next = node.next;
      for (var i = 0; i < count && next !== list2.tail; i++) {
        next = next.next;
      }
      node.next = next;
      next.prev = node;
      list2.length -= i;
    }
    function toArray(list2) {
      var array = [];
      var node = list2.head.next;
      while (node !== list2.tail) {
        array.push(node.value);
        node = node.next;
      }
      return array;
    }
    if (!_self2.document) {
      if (!_self2.addEventListener) {
        return _;
      }
      if (!_.disableWorkerMessageHandler) {
        _self2.addEventListener("message", function(evt) {
          var message = JSON.parse(evt.data);
          var lang2 = message.language;
          var code = message.code;
          var immediateClose = message.immediateClose;
          _self2.postMessage(_.highlight(code, _.languages[lang2], lang2));
          if (immediateClose) {
            _self2.close();
          }
        }, false);
      }
      return _;
    }
    var script = _.util.currentScript();
    if (script) {
      _.filename = script.src;
      if (script.hasAttribute("data-manual")) {
        _.manual = true;
      }
    }
    function highlightAutomaticallyCallback() {
      if (!_.manual) {
        _.highlightAll();
      }
    }
    if (!_.manual) {
      var readyState = document.readyState;
      if (readyState === "loading" || readyState === "interactive" && script && script.defer) {
        document.addEventListener("DOMContentLoaded", highlightAutomaticallyCallback);
      } else {
        if (window.requestAnimationFrame) {
          window.requestAnimationFrame(highlightAutomaticallyCallback);
        } else {
          window.setTimeout(highlightAutomaticallyCallback, 16);
        }
      }
    }
    return _;
  })(_self);
  if (module.exports) {
    module.exports = Prism2;
  }
  if (typeof commonjsGlobal !== "undefined") {
    commonjsGlobal.Prism = Prism2;
  }
  Prism2.languages.markup = {
    "comment": {
      pattern: /<!--(?:(?!<!--)[\s\S])*?-->/,
      greedy: true
    },
    "prolog": {
      pattern: /<\?[\s\S]+?\?>/,
      greedy: true
    },
    "doctype": {
      // https://www.w3.org/TR/xml/#NT-doctypedecl
      pattern: /<!DOCTYPE(?:[^>"'[\]]|"[^"]*"|'[^']*')+(?:\[(?:[^<"'\]]|"[^"]*"|'[^']*'|<(?!!--)|<!--(?:[^-]|-(?!->))*-->)*\]\s*)?>/i,
      greedy: true,
      inside: {
        "internal-subset": {
          pattern: /(^[^\[]*\[)[\s\S]+(?=\]>$)/,
          lookbehind: true,
          greedy: true,
          inside: null
          // see below
        },
        "string": {
          pattern: /"[^"]*"|'[^']*'/,
          greedy: true
        },
        "punctuation": /^<!|>$|[[\]]/,
        "doctype-tag": /^DOCTYPE/i,
        "name": /[^\s<>'"]+/
      }
    },
    "cdata": {
      pattern: /<!\[CDATA\[[\s\S]*?\]\]>/i,
      greedy: true
    },
    "tag": {
      pattern: /<\/?(?!\d)[^\s>\/=$<%]+(?:\s(?:\s*[^\s>\/=]+(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s'">=]+(?=[\s>]))|(?=[\s/>])))+)?\s*\/?>/,
      greedy: true,
      inside: {
        "tag": {
          pattern: /^<\/?[^\s>\/]+/,
          inside: {
            "punctuation": /^<\/?/,
            "namespace": /^[^\s>\/:]+:/
          }
        },
        "special-attr": [],
        "attr-value": {
          pattern: /=\s*(?:"[^"]*"|'[^']*'|[^\s'">=]+)/,
          inside: {
            "punctuation": [
              {
                pattern: /^=/,
                alias: "attr-equals"
              },
              {
                pattern: /^(\s*)["']|["']$/,
                lookbehind: true
              }
            ]
          }
        },
        "punctuation": /\/?>/,
        "attr-name": {
          pattern: /[^\s>\/]+/,
          inside: {
            "namespace": /^[^\s>\/:]+:/
          }
        }
      }
    },
    "entity": [
      {
        pattern: /&[\da-z]{1,8};/i,
        alias: "named-entity"
      },
      /&#x?[\da-f]{1,8};/i
    ]
  };
  Prism2.languages.markup["tag"].inside["attr-value"].inside["entity"] = Prism2.languages.markup["entity"];
  Prism2.languages.markup["doctype"].inside["internal-subset"].inside = Prism2.languages.markup;
  Prism2.hooks.add("wrap", function(env) {
    if (env.type === "entity") {
      env.attributes["title"] = env.content.replace(/&amp;/, "&");
    }
  });
  Object.defineProperty(Prism2.languages.markup.tag, "addInlined", {
    /**
     * Adds an inlined language to markup.
     *
     * An example of an inlined language is CSS with `<style>` tags.
     *
     * @param {string} tagName The name of the tag that contains the inlined language. This name will be treated as
     * case insensitive.
     * @param {string} lang The language key.
     * @example
     * addInlined('style', 'css');
     */
    value: function addInlined(tagName, lang) {
      var includedCdataInside = {};
      includedCdataInside["language-" + lang] = {
        pattern: /(^<!\[CDATA\[)[\s\S]+?(?=\]\]>$)/i,
        lookbehind: true,
        inside: Prism2.languages[lang]
      };
      includedCdataInside["cdata"] = /^<!\[CDATA\[|\]\]>$/i;
      var inside = {
        "included-cdata": {
          pattern: /<!\[CDATA\[[\s\S]*?\]\]>/i,
          inside: includedCdataInside
        }
      };
      inside["language-" + lang] = {
        pattern: /[\s\S]+/,
        inside: Prism2.languages[lang]
      };
      var def2 = {};
      def2[tagName] = {
        pattern: RegExp(/(<__[^>]*>)(?:<!\[CDATA\[(?:[^\]]|\](?!\]>))*\]\]>|(?!<!\[CDATA\[)[\s\S])*?(?=<\/__>)/.source.replace(/__/g, function() {
          return tagName;
        }), "i"),
        lookbehind: true,
        greedy: true,
        inside
      };
      Prism2.languages.insertBefore("markup", "cdata", def2);
    }
  });
  Object.defineProperty(Prism2.languages.markup.tag, "addAttribute", {
    /**
     * Adds an pattern to highlight languages embedded in HTML attributes.
     *
     * An example of an inlined language is CSS with `style` attributes.
     *
     * @param {string} attrName The name of the tag that contains the inlined language. This name will be treated as
     * case insensitive.
     * @param {string} lang The language key.
     * @example
     * addAttribute('style', 'css');
     */
    value: function(attrName, lang) {
      Prism2.languages.markup.tag.inside["special-attr"].push({
        pattern: RegExp(
          /(^|["'\s])/.source + "(?:" + attrName + ")" + /\s*=\s*(?:"[^"]*"|'[^']*'|[^\s'">=]+(?=[\s>]))/.source,
          "i"
        ),
        lookbehind: true,
        inside: {
          "attr-name": /^[^\s=]+/,
          "attr-value": {
            pattern: /=[\s\S]+/,
            inside: {
              "value": {
                pattern: /(^=\s*(["']|(?!["'])))\S[\s\S]*(?=\2$)/,
                lookbehind: true,
                alias: [lang, "language-" + lang],
                inside: Prism2.languages[lang]
              },
              "punctuation": [
                {
                  pattern: /^=/,
                  alias: "attr-equals"
                },
                /"|'/
              ]
            }
          }
        }
      });
    }
  });
  Prism2.languages.html = Prism2.languages.markup;
  Prism2.languages.mathml = Prism2.languages.markup;
  Prism2.languages.svg = Prism2.languages.markup;
  Prism2.languages.xml = Prism2.languages.extend("markup", {});
  Prism2.languages.ssml = Prism2.languages.xml;
  Prism2.languages.atom = Prism2.languages.xml;
  Prism2.languages.rss = Prism2.languages.xml;
  (function(Prism3) {
    var string = /(?:"(?:\\(?:\r\n|[\s\S])|[^"\\\r\n])*"|'(?:\\(?:\r\n|[\s\S])|[^'\\\r\n])*')/;
    Prism3.languages.css = {
      "comment": /\/\*[\s\S]*?\*\//,
      "atrule": {
        pattern: RegExp("@[\\w-](?:" + /[^;{\s"']|\s+(?!\s)/.source + "|" + string.source + ")*?" + /(?:;|(?=\s*\{))/.source),
        inside: {
          "rule": /^@[\w-]+/,
          "selector-function-argument": {
            pattern: /(\bselector\s*\(\s*(?![\s)]))(?:[^()\s]|\s+(?![\s)])|\((?:[^()]|\([^()]*\))*\))+(?=\s*\))/,
            lookbehind: true,
            alias: "selector"
          },
          "keyword": {
            pattern: /(^|[^\w-])(?:and|not|only|or)(?![\w-])/,
            lookbehind: true
          }
          // See rest below
        }
      },
      "url": {
        // https://drafts.csswg.org/css-values-3/#urls
        pattern: RegExp("\\burl\\((?:" + string.source + "|" + /(?:[^\\\r\n()"']|\\[\s\S])*/.source + ")\\)", "i"),
        greedy: true,
        inside: {
          "function": /^url/i,
          "punctuation": /^\(|\)$/,
          "string": {
            pattern: RegExp("^" + string.source + "$"),
            alias: "url"
          }
        }
      },
      "selector": {
        pattern: RegExp(`(^|[{}\\s])[^{}\\s](?:[^{};"'\\s]|\\s+(?![\\s{])|` + string.source + ")*(?=\\s*\\{)"),
        lookbehind: true
      },
      "string": {
        pattern: string,
        greedy: true
      },
      "property": {
        pattern: /(^|[^-\w\xA0-\uFFFF])(?!\s)[-_a-z\xA0-\uFFFF](?:(?!\s)[-\w\xA0-\uFFFF])*(?=\s*:)/i,
        lookbehind: true
      },
      "important": /!important\b/i,
      "function": {
        pattern: /(^|[^-a-z0-9])[-a-z0-9]+(?=\()/i,
        lookbehind: true
      },
      "punctuation": /[(){};:,]/
    };
    Prism3.languages.css["atrule"].inside.rest = Prism3.languages.css;
    var markup = Prism3.languages.markup;
    if (markup) {
      markup.tag.addInlined("style", "css");
      markup.tag.addAttribute("style", "css");
    }
  })(Prism2);
  Prism2.languages.clike = {
    "comment": [
      {
        pattern: /(^|[^\\])\/\*[\s\S]*?(?:\*\/|$)/,
        lookbehind: true,
        greedy: true
      },
      {
        pattern: /(^|[^\\:])\/\/.*/,
        lookbehind: true,
        greedy: true
      }
    ],
    "string": {
      pattern: /(["'])(?:\\(?:\r\n|[\s\S])|(?!\1)[^\\\r\n])*\1/,
      greedy: true
    },
    "class-name": {
      pattern: /(\b(?:class|extends|implements|instanceof|interface|new|trait)\s+|\bcatch\s+\()[\w.\\]+/i,
      lookbehind: true,
      inside: {
        "punctuation": /[.\\]/
      }
    },
    "keyword": /\b(?:break|catch|continue|do|else|finally|for|function|if|in|instanceof|new|null|return|throw|try|while)\b/,
    "boolean": /\b(?:false|true)\b/,
    "function": /\b\w+(?=\()/,
    "number": /\b0x[\da-f]+\b|(?:\b\d+(?:\.\d*)?|\B\.\d+)(?:e[+-]?\d+)?/i,
    "operator": /[<>]=?|[!=]=?=?|--?|\+\+?|&&?|\|\|?|[?*/~^%]/,
    "punctuation": /[{}[\];(),.:]/
  };
  Prism2.languages.javascript = Prism2.languages.extend("clike", {
    "class-name": [
      Prism2.languages.clike["class-name"],
      {
        pattern: /(^|[^$\w\xA0-\uFFFF])(?!\s)[_$A-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*(?=\.(?:constructor|prototype))/,
        lookbehind: true
      }
    ],
    "keyword": [
      {
        pattern: /((?:^|\})\s*)catch\b/,
        lookbehind: true
      },
      {
        pattern: /(^|[^.]|\.\.\.\s*)\b(?:as|assert(?=\s*\{)|async(?=\s*(?:function\b|\(|[$\w\xA0-\uFFFF]|$))|await|break|case|class|const|continue|debugger|default|delete|do|else|enum|export|extends|finally(?=\s*(?:\{|$))|for|from(?=\s*(?:['"]|$))|function|(?:get|set)(?=\s*(?:[#\[$\w\xA0-\uFFFF]|$))|if|implements|import|in|instanceof|interface|let|new|null|of|package|private|protected|public|return|static|super|switch|this|throw|try|typeof|undefined|var|void|while|with|yield)\b/,
        lookbehind: true
      }
    ],
    // Allow for all non-ASCII characters (See http://stackoverflow.com/a/2008444)
    "function": /#?(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*(?=\s*(?:\.\s*(?:apply|bind|call)\s*)?\()/,
    "number": {
      pattern: RegExp(
        /(^|[^\w$])/.source + "(?:" + // constant
        (/NaN|Infinity/.source + "|" + // binary integer
        /0[bB][01]+(?:_[01]+)*n?/.source + "|" + // octal integer
        /0[oO][0-7]+(?:_[0-7]+)*n?/.source + "|" + // hexadecimal integer
        /0[xX][\dA-Fa-f]+(?:_[\dA-Fa-f]+)*n?/.source + "|" + // decimal bigint
        /\d+(?:_\d+)*n/.source + "|" + // decimal number (integer or float) but no bigint
        /(?:\d+(?:_\d+)*(?:\.(?:\d+(?:_\d+)*)?)?|\.\d+(?:_\d+)*)(?:[Ee][+-]?\d+(?:_\d+)*)?/.source) + ")" + /(?![\w$])/.source
      ),
      lookbehind: true
    },
    "operator": /--|\+\+|\*\*=?|=>|&&=?|\|\|=?|[!=]==|<<=?|>>>?=?|[-+*/%&|^!=<>]=?|\.{3}|\?\?=?|\?\.?|[~:]/
  });
  Prism2.languages.javascript["class-name"][0].pattern = /(\b(?:class|extends|implements|instanceof|interface|new)\s+)[\w.\\]+/;
  Prism2.languages.insertBefore("javascript", "keyword", {
    "regex": {
      pattern: RegExp(
        // lookbehind
        // eslint-disable-next-line regexp/no-dupe-characters-character-class
        /((?:^|[^$\w\xA0-\uFFFF."'\])\s]|\b(?:return|yield))\s*)/.source + // Regex pattern:
        // There are 2 regex patterns here. The RegExp set notation proposal added support for nested character
        // classes if the `v` flag is present. Unfortunately, nested CCs are both context-free and incompatible
        // with the only syntax, so we have to define 2 different regex patterns.
        /\//.source + "(?:" + /(?:\[(?:[^\]\\\r\n]|\\.)*\]|\\.|[^/\\\[\r\n])+\/[dgimyus]{0,7}/.source + "|" + // `v` flag syntax. This supports 3 levels of nested character classes.
        /(?:\[(?:[^[\]\\\r\n]|\\.|\[(?:[^[\]\\\r\n]|\\.|\[(?:[^[\]\\\r\n]|\\.)*\])*\])*\]|\\.|[^/\\\[\r\n])+\/[dgimyus]{0,7}v[dgimyus]{0,7}/.source + ")" + // lookahead
        /(?=(?:\s|\/\*(?:[^*]|\*(?!\/))*\*\/)*(?:$|[\r\n,.;:})\]]|\/\/))/.source
      ),
      lookbehind: true,
      greedy: true,
      inside: {
        "regex-source": {
          pattern: /^(\/)[\s\S]+(?=\/[a-z]*$)/,
          lookbehind: true,
          alias: "language-regex",
          inside: Prism2.languages.regex
        },
        "regex-delimiter": /^\/|\/$/,
        "regex-flags": /^[a-z]+$/
      }
    },
    // This must be declared before keyword because we use "function" inside the look-forward
    "function-variable": {
      pattern: /#?(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*(?=\s*[=:]\s*(?:async\s*)?(?:\bfunction\b|(?:\((?:[^()]|\([^()]*\))*\)|(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*)\s*=>))/,
      alias: "function"
    },
    "parameter": [
      {
        pattern: /(function(?:\s+(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*)?\s*\(\s*)(?!\s)(?:[^()\s]|\s+(?![\s)])|\([^()]*\))+(?=\s*\))/,
        lookbehind: true,
        inside: Prism2.languages.javascript
      },
      {
        pattern: /(^|[^$\w\xA0-\uFFFF])(?!\s)[_$a-z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*(?=\s*=>)/i,
        lookbehind: true,
        inside: Prism2.languages.javascript
      },
      {
        pattern: /(\(\s*)(?!\s)(?:[^()\s]|\s+(?![\s)])|\([^()]*\))+(?=\s*\)\s*=>)/,
        lookbehind: true,
        inside: Prism2.languages.javascript
      },
      {
        pattern: /((?:\b|\s|^)(?!(?:as|async|await|break|case|catch|class|const|continue|debugger|default|delete|do|else|enum|export|extends|finally|for|from|function|get|if|implements|import|in|instanceof|interface|let|new|null|of|package|private|protected|public|return|set|static|super|switch|this|throw|try|typeof|undefined|var|void|while|with|yield)(?![$\w\xA0-\uFFFF]))(?:(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*\s*)\(\s*|\]\s*\(\s*)(?!\s)(?:[^()\s]|\s+(?![\s)])|\([^()]*\))+(?=\s*\)\s*\{)/,
        lookbehind: true,
        inside: Prism2.languages.javascript
      }
    ],
    "constant": /\b[A-Z](?:[A-Z_]|\dx?)*\b/
  });
  Prism2.languages.insertBefore("javascript", "string", {
    "hashbang": {
      pattern: /^#!.*/,
      greedy: true,
      alias: "comment"
    },
    "template-string": {
      pattern: /`(?:\\[\s\S]|\$\{(?:[^{}]|\{(?:[^{}]|\{[^}]*\})*\})+\}|(?!\$\{)[^\\`])*`/,
      greedy: true,
      inside: {
        "template-punctuation": {
          pattern: /^`|`$/,
          alias: "string"
        },
        "interpolation": {
          pattern: /((?:^|[^\\])(?:\\{2})*)\$\{(?:[^{}]|\{(?:[^{}]|\{[^}]*\})*\})+\}/,
          lookbehind: true,
          inside: {
            "interpolation-punctuation": {
              pattern: /^\$\{|\}$/,
              alias: "punctuation"
            },
            rest: Prism2.languages.javascript
          }
        },
        "string": /[\s\S]+/
      }
    },
    "string-property": {
      pattern: /((?:^|[,{])[ \t]*)(["'])(?:\\(?:\r\n|[\s\S])|(?!\2)[^\\\r\n])*\2(?=\s*:)/m,
      lookbehind: true,
      greedy: true,
      alias: "property"
    }
  });
  Prism2.languages.insertBefore("javascript", "operator", {
    "literal-property": {
      pattern: /((?:^|[,{])[ \t]*)(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*(?=\s*:)/m,
      lookbehind: true,
      alias: "property"
    }
  });
  if (Prism2.languages.markup) {
    Prism2.languages.markup.tag.addInlined("script", "javascript");
    Prism2.languages.markup.tag.addAttribute(
      /on(?:abort|blur|change|click|composition(?:end|start|update)|dblclick|error|focus(?:in|out)?|key(?:down|up)|load|mouse(?:down|enter|leave|move|out|over|up)|reset|resize|scroll|select|slotchange|submit|unload|wheel)/.source,
      "javascript"
    );
  }
  Prism2.languages.js = Prism2.languages.javascript;
  (function() {
    if (typeof Prism2 === "undefined" || typeof document === "undefined") {
      return;
    }
    if (!Element.prototype.matches) {
      Element.prototype.matches = Element.prototype.msMatchesSelector || Element.prototype.webkitMatchesSelector;
    }
    var LOADING_MESSAGE = "Loading…";
    var FAILURE_MESSAGE = function(status, message) {
      return "✖ Error " + status + " while fetching file: " + message;
    };
    var FAILURE_EMPTY_MESSAGE = "✖ Error: File does not exist or is empty";
    var EXTENSIONS = {
      "js": "javascript",
      "py": "python",
      "rb": "ruby",
      "ps1": "powershell",
      "psm1": "powershell",
      "sh": "bash",
      "bat": "batch",
      "h": "c",
      "tex": "latex"
    };
    var STATUS_ATTR = "data-src-status";
    var STATUS_LOADING = "loading";
    var STATUS_LOADED = "loaded";
    var STATUS_FAILED = "failed";
    var SELECTOR = "pre[data-src]:not([" + STATUS_ATTR + '="' + STATUS_LOADED + '"]):not([' + STATUS_ATTR + '="' + STATUS_LOADING + '"])';
    function loadFile(src, success, error) {
      var xhr = new XMLHttpRequest();
      xhr.open("GET", src, true);
      xhr.onreadystatechange = function() {
        if (xhr.readyState == 4) {
          if (xhr.status < 400 && xhr.responseText) {
            success(xhr.responseText);
          } else {
            if (xhr.status >= 400) {
              error(FAILURE_MESSAGE(xhr.status, xhr.statusText));
            } else {
              error(FAILURE_EMPTY_MESSAGE);
            }
          }
        }
      };
      xhr.send(null);
    }
    function parseRange(range) {
      var m = /^\s*(\d+)\s*(?:(,)\s*(?:(\d+)\s*)?)?$/.exec(range || "");
      if (m) {
        var start = Number(m[1]);
        var comma = m[2];
        var end = m[3];
        if (!comma) {
          return [start, start];
        }
        if (!end) {
          return [start, void 0];
        }
        return [start, Number(end)];
      }
      return void 0;
    }
    Prism2.hooks.add("before-highlightall", function(env) {
      env.selector += ", " + SELECTOR;
    });
    Prism2.hooks.add("before-sanity-check", function(env) {
      var pre = (
        /** @type {HTMLPreElement} */
        env.element
      );
      if (pre.matches(SELECTOR)) {
        env.code = "";
        pre.setAttribute(STATUS_ATTR, STATUS_LOADING);
        var code = pre.appendChild(document.createElement("CODE"));
        code.textContent = LOADING_MESSAGE;
        var src = pre.getAttribute("data-src");
        var language = env.language;
        if (language === "none") {
          var extension = (/\.(\w+)$/.exec(src) || [, "none"])[1];
          language = EXTENSIONS[extension] || extension;
        }
        Prism2.util.setLanguage(code, language);
        Prism2.util.setLanguage(pre, language);
        var autoloader = Prism2.plugins.autoloader;
        if (autoloader) {
          autoloader.loadLanguages(language);
        }
        loadFile(
          src,
          function(text2) {
            pre.setAttribute(STATUS_ATTR, STATUS_LOADED);
            var range = parseRange(pre.getAttribute("data-range"));
            if (range) {
              var lines = text2.split(/\r\n?|\n/g);
              var start = range[0];
              var end = range[1] == null ? lines.length : range[1];
              if (start < 0) {
                start += lines.length;
              }
              start = Math.max(0, Math.min(start - 1, lines.length));
              if (end < 0) {
                end += lines.length;
              }
              end = Math.max(0, Math.min(end, lines.length));
              text2 = lines.slice(start, end).join("\n");
              if (!pre.hasAttribute("data-start")) {
                pre.setAttribute("data-start", String(start + 1));
              }
            }
            code.textContent = text2;
            Prism2.highlightElement(code);
          },
          function(error) {
            pre.setAttribute(STATUS_ATTR, STATUS_FAILED);
            code.textContent = error;
          }
        );
      }
    });
    Prism2.plugins.fileHighlight = {
      /**
       * Executes the File Highlight plugin for all matching `pre` elements under the given container.
       *
       * Note: Elements which are already loaded or currently loading will not be touched by this method.
       *
       * @param {ParentNode} [container=document]
       */
      highlight: function highlight(container) {
        var elements = (container || document).querySelectorAll(SELECTOR);
        for (var i = 0, element2; element2 = elements[i++]; ) {
          Prism2.highlightElement(element2);
        }
      }
    };
    var logged = false;
    Prism2.fileHighlight = function() {
      if (!logged) {
        console.warn("Prism.fileHighlight is deprecated. Use `Prism.plugins.fileHighlight.highlight` instead.");
        logged = true;
      }
      Prism2.plugins.fileHighlight.highlight.apply(this, arguments);
    };
  })();
})(prism);
Prism.languages.python = {
  "comment": {
    pattern: /(^|[^\\])#.*/,
    lookbehind: true,
    greedy: true
  },
  "string-interpolation": {
    pattern: /(?:f|fr|rf)(?:("""|''')[\s\S]*?\1|("|')(?:\\.|(?!\2)[^\\\r\n])*\2)/i,
    greedy: true,
    inside: {
      "interpolation": {
        // "{" <expression> <optional "!s", "!r", or "!a"> <optional ":" format specifier> "}"
        pattern: /((?:^|[^{])(?:\{\{)*)\{(?!\{)(?:[^{}]|\{(?!\{)(?:[^{}]|\{(?!\{)(?:[^{}])+\})+\})+\}/,
        lookbehind: true,
        inside: {
          "format-spec": {
            pattern: /(:)[^:(){}]+(?=\}$)/,
            lookbehind: true
          },
          "conversion-option": {
            pattern: /![sra](?=[:}]$)/,
            alias: "punctuation"
          },
          rest: null
        }
      },
      "string": /[\s\S]+/
    }
  },
  "triple-quoted-string": {
    pattern: /(?:[rub]|br|rb)?("""|''')[\s\S]*?\1/i,
    greedy: true,
    alias: "string"
  },
  "string": {
    pattern: /(?:[rub]|br|rb)?("|')(?:\\.|(?!\1)[^\\\r\n])*\1/i,
    greedy: true
  },
  "function": {
    pattern: /((?:^|\s)def[ \t]+)[a-zA-Z_]\w*(?=\s*\()/g,
    lookbehind: true
  },
  "class-name": {
    pattern: /(\bclass\s+)\w+/i,
    lookbehind: true
  },
  "decorator": {
    pattern: /(^[\t ]*)@\w+(?:\.\w+)*/m,
    lookbehind: true,
    alias: ["annotation", "punctuation"],
    inside: {
      "punctuation": /\./
    }
  },
  "keyword": /\b(?:_(?=\s*:)|and|as|assert|async|await|break|case|class|continue|def|del|elif|else|except|exec|finally|for|from|global|if|import|in|is|lambda|match|nonlocal|not|or|pass|print|raise|return|try|while|with|yield)\b/,
  "builtin": /\b(?:__import__|abs|all|any|apply|ascii|basestring|bin|bool|buffer|bytearray|bytes|callable|chr|classmethod|cmp|coerce|compile|complex|delattr|dict|dir|divmod|enumerate|eval|execfile|file|filter|float|format|frozenset|getattr|globals|hasattr|hash|help|hex|id|input|int|intern|isinstance|issubclass|iter|len|list|locals|long|map|max|memoryview|min|next|object|oct|open|ord|pow|property|range|raw_input|reduce|reload|repr|reversed|round|set|setattr|slice|sorted|staticmethod|str|sum|super|tuple|type|unichr|unicode|vars|xrange|zip)\b/,
  "boolean": /\b(?:False|None|True)\b/,
  "number": /\b0(?:b(?:_?[01])+|o(?:_?[0-7])+|x(?:_?[a-f0-9])+)\b|(?:\b\d+(?:_\d+)*(?:\.(?:\d+(?:_\d+)*)?)?|\B\.\d+(?:_\d+)*)(?:e[+-]?\d+(?:_\d+)*)?j?(?!\w)/i,
  "operator": /[-+%=]=?|!=|:=|\*\*?=?|\/\/?=?|<[<=>]?|>[=>]?|[&|^~]/,
  "punctuation": /[{}[\];(),.:]/
};
Prism.languages.python["string-interpolation"].inside["interpolation"].inside.rest = Prism.languages.python;
Prism.languages.py = Prism.languages.python;
(function(Prism2) {
  var funcPattern = /\\(?:[^a-z()[\]]|[a-z*]+)/i;
  var insideEqu = {
    "equation-command": {
      pattern: funcPattern,
      alias: "regex"
    }
  };
  Prism2.languages.latex = {
    "comment": /%.*/,
    // the verbatim environment prints whitespace to the document
    "cdata": {
      pattern: /(\\begin\{((?:lstlisting|verbatim)\*?)\})[\s\S]*?(?=\\end\{\2\})/,
      lookbehind: true
    },
    /*
     * equations can be between $$ $$ or $ $ or \( \) or \[ \]
     * (all are multiline)
     */
    "equation": [
      {
        pattern: /\$\$(?:\\[\s\S]|[^\\$])+\$\$|\$(?:\\[\s\S]|[^\\$])+\$|\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]/,
        inside: insideEqu,
        alias: "string"
      },
      {
        pattern: /(\\begin\{((?:align|eqnarray|equation|gather|math|multline)\*?)\})[\s\S]*?(?=\\end\{\2\})/,
        lookbehind: true,
        inside: insideEqu,
        alias: "string"
      }
    ],
    /*
     * arguments which are keywords or references are highlighted
     * as keywords
     */
    "keyword": {
      pattern: /(\\(?:begin|cite|documentclass|end|label|ref|usepackage)(?:\[[^\]]+\])?\{)[^}]+(?=\})/,
      lookbehind: true
    },
    "url": {
      pattern: /(\\url\{)[^}]+(?=\})/,
      lookbehind: true
    },
    /*
     * section or chapter headlines are highlighted as bold so that
     * they stand out more
     */
    "headline": {
      pattern: /(\\(?:chapter|frametitle|paragraph|part|section|subparagraph|subsection|subsubparagraph|subsubsection|subsubsubparagraph)\*?(?:\[[^\]]+\])?\{)[^}]+(?=\})/,
      lookbehind: true,
      alias: "class-name"
    },
    "function": {
      pattern: funcPattern,
      alias: "selector"
    },
    "punctuation": /[[\]{}&]/
  };
  Prism2.languages.tex = Prism2.languages.latex;
  Prism2.languages.context = Prism2.languages.latex;
})(Prism);
(function(Prism2) {
  var envVars = "\\b(?:BASH|BASHOPTS|BASH_ALIASES|BASH_ARGC|BASH_ARGV|BASH_CMDS|BASH_COMPLETION_COMPAT_DIR|BASH_LINENO|BASH_REMATCH|BASH_SOURCE|BASH_VERSINFO|BASH_VERSION|COLORTERM|COLUMNS|COMP_WORDBREAKS|DBUS_SESSION_BUS_ADDRESS|DEFAULTS_PATH|DESKTOP_SESSION|DIRSTACK|DISPLAY|EUID|GDMSESSION|GDM_LANG|GNOME_KEYRING_CONTROL|GNOME_KEYRING_PID|GPG_AGENT_INFO|GROUPS|HISTCONTROL|HISTFILE|HISTFILESIZE|HISTSIZE|HOME|HOSTNAME|HOSTTYPE|IFS|INSTANCE|JOB|LANG|LANGUAGE|LC_ADDRESS|LC_ALL|LC_IDENTIFICATION|LC_MEASUREMENT|LC_MONETARY|LC_NAME|LC_NUMERIC|LC_PAPER|LC_TELEPHONE|LC_TIME|LESSCLOSE|LESSOPEN|LINES|LOGNAME|LS_COLORS|MACHTYPE|MAILCHECK|MANDATORY_PATH|NO_AT_BRIDGE|OLDPWD|OPTERR|OPTIND|ORBIT_SOCKETDIR|OSTYPE|PAPERSIZE|PATH|PIPESTATUS|PPID|PS1|PS2|PS3|PS4|PWD|RANDOM|REPLY|SECONDS|SELINUX_INIT|SESSION|SESSIONTYPE|SESSION_MANAGER|SHELL|SHELLOPTS|SHLVL|SSH_AUTH_SOCK|TERM|UID|UPSTART_EVENTS|UPSTART_INSTANCE|UPSTART_JOB|UPSTART_SESSION|USER|WINDOWID|XAUTHORITY|XDG_CONFIG_DIRS|XDG_CURRENT_DESKTOP|XDG_DATA_DIRS|XDG_GREETER_DATA_DIR|XDG_MENU_PREFIX|XDG_RUNTIME_DIR|XDG_SEAT|XDG_SEAT_PATH|XDG_SESSION_DESKTOP|XDG_SESSION_ID|XDG_SESSION_PATH|XDG_SESSION_TYPE|XDG_VTNR|XMODIFIERS)\\b";
  var commandAfterHeredoc = {
    pattern: /(^(["']?)\w+\2)[ \t]+\S.*/,
    lookbehind: true,
    alias: "punctuation",
    // this looks reasonably well in all themes
    inside: null
    // see below
  };
  var insideString = {
    "bash": commandAfterHeredoc,
    "environment": {
      pattern: RegExp("\\$" + envVars),
      alias: "constant"
    },
    "variable": [
      // [0]: Arithmetic Environment
      {
        pattern: /\$?\(\([\s\S]+?\)\)/,
        greedy: true,
        inside: {
          // If there is a $ sign at the beginning highlight $(( and )) as variable
          "variable": [
            {
              pattern: /(^\$\(\([\s\S]+)\)\)/,
              lookbehind: true
            },
            /^\$\(\(/
          ],
          "number": /\b0x[\dA-Fa-f]+\b|(?:\b\d+(?:\.\d*)?|\B\.\d+)(?:[Ee]-?\d+)?/,
          // Operators according to https://www.gnu.org/software/bash/manual/bashref.html#Shell-Arithmetic
          "operator": /--|\+\+|\*\*=?|<<=?|>>=?|&&|\|\||[=!+\-*/%<>^&|]=?|[?~:]/,
          // If there is no $ sign at the beginning highlight (( and )) as punctuation
          "punctuation": /\(\(?|\)\)?|,|;/
        }
      },
      // [1]: Command Substitution
      {
        pattern: /\$\((?:\([^)]+\)|[^()])+\)|`[^`]+`/,
        greedy: true,
        inside: {
          "variable": /^\$\(|^`|\)$|`$/
        }
      },
      // [2]: Brace expansion
      {
        pattern: /\$\{[^}]+\}/,
        greedy: true,
        inside: {
          "operator": /:[-=?+]?|[!\/]|##?|%%?|\^\^?|,,?/,
          "punctuation": /[\[\]]/,
          "environment": {
            pattern: RegExp("(\\{)" + envVars),
            lookbehind: true,
            alias: "constant"
          }
        }
      },
      /\$(?:\w+|[#?*!@$])/
    ],
    // Escape sequences from echo and printf's manuals, and escaped quotes.
    "entity": /\\(?:[abceEfnrtv\\"]|O?[0-7]{1,3}|U[0-9a-fA-F]{8}|u[0-9a-fA-F]{4}|x[0-9a-fA-F]{1,2})/
  };
  Prism2.languages.bash = {
    "shebang": {
      pattern: /^#!\s*\/.*/,
      alias: "important"
    },
    "comment": {
      pattern: /(^|[^"{\\$])#.*/,
      lookbehind: true
    },
    "function-name": [
      // a) function foo {
      // b) foo() {
      // c) function foo() {
      // but not “foo {”
      {
        // a) and c)
        pattern: /(\bfunction\s+)[\w-]+(?=(?:\s*\(?:\s*\))?\s*\{)/,
        lookbehind: true,
        alias: "function"
      },
      {
        // b)
        pattern: /\b[\w-]+(?=\s*\(\s*\)\s*\{)/,
        alias: "function"
      }
    ],
    // Highlight variable names as variables in for and select beginnings.
    "for-or-select": {
      pattern: /(\b(?:for|select)\s+)\w+(?=\s+in\s)/,
      alias: "variable",
      lookbehind: true
    },
    // Highlight variable names as variables in the left-hand part
    // of assignments (“=” and “+=”).
    "assign-left": {
      pattern: /(^|[\s;|&]|[<>]\()\w+(?:\.\w+)*(?=\+?=)/,
      inside: {
        "environment": {
          pattern: RegExp("(^|[\\s;|&]|[<>]\\()" + envVars),
          lookbehind: true,
          alias: "constant"
        }
      },
      alias: "variable",
      lookbehind: true
    },
    // Highlight parameter names as variables
    "parameter": {
      pattern: /(^|\s)-{1,2}(?:\w+:[+-]?)?\w+(?:\.\w+)*(?=[=\s]|$)/,
      alias: "variable",
      lookbehind: true
    },
    "string": [
      // Support for Here-documents https://en.wikipedia.org/wiki/Here_document
      {
        pattern: /((?:^|[^<])<<-?\s*)(\w+)\s[\s\S]*?(?:\r?\n|\r)\2/,
        lookbehind: true,
        greedy: true,
        inside: insideString
      },
      // Here-document with quotes around the tag
      // → No expansion (so no “inside”).
      {
        pattern: /((?:^|[^<])<<-?\s*)(["'])(\w+)\2\s[\s\S]*?(?:\r?\n|\r)\3/,
        lookbehind: true,
        greedy: true,
        inside: {
          "bash": commandAfterHeredoc
        }
      },
      // “Normal” string
      {
        // https://www.gnu.org/software/bash/manual/html_node/Double-Quotes.html
        pattern: /(^|[^\\](?:\\\\)*)"(?:\\[\s\S]|\$\([^)]+\)|\$(?!\()|`[^`]+`|[^"\\`$])*"/,
        lookbehind: true,
        greedy: true,
        inside: insideString
      },
      {
        // https://www.gnu.org/software/bash/manual/html_node/Single-Quotes.html
        pattern: /(^|[^$\\])'[^']*'/,
        lookbehind: true,
        greedy: true
      },
      {
        // https://www.gnu.org/software/bash/manual/html_node/ANSI_002dC-Quoting.html
        pattern: /\$'(?:[^'\\]|\\[\s\S])*'/,
        greedy: true,
        inside: {
          "entity": insideString.entity
        }
      }
    ],
    "environment": {
      pattern: RegExp("\\$?" + envVars),
      alias: "constant"
    },
    "variable": insideString.variable,
    "function": {
      pattern: /(^|[\s;|&]|[<>]\()(?:add|apropos|apt|apt-cache|apt-get|aptitude|aspell|automysqlbackup|awk|basename|bash|bc|bconsole|bg|bzip2|cal|cargo|cat|cfdisk|chgrp|chkconfig|chmod|chown|chroot|cksum|clear|cmp|column|comm|composer|cp|cron|crontab|csplit|curl|cut|date|dc|dd|ddrescue|debootstrap|df|diff|diff3|dig|dir|dircolors|dirname|dirs|dmesg|docker|docker-compose|du|egrep|eject|env|ethtool|expand|expect|expr|fdformat|fdisk|fg|fgrep|file|find|fmt|fold|format|free|fsck|ftp|fuser|gawk|git|gparted|grep|groupadd|groupdel|groupmod|groups|grub-mkconfig|gzip|halt|head|hg|history|host|hostname|htop|iconv|id|ifconfig|ifdown|ifup|import|install|ip|java|jobs|join|kill|killall|less|link|ln|locate|logname|logrotate|look|lpc|lpr|lprint|lprintd|lprintq|lprm|ls|lsof|lynx|make|man|mc|mdadm|mkconfig|mkdir|mke2fs|mkfifo|mkfs|mkisofs|mknod|mkswap|mmv|more|most|mount|mtools|mtr|mutt|mv|nano|nc|netstat|nice|nl|node|nohup|notify-send|npm|nslookup|op|open|parted|passwd|paste|pathchk|ping|pkill|pnpm|podman|podman-compose|popd|pr|printcap|printenv|ps|pushd|pv|quota|quotacheck|quotactl|ram|rar|rcp|reboot|remsync|rename|renice|rev|rm|rmdir|rpm|rsync|scp|screen|sdiff|sed|sendmail|seq|service|sftp|sh|shellcheck|shuf|shutdown|sleep|slocate|sort|split|ssh|stat|strace|su|sudo|sum|suspend|swapon|sync|sysctl|tac|tail|tar|tee|time|timeout|top|touch|tr|traceroute|tsort|tty|umount|uname|unexpand|uniq|units|unrar|unshar|unzip|update-grub|uptime|useradd|userdel|usermod|users|uudecode|uuencode|v|vcpkg|vdir|vi|vim|virsh|vmstat|wait|watch|wc|wget|whereis|which|who|whoami|write|xargs|xdg-open|yarn|yes|zenity|zip|zsh|zypper)(?=$|[)\s;|&])/,
      lookbehind: true
    },
    "keyword": {
      pattern: /(^|[\s;|&]|[<>]\()(?:case|do|done|elif|else|esac|fi|for|function|if|in|select|then|until|while)(?=$|[)\s;|&])/,
      lookbehind: true
    },
    // https://www.gnu.org/software/bash/manual/html_node/Shell-Builtin-Commands.html
    "builtin": {
      pattern: /(^|[\s;|&]|[<>]\()(?:\.|:|alias|bind|break|builtin|caller|cd|command|continue|declare|echo|enable|eval|exec|exit|export|getopts|hash|help|let|local|logout|mapfile|printf|pwd|read|readarray|readonly|return|set|shift|shopt|source|test|times|trap|type|typeset|ulimit|umask|unalias|unset)(?=$|[)\s;|&])/,
      lookbehind: true,
      // Alias added to make those easier to distinguish from strings.
      alias: "class-name"
    },
    "boolean": {
      pattern: /(^|[\s;|&]|[<>]\()(?:false|true)(?=$|[)\s;|&])/,
      lookbehind: true
    },
    "file-descriptor": {
      pattern: /\B&\d\b/,
      alias: "important"
    },
    "operator": {
      // Lots of redirections here, but not just that.
      pattern: /\d?<>|>\||\+=|=[=~]?|!=?|<<[<-]?|[&\d]?>>|\d[<>]&?|[<>][&=]?|&[>&]?|\|[&|]?/,
      inside: {
        "file-descriptor": {
          pattern: /^\d/,
          alias: "important"
        }
      }
    },
    "punctuation": /\$?\(\(?|\)\)?|\.\.|[{}[\];\\]/,
    "number": {
      pattern: /(^|\s)(?:[1-9]\d*|0)(?:[.,]\d+)?\b/,
      lookbehind: true
    }
  };
  commandAfterHeredoc.inside = Prism2.languages.bash;
  var toBeCopied = [
    "comment",
    "function-name",
    "for-or-select",
    "assign-left",
    "parameter",
    "string",
    "environment",
    "function",
    "keyword",
    "builtin",
    "boolean",
    "file-descriptor",
    "operator",
    "punctuation",
    "number"
  ];
  var inside = insideString.variable[1].inside;
  for (var i = 0; i < toBeCopied.length; i++) {
    inside[toBeCopied[i]] = Prism2.languages.bash[toBeCopied[i]];
  }
  Prism2.languages.sh = Prism2.languages.bash;
  Prism2.languages.shell = Prism2.languages.bash;
})(Prism);
Prism.languages.c = Prism.languages.extend("clike", {
  "comment": {
    pattern: /\/\/(?:[^\r\n\\]|\\(?:\r\n?|\n|(?![\r\n])))*|\/\*[\s\S]*?(?:\*\/|$)/,
    greedy: true
  },
  "string": {
    // https://en.cppreference.com/w/c/language/string_literal
    pattern: /"(?:\\(?:\r\n|[\s\S])|[^"\\\r\n])*"/,
    greedy: true
  },
  "class-name": {
    pattern: /(\b(?:enum|struct)\s+(?:__attribute__\s*\(\([\s\S]*?\)\)\s*)?)\w+|\b[a-z]\w*_t\b/,
    lookbehind: true
  },
  "keyword": /\b(?:_Alignas|_Alignof|_Atomic|_Bool|_Complex|_Generic|_Imaginary|_Noreturn|_Static_assert|_Thread_local|__attribute__|asm|auto|break|case|char|const|continue|default|do|double|else|enum|extern|float|for|goto|if|inline|int|long|register|return|short|signed|sizeof|static|struct|switch|typedef|typeof|union|unsigned|void|volatile|while)\b/,
  "function": /\b[a-z_]\w*(?=\s*\()/i,
  "number": /(?:\b0x(?:[\da-f]+(?:\.[\da-f]*)?|\.[\da-f]+)(?:p[+-]?\d+)?|(?:\b\d+(?:\.\d*)?|\B\.\d+)(?:e[+-]?\d+)?)[ful]{0,4}/i,
  "operator": />>=?|<<=?|->|([-+&|:])\1|[?:~]|[-+*/%&|^!=<>]=?/
});
Prism.languages.insertBefore("c", "string", {
  "char": {
    // https://en.cppreference.com/w/c/language/character_constant
    pattern: /'(?:\\(?:\r\n|[\s\S])|[^'\\\r\n]){0,32}'/,
    greedy: true
  }
});
Prism.languages.insertBefore("c", "string", {
  "macro": {
    // allow for multiline macro definitions
    // spaces after the # character compile fine with gcc
    pattern: /(^[\t ]*)#\s*[a-z](?:[^\r\n\\/]|\/(?!\*)|\/\*(?:[^*]|\*(?!\/))*\*\/|\\(?:\r\n|[\s\S]))*/im,
    lookbehind: true,
    greedy: true,
    alias: "property",
    inside: {
      "string": [
        {
          // highlight the path of the include statement as a string
          pattern: /^(#\s*include\s*)<[^>]+>/,
          lookbehind: true
        },
        Prism.languages.c["string"]
      ],
      "char": Prism.languages.c["char"],
      "comment": Prism.languages.c["comment"],
      "macro-name": [
        {
          pattern: /(^#\s*define\s+)\w+\b(?!\()/i,
          lookbehind: true
        },
        {
          pattern: /(^#\s*define\s+)\w+\b(?=\()/i,
          lookbehind: true,
          alias: "function"
        }
      ],
      // highlight macro directives as keywords
      "directive": {
        pattern: /^(#\s*)[a-z]+/,
        lookbehind: true,
        alias: "keyword"
      },
      "directive-hash": /^#/,
      "punctuation": /##|\\(?=[\r\n])/,
      "expression": {
        pattern: /\S[\s\S]*/,
        inside: Prism.languages.c
      }
    }
  }
});
Prism.languages.insertBefore("c", "function", {
  // highlight predefined macros as constants
  "constant": /\b(?:EOF|NULL|SEEK_CUR|SEEK_END|SEEK_SET|__DATE__|__FILE__|__LINE__|__TIMESTAMP__|__TIME__|__func__|stderr|stdin|stdout)\b/
});
delete Prism.languages.c["boolean"];
(function(Prism2) {
  var keyword = /\b(?:alignas|alignof|asm|auto|bool|break|case|catch|char|char16_t|char32_t|char8_t|class|co_await|co_return|co_yield|compl|concept|const|const_cast|consteval|constexpr|constinit|continue|decltype|default|delete|do|double|dynamic_cast|else|enum|explicit|export|extern|final|float|for|friend|goto|if|import|inline|int|int16_t|int32_t|int64_t|int8_t|long|module|mutable|namespace|new|noexcept|nullptr|operator|override|private|protected|public|register|reinterpret_cast|requires|return|short|signed|sizeof|static|static_assert|static_cast|struct|switch|template|this|thread_local|throw|try|typedef|typeid|typename|uint16_t|uint32_t|uint64_t|uint8_t|union|unsigned|using|virtual|void|volatile|wchar_t|while)\b/;
  var modName = /\b(?!<keyword>)\w+(?:\s*\.\s*\w+)*\b/.source.replace(/<keyword>/g, function() {
    return keyword.source;
  });
  Prism2.languages.cpp = Prism2.languages.extend("c", {
    "class-name": [
      {
        pattern: RegExp(/(\b(?:class|concept|enum|struct|typename)\s+)(?!<keyword>)\w+/.source.replace(/<keyword>/g, function() {
          return keyword.source;
        })),
        lookbehind: true
      },
      // This is intended to capture the class name of method implementations like:
      //   void foo::bar() const {}
      // However! The `foo` in the above example could also be a namespace, so we only capture the class name if
      // it starts with an uppercase letter. This approximation should give decent results.
      /\b[A-Z]\w*(?=\s*::\s*\w+\s*\()/,
      // This will capture the class name before destructors like:
      //   Foo::~Foo() {}
      /\b[A-Z_]\w*(?=\s*::\s*~\w+\s*\()/i,
      // This also intends to capture the class name of method implementations but here the class has template
      // parameters, so it can't be a namespace (until C++ adds generic namespaces).
      /\b\w+(?=\s*<(?:[^<>]|<(?:[^<>]|<[^<>]*>)*>)*>\s*::\s*\w+\s*\()/
    ],
    "keyword": keyword,
    "number": {
      pattern: /(?:\b0b[01']+|\b0x(?:[\da-f']+(?:\.[\da-f']*)?|\.[\da-f']+)(?:p[+-]?[\d']+)?|(?:\b[\d']+(?:\.[\d']*)?|\B\.[\d']+)(?:e[+-]?[\d']+)?)[ful]{0,4}/i,
      greedy: true
    },
    "operator": />>=?|<<=?|->|--|\+\+|&&|\|\||[?:~]|<=>|[-+*/%&|^!=<>]=?|\b(?:and|and_eq|bitand|bitor|not|not_eq|or|or_eq|xor|xor_eq)\b/,
    "boolean": /\b(?:false|true)\b/
  });
  Prism2.languages.insertBefore("cpp", "string", {
    "module": {
      // https://en.cppreference.com/w/cpp/language/modules
      pattern: RegExp(
        /(\b(?:import|module)\s+)/.source + "(?:" + // header-name
        /"(?:\\(?:\r\n|[\s\S])|[^"\\\r\n])*"|<[^<>\r\n]*>/.source + "|" + // module name or partition or both
        /<mod-name>(?:\s*:\s*<mod-name>)?|:\s*<mod-name>/.source.replace(/<mod-name>/g, function() {
          return modName;
        }) + ")"
      ),
      lookbehind: true,
      greedy: true,
      inside: {
        "string": /^[<"][\s\S]+/,
        "operator": /:/,
        "punctuation": /\./
      }
    },
    "raw-string": {
      pattern: /R"([^()\\ ]{0,16})\([\s\S]*?\)\1"/,
      alias: "string",
      greedy: true
    }
  });
  Prism2.languages.insertBefore("cpp", "keyword", {
    "generic-function": {
      pattern: /\b(?!operator\b)[a-z_]\w*\s*<(?:[^<>]|<[^<>]*>)*>(?=\s*\()/i,
      inside: {
        "function": /^\w+/,
        "generic": {
          pattern: /<[\s\S]+/,
          alias: "class-name",
          inside: Prism2.languages.cpp
        }
      }
    }
  });
  Prism2.languages.insertBefore("cpp", "operator", {
    "double-colon": {
      pattern: /::/,
      alias: "punctuation"
    }
  });
  Prism2.languages.insertBefore("cpp", "class-name", {
    // the base clause is an optional list of parent classes
    // https://en.cppreference.com/w/cpp/language/class
    "base-clause": {
      pattern: /(\b(?:class|struct)\s+\w+\s*:\s*)[^;{}"'\s]+(?:\s+[^;{}"'\s]+)*(?=\s*[;{])/,
      lookbehind: true,
      greedy: true,
      inside: Prism2.languages.extend("cpp", {})
    }
  });
  Prism2.languages.insertBefore("inside", "double-colon", {
    // All untokenized words that are not namespaces should be class names
    "class-name": /\b[a-z_]\w*\b(?!\s*::)/i
  }, Prism2.languages.cpp["base-clause"]);
})(Prism);
Prism.languages.json = {
  "property": {
    pattern: /(^|[^\\])"(?:\\.|[^\\"\r\n])*"(?=\s*:)/,
    lookbehind: true,
    greedy: true
  },
  "string": {
    pattern: /(^|[^\\])"(?:\\.|[^\\"\r\n])*"(?!\s*:)/,
    lookbehind: true,
    greedy: true
  },
  "comment": {
    pattern: /\/\/.*|\/\*[\s\S]*?(?:\*\/|$)/,
    greedy: true
  },
  "number": /-?\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b/i,
  "punctuation": /[{}[\],]/,
  "operator": /:/,
  "boolean": /\b(?:false|true)\b/,
  "null": {
    pattern: /\bnull\b/,
    alias: "keyword"
  }
};
Prism.languages.webmanifest = Prism.languages.json;
Prism.languages.sql = {
  "comment": {
    pattern: /(^|[^\\])(?:\/\*[\s\S]*?\*\/|(?:--|\/\/|#).*)/,
    lookbehind: true
  },
  "variable": [
    {
      pattern: /@(["'`])(?:\\[\s\S]|(?!\1)[^\\])+\1/,
      greedy: true
    },
    /@[\w.$]+/
  ],
  "string": {
    pattern: /(^|[^@\\])("|')(?:\\[\s\S]|(?!\2)[^\\]|\2\2)*\2/,
    greedy: true,
    lookbehind: true
  },
  "identifier": {
    pattern: /(^|[^@\\])`(?:\\[\s\S]|[^`\\]|``)*`/,
    greedy: true,
    lookbehind: true,
    inside: {
      "punctuation": /^`|`$/
    }
  },
  "function": /\b(?:AVG|COUNT|FIRST|FORMAT|LAST|LCASE|LEN|MAX|MID|MIN|MOD|NOW|ROUND|SUM|UCASE)(?=\s*\()/i,
  // Should we highlight user defined functions too?
  "keyword": /\b(?:ACTION|ADD|AFTER|ALGORITHM|ALL|ALTER|ANALYZE|ANY|APPLY|AS|ASC|AUTHORIZATION|AUTO_INCREMENT|BACKUP|BDB|BEGIN|BERKELEYDB|BIGINT|BINARY|BIT|BLOB|BOOL|BOOLEAN|BREAK|BROWSE|BTREE|BULK|BY|CALL|CASCADED?|CASE|CHAIN|CHAR(?:ACTER|SET)?|CHECK(?:POINT)?|CLOSE|CLUSTERED|COALESCE|COLLATE|COLUMNS?|COMMENT|COMMIT(?:TED)?|COMPUTE|CONNECT|CONSISTENT|CONSTRAINT|CONTAINS(?:TABLE)?|CONTINUE|CONVERT|CREATE|CROSS|CURRENT(?:_DATE|_TIME|_TIMESTAMP|_USER)?|CURSOR|CYCLE|DATA(?:BASES?)?|DATE(?:TIME)?|DAY|DBCC|DEALLOCATE|DEC|DECIMAL|DECLARE|DEFAULT|DEFINER|DELAYED|DELETE|DELIMITERS?|DENY|DESC|DESCRIBE|DETERMINISTIC|DISABLE|DISCARD|DISK|DISTINCT|DISTINCTROW|DISTRIBUTED|DO|DOUBLE|DROP|DUMMY|DUMP(?:FILE)?|DUPLICATE|ELSE(?:IF)?|ENABLE|ENCLOSED|END|ENGINE|ENUM|ERRLVL|ERRORS|ESCAPED?|EXCEPT|EXEC(?:UTE)?|EXISTS|EXIT|EXPLAIN|EXTENDED|FETCH|FIELDS|FILE|FILLFACTOR|FIRST|FIXED|FLOAT|FOLLOWING|FOR(?: EACH ROW)?|FORCE|FOREIGN|FREETEXT(?:TABLE)?|FROM|FULL|FUNCTION|GEOMETRY(?:COLLECTION)?|GLOBAL|GOTO|GRANT|GROUP|HANDLER|HASH|HAVING|HOLDLOCK|HOUR|IDENTITY(?:COL|_INSERT)?|IF|IGNORE|IMPORT|INDEX|INFILE|INNER|INNODB|INOUT|INSERT|INT|INTEGER|INTERSECT|INTERVAL|INTO|INVOKER|ISOLATION|ITERATE|JOIN|KEYS?|KILL|LANGUAGE|LAST|LEAVE|LEFT|LEVEL|LIMIT|LINENO|LINES|LINESTRING|LOAD|LOCAL|LOCK|LONG(?:BLOB|TEXT)|LOOP|MATCH(?:ED)?|MEDIUM(?:BLOB|INT|TEXT)|MERGE|MIDDLEINT|MINUTE|MODE|MODIFIES|MODIFY|MONTH|MULTI(?:LINESTRING|POINT|POLYGON)|NATIONAL|NATURAL|NCHAR|NEXT|NO|NONCLUSTERED|NULLIF|NUMERIC|OFF?|OFFSETS?|ON|OPEN(?:DATASOURCE|QUERY|ROWSET)?|OPTIMIZE|OPTION(?:ALLY)?|ORDER|OUT(?:ER|FILE)?|OVER|PARTIAL|PARTITION|PERCENT|PIVOT|PLAN|POINT|POLYGON|PRECEDING|PRECISION|PREPARE|PREV|PRIMARY|PRINT|PRIVILEGES|PROC(?:EDURE)?|PUBLIC|PURGE|QUICK|RAISERROR|READS?|REAL|RECONFIGURE|REFERENCES|RELEASE|RENAME|REPEAT(?:ABLE)?|REPLACE|REPLICATION|REQUIRE|RESIGNAL|RESTORE|RESTRICT|RETURN(?:ING|S)?|REVOKE|RIGHT|ROLLBACK|ROUTINE|ROW(?:COUNT|GUIDCOL|S)?|RTREE|RULE|SAVE(?:POINT)?|SCHEMA|SECOND|SELECT|SERIAL(?:IZABLE)?|SESSION(?:_USER)?|SET(?:USER)?|SHARE|SHOW|SHUTDOWN|SIMPLE|SMALLINT|SNAPSHOT|SOME|SONAME|SQL|START(?:ING)?|STATISTICS|STATUS|STRIPED|SYSTEM_USER|TABLES?|TABLESPACE|TEMP(?:ORARY|TABLE)?|TERMINATED|TEXT(?:SIZE)?|THEN|TIME(?:STAMP)?|TINY(?:BLOB|INT|TEXT)|TOP?|TRAN(?:SACTIONS?)?|TRIGGER|TRUNCATE|TSEQUAL|TYPES?|UNBOUNDED|UNCOMMITTED|UNDEFINED|UNION|UNIQUE|UNLOCK|UNPIVOT|UNSIGNED|UPDATE(?:TEXT)?|USAGE|USE|USER|USING|VALUES?|VAR(?:BINARY|CHAR|CHARACTER|YING)|VIEW|WAITFOR|WARNINGS|WHEN|WHERE|WHILE|WITH(?: ROLLUP|IN)?|WORK|WRITE(?:TEXT)?|YEAR)\b/i,
  "boolean": /\b(?:FALSE|NULL|TRUE)\b/i,
  "number": /\b0x[\da-f]+\b|\b\d+(?:\.\d*)?|\B\.\d+\b/i,
  "operator": /[-+*\/=%^~]|&&?|\|\|?|!=?|<(?:=>?|<|>)?|>[>=]?|\b(?:AND|BETWEEN|DIV|ILIKE|IN|IS|LIKE|NOT|OR|REGEXP|RLIKE|SOUNDS LIKE|XOR)\b/i,
  "punctuation": /[;[\]()`,.]/
};
(function(Prism2) {
  var keywords = /\b(?:abstract|assert|boolean|break|byte|case|catch|char|class|const|continue|default|do|double|else|enum|exports|extends|final|finally|float|for|goto|if|implements|import|instanceof|int|interface|long|module|native|new|non-sealed|null|open|opens|package|permits|private|protected|provides|public|record(?!\s*[(){}[\]<>=%~.:,;?+\-*/&|^])|requires|return|sealed|short|static|strictfp|super|switch|synchronized|this|throw|throws|to|transient|transitive|try|uses|var|void|volatile|while|with|yield)\b/;
  var classNamePrefix = /(?:[a-z]\w*\s*\.\s*)*(?:[A-Z]\w*\s*\.\s*)*/.source;
  var className = {
    pattern: RegExp(/(^|[^\w.])/.source + classNamePrefix + /[A-Z](?:[\d_A-Z]*[a-z]\w*)?\b/.source),
    lookbehind: true,
    inside: {
      "namespace": {
        pattern: /^[a-z]\w*(?:\s*\.\s*[a-z]\w*)*(?:\s*\.)?/,
        inside: {
          "punctuation": /\./
        }
      },
      "punctuation": /\./
    }
  };
  Prism2.languages.java = Prism2.languages.extend("clike", {
    "string": {
      pattern: /(^|[^\\])"(?:\\.|[^"\\\r\n])*"/,
      lookbehind: true,
      greedy: true
    },
    "class-name": [
      className,
      {
        // variables, parameters, and constructor references
        // this to support class names (or generic parameters) which do not contain a lower case letter (also works for methods)
        pattern: RegExp(/(^|[^\w.])/.source + classNamePrefix + /[A-Z]\w*(?=\s+\w+\s*[;,=()]|\s*(?:\[[\s,]*\]\s*)?::\s*new\b)/.source),
        lookbehind: true,
        inside: className.inside
      },
      {
        // class names based on keyword
        // this to support class names (or generic parameters) which do not contain a lower case letter (also works for methods)
        pattern: RegExp(/(\b(?:class|enum|extends|implements|instanceof|interface|new|record|throws)\s+)/.source + classNamePrefix + /[A-Z]\w*\b/.source),
        lookbehind: true,
        inside: className.inside
      }
    ],
    "keyword": keywords,
    "function": [
      Prism2.languages.clike.function,
      {
        pattern: /(::\s*)[a-z_]\w*/,
        lookbehind: true
      }
    ],
    "number": /\b0b[01][01_]*L?\b|\b0x(?:\.[\da-f_p+-]+|[\da-f_]+(?:\.[\da-f_p+-]+)?)\b|(?:\b\d[\d_]*(?:\.[\d_]*)?|\B\.\d[\d_]*)(?:e[+-]?\d[\d_]*)?[dfl]?/i,
    "operator": {
      pattern: /(^|[^.])(?:<<=?|>>>?=?|->|--|\+\+|&&|\|\||::|[?:~]|[-+*/%&|^!=<>]=?)/m,
      lookbehind: true
    },
    "constant": /\b[A-Z][A-Z_\d]+\b/
  });
  Prism2.languages.insertBefore("java", "string", {
    "triple-quoted-string": {
      // http://openjdk.java.net/jeps/355#Description
      pattern: /"""[ \t]*[\r\n](?:(?:"|"")?(?:\\.|[^"\\]))*"""/,
      greedy: true,
      alias: "string"
    },
    "char": {
      pattern: /'(?:\\.|[^'\\\r\n]){1,6}'/,
      greedy: true
    }
  });
  Prism2.languages.insertBefore("java", "class-name", {
    "annotation": {
      pattern: /(^|[^.])@\w+(?:\s*\.\s*\w+)*/,
      lookbehind: true,
      alias: "punctuation"
    },
    "generics": {
      pattern: /<(?:[\w\s,.?]|&(?!&)|<(?:[\w\s,.?]|&(?!&)|<(?:[\w\s,.?]|&(?!&)|<(?:[\w\s,.?]|&(?!&))*>)*>)*>)*>/,
      inside: {
        "class-name": className,
        "keyword": keywords,
        "punctuation": /[<>(),.:]/,
        "operator": /[?&|]/
      }
    },
    "import": [
      {
        pattern: RegExp(/(\bimport\s+)/.source + classNamePrefix + /(?:[A-Z]\w*|\*)(?=\s*;)/.source),
        lookbehind: true,
        inside: {
          "namespace": className.inside.namespace,
          "punctuation": /\./,
          "operator": /\*/,
          "class-name": /\w+/
        }
      },
      {
        pattern: RegExp(/(\bimport\s+static\s+)/.source + classNamePrefix + /(?:\w+|\*)(?=\s*;)/.source),
        lookbehind: true,
        alias: "static",
        inside: {
          "namespace": className.inside.namespace,
          "static": /\b\w+$/,
          "punctuation": /\./,
          "operator": /\*/,
          "class-name": /\w+/
        }
      }
    ],
    "namespace": {
      pattern: RegExp(
        /(\b(?:exports|import(?:\s+static)?|module|open|opens|package|provides|requires|to|transitive|uses|with)\s+)(?!<keyword>)[a-z]\w*(?:\.[a-z]\w*)*\.?/.source.replace(/<keyword>/g, function() {
          return keywords.source;
        })
      ),
      lookbehind: true,
      inside: {
        "punctuation": /\./
      }
    }
  });
})(Prism);
Prism.languages.go = Prism.languages.extend("clike", {
  "string": {
    pattern: /(^|[^\\])"(?:\\.|[^"\\\r\n])*"|`[^`]*`/,
    lookbehind: true,
    greedy: true
  },
  "keyword": /\b(?:break|case|chan|const|continue|default|defer|else|fallthrough|for|func|go(?:to)?|if|import|interface|map|package|range|return|select|struct|switch|type|var)\b/,
  "boolean": /\b(?:_|false|iota|nil|true)\b/,
  "number": [
    // binary and octal integers
    /\b0(?:b[01_]+|o[0-7_]+)i?\b/i,
    // hexadecimal integers and floats
    /\b0x(?:[a-f\d_]+(?:\.[a-f\d_]*)?|\.[a-f\d_]+)(?:p[+-]?\d+(?:_\d+)*)?i?(?!\w)/i,
    // decimal integers and floats
    /(?:\b\d[\d_]*(?:\.[\d_]*)?|\B\.\d[\d_]*)(?:e[+-]?[\d_]+)?i?(?!\w)/i
  ],
  "operator": /[*\/%^!=]=?|\+[=+]?|-[=-]?|\|[=|]?|&(?:=|&|\^=?)?|>(?:>=?|=)?|<(?:<=?|=|-)?|:=|\.\.\./,
  "builtin": /\b(?:append|bool|byte|cap|close|complex|complex(?:64|128)|copy|delete|error|float(?:32|64)|u?int(?:8|16|32|64)?|imag|len|make|new|panic|print(?:ln)?|real|recover|rune|string|uintptr)\b/
});
Prism.languages.insertBefore("go", "string", {
  "char": {
    pattern: /'(?:\\.|[^'\\\r\n]){0,10}'/,
    greedy: true
  }
});
delete Prism.languages.go["class-name"];
(function(Prism2) {
  var multilineComment = /\/\*(?:[^*/]|\*(?!\/)|\/(?!\*)|<self>)*\*\//.source;
  for (var i = 0; i < 2; i++) {
    multilineComment = multilineComment.replace(/<self>/g, function() {
      return multilineComment;
    });
  }
  multilineComment = multilineComment.replace(/<self>/g, function() {
    return /[^\s\S]/.source;
  });
  Prism2.languages.rust = {
    "comment": [
      {
        pattern: RegExp(/(^|[^\\])/.source + multilineComment),
        lookbehind: true,
        greedy: true
      },
      {
        pattern: /(^|[^\\:])\/\/.*/,
        lookbehind: true,
        greedy: true
      }
    ],
    "string": {
      pattern: /b?"(?:\\[\s\S]|[^\\"])*"|b?r(#*)"(?:[^"]|"(?!\1))*"\1/,
      greedy: true
    },
    "char": {
      pattern: /b?'(?:\\(?:x[0-7][\da-fA-F]|u\{(?:[\da-fA-F]_*){1,6}\}|.)|[^\\\r\n\t'])'/,
      greedy: true
    },
    "attribute": {
      pattern: /#!?\[(?:[^\[\]"]|"(?:\\[\s\S]|[^\\"])*")*\]/,
      greedy: true,
      alias: "attr-name",
      inside: {
        "string": null
        // see below
      }
    },
    // Closure params should not be confused with bitwise OR |
    "closure-params": {
      pattern: /([=(,:]\s*|\bmove\s*)\|[^|]*\||\|[^|]*\|(?=\s*(?:\{|->))/,
      lookbehind: true,
      greedy: true,
      inside: {
        "closure-punctuation": {
          pattern: /^\||\|$/,
          alias: "punctuation"
        },
        rest: null
        // see below
      }
    },
    "lifetime-annotation": {
      pattern: /'\w+/,
      alias: "symbol"
    },
    "fragment-specifier": {
      pattern: /(\$\w+:)[a-z]+/,
      lookbehind: true,
      alias: "punctuation"
    },
    "variable": /\$\w+/,
    "function-definition": {
      pattern: /(\bfn\s+)\w+/,
      lookbehind: true,
      alias: "function"
    },
    "type-definition": {
      pattern: /(\b(?:enum|struct|trait|type|union)\s+)\w+/,
      lookbehind: true,
      alias: "class-name"
    },
    "module-declaration": [
      {
        pattern: /(\b(?:crate|mod)\s+)[a-z][a-z_\d]*/,
        lookbehind: true,
        alias: "namespace"
      },
      {
        pattern: /(\b(?:crate|self|super)\s*)::\s*[a-z][a-z_\d]*\b(?:\s*::(?:\s*[a-z][a-z_\d]*\s*::)*)?/,
        lookbehind: true,
        alias: "namespace",
        inside: {
          "punctuation": /::/
        }
      }
    ],
    "keyword": [
      // https://github.com/rust-lang/reference/blob/master/src/keywords.md
      /\b(?:Self|abstract|as|async|await|become|box|break|const|continue|crate|do|dyn|else|enum|extern|final|fn|for|if|impl|in|let|loop|macro|match|mod|move|mut|override|priv|pub|ref|return|self|static|struct|super|trait|try|type|typeof|union|unsafe|unsized|use|virtual|where|while|yield)\b/,
      // primitives and str
      // https://doc.rust-lang.org/stable/rust-by-example/primitives.html
      /\b(?:bool|char|f(?:32|64)|[ui](?:8|16|32|64|128|size)|str)\b/
    ],
    // functions can technically start with an upper-case letter, but this will introduce a lot of false positives
    // and Rust's naming conventions recommend snake_case anyway.
    // https://doc.rust-lang.org/1.0.0/style/style/naming/README.html
    "function": /\b[a-z_]\w*(?=\s*(?:::\s*<|\())/,
    "macro": {
      pattern: /\b\w+!/,
      alias: "property"
    },
    "constant": /\b[A-Z_][A-Z_\d]+\b/,
    "class-name": /\b[A-Z]\w*\b/,
    "namespace": {
      pattern: /(?:\b[a-z][a-z_\d]*\s*::\s*)*\b[a-z][a-z_\d]*\s*::(?!\s*<)/,
      inside: {
        "punctuation": /::/
      }
    },
    // Hex, oct, bin, dec numbers with visual separators and type suffix
    "number": /\b(?:0x[\dA-Fa-f](?:_?[\dA-Fa-f])*|0o[0-7](?:_?[0-7])*|0b[01](?:_?[01])*|(?:(?:\d(?:_?\d)*)?\.)?\d(?:_?\d)*(?:[Ee][+-]?\d+)?)(?:_?(?:f32|f64|[iu](?:8|16|32|64|size)?))?\b/,
    "boolean": /\b(?:false|true)\b/,
    "punctuation": /->|\.\.=|\.{1,3}|::|[{}[\];(),:]/,
    "operator": /[-+*\/%!^]=?|=[=>]?|&[&=]?|\|[|=]?|<<?=?|>>?=?|[@?]/
  };
  Prism2.languages.rust["closure-params"].inside.rest = Prism2.languages.rust;
  Prism2.languages.rust["attribute"].inside["string"] = Prism2.languages.rust["string"];
})(Prism);
(function(Prism2) {
  var comment = /\/\*[\s\S]*?\*\/|\/\/.*|#(?!\[).*/;
  var constant = [
    {
      pattern: /\b(?:false|true)\b/i,
      alias: "boolean"
    },
    {
      pattern: /(::\s*)\b[a-z_]\w*\b(?!\s*\()/i,
      greedy: true,
      lookbehind: true
    },
    {
      pattern: /(\b(?:case|const)\s+)\b[a-z_]\w*(?=\s*[;=])/i,
      greedy: true,
      lookbehind: true
    },
    /\b(?:null)\b/i,
    /\b[A-Z_][A-Z0-9_]*\b(?!\s*\()/
  ];
  var number = /\b0b[01]+(?:_[01]+)*\b|\b0o[0-7]+(?:_[0-7]+)*\b|\b0x[\da-f]+(?:_[\da-f]+)*\b|(?:\b\d+(?:_\d+)*\.?(?:\d+(?:_\d+)*)?|\B\.\d+)(?:e[+-]?\d+)?/i;
  var operator = /<?=>|\?\?=?|\.{3}|\??->|[!=]=?=?|::|\*\*=?|--|\+\+|&&|\|\||<<|>>|[?~]|[/^|%*&<>.+-]=?/;
  var punctuation2 = /[{}\[\](),:;]/;
  Prism2.languages.php = {
    "delimiter": {
      pattern: /\?>$|^<\?(?:php(?=\s)|=)?/i,
      alias: "important"
    },
    "comment": comment,
    "variable": /\$+(?:\w+\b|(?=\{))/,
    "package": {
      pattern: /(namespace\s+|use\s+(?:function\s+)?)(?:\\?\b[a-z_]\w*)+\b(?!\\)/i,
      lookbehind: true,
      inside: {
        "punctuation": /\\/
      }
    },
    "class-name-definition": {
      pattern: /(\b(?:class|enum|interface|trait)\s+)\b[a-z_]\w*(?!\\)\b/i,
      lookbehind: true,
      alias: "class-name"
    },
    "function-definition": {
      pattern: /(\bfunction\s+)[a-z_]\w*(?=\s*\()/i,
      lookbehind: true,
      alias: "function"
    },
    "keyword": [
      {
        pattern: /(\(\s*)\b(?:array|bool|boolean|float|int|integer|object|string)\b(?=\s*\))/i,
        alias: "type-casting",
        greedy: true,
        lookbehind: true
      },
      {
        pattern: /([(,?]\s*)\b(?:array(?!\s*\()|bool|callable|(?:false|null)(?=\s*\|)|float|int|iterable|mixed|object|self|static|string)\b(?=\s*\$)/i,
        alias: "type-hint",
        greedy: true,
        lookbehind: true
      },
      {
        pattern: /(\)\s*:\s*(?:\?\s*)?)\b(?:array(?!\s*\()|bool|callable|(?:false|null)(?=\s*\|)|float|int|iterable|mixed|never|object|self|static|string|void)\b/i,
        alias: "return-type",
        greedy: true,
        lookbehind: true
      },
      {
        pattern: /\b(?:array(?!\s*\()|bool|float|int|iterable|mixed|object|string|void)\b/i,
        alias: "type-declaration",
        greedy: true
      },
      {
        pattern: /(\|\s*)(?:false|null)\b|\b(?:false|null)(?=\s*\|)/i,
        alias: "type-declaration",
        greedy: true,
        lookbehind: true
      },
      {
        pattern: /\b(?:parent|self|static)(?=\s*::)/i,
        alias: "static-context",
        greedy: true
      },
      {
        // yield from
        pattern: /(\byield\s+)from\b/i,
        lookbehind: true
      },
      // `class` is always a keyword unlike other keywords
      /\bclass\b/i,
      {
        // https://www.php.net/manual/en/reserved.keywords.php
        //
        // keywords cannot be preceded by "->"
        // the complex lookbehind means `(?<!(?:->|::)\s*)`
        pattern: /((?:^|[^\s>:]|(?:^|[^-])>|(?:^|[^:]):)\s*)\b(?:abstract|and|array|as|break|callable|case|catch|clone|const|continue|declare|default|die|do|echo|else|elseif|empty|enddeclare|endfor|endforeach|endif|endswitch|endwhile|enum|eval|exit|extends|final|finally|fn|for|foreach|function|global|goto|if|implements|include|include_once|instanceof|insteadof|interface|isset|list|match|namespace|never|new|or|parent|print|private|protected|public|readonly|require|require_once|return|self|static|switch|throw|trait|try|unset|use|var|while|xor|yield|__halt_compiler)\b/i,
        lookbehind: true
      }
    ],
    "argument-name": {
      pattern: /([(,]\s*)\b[a-z_]\w*(?=\s*:(?!:))/i,
      lookbehind: true
    },
    "class-name": [
      {
        pattern: /(\b(?:extends|implements|instanceof|new(?!\s+self|\s+static))\s+|\bcatch\s*\()\b[a-z_]\w*(?!\\)\b/i,
        greedy: true,
        lookbehind: true
      },
      {
        pattern: /(\|\s*)\b[a-z_]\w*(?!\\)\b/i,
        greedy: true,
        lookbehind: true
      },
      {
        pattern: /\b[a-z_]\w*(?!\\)\b(?=\s*\|)/i,
        greedy: true
      },
      {
        pattern: /(\|\s*)(?:\\?\b[a-z_]\w*)+\b/i,
        alias: "class-name-fully-qualified",
        greedy: true,
        lookbehind: true,
        inside: {
          "punctuation": /\\/
        }
      },
      {
        pattern: /(?:\\?\b[a-z_]\w*)+\b(?=\s*\|)/i,
        alias: "class-name-fully-qualified",
        greedy: true,
        inside: {
          "punctuation": /\\/
        }
      },
      {
        pattern: /(\b(?:extends|implements|instanceof|new(?!\s+self\b|\s+static\b))\s+|\bcatch\s*\()(?:\\?\b[a-z_]\w*)+\b(?!\\)/i,
        alias: "class-name-fully-qualified",
        greedy: true,
        lookbehind: true,
        inside: {
          "punctuation": /\\/
        }
      },
      {
        pattern: /\b[a-z_]\w*(?=\s*\$)/i,
        alias: "type-declaration",
        greedy: true
      },
      {
        pattern: /(?:\\?\b[a-z_]\w*)+(?=\s*\$)/i,
        alias: ["class-name-fully-qualified", "type-declaration"],
        greedy: true,
        inside: {
          "punctuation": /\\/
        }
      },
      {
        pattern: /\b[a-z_]\w*(?=\s*::)/i,
        alias: "static-context",
        greedy: true
      },
      {
        pattern: /(?:\\?\b[a-z_]\w*)+(?=\s*::)/i,
        alias: ["class-name-fully-qualified", "static-context"],
        greedy: true,
        inside: {
          "punctuation": /\\/
        }
      },
      {
        pattern: /([(,?]\s*)[a-z_]\w*(?=\s*\$)/i,
        alias: "type-hint",
        greedy: true,
        lookbehind: true
      },
      {
        pattern: /([(,?]\s*)(?:\\?\b[a-z_]\w*)+(?=\s*\$)/i,
        alias: ["class-name-fully-qualified", "type-hint"],
        greedy: true,
        lookbehind: true,
        inside: {
          "punctuation": /\\/
        }
      },
      {
        pattern: /(\)\s*:\s*(?:\?\s*)?)\b[a-z_]\w*(?!\\)\b/i,
        alias: "return-type",
        greedy: true,
        lookbehind: true
      },
      {
        pattern: /(\)\s*:\s*(?:\?\s*)?)(?:\\?\b[a-z_]\w*)+\b(?!\\)/i,
        alias: ["class-name-fully-qualified", "return-type"],
        greedy: true,
        lookbehind: true,
        inside: {
          "punctuation": /\\/
        }
      }
    ],
    "constant": constant,
    "function": {
      pattern: /(^|[^\\\w])\\?[a-z_](?:[\w\\]*\w)?(?=\s*\()/i,
      lookbehind: true,
      inside: {
        "punctuation": /\\/
      }
    },
    "property": {
      pattern: /(->\s*)\w+/,
      lookbehind: true
    },
    "number": number,
    "operator": operator,
    "punctuation": punctuation2
  };
  var string_interpolation = {
    pattern: /\{\$(?:\{(?:\{[^{}]+\}|[^{}]+)\}|[^{}])+\}|(^|[^\\{])\$+(?:\w+(?:\[[^\r\n\[\]]+\]|->\w+)?)/,
    lookbehind: true,
    inside: Prism2.languages.php
  };
  var string = [
    {
      pattern: /<<<'([^']+)'[\r\n](?:.*[\r\n])*?\1;/,
      alias: "nowdoc-string",
      greedy: true,
      inside: {
        "delimiter": {
          pattern: /^<<<'[^']+'|[a-z_]\w*;$/i,
          alias: "symbol",
          inside: {
            "punctuation": /^<<<'?|[';]$/
          }
        }
      }
    },
    {
      pattern: /<<<(?:"([^"]+)"[\r\n](?:.*[\r\n])*?\1;|([a-z_]\w*)[\r\n](?:.*[\r\n])*?\2;)/i,
      alias: "heredoc-string",
      greedy: true,
      inside: {
        "delimiter": {
          pattern: /^<<<(?:"[^"]+"|[a-z_]\w*)|[a-z_]\w*;$/i,
          alias: "symbol",
          inside: {
            "punctuation": /^<<<"?|[";]$/
          }
        },
        "interpolation": string_interpolation
      }
    },
    {
      pattern: /`(?:\\[\s\S]|[^\\`])*`/,
      alias: "backtick-quoted-string",
      greedy: true
    },
    {
      pattern: /'(?:\\[\s\S]|[^\\'])*'/,
      alias: "single-quoted-string",
      greedy: true
    },
    {
      pattern: /"(?:\\[\s\S]|[^\\"])*"/,
      alias: "double-quoted-string",
      greedy: true,
      inside: {
        "interpolation": string_interpolation
      }
    }
  ];
  Prism2.languages.insertBefore("php", "variable", {
    "string": string,
    "attribute": {
      pattern: /#\[(?:[^"'\/#]|\/(?![*/])|\/\/.*$|#(?!\[).*$|\/\*(?:[^*]|\*(?!\/))*\*\/|"(?:\\[\s\S]|[^\\"])*"|'(?:\\[\s\S]|[^\\'])*')+\](?=\s*[a-z$#])/im,
      greedy: true,
      inside: {
        "attribute-content": {
          pattern: /^(#\[)[\s\S]+(?=\]$)/,
          lookbehind: true,
          // inside can appear subset of php
          inside: {
            "comment": comment,
            "string": string,
            "attribute-class-name": [
              {
                pattern: /([^:]|^)\b[a-z_]\w*(?!\\)\b/i,
                alias: "class-name",
                greedy: true,
                lookbehind: true
              },
              {
                pattern: /([^:]|^)(?:\\?\b[a-z_]\w*)+/i,
                alias: [
                  "class-name",
                  "class-name-fully-qualified"
                ],
                greedy: true,
                lookbehind: true,
                inside: {
                  "punctuation": /\\/
                }
              }
            ],
            "constant": constant,
            "number": number,
            "operator": operator,
            "punctuation": punctuation2
          }
        },
        "delimiter": {
          pattern: /^#\[|\]$/,
          alias: "punctuation"
        }
      }
    }
  });
  Prism2.hooks.add("before-tokenize", function(env) {
    if (!/<\?/.test(env.code)) {
      return;
    }
    var phpPattern = /<\?(?:[^"'/#]|\/(?![*/])|("|')(?:\\[\s\S]|(?!\1)[^\\])*\1|(?:\/\/|#(?!\[))(?:[^?\n\r]|\?(?!>))*(?=$|\?>|[\r\n])|#\[|\/\*(?:[^*]|\*(?!\/))*(?:\*\/|$))*?(?:\?>|$)/g;
    Prism2.languages["markup-templating"].buildPlaceholders(env, "php", phpPattern);
  });
  Prism2.hooks.add("after-tokenize", function(env) {
    Prism2.languages["markup-templating"].tokenizePlaceholders(env, "php");
  });
})(Prism);
(function(Prism2) {
  var anchorOrAlias = /[*&][^\s[\]{},]+/;
  var tag2 = /!(?:<[\w\-%#;/?:@&=+$,.!~*'()[\]]+>|(?:[a-zA-Z\d-]*!)?[\w\-%#;/?:@&=+$.~*'()]+)?/;
  var properties = "(?:" + tag2.source + "(?:[ 	]+" + anchorOrAlias.source + ")?|" + anchorOrAlias.source + "(?:[ 	]+" + tag2.source + ")?)";
  var plainKey = /(?:[^\s\x00-\x08\x0e-\x1f!"#%&'*,\-:>?@[\]`{|}\x7f-\x84\x86-\x9f\ud800-\udfff\ufffe\uffff]|[?:-]<PLAIN>)(?:[ \t]*(?:(?![#:])<PLAIN>|:<PLAIN>))*/.source.replace(/<PLAIN>/g, function() {
    return /[^\s\x00-\x08\x0e-\x1f,[\]{}\x7f-\x84\x86-\x9f\ud800-\udfff\ufffe\uffff]/.source;
  });
  var string = /"(?:[^"\\\r\n]|\\.)*"|'(?:[^'\\\r\n]|\\.)*'/.source;
  function createValuePattern(value, flags) {
    flags = (flags || "").replace(/m/g, "") + "m";
    var pattern = /([:\-,[{]\s*(?:\s<<prop>>[ \t]+)?)(?:<<value>>)(?=[ \t]*(?:$|,|\]|\}|(?:[\r\n]\s*)?#))/.source.replace(/<<prop>>/g, function() {
      return properties;
    }).replace(/<<value>>/g, function() {
      return value;
    });
    return RegExp(pattern, flags);
  }
  Prism2.languages.yaml = {
    "scalar": {
      pattern: RegExp(/([\-:]\s*(?:\s<<prop>>[ \t]+)?[|>])[ \t]*(?:((?:\r?\n|\r)[ \t]+)\S[^\r\n]*(?:\2[^\r\n]+)*)/.source.replace(/<<prop>>/g, function() {
        return properties;
      })),
      lookbehind: true,
      alias: "string"
    },
    "comment": /#.*/,
    "key": {
      pattern: RegExp(/((?:^|[:\-,[{\r\n?])[ \t]*(?:<<prop>>[ \t]+)?)<<key>>(?=\s*:\s)/.source.replace(/<<prop>>/g, function() {
        return properties;
      }).replace(/<<key>>/g, function() {
        return "(?:" + plainKey + "|" + string + ")";
      })),
      lookbehind: true,
      greedy: true,
      alias: "atrule"
    },
    "directive": {
      pattern: /(^[ \t]*)%.+/m,
      lookbehind: true,
      alias: "important"
    },
    "datetime": {
      pattern: createValuePattern(/\d{4}-\d\d?-\d\d?(?:[tT]|[ \t]+)\d\d?:\d{2}:\d{2}(?:\.\d*)?(?:[ \t]*(?:Z|[-+]\d\d?(?::\d{2})?))?|\d{4}-\d{2}-\d{2}|\d\d?:\d{2}(?::\d{2}(?:\.\d*)?)?/.source),
      lookbehind: true,
      alias: "number"
    },
    "boolean": {
      pattern: createValuePattern(/false|true/.source, "i"),
      lookbehind: true,
      alias: "important"
    },
    "null": {
      pattern: createValuePattern(/null|~/.source, "i"),
      lookbehind: true,
      alias: "important"
    },
    "string": {
      pattern: createValuePattern(string),
      lookbehind: true,
      greedy: true
    },
    "number": {
      pattern: createValuePattern(/[+-]?(?:0x[\da-f]+|0o[0-7]+|(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?|\.inf|\.nan)/.source, "i"),
      lookbehind: true
    },
    "tag": tag2,
    "important": anchorOrAlias,
    "punctuation": /---|[:[\]{}\-,|>?]|\.\.\./
  };
  Prism2.languages.yml = Prism2.languages.yaml;
})(Prism);
(function(Prism2) {
  function getPlaceholder(language, index) {
    return "___" + language.toUpperCase() + index + "___";
  }
  Object.defineProperties(Prism2.languages["markup-templating"] = {}, {
    buildPlaceholders: {
      /**
       * Tokenize all inline templating expressions matching `placeholderPattern`.
       *
       * If `replaceFilter` is provided, only matches of `placeholderPattern` for which `replaceFilter` returns
       * `true` will be replaced.
       *
       * @param {object} env The environment of the `before-tokenize` hook.
       * @param {string} language The language id.
       * @param {RegExp} placeholderPattern The matches of this pattern will be replaced by placeholders.
       * @param {(match: string) => boolean} [replaceFilter]
       */
      value: function(env, language, placeholderPattern, replaceFilter) {
        if (env.language !== language) {
          return;
        }
        var tokenStack = env.tokenStack = [];
        env.code = env.code.replace(placeholderPattern, function(match) {
          if (typeof replaceFilter === "function" && !replaceFilter(match)) {
            return match;
          }
          var i = tokenStack.length;
          var placeholder;
          while (env.code.indexOf(placeholder = getPlaceholder(language, i)) !== -1) {
            ++i;
          }
          tokenStack[i] = match;
          return placeholder;
        });
        env.grammar = Prism2.languages.markup;
      }
    },
    tokenizePlaceholders: {
      /**
       * Replace placeholders with proper tokens after tokenizing.
       *
       * @param {object} env The environment of the `after-tokenize` hook.
       * @param {string} language The language id.
       */
      value: function(env, language) {
        if (env.language !== language || !env.tokenStack) {
          return;
        }
        env.grammar = Prism2.languages[language];
        var j = 0;
        var keys = Object.keys(env.tokenStack);
        function walkTokens(tokens) {
          for (var i = 0; i < tokens.length; i++) {
            if (j >= keys.length) {
              break;
            }
            var token = tokens[i];
            if (typeof token === "string" || token.content && typeof token.content === "string") {
              var k = keys[j];
              var t = env.tokenStack[k];
              var s = typeof token === "string" ? token : token.content;
              var placeholder = getPlaceholder(language, k);
              var index = s.indexOf(placeholder);
              if (index > -1) {
                ++j;
                var before = s.substring(0, index);
                var middle = new Prism2.Token(language, Prism2.tokenize(t, env.grammar), "language-" + language, t);
                var after = s.substring(index + placeholder.length);
                var replacement = [];
                if (before) {
                  replacement.push.apply(replacement, walkTokens([before]));
                }
                replacement.push(middle);
                if (after) {
                  replacement.push.apply(replacement, walkTokens([after]));
                }
                if (typeof token === "string") {
                  tokens.splice.apply(tokens, [i, 1].concat(replacement));
                } else {
                  token.content = replacement;
                }
              }
            } else if (token.content) {
              walkTokens(token.content);
            }
          }
          return tokens;
        }
        walkTokens(env.tokens);
      }
    }
  });
})(Prism);
new BananaSlug();
const mergeMaps = (maps) => {
  const merged = {};
  for (let i = 0, l = maps.length; i < l; i++) {
    const map = maps[i];
    for (const key in map) {
      if (!merged[key]) {
        merged[key] = map[key];
      } else {
        merged[key] = merged[key].concat(map[key]);
      }
    }
  }
  return merged;
};
const HTML_ATTRIBUTES_ALLOW = [
  "abbr",
  "accept",
  "accept-charset",
  "accesskey",
  "action",
  "align",
  "alink",
  "allow",
  "allowfullscreen",
  "alt",
  "anchor",
  "archive",
  "as",
  "async",
  "autocapitalize",
  "autocomplete",
  "autocorrect",
  "autofocus",
  "autopictureinpicture",
  "autoplay",
  "axis",
  "background",
  "behavior",
  "bgcolor",
  "border",
  "bordercolor",
  "capture",
  "cellpadding",
  "cellspacing",
  "challenge",
  "char",
  "charoff",
  "charset",
  "checked",
  "cite",
  "class",
  "classid",
  "clear",
  "code",
  "codebase",
  "codetype",
  "color",
  "cols",
  "colspan",
  "compact",
  "content",
  "contenteditable",
  "controls",
  "controlslist",
  "conversiondestination",
  "coords",
  "crossorigin",
  "csp",
  "data",
  "datetime",
  "declare",
  "decoding",
  "default",
  "defer",
  "dir",
  "direction",
  "dirname",
  "disabled",
  "disablepictureinpicture",
  "disableremoteplayback",
  "disallowdocumentaccess",
  "download",
  "draggable",
  "elementtiming",
  "enctype",
  "end",
  "enterkeyhint",
  "event",
  "exportparts",
  "face",
  "for",
  "form",
  "formaction",
  "formenctype",
  "formmethod",
  "formnovalidate",
  "formtarget",
  "frame",
  "frameborder",
  "headers",
  "height",
  "hidden",
  "high",
  "href",
  "hreflang",
  "hreftranslate",
  "hspace",
  "http-equiv",
  "id",
  "imagesizes",
  "imagesrcset",
  "importance",
  "impressiondata",
  "impressionexpiry",
  "incremental",
  "inert",
  "inputmode",
  "integrity",
  "invisible",
  "ismap",
  "keytype",
  "kind",
  "label",
  "lang",
  "language",
  "latencyhint",
  "leftmargin",
  "link",
  "list",
  "loading",
  "longdesc",
  "loop",
  "low",
  "lowsrc",
  "manifest",
  "marginheight",
  "marginwidth",
  "max",
  "maxlength",
  "mayscript",
  "media",
  "method",
  "min",
  "minlength",
  "multiple",
  "muted",
  "name",
  "nohref",
  "nomodule",
  "nonce",
  "noresize",
  "noshade",
  "novalidate",
  "nowrap",
  "object",
  "open",
  "optimum",
  "part",
  "pattern",
  "ping",
  "placeholder",
  "playsinline",
  "policy",
  "poster",
  "preload",
  "pseudo",
  "readonly",
  "referrerpolicy",
  "rel",
  "reportingorigin",
  "required",
  "resources",
  "rev",
  "reversed",
  "role",
  "rows",
  "rowspan",
  "rules",
  "sandbox",
  "scheme",
  "scope",
  "scopes",
  "scrollamount",
  "scrolldelay",
  "scrolling",
  "select",
  "selected",
  "shadowroot",
  "shadowrootdelegatesfocus",
  "shape",
  "size",
  "sizes",
  "slot",
  "span",
  "spellcheck",
  "src",
  "srclang",
  "srcset",
  "standby",
  "start",
  "step",
  "style",
  "summary",
  "tabindex",
  "target",
  "text",
  "title",
  "topmargin",
  "translate",
  "truespeed",
  "trusttoken",
  "type",
  "usemap",
  "valign",
  "value",
  "valuetype",
  "version",
  "virtualkeyboardpolicy",
  "vlink",
  "vspace",
  "webkitdirectory",
  "width",
  "wrap"
];
const SVG_ATTRIBUTES_ALLOW = [
  "accent-height",
  "accumulate",
  "additive",
  "alignment-baseline",
  "ascent",
  "attributename",
  "attributetype",
  "azimuth",
  "basefrequency",
  "baseline-shift",
  "begin",
  "bias",
  "by",
  "class",
  "clip",
  "clippathunits",
  "clip-path",
  "clip-rule",
  "color",
  "color-interpolation",
  "color-interpolation-filters",
  "color-profile",
  "color-rendering",
  "cx",
  "cy",
  "d",
  "dx",
  "dy",
  "diffuseconstant",
  "direction",
  "display",
  "divisor",
  "dominant-baseline",
  "dur",
  "edgemode",
  "elevation",
  "end",
  "fill",
  "fill-opacity",
  "fill-rule",
  "filter",
  "filterunits",
  "flood-color",
  "flood-opacity",
  "font-family",
  "font-size",
  "font-size-adjust",
  "font-stretch",
  "font-style",
  "font-variant",
  "font-weight",
  "fx",
  "fy",
  "g1",
  "g2",
  "glyph-name",
  "glyphref",
  "gradientunits",
  "gradienttransform",
  "height",
  "href",
  "id",
  "image-rendering",
  "in",
  "in2",
  "k",
  "k1",
  "k2",
  "k3",
  "k4",
  "kerning",
  "keypoints",
  "keysplines",
  "keytimes",
  "lang",
  "lengthadjust",
  "letter-spacing",
  "kernelmatrix",
  "kernelunitlength",
  "lighting-color",
  "local",
  "marker-end",
  "marker-mid",
  "marker-start",
  "markerheight",
  "markerunits",
  "markerwidth",
  "maskcontentunits",
  "maskunits",
  "max",
  "mask",
  "media",
  "method",
  "mode",
  "min",
  "name",
  "numoctaves",
  "offset",
  "operator",
  "opacity",
  "order",
  "orient",
  "orientation",
  "origin",
  "overflow",
  "paint-order",
  "path",
  "pathlength",
  "patterncontentunits",
  "patterntransform",
  "patternunits",
  "points",
  "preservealpha",
  "preserveaspectratio",
  "primitiveunits",
  "r",
  "rx",
  "ry",
  "radius",
  "refx",
  "refy",
  "repeatcount",
  "repeatdur",
  "restart",
  "result",
  "rotate",
  "scale",
  "seed",
  "shape-rendering",
  "specularconstant",
  "specularexponent",
  "spreadmethod",
  "startoffset",
  "stddeviation",
  "stitchtiles",
  "stop-color",
  "stop-opacity",
  "stroke-dasharray",
  "stroke-dashoffset",
  "stroke-linecap",
  "stroke-linejoin",
  "stroke-miterlimit",
  "stroke-opacity",
  "stroke",
  "stroke-width",
  "style",
  "surfacescale",
  "systemlanguage",
  "tabindex",
  "targetx",
  "targety",
  "transform",
  "transform-origin",
  "text-anchor",
  "text-decoration",
  "text-rendering",
  "textlength",
  "type",
  "u1",
  "u2",
  "unicode",
  "values",
  "viewbox",
  "visibility",
  "version",
  "vert-adv-y",
  "vert-origin-x",
  "vert-origin-y",
  "width",
  "word-spacing",
  "wrap",
  "writing-mode",
  "xchannelselector",
  "ychannelselector",
  "x",
  "x1",
  "x2",
  "xmlns",
  "y",
  "y1",
  "y2",
  "z",
  "zoomandpan"
];
const MATH_ATTRIBUTES_ALLOW = [
  "accent",
  "accentunder",
  "align",
  "bevelled",
  "close",
  "columnsalign",
  "columnlines",
  "columnspan",
  "denomalign",
  "depth",
  "dir",
  "display",
  "displaystyle",
  "encoding",
  "fence",
  "frame",
  "height",
  "href",
  "id",
  "largeop",
  "length",
  "linethickness",
  "lspace",
  "lquote",
  "mathbackground",
  "mathcolor",
  "mathsize",
  "mathvariant",
  "maxsize",
  "minsize",
  "movablelimits",
  "notation",
  "numalign",
  "open",
  "rowalign",
  "rowlines",
  "rowspacing",
  "rowspan",
  "rspace",
  "rquote",
  "scriptlevel",
  "scriptminsize",
  "scriptsizemultiplier",
  "selection",
  "separator",
  "separators",
  "stretchy",
  "subscriptshift",
  "supscriptshift",
  "symmetric",
  "voffset",
  "width",
  "xmlns"
];
({
  allowAttributes: mergeMaps([
    Object.fromEntries(HTML_ATTRIBUTES_ALLOW.map((name) => [name, ["*"]])),
    Object.fromEntries(SVG_ATTRIBUTES_ALLOW.map((name) => [name, ["svg:*"]])),
    Object.fromEntries(MATH_ATTRIBUTES_ALLOW.map((name) => [name, ["math:*"]]))
  ])
});
const color_values = [
  { color: "red", primary: 600, secondary: 100 },
  { color: "green", primary: 600, secondary: 100 },
  { color: "blue", primary: 600, secondary: 100 },
  { color: "yellow", primary: 500, secondary: 100 },
  { color: "purple", primary: 600, secondary: 100 },
  { color: "teal", primary: 600, secondary: 100 },
  { color: "orange", primary: 600, secondary: 100 },
  { color: "cyan", primary: 600, secondary: 100 },
  { color: "lime", primary: 500, secondary: 100 },
  { color: "pink", primary: 600, secondary: 100 }
];
const tw_colors = {
  inherit: "inherit",
  current: "currentColor",
  transparent: "transparent",
  black: "#000",
  white: "#fff",
  slate: {
    50: "#f8fafc",
    100: "#f1f5f9",
    200: "#e2e8f0",
    300: "#cbd5e1",
    400: "#94a3b8",
    500: "#64748b",
    600: "#475569",
    700: "#334155",
    800: "#1e293b",
    900: "#0f172a",
    950: "#020617"
  },
  gray: {
    50: "#f9fafb",
    100: "#f3f4f6",
    200: "#e5e7eb",
    300: "#d1d5db",
    400: "#9ca3af",
    500: "#6b7280",
    600: "#4b5563",
    700: "#374151",
    800: "#1f2937",
    900: "#111827",
    950: "#030712"
  },
  zinc: {
    50: "#fafafa",
    100: "#f4f4f5",
    200: "#e4e4e7",
    300: "#d4d4d8",
    400: "#a1a1aa",
    500: "#71717a",
    600: "#52525b",
    700: "#3f3f46",
    800: "#27272a",
    900: "#18181b",
    950: "#09090b"
  },
  neutral: {
    50: "#fafafa",
    100: "#f5f5f5",
    200: "#e5e5e5",
    300: "#d4d4d4",
    400: "#a3a3a3",
    500: "#737373",
    600: "#525252",
    700: "#404040",
    800: "#262626",
    900: "#171717",
    950: "#0a0a0a"
  },
  stone: {
    50: "#fafaf9",
    100: "#f5f5f4",
    200: "#e7e5e4",
    300: "#d6d3d1",
    400: "#a8a29e",
    500: "#78716c",
    600: "#57534e",
    700: "#44403c",
    800: "#292524",
    900: "#1c1917",
    950: "#0c0a09"
  },
  red: {
    50: "#fef2f2",
    100: "#fee2e2",
    200: "#fecaca",
    300: "#fca5a5",
    400: "#f87171",
    500: "#ef4444",
    600: "#dc2626",
    700: "#b91c1c",
    800: "#991b1b",
    900: "#7f1d1d",
    950: "#450a0a"
  },
  orange: {
    50: "#fff7ed",
    100: "#ffedd5",
    200: "#fed7aa",
    300: "#fdba74",
    400: "#fb923c",
    500: "#f97316",
    600: "#ea580c",
    700: "#c2410c",
    800: "#9a3412",
    900: "#7c2d12",
    950: "#431407"
  },
  amber: {
    50: "#fffbeb",
    100: "#fef3c7",
    200: "#fde68a",
    300: "#fcd34d",
    400: "#fbbf24",
    500: "#f59e0b",
    600: "#d97706",
    700: "#b45309",
    800: "#92400e",
    900: "#78350f",
    950: "#451a03"
  },
  yellow: {
    50: "#fefce8",
    100: "#fef9c3",
    200: "#fef08a",
    300: "#fde047",
    400: "#facc15",
    500: "#eab308",
    600: "#ca8a04",
    700: "#a16207",
    800: "#854d0e",
    900: "#713f12",
    950: "#422006"
  },
  lime: {
    50: "#f7fee7",
    100: "#ecfccb",
    200: "#d9f99d",
    300: "#bef264",
    400: "#a3e635",
    500: "#84cc16",
    600: "#65a30d",
    700: "#4d7c0f",
    800: "#3f6212",
    900: "#365314",
    950: "#1a2e05"
  },
  green: {
    50: "#f0fdf4",
    100: "#dcfce7",
    200: "#bbf7d0",
    300: "#86efac",
    400: "#4ade80",
    500: "#22c55e",
    600: "#16a34a",
    700: "#15803d",
    800: "#166534",
    900: "#14532d",
    950: "#052e16"
  },
  emerald: {
    50: "#ecfdf5",
    100: "#d1fae5",
    200: "#a7f3d0",
    300: "#6ee7b7",
    400: "#34d399",
    500: "#10b981",
    600: "#059669",
    700: "#047857",
    800: "#065f46",
    900: "#064e3b",
    950: "#022c22"
  },
  teal: {
    50: "#f0fdfa",
    100: "#ccfbf1",
    200: "#99f6e4",
    300: "#5eead4",
    400: "#2dd4bf",
    500: "#14b8a6",
    600: "#0d9488",
    700: "#0f766e",
    800: "#115e59",
    900: "#134e4a",
    950: "#042f2e"
  },
  cyan: {
    50: "#ecfeff",
    100: "#cffafe",
    200: "#a5f3fc",
    300: "#67e8f9",
    400: "#22d3ee",
    500: "#06b6d4",
    600: "#0891b2",
    700: "#0e7490",
    800: "#155e75",
    900: "#164e63",
    950: "#083344"
  },
  sky: {
    50: "#f0f9ff",
    100: "#e0f2fe",
    200: "#bae6fd",
    300: "#7dd3fc",
    400: "#38bdf8",
    500: "#0ea5e9",
    600: "#0284c7",
    700: "#0369a1",
    800: "#075985",
    900: "#0c4a6e",
    950: "#082f49"
  },
  blue: {
    50: "#eff6ff",
    100: "#dbeafe",
    200: "#bfdbfe",
    300: "#93c5fd",
    400: "#60a5fa",
    500: "#3b82f6",
    600: "#2563eb",
    700: "#1d4ed8",
    800: "#1e40af",
    900: "#1e3a8a",
    950: "#172554"
  },
  indigo: {
    50: "#eef2ff",
    100: "#e0e7ff",
    200: "#c7d2fe",
    300: "#a5b4fc",
    400: "#818cf8",
    500: "#6366f1",
    600: "#4f46e5",
    700: "#4338ca",
    800: "#3730a3",
    900: "#312e81",
    950: "#1e1b4b"
  },
  violet: {
    50: "#f5f3ff",
    100: "#ede9fe",
    200: "#ddd6fe",
    300: "#c4b5fd",
    400: "#a78bfa",
    500: "#8b5cf6",
    600: "#7c3aed",
    700: "#6d28d9",
    800: "#5b21b6",
    900: "#4c1d95",
    950: "#2e1065"
  },
  purple: {
    50: "#faf5ff",
    100: "#f3e8ff",
    200: "#e9d5ff",
    300: "#d8b4fe",
    400: "#c084fc",
    500: "#a855f7",
    600: "#9333ea",
    700: "#7e22ce",
    800: "#6b21a8",
    900: "#581c87",
    950: "#3b0764"
  },
  fuchsia: {
    50: "#fdf4ff",
    100: "#fae8ff",
    200: "#f5d0fe",
    300: "#f0abfc",
    400: "#e879f9",
    500: "#d946ef",
    600: "#c026d3",
    700: "#a21caf",
    800: "#86198f",
    900: "#701a75",
    950: "#4a044e"
  },
  pink: {
    50: "#fdf2f8",
    100: "#fce7f3",
    200: "#fbcfe8",
    300: "#f9a8d4",
    400: "#f472b6",
    500: "#ec4899",
    600: "#db2777",
    700: "#be185d",
    800: "#9d174d",
    900: "#831843",
    950: "#500724"
  },
  rose: {
    50: "#fff1f2",
    100: "#ffe4e6",
    200: "#fecdd3",
    300: "#fda4af",
    400: "#fb7185",
    500: "#f43f5e",
    600: "#e11d48",
    700: "#be123c",
    800: "#9f1239",
    900: "#881337",
    950: "#4c0519"
  }
};
color_values.reduce(
  (acc, { color, primary, secondary }) => ({
    ...acc,
    [color]: {
      primary: tw_colors[color][primary],
      secondary: tw_colors[color][secondary]
    }
  }),
  {}
);
new Intl.Collator(0, { numeric: 1 }).compare;
typeof process !== "undefined" && process.versions && process.versions.node;
class TextLineStream extends TransformStream {
  /** Constructs a new instance. */
  constructor(options = { allowCR: false }) {
    super({
      transform: (chars, controller) => {
        chars = __privateGet(this, _currentLine) + chars;
        while (true) {
          const lfIndex = chars.indexOf("\n");
          const crIndex = options.allowCR ? chars.indexOf("\r") : -1;
          if (crIndex !== -1 && crIndex !== chars.length - 1 && (lfIndex === -1 || lfIndex - 1 > crIndex)) {
            controller.enqueue(chars.slice(0, crIndex));
            chars = chars.slice(crIndex + 1);
            continue;
          }
          if (lfIndex === -1)
            break;
          const endIndex = chars[lfIndex - 1] === "\r" ? lfIndex - 1 : lfIndex;
          controller.enqueue(chars.slice(0, endIndex));
          chars = chars.slice(lfIndex + 1);
        }
        __privateSet(this, _currentLine, chars);
      },
      flush: (controller) => {
        if (__privateGet(this, _currentLine) === "")
          return;
        const currentLine = options.allowCR && __privateGet(this, _currentLine).endsWith("\r") ? __privateGet(this, _currentLine).slice(0, -1) : __privateGet(this, _currentLine);
        controller.enqueue(currentLine);
      }
    });
    __privateAdd(this, _currentLine, "");
  }
}
_currentLine = /* @__PURE__ */ new WeakMap();
function SelectionButtons2($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let is_first_position, direction;
    let position = $$props["position"];
    let coords = $$props["coords"];
    let on_click = fallback($$props["on_click"], null);
    is_first_position = position === "column" ? coords[0] === 0 : coords[1] === 0;
    direction = position === "column" ? is_first_position ? "down" : "up" : is_first_position ? "right" : "left";
    $$renderer2.push(`<button${attr_class(`selection-button selection-button-${stringify(position)} ${stringify(is_first_position ? `move-${direction}` : "")}`, "svelte-fit6p6")}${attr("aria-label", `Select ${position}`)}><span${attr_class(clsx(direction), "svelte-fit6p6")}><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" class="svelte-fit6p6"><path d="m16.707 13.293-4-4a1 1 0 0 0-1.414 0l-4 4A1 1 0 0 0 8 15h8a1 1 0 0 0 .707-1.707z"${attr("data-name", direction)}></path></svg></span></button>`);
    bind_props($$props, { position, coords, on_click });
  });
}
function BooleanCell2($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let value = fallback($$props["value"], false);
    let editable = fallback($$props["editable"], true);
    let on_change = $$props["on_change"];
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      $$renderer3.push(`<div class="bool-cell svelte-1d51ufy" role="button" tabindex="-1">`);
      Checkbox($$renderer3, {
        label: "",
        interactive: editable,
        get value() {
          return value;
        },
        set value($$value) {
          value = $$value;
          $$settled = false;
        }
      });
      $$renderer3.push(`<!----></div>`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
    bind_props($$props, { value, editable, on_change });
  });
}
function EditableCell2($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let should_truncate, display_content, display_text;
    let edit2 = $$props["edit"];
    let value = fallback($$props["value"], "");
    let display_value = fallback($$props["display_value"], null);
    let styling = fallback($$props["styling"], "");
    let header = fallback($$props["header"], false);
    let datatype = fallback($$props["datatype"], "str");
    let latex_delimiters = $$props["latex_delimiters"];
    let line_breaks = fallback($$props["line_breaks"], true);
    let editable = fallback($$props["editable"], true);
    let is_static = fallback($$props["is_static"], false);
    let max_chars = fallback($$props["max_chars"], null);
    let components = fallback($$props["components"], () => ({}), true);
    let i18n = $$props["i18n"];
    let is_dragging = fallback($$props["is_dragging"], false);
    let wrap_text = fallback($$props["wrap_text"], false);
    let show_selection_buttons = fallback($$props["show_selection_buttons"], false);
    let coords = $$props["coords"];
    let on_select_column = fallback($$props["on_select_column"], null);
    let on_select_row = fallback($$props["on_select_row"], null);
    let el = $$props["el"];
    function truncate_text2(text2, max_length = null, is_image = false) {
      if (is_image) return String(text2);
      const str = String(text2);
      if (!max_length || max_length <= 0) return str;
      if (str.length <= max_length) return str;
      return str.slice(0, max_length) + "...";
    }
    function commit_change(checked) {
    }
    should_truncate = !edit2 && max_chars !== null && max_chars > 0;
    display_content = editable ? value : display_value !== null ? display_value : value;
    display_text = should_truncate ? truncate_text2(display_content, max_chars, datatype === "image") : display_content;
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      if (edit2 && datatype !== "bool") {
        $$renderer3.push("<!--[-->");
        $$renderer3.push(`<textarea${attr("readonly", is_static, true)}${attr("aria-readonly", is_static)}${attr("aria-label", is_static ? "Cell is read-only" : "Edit cell")} tabindex="-1"${attr_class("svelte-odwpey", void 0, { "header": header })}>`);
        const $$body = escape_html(value);
        if ($$body) {
          $$renderer3.push(`${$$body}`);
        }
        $$renderer3.push(`</textarea>`);
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--> `);
      if (datatype === "bool" && typeof value === "boolean") {
        $$renderer3.push("<!--[-->");
        BooleanCell2($$renderer3, {
          editable,
          on_change: commit_change,
          get value() {
            return value;
          },
          set value($$value) {
            value = $$value;
            $$settled = false;
          }
        });
      } else {
        $$renderer3.push("<!--[!-->");
        $$renderer3.push(`<span tabindex="0" role="button"${attr_style(styling)}${attr("data-editable", editable)}${attr("data-max-chars", max_chars)}${attr("data-expanded", edit2)} placeholder=" "${attr_class("svelte-odwpey", void 0, {
          "dragging": is_dragging,
          "edit": edit2,
          "expanded": edit2,
          "multiline": header,
          "text": datatype === "str",
          "wrap": wrap_text
        })}>`);
        if (datatype === "image" && components.image) {
          $$renderer3.push("<!--[-->");
          $$renderer3.push(`<!---->`);
          components.image?.($$renderer3, {
            value: { url: display_text },
            show_label: false,
            label: "cell-image",
            show_download_button: false,
            i18n,
            gradio: { dispatch: () => {
            } }
          });
          $$renderer3.push(`<!---->`);
        } else {
          $$renderer3.push("<!--[!-->");
          if (datatype === "html") {
            $$renderer3.push("<!--[-->");
            $$renderer3.push(`${html$1(display_text)}`);
          } else {
            $$renderer3.push("<!--[!-->");
            if (datatype === "markdown") {
              $$renderer3.push("<!--[-->");
              MarkdownCode($$renderer3, {
                message: display_text.toLocaleString(),
                latex_delimiters,
                line_breaks,
                chatbot: false
              });
            } else {
              $$renderer3.push("<!--[!-->");
              $$renderer3.push(`${escape_html(display_text)}`);
            }
            $$renderer3.push(`<!--]-->`);
          }
          $$renderer3.push(`<!--]-->`);
        }
        $$renderer3.push(`<!--]--></span>`);
      }
      $$renderer3.push(`<!--]--> `);
      if (show_selection_buttons && coords && on_select_column && on_select_row) {
        $$renderer3.push("<!--[-->");
        SelectionButtons2($$renderer3, {
          position: "column",
          coords,
          on_click: () => on_select_column(coords[1])
        });
        $$renderer3.push(`<!----> `);
        SelectionButtons2($$renderer3, {
          position: "row",
          coords,
          on_click: () => on_select_row(coords[0])
        });
        $$renderer3.push(`<!---->`);
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]-->`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
    bind_props($$props, {
      edit: edit2,
      value,
      display_value,
      styling,
      header,
      datatype,
      latex_delimiters,
      line_breaks,
      editable,
      is_static,
      max_chars,
      components,
      i18n,
      is_dragging,
      wrap_text,
      show_selection_buttons,
      coords,
      on_select_column,
      on_select_row,
      el
    });
  });
}
function RowNumber2($$renderer, $$props) {
  let index = fallback($$props["index"], null);
  let is_header = fallback($$props["is_header"], false);
  if (is_header) {
    $$renderer.push("<!--[-->");
    $$renderer.push(`<th tabindex="-1" class="row-number svelte-11p754p"><div class="cell-wrap"><div class="header-content"><div class="header-text"></div></div></div></th>`);
  } else {
    $$renderer.push("<!--[!-->");
    $$renderer.push(`<td class="row-number svelte-11p754p" tabindex="-1"${attr("data-row", index)} data-col="row-number">${escape_html(index !== null ? index + 1 : "")}</td>`);
  }
  $$renderer.push(`<!--]-->`);
  bind_props($$props, { index, is_header });
}
function CellMenuButton2($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let on_click = $$props["on_click"];
    $$renderer2.push(`<button aria-label="Open cell menu" class="cell-menu-button svelte-j6wkfp" aria-haspopup="menu">⋮</button>`);
    bind_props($$props, { on_click });
  });
}
function filter_data(data, filter_columns) {
  if (!data || !data.length || !data[0]) {
    return [];
  }
  let row_indices = [...Array(data.length)].map((_, i) => i);
  if (filter_columns.length > 0) {
    filter_columns.forEach((column) => {
      if (column.datatype === "string") {
        switch (column.filter) {
          case "Contains":
            row_indices = row_indices.filter(
              (i) => data[i][column.col]?.value.toString().includes(column.value)
            );
            break;
          case "Does not contain":
            row_indices = row_indices.filter(
              (i) => !data[i][column.col]?.value.toString().includes(column.value)
            );
            break;
          case "Starts with":
            row_indices = row_indices.filter(
              (i) => data[i][column.col]?.value.toString().startsWith(column.value)
            );
            break;
          case "Ends with":
            row_indices = row_indices.filter(
              (i) => data[i][column.col]?.value.toString().endsWith(column.value)
            );
            break;
          case "Is":
            row_indices = row_indices.filter(
              (i) => data[i][column.col]?.value.toString() === column.value
            );
            break;
          case "Is not":
            row_indices = row_indices.filter(
              (i) => !(data[i][column.col]?.value.toString() === column.value)
            );
            break;
          case "Is empty":
            row_indices = row_indices.filter(
              (i) => data[i][column.col]?.value.toString() === ""
            );
            break;
          case "Is not empty":
            row_indices = row_indices.filter(
              (i) => !(data[i][column.col]?.value.toString() === "")
            );
            break;
        }
      } else if (column.datatype === "number") {
        switch (column.filter) {
          case "=":
            row_indices = row_indices.filter((i) => {
              if (!isNaN(Number(data[i][column.col]?.value)) && !isNaN(Number(column.value))) {
                return Number(data[i][column.col]?.value) === Number(column.value);
              }
              return false;
            });
            break;
          case "≠":
            row_indices = row_indices.filter((i) => {
              if (!isNaN(Number(data[i][column.col]?.value)) && !isNaN(Number(column.value))) {
                return !(Number(data[i][column.col]?.value) === Number(column.value));
              }
              return false;
            });
            break;
          case ">":
            row_indices = row_indices.filter((i) => {
              if (!isNaN(Number(data[i][column.col]?.value)) && !isNaN(Number(column.value))) {
                return Number(data[i][column.col]?.value) > Number(column.value);
              }
              return false;
            });
            break;
          case "<":
            row_indices = row_indices.filter((i) => {
              if (!isNaN(Number(data[i][column.col]?.value)) && !isNaN(Number(column.value))) {
                return Number(data[i][column.col]?.value) < Number(column.value);
              }
              return false;
            });
            break;
          case "≥":
            row_indices = row_indices.filter((i) => {
              if (!isNaN(Number(data[i][column.col]?.value)) && !isNaN(Number(column.value))) {
                return Number(data[i][column.col]?.value) >= Number(column.value);
              }
              return false;
            });
            break;
          case "≤":
            row_indices = row_indices.filter((i) => {
              if (!isNaN(Number(data[i][column.col]?.value)) && !isNaN(Number(column.value))) {
                return Number(data[i][column.col]?.value) <= Number(column.value);
              }
              return false;
            });
            break;
          case "Is empty":
            row_indices = row_indices.filter(
              (i) => data[i][column.col]?.value.toString() === ""
            );
            break;
          case "Is not empty":
            row_indices = row_indices.filter((i) => {
              if (!isNaN(Number(data[i][column.col]?.value))) {
                return !(data[i][column.col]?.value.toString() === "");
              }
              return false;
            });
            break;
        }
      }
    });
    return row_indices;
  }
  return [...Array(data.length)].map((_, i) => i);
}
function filter_data_and_preserve_selection(data, display_value, styling, filter_columns, selected, get_current_indices2, original_data, original_display_value, original_styling) {
  let id = null;
  if (selected && selected[0] in data && selected[1] in data[selected[0]]) {
    id = data[selected[0]][selected[1]].id;
  }
  filter_table_data(
    data,
    display_value,
    styling,
    filter_columns,
    original_data,
    original_display_value,
    original_styling
  );
  let new_selected = selected;
  if (id) {
    const [i, j] = get_current_indices2(id, data);
    new_selected = [i, j];
  }
  return { data, selected: new_selected };
}
function get_max(data) {
  if (!data || !data.length) return [];
  let max = data[0].slice();
  for (let i = 0; i < data.length; i++) {
    for (let j = 0; j < data[i].length; j++) {
      if (`${max[j].value}`.length < `${data[i][j].value}`.length) {
        max[j] = data[i][j];
      }
    }
  }
  return max;
}
function sort_table_data(data, display_value, styling, sort_columns) {
  if (!sort_columns.length) return;
  if (!data || !data.length) return;
  const indices = sort_data(data, sort_columns);
  const new_data = indices.map((i) => data[i]);
  data.splice(0, data.length, ...new_data);
  if (display_value) {
    const new_display = indices.map((i) => display_value[i]);
    display_value.splice(0, display_value.length, ...new_display);
  }
  if (styling) {
    const new_styling = indices.map((i) => styling[i]);
    styling.splice(0, styling.length, ...new_styling);
  }
}
function filter_table_data(data, display_value, styling, filter_columns, original_data, original_display_value, original_styling) {
  const base_data = original_data ?? data;
  const base_display_value = original_display_value ?? display_value;
  const base_styling = original_styling ?? styling;
  if (!filter_columns.length) {
    data.splice(0, data.length, ...base_data.map((row) => [...row]));
    if (display_value && base_display_value) {
      display_value.splice(
        0,
        display_value.length,
        ...base_display_value.map((row) => [...row])
      );
    }
    if (styling && base_styling) {
      styling.splice(0, styling.length, ...base_styling.map((row) => [...row]));
    }
    return;
  }
  if (!data || !data.length) return;
  const indices = filter_data(base_data, filter_columns);
  const new_data = indices.map((i) => base_data[i]);
  data.splice(0, data.length, ...new_data);
  if (display_value && base_display_value) {
    const new_display = indices.map((i) => base_display_value[i]);
    display_value.splice(0, display_value.length, ...new_display);
  }
  if (styling && base_styling) {
    const new_styling = indices.map((i) => base_styling[i]);
    styling.splice(0, styling.length, ...new_styling);
  }
}
async function copy_table_data(data, selected_cells) {
  if (!data || !data.length) return;
  const cells_to_copy = data.flatMap((row, r) => row.map((_, c) => [r, c]));
  const csv = cells_to_copy.reduce(
    (acc, [row, col]) => {
      acc[row] = acc[row] || {};
      const value = String(data[row][col].value);
      acc[row][col] = value.includes(",") || value.includes('"') || value.includes("\n") ? `"${value.replace(/"/g, '""')}"` : value;
      return acc;
    },
    {}
  );
  const rows = Object.keys(csv).sort((a, b) => +a - +b);
  if (!rows.length) return;
  const cols = Object.keys(csv[rows[0]]).sort((a, b) => +a - +b);
  const text2 = rows.map((r) => cols.map((c) => csv[r][c] || "").join(",")).join("\n");
  try {
    await navigator.clipboard.writeText(text2);
  } catch (err) {
    throw new Error("Failed to copy to clipboard: " + err.message);
  }
}
function get_sort_status(name, sort_columns, headers) {
  if (!sort_columns.length) return "none";
  const sort_item = sort_columns.find((item) => {
    const col = item.col;
    if (col < 0 || col >= headers.length) return false;
    return headers[col] === name;
  });
  if (!sort_item) return "none";
  return sort_item.direction;
}
function sort_data(data, sort_columns) {
  if (!data || !data.length || !data[0]) {
    return [];
  }
  if (sort_columns.length > 0) {
    const row_indices = [...Array(data.length)].map((_, i) => i);
    row_indices.sort((row_a_idx, row_b_idx) => {
      const row_a = data[row_a_idx];
      const row_b = data[row_b_idx];
      for (const { col: sort_by, direction } of sort_columns) {
        if (!row_a || !row_b || sort_by < 0 || sort_by >= row_a.length || sort_by >= row_b.length || !row_a[sort_by] || !row_b[sort_by]) {
          continue;
        }
        const val_a = row_a[sort_by].value;
        const val_b = row_b[sort_by].value;
        const comparison = val_a < val_b ? -1 : val_a > val_b ? 1 : 0;
        if (comparison !== 0) {
          return direction === "asc" ? comparison : -comparison;
        }
      }
      return 0;
    });
    return row_indices;
  }
  return [...Array(data.length)].map((_, i) => i);
}
function sort_data_and_preserve_selection(data, display_value, styling, sort_columns, selected, get_current_indices2) {
  let id = null;
  if (selected && selected[0] in data && selected[1] in data[selected[0]]) {
    id = data[selected[0]][selected[1]].id;
  }
  sort_table_data(data, display_value, styling, sort_columns);
  let new_selected = selected;
  if (id) {
    const [i, j] = get_current_indices2(id, data);
    new_selected = [i, j];
  }
  return { data, selected: new_selected };
}
function Padlock2($$renderer) {
  $$renderer.push(`<div class="wrapper svelte-4dxgzr" aria-label="Static column"><svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect><path d="M7 11V7a5 5 0 0 1 10 0v4"></path></svg></div>`);
}
function SortArrowUp2($$renderer, $$props) {
  let size = fallback($$props["size"], 16);
  $$renderer.push(`<svg${attr("width", size)}${attr("height", size)} viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M4 8L8 4L12 8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path><path d="M8 4V12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"></path></svg>`);
  bind_props($$props, { size });
}
function SortArrowDown2($$renderer, $$props) {
  let size = fallback($$props["size"], 16);
  $$renderer.push(`<svg${attr("width", size)}${attr("height", size)} viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M4 8L8 12L12 8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path><path d="M8 12V4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"></path></svg>`);
  bind_props($$props, { size });
}
function CellMenuIcons2($$renderer, $$props) {
  let icon = $$props["icon"];
  if (icon == "add-column-right") {
    $$renderer.push("<!--[-->");
    $$renderer.push(`<svg viewBox="0 0 24 24" width="16" height="16"><rect x="4" y="6" width="4" height="12" stroke="currentColor" stroke-width="2" fill="none"></rect><path d="M12 12H19M16 8L19 12L16 16" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round"></path></svg>`);
  } else {
    $$renderer.push("<!--[!-->");
    if (icon == "add-column-left") {
      $$renderer.push("<!--[-->");
      $$renderer.push(`<svg viewBox="0 0 24 24" width="16" height="16"><rect x="16" y="6" width="4" height="12" stroke="currentColor" stroke-width="2" fill="none"></rect><path d="M12 12H5M8 8L5 12L8 16" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round"></path></svg>`);
    } else {
      $$renderer.push("<!--[!-->");
      if (icon == "add-row-above") {
        $$renderer.push("<!--[-->");
        $$renderer.push(`<svg viewBox="0 0 24 24" width="16" height="16"><rect x="6" y="16" width="12" height="4" stroke="currentColor" stroke-width="2"></rect><path d="M12 12V5M8 8L12 5L16 8" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round"></path></svg>`);
      } else {
        $$renderer.push("<!--[!-->");
        if (icon == "add-row-below") {
          $$renderer.push("<!--[-->");
          $$renderer.push(`<svg viewBox="0 0 24 24" width="16" height="16"><rect x="6" y="4" width="12" height="4" stroke="currentColor" stroke-width="2"></rect><path d="M12 12V19M8 16L12 19L16 16" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round"></path></svg>`);
        } else {
          $$renderer.push("<!--[!-->");
          if (icon == "delete-row") {
            $$renderer.push("<!--[-->");
            $$renderer.push(`<svg viewBox="0 0 24 24" width="16" height="16"><rect x="5" y="10" width="14" height="4" stroke="currentColor" stroke-width="2"></rect><path d="M8 7L16 17M16 7L8 17" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path></svg>`);
          } else {
            $$renderer.push("<!--[!-->");
            if (icon == "delete-column") {
              $$renderer.push("<!--[-->");
              $$renderer.push(`<svg viewBox="0 0 24 24" width="16" height="16"><rect x="10" y="5" width="4" height="14" stroke="currentColor" stroke-width="2"></rect><path d="M7 8L17 16M17 8L7 16" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path></svg>`);
            } else {
              $$renderer.push("<!--[!-->");
              if (icon == "sort-asc") {
                $$renderer.push("<!--[-->");
                $$renderer.push(`<svg viewBox="0 0 24 24" width="16" height="16"><path d="M8 16L12 12L16 16" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"></path><path d="M12 12V19" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path><path d="M5 7H19" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path></svg>`);
              } else {
                $$renderer.push("<!--[!-->");
                if (icon == "sort-desc") {
                  $$renderer.push("<!--[-->");
                  $$renderer.push(`<svg viewBox="0 0 24 24" width="16" height="16"><path d="M8 12L12 16L16 12" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"></path><path d="M12 16V9" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path><path d="M5 5H19" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path></svg>`);
                } else {
                  $$renderer.push("<!--[!-->");
                  if (icon == "clear-sort") {
                    $$renderer.push("<!--[-->");
                    $$renderer.push(`<svg viewBox="0 0 24 24" width="16" height="16"><path d="M5 5H19" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path><path d="M5 9H15" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path><path d="M5 13H11" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path><path d="M5 17H7" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path><path d="M17 17L21 21M21 17L17 21" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path></svg>`);
                  } else {
                    $$renderer.push("<!--[!-->");
                    if (icon == "filter") {
                      $$renderer.push("<!--[-->");
                      $$renderer.push(`<svg viewBox="0 0 24 24" width="16" height="16"><path d="M5 5H19" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path><path d="M8 9H16" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path><path d="M11 13H13" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path></svg>`);
                    } else {
                      $$renderer.push("<!--[!-->");
                      if (icon == "clear-filter") {
                        $$renderer.push("<!--[-->");
                        $$renderer.push(`<svg viewBox="0 0 24 24" width="16" height="16"><path d="M5 5H19" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path><path d="M8 9H16" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path><path d="M11 13H13" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path><path d="M17 17L21 21M21 17L17 21" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path></svg>`);
                      } else {
                        $$renderer.push("<!--[!-->");
                      }
                      $$renderer.push(`<!--]-->`);
                    }
                    $$renderer.push(`<!--]-->`);
                  }
                  $$renderer.push(`<!--]-->`);
                }
                $$renderer.push(`<!--]-->`);
              }
              $$renderer.push(`<!--]-->`);
            }
            $$renderer.push(`<!--]-->`);
          }
          $$renderer.push(`<!--]-->`);
        }
        $$renderer.push(`<!--]-->`);
      }
      $$renderer.push(`<!--]-->`);
    }
    $$renderer.push(`<!--]-->`);
  }
  $$renderer.push(`<!--]-->`);
  bind_props($$props, { icon });
}
function TableHeader2($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let can_add_columns, is_bool_column, select_all_state, sort_index, filter_index, sort_priority, current_direction;
    let value = $$props["value"];
    let i = $$props["i"];
    let datatype = fallback($$props["datatype"], "str");
    let actual_pinned_columns = $$props["actual_pinned_columns"];
    let header_edit = $$props["header_edit"];
    let selected_header = $$props["selected_header"];
    let headers = $$props["headers"];
    let get_cell_width2 = $$props["get_cell_width"];
    let handle_header_click = $$props["handle_header_click"];
    let toggle_header_menu = $$props["toggle_header_menu"];
    let end_header_edit = $$props["end_header_edit"];
    let sort_columns = fallback($$props["sort_columns"], () => [], true);
    let filter_columns = fallback($$props["filter_columns"], () => [], true);
    let latex_delimiters = $$props["latex_delimiters"];
    let line_breaks = $$props["line_breaks"];
    let max_chars = $$props["max_chars"];
    let editable = $$props["editable"];
    let i18n = $$props["i18n"];
    let el = $$props["el"];
    let is_static = $$props["is_static"];
    let col_count = $$props["col_count"];
    let data = fallback($$props["data"], () => [], true);
    let on_select_all = fallback($$props["on_select_all"], void 0);
    function get_header_position(col_index) {
      if (col_index >= actual_pinned_columns) {
        return "auto";
      }
      if (col_index === 0) {
        return "0";
      }
      const previous_widths = Array(col_index).fill(0).map((_, idx) => {
        return get_cell_width2(idx);
      }).join(" + ");
      return `calc(${previous_widths})`;
    }
    can_add_columns = col_count && col_count[1] === "dynamic";
    is_bool_column = datatype === "bool";
    select_all_state = (() => {
      if (!is_bool_column || data.length === 0) return "unchecked";
      const true_count = data.filter((row) => row[i]?.value === true || row[i]?.value === "true").length;
      if (true_count === 0) return "unchecked";
      if (true_count === data.length) return "checked";
      return "indeterminate";
    })();
    sort_index = sort_columns.findIndex((item) => item.col === i);
    filter_index = filter_columns.findIndex((item) => item.col === i);
    sort_priority = sort_index !== -1 ? sort_index + 1 : null;
    current_direction = sort_index !== -1 ? sort_columns[sort_index].direction : null;
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      $$renderer3.push(`<th${attr("aria-sort", get_sort_status(value, sort_columns, headers) === "none" ? "none" : get_sort_status(value, sort_columns, headers) === "asc" ? "ascending" : "descending")}${attr_style(`width: ${stringify(get_cell_width2(i))}; left: ${stringify(get_header_position(i))};`)}${attr("title", value)}${attr_class("svelte-1yuystx", void 0, {
        "pinned-column": i < actual_pinned_columns,
        "last-pinned": i === actual_pinned_columns - 1,
        "focus": header_edit === i || selected_header === i,
        "sorted": sort_index !== -1,
        "filtered": filter_index !== -1
      })}><div class="cell-wrap svelte-1yuystx"><div class="header-content svelte-1yuystx">`);
      if (is_bool_column && editable && on_select_all) {
        $$renderer3.push("<!--[-->");
        $$renderer3.push(`<div class="select-all-checkbox svelte-1yuystx" role="button" tabindex="0">`);
        Checkbox($$renderer3, {
          value: select_all_state === "checked",
          indeterminate: select_all_state === "indeterminate",
          label: "",
          interactive: true
        });
        $$renderer3.push(`<!----></div>`);
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--> <button class="header-button svelte-1yuystx"${attr("title", value)}>`);
      EditableCell2($$renderer3, {
        max_chars,
        latex_delimiters,
        line_breaks,
        edit: header_edit === i,
        header: true,
        editable,
        is_static,
        i18n,
        coords: [i, 0],
        get value() {
          return value;
        },
        set value($$value) {
          value = $$value;
          $$settled = false;
        },
        get el() {
          return el;
        },
        set el($$value) {
          el = $$value;
          $$settled = false;
        }
      });
      $$renderer3.push(`<!----> `);
      if (sort_index !== -1) {
        $$renderer3.push("<!--[-->");
        $$renderer3.push(`<div class="sort-indicators svelte-1yuystx"><span class="sort-arrow svelte-1yuystx">`);
        if (current_direction === "asc") {
          $$renderer3.push("<!--[-->");
          SortArrowUp2($$renderer3, { size: 12 });
        } else {
          $$renderer3.push("<!--[!-->");
          SortArrowDown2($$renderer3, { size: 12 });
        }
        $$renderer3.push(`<!--]--></span> `);
        if (sort_columns.length > 1) {
          $$renderer3.push("<!--[-->");
          $$renderer3.push(`<span class="sort-priority svelte-1yuystx">${escape_html(sort_priority)}</span>`);
        } else {
          $$renderer3.push("<!--[!-->");
        }
        $$renderer3.push(`<!--]--></div>`);
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--> `);
      if (filter_index !== -1) {
        $$renderer3.push("<!--[-->");
        $$renderer3.push(`<div class="filter-indicators svelte-1yuystx"><span class="filter-icon svelte-1yuystx">`);
        CellMenuIcons2($$renderer3, { icon: "filter" });
        $$renderer3.push(`<!----></span></div>`);
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--></button> `);
      if (is_static) {
        $$renderer3.push("<!--[-->");
        Padlock2($$renderer3);
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--></div> `);
      if (can_add_columns) {
        $$renderer3.push("<!--[-->");
        CellMenuButton2($$renderer3, { on_click: (event) => toggle_header_menu(event, i) });
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--></div></th>`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
    bind_props($$props, {
      value,
      i,
      datatype,
      actual_pinned_columns,
      header_edit,
      selected_header,
      headers,
      get_cell_width: get_cell_width2,
      handle_header_click,
      toggle_header_menu,
      end_header_edit,
      sort_columns,
      filter_columns,
      latex_delimiters,
      line_breaks,
      max_chars,
      editable,
      i18n,
      el,
      is_static,
      col_count,
      data,
      on_select_all
    });
  });
}
function is_cell_in_selection(coords, selected_cells) {
  const [row, col] = coords;
  return selected_cells.some(([r, c]) => r === row && c === col);
}
function is_cell_selected(cell, selected_cells) {
  const [row, col] = cell;
  if (!selected_cells.some(([r, c]) => r === row && c === col)) return "";
  const up = selected_cells.some(([r, c]) => r === row - 1 && c === col);
  const down = selected_cells.some(([r, c]) => r === row + 1 && c === col);
  const left = selected_cells.some(([r, c]) => r === row && c === col - 1);
  const right = selected_cells.some(([r, c]) => r === row && c === col + 1);
  return `cell-selected${up ? " no-top" : ""}${down ? " no-bottom" : ""}${left ? " no-left" : ""}${right ? " no-right" : ""}`;
}
function get_range_selection(start, end) {
  const [start_row, start_col] = start;
  const [end_row, end_col] = end;
  const min_row = Math.min(start_row, end_row);
  const max_row = Math.max(start_row, end_row);
  const min_col = Math.min(start_col, end_col);
  const max_col = Math.max(start_col, end_col);
  const cells = [];
  cells.push(start);
  for (let i = min_row; i <= max_row; i++) {
    for (let j = min_col; j <= max_col; j++) {
      if (i === start_row && j === start_col) continue;
      cells.push([i, j]);
    }
  }
  return cells;
}
function handle_selection(current, selected_cells, event) {
  if (event.shiftKey && selected_cells.length > 0) {
    return get_range_selection(
      selected_cells[selected_cells.length - 1],
      current
    );
  }
  if (event.metaKey || event.ctrlKey) {
    const is_cell_match = ([r, c]) => r === current[0] && c === current[1];
    const index = selected_cells.findIndex(is_cell_match);
    return index === -1 ? [...selected_cells, current] : selected_cells.filter((_, i) => i !== index);
  }
  return [current];
}
function should_show_cell_menu(cell, selected_cells, editable) {
  const [row, col] = cell;
  return editable && selected_cells.length === 1 && selected_cells[0][0] === row && selected_cells[0][1] === col;
}
function get_next_cell_coordinates(current, data, shift_key) {
  const [row, col] = current;
  const direction = shift_key ? -1 : 1;
  if (data[row]?.[col + direction]) {
    return [row, col + direction];
  }
  const next_row = row + (direction > 0 ? 1 : 0);
  const prev_row = row + (direction < 0 ? -1 : 0);
  if (direction > 0 && data[next_row]?.[0]) {
    return [next_row, 0];
  }
  if (direction < 0 && data[prev_row]?.[data[0].length - 1]) {
    return [prev_row, data[0].length - 1];
  }
  return false;
}
function move_cursor(event, current_coords, data) {
  const key = event.key;
  const dir = {
    ArrowRight: [0, 1],
    ArrowLeft: [0, -1],
    ArrowDown: [1, 0],
    ArrowUp: [-1, 0]
  }[key];
  let i, j;
  if (event.metaKey || event.ctrlKey) {
    if (key === "ArrowRight") {
      i = current_coords[0];
      j = data[0].length - 1;
    } else if (key === "ArrowLeft") {
      i = current_coords[0];
      j = 0;
    } else if (key === "ArrowDown") {
      i = data.length - 1;
      j = current_coords[1];
    } else if (key === "ArrowUp") {
      i = 0;
      j = current_coords[1];
    } else {
      return false;
    }
  } else {
    i = current_coords[0] + dir[0];
    j = current_coords[1] + dir[1];
  }
  if (i < 0 && j <= 0) {
    return false;
  }
  const is_data = data[i]?.[j];
  if (is_data) {
    return [i, j];
  }
  return false;
}
function get_current_indices(id, data) {
  return data.reduce(
    (acc, arr, i) => {
      const j = arr.reduce(
        (_acc, _data, k) => id === _data.id ? k : _acc,
        -1
      );
      return j === -1 ? acc : [i, j];
    },
    [-1, -1]
  );
}
function calculate_selection_positions(selected, data, els, parent, table) {
  const [row, col] = selected;
  if (!data[row]?.[col]) {
    return { col_pos: "0px", row_pos: void 0 };
  }
  const cell_id = data[row][col].id;
  const cell_el = els[cell_id]?.cell;
  if (!cell_el) {
    return { col_pos: "0px", row_pos: void 0 };
  }
  const cell_rect = cell_el.getBoundingClientRect();
  const table_rect = table.getBoundingClientRect();
  const col_pos = `${cell_rect.left - table_rect.left + cell_rect.width / 2}px`;
  const row_pos = `${cell_rect.top - table_rect.top + cell_rect.height / 2}px`;
  return { col_pos, row_pos };
}
function TableCell2($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let cell_classes, is_in_selection, has_no_top, has_no_bottom, has_no_left, has_no_right;
    let value = $$props["value"];
    let index = $$props["index"];
    let j = $$props["j"];
    let actual_pinned_columns = $$props["actual_pinned_columns"];
    let get_cell_width2 = $$props["get_cell_width"];
    let handle_cell_click = $$props["handle_cell_click"];
    let handle_blur = $$props["handle_blur"];
    let toggle_cell_menu = $$props["toggle_cell_menu"];
    let is_cell_selected2 = $$props["is_cell_selected"];
    let should_show_cell_menu2 = $$props["should_show_cell_menu"];
    let selected_cells = $$props["selected_cells"];
    let copy_flash = $$props["copy_flash"];
    let active_cell_menu = $$props["active_cell_menu"];
    let styling = $$props["styling"];
    let latex_delimiters = $$props["latex_delimiters"];
    let line_breaks = $$props["line_breaks"];
    let datatype = $$props["datatype"];
    let editing = $$props["editing"];
    let max_chars = $$props["max_chars"];
    let editable = $$props["editable"];
    let is_static = fallback($$props["is_static"], false);
    let i18n = $$props["i18n"];
    let components = fallback($$props["components"], () => ({}), true);
    let el = $$props["el"];
    let handle_select_column = $$props["handle_select_column"];
    let handle_select_row = $$props["handle_select_row"];
    let is_dragging = $$props["is_dragging"];
    let display_value = $$props["display_value"];
    let wrap = fallback($$props["wrap"], false);
    function get_cell_position(col_index) {
      if (col_index >= actual_pinned_columns) {
        return "auto";
      }
      if (col_index === 0) {
        return "0";
      }
      const previous_widths = Array(col_index).fill(0).map((_, idx) => {
        return get_cell_width2(idx);
      }).join(" + ");
      return `calc(${previous_widths})`;
    }
    cell_classes = is_cell_selected2([index, j], selected_cells || []);
    is_in_selection = is_cell_in_selection([index, j], selected_cells);
    has_no_top = cell_classes.includes("no-top");
    has_no_bottom = cell_classes.includes("no-bottom");
    has_no_left = cell_classes.includes("no-left");
    has_no_right = cell_classes.includes("no-right");
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      $$renderer3.push(`<td${attr("tabindex", j < actual_pinned_columns ? -1 : 0)}${attr("data-row", index)}${attr("data-col", j)}${attr("data-testid", `cell-${index}-${j}`)}${attr_style(`width: ${stringify(get_cell_width2(j))}; left: ${stringify(get_cell_position(j))}; ${stringify(styling || "")}`)}${attr_class("svelte-18bs22a", void 0, {
        "pinned-column": j < actual_pinned_columns,
        "last-pinned": j === actual_pinned_columns - 1,
        "flash": copy_flash && is_in_selection,
        "cell-selected": is_in_selection,
        "no-top": has_no_top,
        "no-bottom": has_no_bottom,
        "no-left": has_no_left,
        "no-right": has_no_right,
        "menu-active": active_cell_menu && active_cell_menu.row === index && active_cell_menu.col === j,
        "dragging": is_dragging
      })}><div class="cell-wrap svelte-18bs22a">`);
      EditableCell2($$renderer3, {
        display_value: display_value !== void 0 ? display_value : String(value),
        latex_delimiters,
        line_breaks,
        editable,
        is_static,
        edit: editing && editing[0] === index && editing[1] === j,
        datatype,
        max_chars,
        i18n,
        components,
        show_selection_buttons: selected_cells.length === 1 && selected_cells[0][0] === index && selected_cells[0][1] === j,
        coords: [index, j],
        on_select_column: handle_select_column,
        on_select_row: handle_select_row,
        is_dragging,
        wrap_text: wrap,
        get value() {
          return value;
        },
        set value($$value) {
          value = $$value;
          $$settled = false;
        },
        get el() {
          return el.input;
        },
        set el($$value) {
          el.input = $$value;
          $$settled = false;
        }
      });
      $$renderer3.push(`<!----> `);
      if (editable && should_show_cell_menu2([index, j], selected_cells, editable)) {
        $$renderer3.push("<!--[-->");
        CellMenuButton2($$renderer3, { on_click: (event) => toggle_cell_menu(event, index, j) });
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--></div></td>`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
    bind_props($$props, {
      value,
      index,
      j,
      actual_pinned_columns,
      get_cell_width: get_cell_width2,
      handle_cell_click,
      handle_blur,
      toggle_cell_menu,
      is_cell_selected: is_cell_selected2,
      should_show_cell_menu: should_show_cell_menu2,
      selected_cells,
      copy_flash,
      active_cell_menu,
      styling,
      latex_delimiters,
      line_breaks,
      datatype,
      editing,
      max_chars,
      editable,
      is_static,
      i18n,
      components,
      el,
      handle_select_column,
      handle_select_row,
      is_dragging,
      display_value,
      wrap
    });
  });
}
function EmptyRowButton2($$renderer, $$props) {
  let on_click = $$props["on_click"];
  $$renderer.push(`<button class="add-row-button svelte-1at77md" aria-label="Add row">+</button>`);
  bind_props($$props, { on_click });
}
function VirtualTable2($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let sortedItems;
    let items2 = fallback($$props["items"], () => [], true);
    let max_height = $$props["max_height"];
    let actual_height = $$props["actual_height"];
    let table_scrollbar_width = $$props["table_scrollbar_width"];
    let start = fallback($$props["start"], 0);
    let end = fallback($$props["end"], 20);
    let selected = $$props["selected"];
    let disable_scroll = fallback($$props["disable_scroll"], false);
    let show_scroll_button = fallback($$props["show_scroll_button"], false);
    let viewport = $$props["viewport"];
    let label = fallback($$props["label"], null);
    let height2 = "100%";
    let average_height = 30;
    let bottom = 0;
    let head_height = 0;
    let foot_height = 0;
    let rows;
    let top = 0;
    let viewport_height = 200;
    let visible = [];
    const is_browser2 = typeof window !== "undefined";
    const raf2 = is_browser2 ? window.requestAnimationFrame : (cb) => cb();
    async function scroll_and_render(n) {
      raf2(async () => {
        if (typeof n !== "number") return;
        const direction = typeof n !== "number" ? false : is_in_view(n);
        if (direction === true) {
          return;
        }
        if (direction === "back") {
          await scroll_to_index(n, { behavior: "instant" });
        }
        if (direction === "forwards") {
          await scroll_to_index(n, { behavior: "instant" }, true);
        }
      });
    }
    function is_in_view(n) {
      const current = rows;
      if (n < start) {
        return "back";
      }
      if (n >= end - 1) {
        return "forwards";
      }
      const { top: viewport_top } = viewport.getBoundingClientRect();
      const { top: top2, bottom: bottom2 } = current.getBoundingClientRect();
      if (top2 - viewport_top < 37) {
        return "back";
      }
      if (bottom2 - viewport_top > viewport_height) {
        return "forwards";
      }
      return true;
    }
    async function scroll_to_index(index, opts, align_end = false) {
      await tick();
      const _itemHeight = average_height;
      let distance = index * _itemHeight;
      if (align_end) {
        distance = distance - viewport_height + _itemHeight + head_height;
      }
      const scrollbar_height = viewport.offsetHeight - viewport.clientHeight;
      if (scrollbar_height > 0) {
        distance += scrollbar_height;
      }
      const _opts = { top: distance, behavior: "smooth", ...opts };
      viewport.scrollTo(_opts);
    }
    viewport_height = 200;
    sortedItems = items2;
    scroll_and_render(selected);
    visible = is_browser2 ? sortedItems.slice(start, end).map((data, i) => {
      return { index: i + start, data };
    }) : sortedItems.slice(0, max_height / sortedItems.length * average_height + 1).map((data, i) => {
      return { index: i + start, data };
    });
    $$renderer2.push(`<svelte-virtual-table-viewport><div><table${attr_class("table svelte-fhd0lr", void 0, { "disable-scroll": disable_scroll })}${attr_style(`height: ${stringify(height2)}; --bw-svt-p-top: ${stringify(top)}px; --bw-svt-p-bottom: ${stringify(bottom)}px; --bw-svt-head-height: ${stringify(head_height)}px; --bw-svt-foot-height: ${stringify(foot_height)}px; --bw-svt-avg-row-height: ${stringify(average_height)}px; --max-height: ${stringify(max_height)}px`)}>`);
    if (label && label.length !== 0) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<caption class="sr-only">${escape_html(label)}</caption>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--><thead class="thead svelte-fhd0lr"><!--[-->`);
    slot($$renderer2, $$props, "thead", {}, null);
    $$renderer2.push(`<!--]--></thead><tbody class="tbody svelte-fhd0lr">`);
    if (visible.length && visible[0].data.length) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<!--[-->`);
      const each_array = ensure_array_like(visible);
      for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
        let item = each_array[$$index];
        $$renderer2.push(`<!--[-->`);
        slot($$renderer2, $$props, "tbody", { item: item.data, index: item.index }, () => {
          $$renderer2.push(`<tr><td>Missing Table Row</td></tr>`);
        });
        $$renderer2.push(`<!--]-->`);
      }
      $$renderer2.push(`<!--]-->`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--></tbody><tfoot class="tfoot svelte-fhd0lr"><!--[-->`);
    slot($$renderer2, $$props, "tfoot", {}, null);
    $$renderer2.push(`<!--]--></tfoot></table></div></svelte-virtual-table-viewport>`);
    bind_props($$props, {
      items: items2,
      max_height,
      actual_height,
      table_scrollbar_width,
      start,
      end,
      selected,
      disable_scroll,
      show_scroll_button,
      viewport,
      label,
      scroll_to_index
    });
  });
}
function CellMenu2($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let is_header, can_add_rows, can_add_columns;
    let x = $$props["x"];
    let y = $$props["y"];
    let on_add_row_above = $$props["on_add_row_above"];
    let on_add_row_below = $$props["on_add_row_below"];
    let on_add_column_left = $$props["on_add_column_left"];
    let on_add_column_right = $$props["on_add_column_right"];
    let row = $$props["row"];
    let col_count = $$props["col_count"];
    let row_count = $$props["row_count"];
    let on_delete_row = $$props["on_delete_row"];
    let on_delete_col = $$props["on_delete_col"];
    let can_delete_rows = $$props["can_delete_rows"];
    let can_delete_cols = $$props["can_delete_cols"];
    let on_sort = fallback($$props["on_sort"], () => {
    });
    let on_clear_sort = fallback($$props["on_clear_sort"], () => {
    });
    let sort_direction = fallback($$props["sort_direction"], null);
    let sort_priority = fallback($$props["sort_priority"], null);
    let on_filter = fallback($$props["on_filter"], () => {
    });
    let on_clear_filter = fallback($$props["on_clear_filter"], () => {
    });
    let filter_active = fallback($$props["filter_active"], null);
    let editable = fallback($$props["editable"], true);
    let i18n = $$props["i18n"];
    let active_filter_menu = null;
    is_header = row === -1;
    can_add_rows = editable && row_count[1] === "dynamic";
    can_add_columns = editable && col_count[1] === "dynamic";
    $$renderer2.push(`<div class="cell-menu svelte-1v4jjjx" role="menu">`);
    if (is_header) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<button role="menuitem"${attr_class("svelte-1v4jjjx", void 0, { "active": sort_direction === "asc" })}>`);
      CellMenuIcons2($$renderer2, { icon: "sort-asc" });
      $$renderer2.push(`<!----> ${escape_html(i18n("dataframe.sort_ascending"))} `);
      if (sort_direction === "asc" && sort_priority !== null) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<span class="priority svelte-1v4jjjx">${escape_html(sort_priority)}</span>`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]--></button> <button role="menuitem"${attr_class("svelte-1v4jjjx", void 0, { "active": sort_direction === "desc" })}>`);
      CellMenuIcons2($$renderer2, { icon: "sort-desc" });
      $$renderer2.push(`<!----> ${escape_html(i18n("dataframe.sort_descending"))} `);
      if (sort_direction === "desc" && sort_priority !== null) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<span class="priority svelte-1v4jjjx">${escape_html(sort_priority)}</span>`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]--></button> <button role="menuitem" class="svelte-1v4jjjx">`);
      CellMenuIcons2($$renderer2, { icon: "clear-sort" });
      $$renderer2.push(`<!----> ${escape_html(i18n("dataframe.clear_sort"))}</button> <button role="menuitem"${attr_class("svelte-1v4jjjx", void 0, { "active": filter_active || active_filter_menu })}>`);
      CellMenuIcons2($$renderer2, { icon: "filter" });
      $$renderer2.push(`<!----> ${escape_html(i18n("dataframe.filter"))} `);
      if (filter_active) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<span class="priority svelte-1v4jjjx">1</span>`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]--></button> <button role="menuitem" class="svelte-1v4jjjx">`);
      CellMenuIcons2($$renderer2, { icon: "clear-filter" });
      $$renderer2.push(`<!----> ${escape_html(i18n("dataframe.clear_filter"))}</button>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--> `);
    if (!is_header && can_add_rows) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<button role="menuitem" aria-label="Add row above" class="svelte-1v4jjjx">`);
      CellMenuIcons2($$renderer2, { icon: "add-row-above" });
      $$renderer2.push(`<!----> ${escape_html(i18n("dataframe.add_row_above"))}</button> <button role="menuitem" aria-label="Add row below" class="svelte-1v4jjjx">`);
      CellMenuIcons2($$renderer2, { icon: "add-row-below" });
      $$renderer2.push(`<!----> ${escape_html(i18n("dataframe.add_row_below"))}</button> `);
      if (can_delete_rows) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<button role="menuitem" class="delete svelte-1v4jjjx" aria-label="Delete row">`);
        CellMenuIcons2($$renderer2, { icon: "delete-row" });
        $$renderer2.push(`<!----> ${escape_html(i18n("dataframe.delete_row"))}</button>`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]-->`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--> `);
    if (can_add_columns) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<button role="menuitem" aria-label="Add column to the left" class="svelte-1v4jjjx">`);
      CellMenuIcons2($$renderer2, { icon: "add-column-left" });
      $$renderer2.push(`<!----> ${escape_html(i18n("dataframe.add_column_left"))}</button> <button role="menuitem" aria-label="Add column to the right" class="svelte-1v4jjjx">`);
      CellMenuIcons2($$renderer2, { icon: "add-column-right" });
      $$renderer2.push(`<!----> ${escape_html(i18n("dataframe.add_column_right"))}</button> `);
      if (can_delete_cols) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<button role="menuitem" class="delete svelte-1v4jjjx" aria-label="Delete column">`);
        CellMenuIcons2($$renderer2, { icon: "delete-column" });
        $$renderer2.push(`<!----> ${escape_html(i18n("dataframe.delete_column"))}</button>`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]-->`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--></div> `);
    {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]-->`);
    bind_props($$props, {
      x,
      y,
      on_add_row_above,
      on_add_row_below,
      on_add_column_left,
      on_add_column_right,
      row,
      col_count,
      row_count,
      on_delete_row,
      on_delete_col,
      can_delete_rows,
      can_delete_cols,
      on_sort,
      on_clear_sort,
      sort_direction,
      sort_priority,
      on_filter,
      on_clear_filter,
      filter_active,
      editable,
      i18n
    });
  });
}
function Toolbar2($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let show_fullscreen_button = fallback($$props["show_fullscreen_button"], false);
    let show_copy_button = fallback($$props["show_copy_button"], false);
    let show_search = fallback($$props["show_search"], "none");
    let fullscreen = fallback($$props["fullscreen"], false);
    let on_copy = $$props["on_copy"];
    let on_commit_filter = $$props["on_commit_filter"];
    let copied = false;
    let timer;
    let current_search_query = fallback($$props["current_search_query"], null);
    function copy_feedback() {
      copied = true;
      if (timer) clearTimeout(timer);
      timer = setTimeout(
        () => {
          copied = false;
        },
        2e3
      );
    }
    async function handle_copy() {
      await on_copy();
      copy_feedback();
    }
    onDestroy(() => {
      if (timer) clearTimeout(timer);
    });
    $$renderer2.push(`<div class="toolbar svelte-1rajnm3" role="toolbar" aria-label="Table actions"><div class="toolbar-buttons svelte-1rajnm3">`);
    if (show_search !== "none") {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<div class="search-container svelte-1rajnm3"><input type="text"${attr("value", current_search_query || "")}${attr("placeholder", show_search === "filter" ? "Filter..." : "Search...")}${attr_class("search-input svelte-1rajnm3", void 0, { "filter-mode": show_search === "filter" })}${attr("title", `Enter text to ${show_search} the table`)}/> `);
      if (current_search_query && show_search === "filter") {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<button class="toolbar-button check-button svelte-1rajnm3" aria-label="Apply filter and update dataframe values" title="Apply filter and update dataframe values">`);
        Check($$renderer2);
        $$renderer2.push(`<!----></button>`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]--></div>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--> `);
    if (show_copy_button) {
      $$renderer2.push("<!--[-->");
      IconButton($$renderer2, {
        Icon: copied ? Check : Copy,
        label: copied ? "Copied to clipboard" : "Copy table data",
        onclick: handle_copy
      });
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--> `);
    if (show_fullscreen_button) {
      $$renderer2.push("<!--[-->");
      FullscreenButton($$renderer2, { fullscreen });
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--></div></div>`);
    bind_props($$props, {
      show_fullscreen_button,
      show_copy_button,
      show_search,
      fullscreen,
      on_copy,
      on_commit_filter,
      current_search_query
    });
  });
}
function cast_value_to_type(v, t) {
  if (v === null || v === void 0) {
    return v;
  }
  if (t === "number") {
    const n = Number(v);
    return isNaN(n) ? v : n;
  }
  if (t === "bool") {
    if (typeof v === "boolean") return v;
    if (typeof v === "number") return v !== 0;
    const s = String(v).toLowerCase();
    if (s === "true" || s === "1") return true;
    if (s === "false" || s === "0") return false;
    return v;
  }
  if (t === "date") {
    const d = new Date(v);
    return isNaN(d.getTime()) ? v : d.toISOString();
  }
  return v;
}
function make_headers(_head, col_count, els, make_id2) {
  let _h = _head || [];
  if (col_count[1] === "fixed" && _h.length < col_count[0]) {
    const fill = Array(col_count[0] - _h.length).fill("").map((_, i) => `${i + _h.length}`);
    _h = _h.concat(fill);
  }
  if (!_h || _h.length === 0) {
    return Array(col_count[0]).fill(0).map((_, i) => {
      const _id = make_id2();
      els[_id] = { cell: null, input: null };
      return { id: _id, value: JSON.stringify(i + 1) };
    });
  }
  return _h.map((h, i) => {
    const _id = make_id2();
    els[_id] = { cell: null, input: null };
    return { id: _id, value: h ?? "" };
  });
}
function process_data(values, els, data_binding, make_id2, display_value = null, datatype) {
  if (!values || values.length === 0) {
    return [];
  }
  const result = values.map((row, i) => {
    return row.map((value, j) => {
      const _id = make_id2();
      els[_id] = { cell: null, input: null };
      data_binding[_id] = value;
      let display = display_value?.[i]?.[j];
      if (display === void 0) {
        display = String(value);
      }
      const dtype = Array.isArray(datatype) ? datatype[j] : datatype;
      return {
        id: _id,
        value: cast_value_to_type(value, dtype),
        display_value: display
      };
    });
  });
  return result;
}
async function save_cell_value(input_value, ctx, row, col) {
  if (!ctx.data || !ctx.data[row] || !ctx.data[row][col]) return;
  const old_value = ctx.data[row][col].value;
  ctx.data[row][col].value = input_value;
  if (ctx.data[row][col].display_value !== void 0) {
    ctx.data[row][col].display_value = input_value;
  }
  if (old_value !== input_value && ctx.dispatch) {
    ctx.dispatch("change", {
      data: ctx.data.map((row2) => row2.map((cell) => cell.value)),
      headers: ctx.headers?.map((h) => h.value) || [],
      metadata: null
    });
  }
  ctx.actions.set_selected([row, col]);
}
async function handle_cell_blur(event, ctx, coords) {
  if (!ctx.data || !ctx.headers || !ctx.els) return;
  const input_el = event.target;
  if (!input_el || input_el.value === void 0) return;
  await save_cell_value(input_el.value, ctx, coords[0], coords[1]);
}
const DATAFRAME_KEY = /* @__PURE__ */ Symbol("dataframe");
function create_actions(state, context) {
  const update_state = (updater) => state.update((s) => ({ ...s, ...updater(s) }));
  const add_row = (data, make_id2, index) => {
    const new_row = data[0]?.length ? Array(data[0].length).fill(null).map(() => ({ value: "", id: make_id2() })) : [{ value: "", id: make_id2() }];
    const new_data = [...data];
    index !== void 0 ? new_data.splice(index, 0, new_row) : new_data.push(new_row);
    return new_data;
  };
  const add_col = (data, headers, make_id2, index) => {
    const new_headers = [...headers, `Header ${headers.length + 1}`];
    const new_data = data.map((row) => [...row, { value: "", id: make_id2() }]);
    if (index !== void 0) {
      new_headers.splice(index, 0, new_headers.pop());
      new_data.forEach((row) => row.splice(index, 0, row.pop()));
    }
    return { data: new_data, headers: new_headers };
  };
  const update_array = (source, target) => {
    if (source && target) {
      target.splice(0, target.length, ...JSON.parse(JSON.stringify(source)));
    }
  };
  return {
    handle_search: (query) => update_state((s) => ({ current_search_query: query })),
    handle_sort: (col, direction) => update_state((s) => {
      const sort_cols = s.sort_state.sort_columns.filter(
        (c) => c.col !== col
      );
      if (!s.sort_state.sort_columns.some(
        (c) => c.col === col && c.direction === direction
      )) {
        sort_cols.push({ col, direction });
      }
      const initial_data = s.sort_state.initial_data || (context.data && sort_cols.length > 0 ? {
        data: JSON.parse(JSON.stringify(context.data)),
        display_value: context.display_value ? JSON.parse(JSON.stringify(context.display_value)) : null,
        styling: context.styling ? JSON.parse(JSON.stringify(context.styling)) : null
      } : null);
      return {
        sort_state: {
          ...s.sort_state,
          sort_columns: sort_cols.slice(-3),
          initial_data
        }
      };
    }),
    handle_filter: (col, datatype, filter, value) => update_state((s) => {
      const filter_cols = s.filter_state.filter_columns.some(
        (c) => c.col === col
      ) ? s.filter_state.filter_columns.filter((c) => c.col !== col) : [
        ...s.filter_state.filter_columns,
        { col, datatype, filter, value }
      ];
      const initial_data = s.filter_state.initial_data || (context.data && filter_cols.length > 0 ? {
        data: JSON.parse(JSON.stringify(context.data)),
        display_value: context.display_value ? JSON.parse(JSON.stringify(context.display_value)) : null,
        styling: context.styling ? JSON.parse(JSON.stringify(context.styling)) : null
      } : null);
      return {
        filter_state: {
          ...s.filter_state,
          filter_columns: filter_cols,
          initial_data
        }
      };
    }),
    get_sort_status: (name, headers) => {
      const s = get(state);
      const sort_item = s.sort_state.sort_columns.find(
        (item) => headers[item.col] === name
      );
      return sort_item ? sort_item.direction : "none";
    },
    sort_data: (data, display_value, styling) => {
      const {
        sort_state: { sort_columns }
      } = get(state);
      if (sort_columns.length)
        sort_table_data(data, display_value, styling, sort_columns);
    },
    update_row_order: (data) => update_state((s) => ({
      sort_state: {
        ...s.sort_state,
        row_order: s.sort_state.sort_columns.length && data[0] ? [...Array(data.length)].map((_, i) => i).sort((a, b) => {
          for (const { col, direction } of s.sort_state.sort_columns) {
            const comp = (data[a]?.[col]?.value ?? "") < (data[b]?.[col]?.value ?? "") ? -1 : 1;
            return direction === "asc" ? comp : -comp;
          }
          return 0;
        }) : [...Array(data.length)].map((_, i) => i)
      }
    })),
    filter_data: (data) => {
      const query = get(state).current_search_query?.toLowerCase();
      return query ? data.filter(
        (row) => row.some(
          (cell) => String(cell?.value).toLowerCase().includes(query)
        )
      ) : data;
    },
    add_row,
    add_col,
    add_row_at: (data, index, position, make_id2) => add_row(data, make_id2, position === "above" ? index : index + 1),
    add_col_at: (data, headers, index, position, make_id2) => add_col(data, headers, make_id2, position === "left" ? index : index + 1),
    delete_row: (data, index) => data.length > 1 ? data.filter((_, i) => i !== index) : data,
    delete_col: (data, headers, index) => headers.length > 1 ? {
      data: data.map((row) => row.filter((_, i) => i !== index)),
      headers: headers.filter((_, i) => i !== index)
    } : { data, headers },
    delete_row_at: (data, index) => data.length > 1 ? [...data.slice(0, index), ...data.slice(index + 1)] : data,
    delete_col_at: (data, headers, index) => headers.length > 1 ? {
      data: data.map((row) => [
        ...row.slice(0, index),
        ...row.slice(index + 1)
      ]),
      headers: [...headers.slice(0, index), ...headers.slice(index + 1)]
    } : { data, headers },
    trigger_change: async (data, headers, previous_data, previous_headers, value_is_output, dispatch2) => {
      const s = get(state);
      if (s.current_search_query) return;
      const current_headers = headers.map((h) => h.value);
      const current_data = data.map((row) => row.map((cell) => cell.value));
      if (!dequal$1(current_data, previous_data) || !dequal$1(current_headers, previous_headers)) {
        if (!dequal$1(current_headers, previous_headers)) {
          update_state((s2) => ({
            sort_state: { sort_columns: [], row_order: [], initial_data: null },
            filter_state: { filter_columns: [], initial_data: null }
          }));
        }
        dispatch2("change", {
          data: data.map((row) => row.map((cell) => cell.value)),
          headers: current_headers,
          metadata: null
        });
        const index = s.ui_state.selected;
        if (index) {
          dispatch2("edit", {
            index,
            value: data[index[0]][index[1]].value,
            previous_value: previous_data[index[0]][index[1]]
          });
        }
        if (!value_is_output) dispatch2("input");
      }
    },
    reset_sort_state: () => update_state((s) => {
      if (s.sort_state.initial_data && context.data) {
        const original = s.sort_state.initial_data;
        update_array(original.data, context.data);
        update_array(original.display_value, context.display_value);
        update_array(original.styling, context.styling);
      }
      return {
        sort_state: { sort_columns: [], row_order: [], initial_data: null }
      };
    }),
    reset_filter_state: () => update_state((s) => {
      if (s.filter_state.initial_data && context.data) {
        const original = s.filter_state.initial_data;
        update_array(original.data, context.data);
        update_array(original.display_value, context.display_value);
        update_array(original.styling, context.styling);
      }
      return {
        filter_state: { filter_columns: [], initial_data: null }
      };
    }),
    set_active_cell_menu: (menu) => update_state((s) => ({
      ui_state: { ...s.ui_state, active_cell_menu: menu }
    })),
    set_active_header_menu: (menu) => update_state((s) => ({
      ui_state: { ...s.ui_state, active_header_menu: menu }
    })),
    set_selected_cells: (cells) => update_state((s) => ({
      ui_state: { ...s.ui_state, selected_cells: cells }
    })),
    set_selected: (selected) => update_state((s) => ({ ui_state: { ...s.ui_state, selected } })),
    set_editing: (editing) => update_state((s) => ({ ui_state: { ...s.ui_state, editing } })),
    clear_ui_state: () => update_state((s) => ({
      ui_state: {
        active_cell_menu: null,
        active_header_menu: null,
        selected_cells: [],
        selected: false,
        editing: false,
        header_edit: false,
        selected_header: false,
        active_button: null,
        copy_flash: false
      }
    })),
    set_header_edit: (header_index) => update_state((s) => ({
      ui_state: {
        ...s.ui_state,
        selected_cells: [],
        selected_header: header_index,
        header_edit: header_index
      }
    })),
    set_selected_header: (header_index) => update_state((s) => ({
      ui_state: {
        ...s.ui_state,
        selected_header: header_index,
        selected: false,
        selected_cells: []
      }
    })),
    handle_header_click: (col, editable) => update_state((s) => ({
      ui_state: {
        ...s.ui_state,
        active_cell_menu: null,
        active_header_menu: null,
        selected: false,
        selected_cells: [],
        selected_header: col,
        header_edit: editable ? col : false
      }
    })),
    end_header_edit: (key) => {
      if (["Escape", "Enter", "Tab"].includes(key)) {
        update_state((s) => ({
          ui_state: { ...s.ui_state, selected: false, header_edit: false }
        }));
      }
    },
    get_selected_cells: () => get(state).ui_state.selected_cells,
    get_active_cell_menu: () => get(state).ui_state.active_cell_menu,
    get_active_button: () => get(state).ui_state.active_button,
    set_active_button: (button) => update_state((s) => ({
      ui_state: { ...s.ui_state, active_button: button }
    })),
    set_copy_flash: (value) => update_state((s) => ({ ui_state: { ...s.ui_state, copy_flash: value } })),
    handle_cell_click: (event, row, col) => {
      event.preventDefault();
      event.stopPropagation();
      const s = get(state);
      if (s.config.show_row_numbers && col === -1) return;
      let actual_row = row;
      if (s.current_search_query && context.data) {
        const filtered_indices = [];
        context.data.forEach((dataRow, idx) => {
          if (dataRow.some(
            (cell) => String(cell?.value).toLowerCase().includes(s.current_search_query?.toLowerCase() || "")
          )) {
            filtered_indices.push(idx);
          }
        });
        actual_row = filtered_indices[row] ?? row;
      }
      const cells = handle_selection(
        [actual_row, col],
        s.ui_state.selected_cells,
        event
      );
      update_state((s2) => ({
        ui_state: {
          ...s2.ui_state,
          active_cell_menu: null,
          active_header_menu: null,
          selected_header: false,
          header_edit: false,
          selected_cells: cells,
          selected: cells[0]
        }
      }));
      if (s.config.editable && cells.length === 1) {
        update_state((s2) => ({
          ui_state: { ...s2.ui_state, editing: [actual_row, col] }
        }));
        tick().then(
          () => context.els[context.data[actual_row][col].id]?.input?.focus()
        );
      } else {
        tick().then(() => {
          if (context.parent_element) {
            context.parent_element.focus();
          }
        });
      }
      context.dispatch?.("select", {
        index: [actual_row, col],
        col_value: context.get_column(col),
        row_value: context.get_row(actual_row),
        value: context.get_data_at(actual_row, col)
      });
    },
    toggle_cell_menu: (event, row, col) => {
      event.stopPropagation();
      const current_menu = get(state).ui_state.active_cell_menu;
      if (current_menu?.row === row && current_menu.col === col) {
        update_state((s) => ({
          ui_state: { ...s.ui_state, active_cell_menu: null }
        }));
      } else {
        const cell = event.target.closest("td");
        if (cell) {
          const rect = cell.getBoundingClientRect();
          update_state((s) => ({
            ui_state: {
              ...s.ui_state,
              active_cell_menu: { row, col, x: rect.right, y: rect.bottom }
            }
          }));
        }
      }
    },
    toggle_cell_button: (row, col) => {
      const current_button = get(state).ui_state.active_button;
      const new_button = current_button?.type === "cell" && current_button.row === row && current_button.col === col ? null : { type: "cell", row, col };
      update_state((s) => ({
        ui_state: { ...s.ui_state, active_button: new_button }
      }));
    },
    handle_select_column: (col) => {
      if (!context.data) return;
      const cells = context.data.map((_, row) => [row, col]);
      update_state((s) => ({
        ui_state: {
          ...s.ui_state,
          selected_cells: cells,
          selected: cells[0],
          editing: false
        }
      }));
      setTimeout(() => context.parent_element?.focus(), 0);
    },
    handle_select_row: (row) => {
      if (!context.data || !context.data[0]) return;
      const cells = context.data[0].map(
        (_, col) => [row, col]
      );
      update_state((s) => ({
        ui_state: {
          ...s.ui_state,
          selected_cells: cells,
          selected: cells[0],
          editing: false
        }
      }));
      setTimeout(() => context.parent_element?.focus(), 0);
    },
    get_next_cell_coordinates,
    get_range_selection,
    move_cursor
  };
}
function create_dataframe_context(config) {
  const state = writable({
    config,
    current_search_query: null,
    sort_state: { sort_columns: [], row_order: [], initial_data: null },
    filter_state: { filter_columns: [], initial_data: null },
    ui_state: {
      active_cell_menu: null,
      active_header_menu: null,
      selected_cells: [],
      selected: false,
      editing: false,
      header_edit: false,
      selected_header: false,
      active_button: null,
      copy_flash: false
    }
  });
  const context = { state, actions: null };
  context.actions = create_actions(state, context);
  const instance_id = /* @__PURE__ */ Symbol(
    `dataframe_${Math.random().toString(36).substring(2)}`
  );
  setContext(instance_id, context);
  setContext(DATAFRAME_KEY, { instance_id, context });
  return context;
}
function Table2($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    var $$store_subs;
    let selected_cells, selected, editing, header_edit, selected_header, active_cell_menu, active_header_menu, copy_flash, actual_pinned_columns, max, selected_index, handle_mouse_down;
    let datatype = $$props["datatype"];
    let label = fallback($$props["label"], null);
    let show_label = fallback($$props["show_label"], true);
    let headers = fallback($$props["headers"], () => [], true);
    let values = fallback($$props["values"], () => [], true);
    let col_count = $$props["col_count"];
    let row_count = $$props["row_count"];
    let latex_delimiters = $$props["latex_delimiters"];
    let components = fallback($$props["components"], () => ({}), true);
    let editable = fallback($$props["editable"], true);
    let wrap = fallback($$props["wrap"], false);
    let root = $$props["root"];
    let i18n = $$props["i18n"];
    let max_height = fallback($$props["max_height"], 500);
    let line_breaks = fallback($$props["line_breaks"], true);
    let column_widths = fallback($$props["column_widths"], () => [], true);
    let show_row_numbers = fallback($$props["show_row_numbers"], false);
    let upload = $$props["upload"];
    let stream_handler = $$props["stream_handler"];
    let buttons = fallback($$props["buttons"], null);
    let value_is_output = fallback($$props["value_is_output"], false);
    let max_chars = fallback($$props["max_chars"], void 0);
    let show_search = fallback($$props["show_search"], "none");
    let pinned_columns = fallback($$props["pinned_columns"], 0);
    let static_columns = fallback($$props["static_columns"], () => [], true);
    let fullscreen = fallback($$props["fullscreen"], false);
    const df_ctx = create_dataframe_context({
      show_fullscreen_button: buttons === null ? true : buttons.includes("fullscreen"),
      show_copy_button: buttons === null ? true : buttons.includes("copy"),
      show_search,
      show_row_numbers,
      editable,
      pinned_columns,
      show_label,
      line_breaks,
      wrap,
      max_height,
      column_widths,
      max_chars,
      static_columns
    });
    const { state: df_state, actions: df_actions } = df_ctx;
    const dispatch2 = createEventDispatcher();
    let els = {};
    let data_binding = {};
    let _headers = make_headers(headers, col_count, els, make_id2);
    let old_headers = headers;
    let data = [[]];
    let old_val = void 0;
    let search_results = [[]];
    let dragging = false;
    let filtered_to_original_map = [];
    function make_id2() {
      return Math.random().toString(36).substring(2, 15);
    }
    let display_value = fallback($$props["display_value"], null);
    let styling = fallback($$props["styling"], null);
    let previous_headers = _headers.map((h) => h.value);
    let previous_data = data.map((row) => row.map((cell) => cell.value));
    function handle_sort(col, direction) {
      df_actions.handle_sort(col, direction);
      sort_data2(data, display_value, styling);
    }
    function clear_sort() {
      df_actions.reset_sort_state();
      sort_data2(data, display_value, styling);
    }
    function handle_filter(col, datatype2, filter, value) {
      df_actions.handle_filter(col, datatype2, filter, value);
      filter_data2(data, display_value, styling);
    }
    function clear_filter() {
      df_actions.reset_filter_state();
      filter_data2(data, display_value, styling);
    }
    async function edit_header(i, _select = false) {
      if (!editable || header_edit === i || col_count[1] !== "dynamic") return;
      df_actions.set_header_edit(i);
    }
    function handle_header_click(event, col) {
      if (event.target instanceof HTMLAnchorElement) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      if (!editable) return;
      df_actions.set_editing(false);
      df_actions.handle_header_click(col, editable);
      parent.focus();
    }
    function end_header_edit(event) {
      if (!editable) return;
      df_actions.end_header_edit(event.detail.key);
      parent.focus();
    }
    async function add_row(index) {
      parent.focus();
      if (row_count[1] !== "dynamic") return;
      const new_row = Array(data[0]?.length || headers.length).fill(0).map((_, i) => {
        const _id = make_id2();
        els[_id] = { cell: null, input: null };
        return { id: _id, value: "" };
      });
      if (data.length === 0) {
        data = [new_row];
      } else if (index !== void 0 && index >= 0 && index <= data.length) {
        data.splice(index, 0, new_row);
      } else {
        data.push(new_row);
      }
      selected = [index !== void 0 ? index : data.length - 1, 0];
    }
    async function add_col(index) {
      parent.focus();
      if (col_count[1] !== "dynamic") return;
      const result = df_actions.add_col(data, headers, make_id2, index);
      result.data.forEach((row) => {
        row.forEach((cell) => {
          if (!els[cell.id]) {
            els[cell.id] = { cell: null, input: null };
          }
        });
      });
      data = result.data;
      headers = result.headers;
      await tick();
      requestAnimationFrame(() => {
        edit_header(index !== void 0 ? index : data[0].length - 1, true);
        const new_w = parent.querySelectorAll("tbody")[1].offsetWidth;
        parent.querySelectorAll("table")[1].scrollTo({ left: new_w });
      });
    }
    let parent;
    let table;
    function get_cell_width2(index) {
      return `var(--cell-width-${index})`;
    }
    let table_height = values.slice(0, max_height / values.length * 37).length * 37 + 37;
    let scrollbar_width = 0;
    function sort_data2(_data, _display_value, _styling) {
      const result = sort_data_and_preserve_selection(_data, _display_value, _styling, store_get($$store_subs ??= {}, "$df_state", df_state).sort_state.sort_columns, selected, get_current_indices);
      data = result.data;
      selected = result.selected;
    }
    function filter_data2(_data, _display_value, _styling) {
      const result = filter_data_and_preserve_selection(_data, _display_value, _styling, store_get($$store_subs ??= {}, "$df_state", df_state).filter_state.filter_columns, selected, get_current_indices, store_get($$store_subs ??= {}, "$df_state", df_state).filter_state.initial_data?.data, store_get($$store_subs ??= {}, "$df_state", df_state).filter_state.initial_data?.display_value, store_get($$store_subs ??= {}, "$df_state", df_state).filter_state.initial_data?.styling);
      data = result.data;
      selected = result.selected;
    }
    const set_copy_flash = (value) => {
      df_actions.set_copy_flash(value);
    };
    let previous_selected_cells = [];
    function handle_blur(event) {
      const { blur_event, coords } = event.detail;
      handle_cell_blur(blur_event, df_ctx, coords);
    }
    function toggle_header_menu(event, col) {
      event.stopPropagation();
      if (active_header_menu && active_header_menu.col === col) {
        df_actions.set_active_header_menu(null);
      } else {
        const header = event.target.closest("th");
        if (header) {
          const rect = header.getBoundingClientRect();
          df_actions.set_active_header_menu({ col, x: rect.right, y: rect.bottom });
        }
      }
    }
    function delete_col_at(index) {
      if (col_count[1] !== "dynamic") return;
      if (data[0].length <= 1) return;
      const result = df_actions.delete_col_at(data, headers, index);
      data = result.data;
      headers = result.headers;
      _headers = make_headers(headers, col_count, els, make_id2);
      df_actions.set_active_cell_menu(null);
      df_actions.set_active_header_menu(null);
      df_actions.set_selected(false);
      df_actions.set_selected_cells([]);
      df_actions.set_editing(false);
    }
    function delete_row_at(index) {
      data = df_actions.delete_row_at(data, index);
      df_actions.set_active_cell_menu(null);
      df_actions.set_active_header_menu(null);
    }
    let selected_cell_coords;
    function commit_filter() {
      if (store_get($$store_subs ??= {}, "$df_state", df_state).current_search_query && show_search === "filter") {
        search_results.forEach((row) => {
          const data_row = [];
          const display_row = [];
          const styling_row = [];
          row.forEach((cell) => {
            data_row.push(cell.value);
            display_row.push(cell.display_value !== void 0 ? cell.display_value : String(cell.value));
            styling_row.push(cell.styling || "");
          });
        });
        ({
          headers: _headers.map((h) => h.value)});
        df_actions.handle_search(null);
      }
    }
    let viewport;
    let show_scroll_button = false;
    function add_row_at(index, position) {
      const row_index = position === "above" ? index : index + 1;
      add_row(row_index);
      active_cell_menu = null;
      active_header_menu = null;
    }
    function add_col_at(index, position) {
      const col_index = position === "left" ? index : index + 1;
      add_col(col_index);
      active_cell_menu = null;
      active_header_menu = null;
    }
    function reset_sort_state() {
      df_actions.reset_sort_state();
    }
    function handle_select_all(col, checked) {
      data = data.map((row) => {
        const new_row = [...row];
        if (new_row[col]) {
          new_row[col] = { ...new_row[col], value: checked };
        }
        return new_row;
      });
    }
    let is_dragging = false;
    const drag_state = { is_dragging};
    function get_cell_display_value(row, col) {
      const is_search_active = store_get($$store_subs ??= {}, "$df_state", df_state).current_search_query !== void 0;
      if (is_search_active && search_results?.[row]?.[col]) {
        return search_results[row][col].display_value !== void 0 ? search_results[row][col].display_value : String(search_results[row][col].value);
      }
      if (data?.[row]?.[col]) {
        return data[row][col].display_value !== void 0 ? data[row][col].display_value : String(data[row][col].value);
      }
      return "";
    }
    selected_cells = store_get($$store_subs ??= {}, "$df_state", df_state).ui_state.selected_cells;
    selected = store_get($$store_subs ??= {}, "$df_state", df_state).ui_state.selected;
    editing = store_get($$store_subs ??= {}, "$df_state", df_state).ui_state.editing;
    header_edit = store_get($$store_subs ??= {}, "$df_state", df_state).ui_state.header_edit;
    selected_header = store_get($$store_subs ??= {}, "$df_state", df_state).ui_state.selected_header;
    active_cell_menu = store_get($$store_subs ??= {}, "$df_state", df_state).ui_state.active_cell_menu;
    active_header_menu = store_get($$store_subs ??= {}, "$df_state", df_state).ui_state.active_header_menu;
    copy_flash = store_get($$store_subs ??= {}, "$df_state", df_state).ui_state.copy_flash;
    if (!dequal(values, old_val)) {
      const is_reset = values.length === 0 || values.length === 1 && values[0].length === 0;
      const is_different_structure = old_val !== void 0 && (values.length !== old_val.length || values[0] && old_val[0] && values[0].length !== old_val[0].length);
      data = process_data(values, els, data_binding, make_id2, display_value, datatype);
      old_val = JSON.parse(JSON.stringify(values));
      if (is_reset || is_different_structure) {
        df_actions.reset_sort_state();
      } else if (store_get($$store_subs ??= {}, "$df_state", df_state).sort_state.sort_columns.length > 0) {
        sort_data2(data, display_value, styling);
      } else {
        df_actions.handle_sort(-1, "asc");
        df_actions.reset_sort_state();
      }
      if (store_get($$store_subs ??= {}, "$df_state", df_state).filter_state.filter_columns.length > 0) {
        filter_data2(data, display_value, styling);
      } else {
        df_actions.reset_filter_state();
      }
      if (store_get($$store_subs ??= {}, "$df_state", df_state).current_search_query) {
        df_actions.handle_search(null);
      }
    }
    actual_pinned_columns = pinned_columns && data?.[0]?.length ? Math.min(pinned_columns, data[0].length) : 0;
    {
      if (!dequal(headers, old_headers)) {
        _headers = make_headers(headers, col_count, els, make_id2);
        old_headers = JSON.parse(JSON.stringify(headers));
      }
    }
    {
      if (data || _headers || els) {
        df_ctx.data = data;
        df_ctx.headers = _headers;
        df_ctx.els = els;
        df_ctx.display_value = display_value;
        df_ctx.styling = styling;
      }
    }
    if (store_get($$store_subs ??= {}, "$df_state", df_state).current_search_query !== void 0) {
      const cell_map = /* @__PURE__ */ new Map();
      filtered_to_original_map = [];
      data.forEach((row, row_idx) => {
        if (row.some((cell) => String(cell?.value).toLowerCase().includes(store_get($$store_subs ??= {}, "$df_state", df_state).current_search_query?.toLowerCase() || ""))) {
          filtered_to_original_map.push(row_idx);
        }
        row.forEach((cell, col_idx) => {
          cell_map.set(cell.id, {
            value: cell.value,
            display_value: cell.display_value !== void 0 ? cell.display_value : String(cell.value),
            styling: styling?.[row_idx]?.[col_idx] || ""
          });
        });
      });
      const filtered = df_actions.filter_data(data);
      search_results = filtered.map((row) => row.map((cell) => {
        const original = cell_map.get(cell.id);
        return {
          ...cell,
          display_value: original?.display_value !== void 0 ? original.display_value : String(cell.value),
          styling: original?.styling || ""
        };
      }));
    } else {
      filtered_to_original_map = [];
    }
    {
      if (data || _headers) {
        df_actions.trigger_change(data, _headers, previous_data, previous_headers, value_is_output, dispatch2);
        previous_data = data.map((row) => row.map((cell) => cell.value));
        previous_headers = _headers.map((h) => h.value);
      }
    }
    {
      if (store_get($$store_subs ??= {}, "$df_state", df_state).filter_state.filter_columns.length > 0) {
        filter_data2(data, display_value, styling);
      }
      if (store_get($$store_subs ??= {}, "$df_state", df_state).sort_state.sort_columns.length > 0) {
        sort_data2(data, display_value, styling);
        df_actions.update_row_order(data);
      }
    }
    max = get_max(data);
    selected_index = !!selected && selected[0];
    {
      if (copy_flash && !dequal(selected_cells, previous_selected_cells)) {
        set_copy_flash(false);
      }
      previous_selected_cells = selected_cells;
    }
    if (selected !== false) selected_cell_coords = selected;
    if (selected !== false) {
      const positions = calculate_selection_positions(selected, data, els, parent, table);
      document.documentElement.style.setProperty("--selected-col-pos", positions.col_pos);
      document.documentElement.style.setProperty("--selected-row-pos", positions.row_pos || "0px");
    }
    {
      is_dragging = drag_state.is_dragging;
    }
    handle_mouse_down = (() => {
    });
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      $$renderer3.push(`<div class="table-container svelte-2balj6">`);
      if (label && label.length !== 0 && show_label || (buttons === null ? true : buttons.includes("fullscreen")) || (buttons === null ? true : buttons.includes("copy")) || show_search !== "none") {
        $$renderer3.push("<!--[-->");
        $$renderer3.push(`<div class="header-row svelte-2balj6">`);
        if (label && label.length !== 0 && show_label) {
          $$renderer3.push("<!--[-->");
          $$renderer3.push(`<div class="label svelte-2balj6"><p class="svelte-2balj6">${escape_html(label)}</p></div>`);
        } else {
          $$renderer3.push("<!--[!-->");
        }
        $$renderer3.push(`<!--]--> `);
        Toolbar2($$renderer3, {
          show_fullscreen_button: buttons === null ? true : buttons.includes("fullscreen"),
          fullscreen,
          on_copy: async () => await copy_table_data(data),
          show_copy_button: buttons === null ? true : buttons.includes("copy"),
          show_search,
          on_commit_filter: commit_filter,
          current_search_query: store_get($$store_subs ??= {}, "$df_state", df_state).current_search_query
        });
        $$renderer3.push(`<!----></div>`);
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--> <div${attr_class("table-wrap svelte-2balj6", void 0, {
        "dragging": is_dragging,
        "no-wrap": !wrap,
        "menu-open": active_cell_menu || active_header_menu
      })} role="grid" tabindex="0"><table aria-hidden="true" class="svelte-2balj6">`);
      if (label && label.length !== 0) {
        $$renderer3.push("<!--[-->");
        $$renderer3.push(`<caption class="sr-only svelte-2balj6">${escape_html(label)}</caption>`);
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--><thead class="svelte-2balj6"><tr class="svelte-2balj6">`);
      if (show_row_numbers) {
        $$renderer3.push("<!--[-->");
        RowNumber2($$renderer3, { is_header: true });
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--><!--[-->`);
      const each_array = ensure_array_like(_headers);
      for (let i = 0, $$length = each_array.length; i < $$length; i++) {
        let { value, id } = each_array[i];
        TableHeader2($$renderer3, {
          i,
          actual_pinned_columns,
          header_edit,
          selected_header,
          headers,
          get_cell_width: get_cell_width2,
          handle_header_click,
          toggle_header_menu,
          end_header_edit,
          sort_columns: store_get($$store_subs ??= {}, "$df_state", df_state).sort_state.sort_columns,
          filter_columns: store_get($$store_subs ??= {}, "$df_state", df_state).filter_state.filter_columns,
          latex_delimiters,
          line_breaks,
          max_chars,
          editable,
          is_static: static_columns.includes(i),
          i18n,
          col_count,
          datatype: Array.isArray(datatype) ? datatype[i] : datatype,
          data,
          on_select_all: handle_select_all,
          get value() {
            return _headers[i].value;
          },
          set value($$value) {
            _headers[i].value = $$value;
            $$settled = false;
          },
          get el() {
            return els[id].input;
          },
          set el($$value) {
            els[id].input = $$value;
            $$settled = false;
          }
        });
      }
      $$renderer3.push(`<!--]--></tr></thead><tbody class="svelte-2balj6"><tr class="svelte-2balj6">`);
      if (show_row_numbers) {
        $$renderer3.push("<!--[-->");
        RowNumber2($$renderer3, { index: 0 });
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--><!--[-->`);
      const each_array_1 = ensure_array_like(max);
      for (let j = 0, $$length = each_array_1.length; j < $$length; j++) {
        let { value, id } = each_array_1[j];
        $$renderer3.push(`<td tabindex="-1" class="svelte-2balj6"><div class="cell-wrap svelte-2balj6">`);
        EditableCell2($$renderer3, {
          value,
          latex_delimiters,
          line_breaks,
          datatype: Array.isArray(datatype) ? datatype[j] : datatype,
          edit: false,
          el: null,
          editable,
          i18n,
          show_selection_buttons: selected_cells.length === 1 && selected_cells[0][0] === 0 && selected_cells[0][1] === j,
          coords: selected_cell_coords,
          on_select_column: df_actions.handle_select_column,
          on_select_row: df_actions.handle_select_row,
          is_dragging
        });
        $$renderer3.push(`<!----></div></td>`);
      }
      $$renderer3.push(`<!--]--></tr></tbody></table> `);
      Upload($$renderer3, {
        upload,
        stream_handler,
        flex: false,
        center: false,
        boundedheight: false,
        disable_click: true,
        root,
        aria_label: i18n("dataframe.drop_to_upload"),
        get dragging() {
          return dragging;
        },
        set dragging($$value) {
          dragging = $$value;
          $$settled = false;
        },
        children: ($$renderer4) => {
          $$renderer4.push(`<div class="table-wrap svelte-2balj6">`);
          VirtualTable2($$renderer4, {
            max_height,
            selected: selected_index,
            disable_scroll: active_cell_menu !== null || active_header_menu !== null,
            label,
            get items() {
              return search_results;
            },
            set items($$value) {
              search_results = $$value;
              $$settled = false;
            },
            get actual_height() {
              return table_height;
            },
            set actual_height($$value) {
              table_height = $$value;
              $$settled = false;
            },
            get table_scrollbar_width() {
              return scrollbar_width;
            },
            set table_scrollbar_width($$value) {
              scrollbar_width = $$value;
              $$settled = false;
            },
            get viewport() {
              return viewport;
            },
            set viewport($$value) {
              viewport = $$value;
              $$settled = false;
            },
            get show_scroll_button() {
              return show_scroll_button;
            },
            set show_scroll_button($$value) {
              show_scroll_button = $$value;
              $$settled = false;
            },
            $$slots: {
              thead: ($$renderer5) => {
                $$renderer5.push(`<tr slot="thead" class="svelte-2balj6">`);
                if (show_row_numbers) {
                  $$renderer5.push("<!--[-->");
                  RowNumber2($$renderer5, { is_header: true });
                } else {
                  $$renderer5.push("<!--[!-->");
                }
                $$renderer5.push(`<!--]--><!--[-->`);
                const each_array_2 = ensure_array_like(_headers);
                for (let i = 0, $$length = each_array_2.length; i < $$length; i++) {
                  let { value, id } = each_array_2[i];
                  TableHeader2($$renderer5, {
                    i,
                    actual_pinned_columns,
                    header_edit,
                    selected_header,
                    headers,
                    get_cell_width: get_cell_width2,
                    handle_header_click,
                    toggle_header_menu,
                    end_header_edit,
                    sort_columns: store_get($$store_subs ??= {}, "$df_state", df_state).sort_state.sort_columns,
                    filter_columns: store_get($$store_subs ??= {}, "$df_state", df_state).filter_state.filter_columns,
                    latex_delimiters,
                    line_breaks,
                    max_chars,
                    editable,
                    is_static: static_columns.includes(i),
                    i18n,
                    col_count,
                    datatype: Array.isArray(datatype) ? datatype[i] : datatype,
                    data,
                    on_select_all: handle_select_all,
                    get value() {
                      return _headers[i].value;
                    },
                    set value($$value) {
                      _headers[i].value = $$value;
                      $$settled = false;
                    },
                    get el() {
                      return els[id].input;
                    },
                    set el($$value) {
                      els[id].input = $$value;
                      $$settled = false;
                    }
                  });
                }
                $$renderer5.push(`<!--]--></tr>`);
              },
              tbody: ($$renderer5, { item, index }) => {
                $$renderer5.push(`<tr slot="tbody"${attr_class("svelte-2balj6", void 0, { "row-odd": index % 2 === 0 })}>`);
                if (show_row_numbers) {
                  $$renderer5.push("<!--[-->");
                  RowNumber2($$renderer5, { index });
                } else {
                  $$renderer5.push("<!--[!-->");
                }
                $$renderer5.push(`<!--]--><!--[-->`);
                const each_array_3 = ensure_array_like(item);
                for (let j = 0, $$length = each_array_3.length; j < $$length; j++) {
                  let { value, id } = each_array_3[j];
                  TableCell2($$renderer5, {
                    display_value: get_cell_display_value(index, j),
                    index: store_get($$store_subs ??= {}, "$df_state", df_state).current_search_query !== void 0 && filtered_to_original_map[index] !== void 0 ? filtered_to_original_map[index] : index,
                    j,
                    actual_pinned_columns,
                    get_cell_width: get_cell_width2,
                    handle_cell_click: (e, r, c) => handle_mouse_down(e, r, c),
                    handle_blur,
                    toggle_cell_menu: df_actions.toggle_cell_menu,
                    is_cell_selected,
                    should_show_cell_menu,
                    selected_cells,
                    copy_flash,
                    active_cell_menu,
                    styling: search_results[index][j].styling,
                    latex_delimiters,
                    line_breaks,
                    datatype: Array.isArray(datatype) ? datatype[j] : datatype,
                    editing,
                    max_chars,
                    editable,
                    is_static: static_columns.includes(j),
                    i18n,
                    components,
                    handle_select_column: df_actions.handle_select_column,
                    handle_select_row: df_actions.handle_select_row,
                    is_dragging,
                    wrap,
                    get value() {
                      return search_results[index][j].value;
                    },
                    set value($$value) {
                      search_results[index][j].value = $$value;
                      $$settled = false;
                    },
                    get el() {
                      return els[id];
                    },
                    set el($$value) {
                      els[id] = $$value;
                      $$settled = false;
                    }
                  });
                }
                $$renderer5.push(`<!--]--></tr>`);
              }
            }
          });
          $$renderer4.push(`<!----></div>`);
        },
        $$slots: { default: true }
      });
      $$renderer3.push(`<!----> `);
      if (show_scroll_button) {
        $$renderer3.push("<!--[-->");
        $$renderer3.push(`<button class="scroll-top-button svelte-2balj6">↑</button>`);
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--></div></div> `);
      if (data.length === 0 && editable && row_count[1] === "dynamic") {
        $$renderer3.push("<!--[-->");
        EmptyRowButton2($$renderer3, { on_click: () => add_row() });
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--> `);
      if (active_cell_menu || active_header_menu) {
        $$renderer3.push("<!--[-->");
        CellMenu2($$renderer3, {
          x: active_cell_menu?.x ?? active_header_menu?.x ?? 0,
          y: active_cell_menu?.y ?? active_header_menu?.y ?? 0,
          row: active_header_menu ? -1 : active_cell_menu?.row ?? 0,
          col_count,
          row_count,
          on_add_row_above: () => add_row_at(active_cell_menu?.row ?? -1, "above"),
          on_add_row_below: () => add_row_at(active_cell_menu?.row ?? -1, "below"),
          on_add_column_left: () => add_col_at(active_cell_menu?.col ?? active_header_menu?.col ?? -1, "left"),
          on_add_column_right: () => add_col_at(active_cell_menu?.col ?? active_header_menu?.col ?? -1, "right"),
          on_delete_row: () => delete_row_at(active_cell_menu?.row ?? -1),
          on_delete_col: () => delete_col_at(active_cell_menu?.col ?? active_header_menu?.col ?? -1),
          editable,
          can_delete_rows: !active_header_menu && data.length > 1 && editable,
          can_delete_cols: data.length > 0 && data[0]?.length > 1 && editable,
          i18n,
          on_sort: active_header_menu ? (direction) => {
            if (active_header_menu) {
              handle_sort(active_header_menu.col, direction);
              df_actions.set_active_header_menu(null);
            }
          } : void 0,
          on_clear_sort: active_header_menu ? () => {
            clear_sort();
            df_actions.set_active_header_menu(null);
          } : void 0,
          sort_direction: active_header_menu ? store_get($$store_subs ??= {}, "$df_state", df_state).sort_state.sort_columns.find((item) => item.col === (active_header_menu?.col ?? -1))?.direction ?? null : null,
          sort_priority: active_header_menu ? store_get($$store_subs ??= {}, "$df_state", df_state).sort_state.sort_columns.findIndex((item) => item.col === (active_header_menu?.col ?? -1)) + 1 || null : null,
          on_filter: active_header_menu ? (datatype2, filter, value) => {
            if (active_header_menu) {
              handle_filter(active_header_menu.col, datatype2, filter, value);
              df_actions.set_active_header_menu(null);
            }
          } : void 0,
          on_clear_filter: active_header_menu ? () => {
            clear_filter();
            df_actions.set_active_header_menu(null);
          } : void 0,
          filter_active: active_header_menu ? store_get($$store_subs ??= {}, "$df_state", df_state).filter_state.filter_columns.some((c) => c.col === (active_header_menu?.col ?? -1)) : null
        });
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]-->`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
    if ($$store_subs) unsubscribe_stores($$store_subs);
    bind_props($$props, {
      datatype,
      label,
      show_label,
      headers,
      values,
      col_count,
      row_count,
      latex_delimiters,
      components,
      editable,
      wrap,
      root,
      i18n,
      max_height,
      line_breaks,
      column_widths,
      show_row_numbers,
      upload,
      stream_handler,
      buttons,
      value_is_output,
      max_chars,
      show_search,
      pinned_columns,
      static_columns,
      fullscreen,
      display_value,
      styling,
      reset_sort_state
    });
  });
}
function Index4($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let prev_data = null;
    class DataframeGradio extends Gradio {
      async set_data(data) {
        if (data.value) ;
        super.set_data(data);
        if (data.value && dequal$1(data.value, JSON.parse(prev_data)) === false) {
          prev_data = JSON.stringify(data.value);
        }
      }
    }
    let { $$slots, $$events, ...props } = $$props;
    let gradio = new DataframeGradio(props);
    ({
      i18n: gradio.i18n,
      client: gradio.shared.client,
      autoscroll: gradio.shared.autoscroll
    });
    $$renderer2.push(`<div></div>`);
  });
}
const Index$1 = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  BaseDataFrame: Table2,
  BaseExample: Example,
  default: Index4
}, Symbol.toStringTag, { value: "Module" }));

export { Index$1 as I, commonjsGlobal as c };
//# sourceMappingURL=Index20-nJ-tmXTH.js.map
