import './async-D55cHugf.js';
import { f as attr_class } from './index-K3l_dLem.js';
import { c as create_marked } from './MarkdownCode.svelte_svelte_type_style_lang-B2xYMNIW.js';
import { s as sanitize_html_ } from './index35-BGR9YwH8.js';
import { h as html } from './html-CfyvkLET.js';

function sanitize(source) {
  return sanitize_html_(source);
}
const standardHtmlTags = [
  "!--",
  "!doctype",
  "a",
  "abbr",
  "acronym",
  "address",
  "applet",
  "area",
  "article",
  "aside",
  "audio",
  "b",
  "base",
  "basefont",
  "bdi",
  "bdo",
  "big",
  "blockquote",
  "body",
  "br",
  "button",
  "canvas",
  "caption",
  "center",
  "cite",
  "code",
  "col",
  "colgroup",
  "data",
  "datalist",
  "dd",
  "del",
  "details",
  "dfn",
  "dialog",
  "dir",
  "div",
  "dl",
  "dt",
  "em",
  "embed",
  "fieldset",
  "figcaption",
  "figure",
  "font",
  "footer",
  "form",
  "frame",
  "frameset",
  "h1",
  "h2",
  "h3",
  "h4",
  "h5",
  "h6",
  "head",
  "header",
  "hgroup",
  "hr",
  "html",
  "i",
  "iframe",
  "img",
  "input",
  "ins",
  "kbd",
  "label",
  "legend",
  "li",
  "link",
  "main",
  "map",
  "mark",
  "menu",
  "meta",
  "meter",
  "nav",
  "noframes",
  "noscript",
  "object",
  "ol",
  "optgroup",
  "option",
  "output",
  "p",
  "param",
  "picture",
  "pre",
  "progress",
  "q",
  "rp",
  "rt",
  "ruby",
  "s",
  "samp",
  "script",
  "search",
  "section",
  "select",
  "small",
  "source",
  "span",
  "strike",
  "strong",
  "style",
  "sub",
  "summary",
  "sup",
  "svg",
  "table",
  "tbody",
  "td",
  "template",
  "textarea",
  "tfoot",
  "th",
  "thead",
  "time",
  "title",
  "tr",
  "track",
  "tt",
  "u",
  "ul",
  "var",
  "video",
  "wbr"
];
const svgTags = [
  // Base structural elements
  "g",
  "defs",
  "use",
  "symbol",
  // Shape elements
  "rect",
  "circle",
  "ellipse",
  "line",
  "polyline",
  "polygon",
  "path",
  "image",
  // Text elements
  "text",
  "tspan",
  "textPath",
  // Gradient and effects
  "linearGradient",
  "radialGradient",
  "stop",
  "pattern",
  "clipPath",
  "mask",
  "filter",
  // Filter effects
  "feBlend",
  "feColorMatrix",
  "feComponentTransfer",
  "feComposite",
  "feConvolveMatrix",
  "feDiffuseLighting",
  "feDisplacementMap",
  "feGaussianBlur",
  "feMerge",
  "feMorphology",
  "feOffset",
  "feSpecularLighting",
  "feTurbulence",
  "feMergeNode",
  "feFuncR",
  "feFuncG",
  "feFuncB",
  "feFuncA",
  "feDistantLight",
  "fePointLight",
  "feSpotLight",
  "feFlood",
  "feTile",
  // Animation elements
  "animate",
  "animateTransform",
  "animateMotion",
  "mpath",
  "set",
  // Interactive and other elements
  "view",
  "cursor",
  "foreignObject",
  "desc",
  "title",
  "metadata",
  "switch"
];
const standardHtmlAndSvgTags = [
  ...standardHtmlTags,
  ...svgTags.filter((tag) => !standardHtmlTags.includes(tag))
];
function MarkdownCode($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      chatbot = true,
      message,
      sanitize_html = true,
      latex_delimiters = [],
      render_markdown = true,
      line_breaks = true,
      header_links = false,
      allow_tags = false,
      theme_mode = "system"
    } = $$props;
    const marked = create_marked({
      header_links,
      line_breaks,
      latex_delimiters: latex_delimiters || []
    });
    let html$1 = (() => {
      if (message && message.trim()) {
        return process_message(message);
      } else {
        return "";
      }
    })();
    function escapeRegExp(string) {
      return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    }
    function escapeTags(content, tagsToEscape) {
      if (tagsToEscape === true) {
        const tagRegex = /<\/?([a-zA-Z][a-zA-Z0-9-]*)([\s>])/g;
        return content.replace(tagRegex, (match, tagName, endChar) => {
          if (!standardHtmlAndSvgTags.includes(tagName.toLowerCase())) {
            return match.replace(/</g, "&lt;").replace(/>/g, "&gt;");
          }
          return match;
        });
      }
      if (Array.isArray(tagsToEscape)) {
        const tagPattern = tagsToEscape.map((tag) => ({
          open: new RegExp(`<(${tag})(\\s+[^>]*)?>`, "gi"),
          close: new RegExp(`</(${tag})>`, "gi")
        }));
        let result = content;
        tagPattern.forEach((pattern) => {
          result = result.replace(pattern.open, (match) => match.replace(/</g, "&lt;").replace(/>/g, "&gt;"));
          result = result.replace(pattern.close, (match) => match.replace(/</g, "&lt;").replace(/>/g, "&gt;"));
        });
        return result;
      }
      return content;
    }
    function process_message(value) {
      let parsedValue = value;
      if (render_markdown) {
        const latexBlocks = [];
        latex_delimiters.forEach((delimiter, index) => {
          const leftDelimiter = escapeRegExp(delimiter.left);
          const rightDelimiter = escapeRegExp(delimiter.right);
          const regex = new RegExp(`${leftDelimiter}([\\s\\S]+?)${rightDelimiter}`, "g");
          parsedValue = parsedValue.replace(regex, (match, p1) => {
            latexBlocks.push(match);
            return `%%%LATEX_BLOCK_${latexBlocks.length - 1}%%%`;
          });
        });
        parsedValue = marked.parse(parsedValue);
        parsedValue = parsedValue.replace(/%%%LATEX_BLOCK_(\d+)%%%/g, (match, p1) => latexBlocks[parseInt(p1, 10)]);
      }
      if (allow_tags) {
        parsedValue = escapeTags(parsedValue, allow_tags);
      }
      if (sanitize_html && sanitize) {
        parsedValue = sanitize(parsedValue);
      }
      return parsedValue;
    }
    $$renderer2.push(`<span${attr_class("md svelte-1hf8a14", void 0, { "chatbot": chatbot, "prose": render_markdown })}>${html(html$1)}</span>`);
  });
}

export { MarkdownCode as M };
//# sourceMappingURL=MarkdownCode-ucE6Lq0M.js.map
