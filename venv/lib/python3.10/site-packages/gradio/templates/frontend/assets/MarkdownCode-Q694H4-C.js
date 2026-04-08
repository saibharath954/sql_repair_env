const __vite__mapDeps=(i,m=__vite__mapDeps,d=(m.f||(m.f=["./katex-DvMfJMDS.css","./auto-render-YkwsO5wJ.js","./katex-CnjCUzRs.js","./mermaid.core-J5-DjKr9.js","./index-CDZuCcOm.js","./index-DYz7DaJH.css","./i18n-dpAHICcw.js","./i18n-UT_CQGRO.css","./step-TZOpqHBK.js","./dispatch-tQmgj1It.js","./select-k8gDf_61.js"])))=>i.map(i=>d[i]);
import { R as push, y as user_effect, t as template_effect, a as append, T as pop, W as from_html, w as get, a5 as user_derived, V as child, ag as __vitePreload, N as tick, Y as reset } from './index-CDZuCcOm.js';
import { p as prop, d as bind_this, b as set_class } from './i18n-dpAHICcw.js';
import { h as html } from './html-h_YSgefI.js';
import { A as Amuchina, c as create_marked } from './MarkdownCode.svelte_svelte_type_style_lang-GeNUGzep.js';

const is_external_url = (link, root = location.href) => {
  try {
    return !!link && new URL(link).origin !== new URL(root).origin;
  } catch (e) {
    return false;
  }
};
function sanitize(source) {
  const amuchina = new Amuchina();
  const node = new DOMParser().parseFromString(source, "text/html");
  walk_nodes(node.body, "A", (node2) => {
    if (node2 instanceof HTMLElement && "target" in node2) {
      if (is_external_url(node2.getAttribute("href"), location.href)) {
        node2.setAttribute("target", "_blank");
        node2.setAttribute("rel", "noopener noreferrer");
      }
    }
  });
  return amuchina.sanitize(node).body.innerHTML;
}
function walk_nodes(node, test, callback) {
  if (node && (node.nodeName === test || typeof test === "function")) {
    callback(node);
  }
  const children = node?.childNodes || [];
  for (let i = 0; i < children.length; i++) {
    walk_nodes(children[i], test, callback);
  }
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

var root = from_html(`<span><!></span>`);

function MarkdownCode($$anchor, $$props) {
	push($$props, true);

	let chatbot = prop($$props, 'chatbot', 3, true),
		sanitize_html = prop($$props, 'sanitize_html', 3, true),
		latex_delimiters = prop($$props, 'latex_delimiters', 19, () => []),
		render_markdown = prop($$props, 'render_markdown', 3, true),
		line_breaks = prop($$props, 'line_breaks', 3, true),
		header_links = prop($$props, 'header_links', 3, false),
		allow_tags = prop($$props, 'allow_tags', 3, false),
		theme_mode = prop($$props, 'theme_mode', 3, "system");

	let el;

	const marked = create_marked({
		header_links: header_links(),
		line_breaks: line_breaks(),
		latex_delimiters: latex_delimiters() || []
	});

	let html$1 = user_derived(() => {
		if ($$props.message && $$props.message.trim()) {
			return process_message($$props.message);
		} else {
			return "";
		}
	});

	let katex_loaded = false;

	function has_math_syntax(text) {
		if (!latex_delimiters() || latex_delimiters().length === 0) {
			return false;
		}

		return latex_delimiters().some((delimiter) => text.includes(delimiter.left) && text.includes(delimiter.right));
	}

	function escapeRegExp(string) {
		return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
	}

	function escapeTags(content, tagsToEscape) {
		if (tagsToEscape === true) {
			// https://www.w3schools.com/tags/
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

		if (render_markdown()) {
			const latexBlocks = [];

			latex_delimiters().forEach((delimiter, index) => {
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

		if (allow_tags()) {
			parsedValue = escapeTags(parsedValue, allow_tags());
		}

		if (sanitize_html() && sanitize) {
			parsedValue = sanitize(parsedValue);
		}

		return parsedValue;
	}

	async function render_html(value) {
		if (latex_delimiters().length > 0 && value && has_math_syntax(value)) {
			if (!katex_loaded) {
				await Promise.all([
					__vitePreload(() => Promise.resolve({}),true              ?__vite__mapDeps([0]):void 0,import.meta.url),
					__vitePreload(() => import('./auto-render-YkwsO5wJ.js'),true              ?__vite__mapDeps([1,2]):void 0,import.meta.url)
				]).then(([, { default: render_math_in_element }]) => {
					katex_loaded = true;
					render_math_in_element(el, { delimiters: latex_delimiters(), throwOnError: false });
				});
			} else {
				const { default: render_math_in_element } = await __vitePreload(async () => { const { default: render_math_in_element } = await import('./auto-render-YkwsO5wJ.js');return { default: render_math_in_element }},true              ?__vite__mapDeps([1,2]):void 0,import.meta.url);

				render_math_in_element(el, { delimiters: latex_delimiters(), throwOnError: false });
			}
		}

		if (el) {
			const mermaidDivs = el.querySelectorAll(".mermaid");

			if (mermaidDivs.length > 0) {
				await tick();

				const { default: mermaid } = await __vitePreload(async () => { const { default: mermaid } = await import('./mermaid.core-J5-DjKr9.js').then(n => n.b5);return { default: mermaid }},true              ?__vite__mapDeps([3,4,5,6,7,8,9,10]):void 0,import.meta.url);

				mermaid.initialize({
					startOnLoad: false,
					theme: theme_mode() === "dark" ? "dark" : "default",
					securityLevel: "antiscript"
				});

				await mermaid.run({ nodes: Array.from(mermaidDivs).map((node) => node) });
			}
		}
	}

	user_effect(() => {
		if (el && document.body.contains(el)) {
			render_html($$props.message);
		} else {
			console.error("Element is not in the DOM");
		}
	});

	var span = root();
	let classes;
	var node_1 = child(span);

	html(node_1, () => get(html$1));
	reset(span);
	bind_this(span, ($$value) => el = $$value, () => el);
	template_effect(() => classes = set_class(span, 1, 'md svelte-1hf8a14', null, classes, { chatbot: chatbot(), prose: render_markdown() }));
	append($$anchor, span);
	pop();
}

export { MarkdownCode as M };
//# sourceMappingURL=MarkdownCode-Q694H4-C.js.map
