import { p as prop } from './i18n-dpAHICcw.js';
import { R as push, a as append, T as pop, U as flushSync, V as child, W as from_html, Y as reset } from './index-CDZuCcOm.js';
import { M as MarkdownCode } from './MarkdownCode-Q694H4-C.js';
import './ScrollFade.svelte_svelte_type_style_lang-DUvCSfRk.js';

var root = from_html(`<div class="svelte-9hc4ua"><!></div>`);

function Info($$anchor, $$props) {
	push($$props, false);

	let info = prop($$props, 'info', 12);

	var $$exports = {
		get info() {
			return info();
		},

		set info($$value) {
			info($$value);
			flushSync();
		}
	};

	var div = root();
	var node = child(div);

	MarkdownCode(node, {
		get message() {
			return info();
		},
		sanitize_html: true
	});

	reset(div);
	append($$anchor, div);

	return pop($$exports);
}

export { Info as I };
//# sourceMappingURL=Info-CLoErKII.js.map
