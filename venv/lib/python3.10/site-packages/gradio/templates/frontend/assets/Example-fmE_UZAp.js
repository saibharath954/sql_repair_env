import { p as prop, e as init, b as set_class } from './i18n-dpAHICcw.js';
import { R as push, t as template_effect, a as append, T as pop, U as flushSync, V as child, W as from_html, w as get, ae as derived_safe_equal, I as deep_read_state, z as untrack, Y as reset } from './index-CDZuCcOm.js';
import './ScrollFade.svelte_svelte_type_style_lang-DUvCSfRk.js';
import './MarkdownCode.svelte_svelte_type_style_lang-GeNUGzep.js';
import { I as Image } from './Image-CJziNDBt.js';
/* empty css                                                    */
import './StreamingBar.svelte_svelte_type_style_lang-BxBb9ZZb.js';
/* empty css                                                     */
import './Upload-BlCAnBIo.js';
/* empty css                                               */
import './snippet-DVkMfmSq.js';
import './prism-python-C_fanlsZ.js';
import './misc-C2MjMwBX.js';
/* empty css                                             */
import './html-h_YSgefI.js';
import './actions-BTh6ZJJ8.js';

var root = from_html(`<div><!></div>`);

function Example($$anchor, $$props) {
	push($$props, false);

	let value = prop($$props, 'value', 12);
	let type = prop($$props, 'type', 12);
	let selected = prop($$props, 'selected', 12, false);

	var $$exports = {
		get value() {
			return value();
		},

		set value($$value) {
			value($$value);
			flushSync();
		},

		get type() {
			return type();
		},

		set type($$value) {
			type($$value);
			flushSync();
		},

		get selected() {
			return selected();
		},

		set selected($$value) {
			selected($$value);
			flushSync();
		}
	};

	init();

	var div = root();
	let classes;
	var node = child(div);

	{
		let $0 = derived_safe_equal(() => (
			deep_read_state(value()),
			untrack(() => value().composite?.url || value().background?.url)
		));

		Image(node, {
			get src() {
				return get($0);
			},
			alt: ''
		});
	}

	reset(div);

	template_effect(() => classes = set_class(div, 1, 'container svelte-ous74z', null, classes, {
		table: type() === "table",
		gallery: type() === "gallery",
		selected: selected()
	}));

	append($$anchor, div);

	return pop($$exports);
}

export { Example as default };
//# sourceMappingURL=Example-fmE_UZAp.js.map
