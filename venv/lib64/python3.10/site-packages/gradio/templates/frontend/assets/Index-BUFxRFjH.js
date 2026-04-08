import { p as prop, s as slot } from './i18n-dpAHICcw.js';
import { R as push, a6 as comment, S as first_child, a as append, T as pop, U as flushSync } from './index-CDZuCcOm.js';
import { B as Block } from './Block-DntE23uJ.js';
import './MarkdownCode.svelte_svelte_type_style_lang-GeNUGzep.js';
import './ScrollFade.svelte_svelte_type_style_lang-DUvCSfRk.js';
import './prism-python-C_fanlsZ.js';
import './snippet-DVkMfmSq.js';

function Index($$anchor, $$props) {
	push($$props, false);

	let elem_id = prop($$props, 'elem_id', 12);
	let elem_classes = prop($$props, 'elem_classes', 12);
	let visible = prop($$props, 'visible', 12, true);

	var $$exports = {
		get elem_id() {
			return elem_id();
		},

		set elem_id($$value) {
			elem_id($$value);
			flushSync();
		},

		get elem_classes() {
			return elem_classes();
		},

		set elem_classes($$value) {
			elem_classes($$value);
			flushSync();
		},

		get visible() {
			return visible();
		},

		set visible($$value) {
			visible($$value);
			flushSync();
		}
	};

	Block($$anchor, {
		get elem_id() {
			return elem_id();
		},

		get elem_classes() {
			return elem_classes();
		},

		get visible() {
			return visible();
		},
		explicit_call: true,
		children: ($$anchor, $$slotProps) => {
			var fragment_1 = comment();
			var node = first_child(fragment_1);

			slot(node, $$props, 'default', {}, null);
			append($$anchor, fragment_1);
		},
		$$slots: { default: true }
	});

	return pop($$exports);
}

export { Index as default };
//# sourceMappingURL=Index-BUFxRFjH.js.map
