import { g as spread_props, r as rest_props, s as slot } from './i18n-dpAHICcw.js';
import { R as push, T as pop, a6 as comment, S as first_child, a as append } from './index-CDZuCcOm.js';
import { G as Gradio } from './utils.svelte-CyWLYi-B.js';
import { B as BaseColumn } from './Index.svelte_svelte_type_style_lang-CXhnGNdZ.js';
import './clone-dZfS06Ds.js';
import './index-DyDpuTN9.js';
import './StreamingBar.svelte_svelte_type_style_lang-BxBb9ZZb.js';
import './html-h_YSgefI.js';
import './ScrollFade.svelte_svelte_type_style_lang-DUvCSfRk.js';
import './snippet-DVkMfmSq.js';
import './MarkdownCode.svelte_svelte_type_style_lang-GeNUGzep.js';
import './prism-python-C_fanlsZ.js';
import './Clear-tvJMRS4J.js';

function Index($$anchor, $$props) {
	push($$props, true);

	let props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	const gradio = new Gradio(props);

	BaseColumn($$anchor, spread_props(() => gradio.shared, {
		children: ($$anchor, $$slotProps) => {
			var fragment_1 = comment();
			var node = first_child(fragment_1);

			slot(node, $$props, 'default', {}, null);
			append($$anchor, fragment_1);
		},
		$$slots: { default: true }
	}));

	pop();
}

export { BaseColumn, Index as default };
//# sourceMappingURL=Index-4EGr9mxP.js.map
