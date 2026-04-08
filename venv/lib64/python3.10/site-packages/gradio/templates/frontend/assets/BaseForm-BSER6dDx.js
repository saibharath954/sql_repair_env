import { p as prop, s as slot, b as set_class, a as set_attribute, f as set_style } from './i18n-dpAHICcw.js';
import { t as template_effect, a as append, V as child, W as from_html, Y as reset } from './index-CDZuCcOm.js';

var root = from_html(`<div><!></div>`);

function BaseForm($$anchor, $$props) {

	let label = prop($$props, 'label', 3, undefined);
	var div = root();
	let classes;
	let styles;
	var node = child(div);

	slot(node, $$props, 'default', {}, null);
	reset(div);

	template_effect(() => {
		classes = set_class(div, 1, 'form svelte-d5xbca', null, classes, {
			hidden: $$props.visible === false,
			'hidden-css': $$props.visible === "hidden"
		});

		set_attribute(div, 'role', label() ? "group" : undefined);
		set_attribute(div, 'aria-label', label());

		styles = set_style(div, '', styles, {
			'flex-grow': $$props.scale,
			'min-width': `calc(min(${$$props.min_width}px, 100%))`
		});
	});

	append($$anchor, div);
}

export { BaseForm as B };
//# sourceMappingURL=BaseForm-BSER6dDx.js.map
