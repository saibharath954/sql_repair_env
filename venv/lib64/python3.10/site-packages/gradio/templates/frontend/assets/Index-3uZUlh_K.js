import { r as rest_props } from './i18n-dpAHICcw.js';
import { R as push, y as user_effect, T as pop } from './index-CDZuCcOm.js';
import { G as Gradio } from './utils.svelte-CyWLYi-B.js';
import './clone-dZfS06Ds.js';

function Index($$anchor, $$props) {
	push($$props, true);

	let props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	const gradio = new Gradio(props);

	user_effect(() => {
		gradio.props.value && gradio.dispatch("change");
	});

	pop();
}

export { Index as default };
//# sourceMappingURL=Index-3uZUlh_K.js.map
