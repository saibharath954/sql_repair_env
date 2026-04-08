import { i as if_block, g as spread_props, r as rest_props } from './i18n-dpAHICcw.js';
import { R as push, u as state, v as proxy, y as user_effect, w as get, x as set, S as first_child, a as append, X as sibling, a5 as user_derived, T as pop, W as from_html, V as child, Y as reset } from './index-CDZuCcOm.js';
import { b as bind_element_size } from './size-CuuZBRle.js';
import { G as Gradio } from './utils.svelte-CyWLYi-B.js';
import { J as JSON_1, a as JSON } from './JSON-CeWOiUsf.js';
import { B as Block } from './Block-DntE23uJ.js';
import './MarkdownCode.svelte_svelte_type_style_lang-GeNUGzep.js';
import './ScrollFade.svelte_svelte_type_style_lang-DUvCSfRk.js';
import { B as BlockLabel } from './BlockLabel-D4yjUUAn.js';
import { S as Static } from './index-DyDpuTN9.js';
import './StreamingBar.svelte_svelte_type_style_lang-BxBb9ZZb.js';
import './clone-dZfS06Ds.js';
import './Check-4kogBHUX.js';
import './Copy-C8W4pNlO.js';
import './Empty-617iGDfy.js';
import './IconButtonWrapper-KjCt2Pl8.js';
import './snippet-DVkMfmSq.js';
import './prism-python-C_fanlsZ.js';
import './Clear-tvJMRS4J.js';
import './html-h_YSgefI.js';

var root_1 = from_html(`<div><!></div> <!> <!>`, 1);

function Index($$anchor, $$props) {
	push($$props, true);

	const props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	const gradio = new Gradio(props);
	let old_value = state(proxy(gradio.props.value));

	user_effect(() => {
		if (get(old_value) !== gradio.props.value) {
			set(old_value, gradio.props.value, true);
			gradio.dispatch("change");
		}
	});

	let label_height = state(0);

	Block($$anchor, {
		get visible() {
			return gradio.shared.visible;
		},
		test_id: 'json',
		get elem_id() {
			return gradio.shared.elem_id;
		},

		get elem_classes() {
			return gradio.shared.elem_classes;
		},

		get container() {
			return gradio.shared.container;
		},

		get scale() {
			return gradio.shared.scale;
		},

		get min_width() {
			return gradio.shared.min_width;
		},
		padding: false,
		allow_overflow: true,
		overflow_behavior: 'auto',
		get height() {
			return gradio.props.height;
		},

		get min_height() {
			return gradio.props.min_height;
		},

		get max_height() {
			return gradio.props.max_height;
		},

		children: ($$anchor, $$slotProps) => {
			var fragment_1 = root_1();
			var div = first_child(fragment_1);
			var node = child(div);

			{
				var consequent = ($$anchor) => {
					{
						let $0 = user_derived(() => gradio.shared.container === false);

						BlockLabel($$anchor, {
							get Icon() {
								return JSON;
							},

							get show_label() {
								return gradio.shared.show_label;
							},

							get label() {
								return gradio.shared.label;
							},
							float: false,
							get disable() {
								return get($0);
							}
						});
					}
				};

				if_block(node, ($$render) => {
					if (gradio.shared.label) $$render(consequent);
				});
			}

			reset(div);

			var node_1 = sibling(div, 2);

			Static(node_1, spread_props(
				{
					get autoscroll() {
						return gradio.shared.autoscroll;
					},

					get i18n() {
						return gradio.i18n;
					}
				},
				() => gradio.shared.loading_status,
				{
					on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
				}
			));

			var node_2 = sibling(node_1, 2);

			{
				let $0 = user_derived(() => gradio.props.buttons == null
					? true
					: gradio.props.buttons.some((btn) => typeof btn === "string" && btn === "copy"));

				JSON_1(node_2, {
					get value() {
						return gradio.props.value;
					},

					get open() {
						return gradio.props.open;
					},

					get theme_mode() {
						return gradio.props.theme_mode;
					},

					get show_indices() {
						return gradio.props.show_indices;
					},

					get show_copy_button() {
						return get($0);
					},

					get buttons() {
						return gradio.props.buttons;
					},

					on_custom_button_click: (id) => {
						gradio.dispatch("custom_button_click", { id });
					},

					get label_height() {
						return get(label_height);
					}
				});
			}

			bind_element_size(div, 'clientHeight', ($$value) => set(label_height, $$value));
			append($$anchor, fragment_1);
		},
		$$slots: { default: true }
	});

	pop();
}

export { JSON_1 as BaseJSON, Index as default };
//# sourceMappingURL=Index-BJGdaFJX.js.map
