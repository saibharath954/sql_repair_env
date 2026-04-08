import { i as if_block, r as rest_props, g as spread_props } from './i18n-dpAHICcw.js';
import { R as push, u as state, v as proxy, y as user_effect, S as first_child, w as get, a as append, X as sibling, a5 as user_derived, T as pop, x as set, N as tick, W as from_html } from './index-CDZuCcOm.js';
import { s as snapshot } from './clone-dZfS06Ds.js';
import { T as Textbox } from './Textbox-D1wbJ-Bi.js';
import { S as Static } from './index-DyDpuTN9.js';
import './StreamingBar.svelte_svelte_type_style_lang-BxBb9ZZb.js';
import { B as Block } from './Block-DntE23uJ.js';
import './MarkdownCode.svelte_svelte_type_style_lang-GeNUGzep.js';
import './ScrollFade.svelte_svelte_type_style_lang-DUvCSfRk.js';
import { G as Gradio } from './utils.svelte-CyWLYi-B.js';
export { default as BaseExample } from './Example-BF3XR3cU.js';
import './actions-BTh6ZJJ8.js';
import './input-UUW65DyE.js';
import './BlockTitle-Xgz-MKYS.js';
import './Info-CLoErKII.js';
import './MarkdownCode-Q694H4-C.js';
import './html-h_YSgefI.js';
import './Check-4kogBHUX.js';
import './Copy-C8W4pNlO.js';
import './Send-DHvsoBjG.js';
import './Square-Bg2evxzG.js';
import './IconButtonWrapper-KjCt2Pl8.js';
import './snippet-DVkMfmSq.js';
import './Clear-tvJMRS4J.js';
import './prism-python-C_fanlsZ.js';
import './size-CuuZBRle.js';
/* empty css                                               */

var root_1 = from_html(`<!> <!>`, 1);

function Index($$anchor, $$props) {
	push($$props, true);

	let _props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	const gradio = new Gradio(_props);
	let label = user_derived(() => gradio.shared.label || "Textbox");

	// Need to set the value to "" otherwise a change event gets
	// dispatched when the child sets it to ""
	gradio.props.value = gradio.props.value ?? "";

	let old_value = state(proxy(gradio.props.value));

	async function dispatch_change() {
		if (get(old_value) !== gradio.props.value) {
			set(old_value, gradio.props.value, true);
			await tick();
			gradio.dispatch("change", snapshot(gradio.props.value));
		}
	}

	async function handle_input(value) {
		if (!gradio.shared || !gradio.props) return;

		gradio.props.validation_error = null;
		gradio.props.value = value;
		await tick();
		gradio.dispatch("input");
	}

	user_effect(() => {
		dispatch_change();
	});

	function handle_change(value) {
		if (!gradio.shared || !gradio.props) return;

		gradio.props.validation_error = null;
		gradio.props.value = value;
	}

	Block($$anchor, {
		get visible() {
			return gradio.shared.visible;
		},

		get elem_id() {
			return gradio.shared.elem_id;
		},

		get elem_classes() {
			return gradio.shared.elem_classes;
		},

		get scale() {
			return gradio.shared.scale;
		},

		get min_width() {
			return gradio.shared.min_width;
		},
		allow_overflow: false,
		get padding() {
			return gradio.shared.container;
		},

		get rtl() {
			return gradio.props.rtl;
		},

		children: ($$anchor, $$slotProps) => {
			var fragment_1 = root_1();
			var node = first_child(fragment_1);

			{
				var consequent = ($$anchor) => {
					Static($$anchor, spread_props(
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
							show_validation_error: false,
							on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
						}
					));
				};

				if_block(node, ($$render) => {
					if (gradio.shared.loading_status) $$render(consequent);
				});
			}

			var node_1 = sibling(node, 2);

			{
				let $0 = user_derived(() => gradio.shared?.loading_status?.validation_error || gradio.shared?.validation_error);
				let $1 = user_derived(() => !gradio.shared.interactive);

				Textbox(node_1, {
					get label() {
						return get(label);
					},

					get info() {
						return gradio.props.info;
					},

					get show_label() {
						return gradio.shared.show_label;
					},

					get lines() {
						return gradio.props.lines;
					},

					get type() {
						return gradio.props.type;
					},

					get rtl() {
						return gradio.props.rtl;
					},

					get text_align() {
						return gradio.props.text_align;
					},

					get max_lines() {
						return gradio.props.max_lines;
					},

					get placeholder() {
						return gradio.props.placeholder;
					},

					get submit_btn() {
						return gradio.props.submit_btn;
					},

					get stop_btn() {
						return gradio.props.stop_btn;
					},

					get buttons() {
						return gradio.props.buttons;
					},

					get autofocus() {
						return gradio.props.autofocus;
					},

					get container() {
						return gradio.shared.container;
					},

					get autoscroll() {
						return gradio.shared.autoscroll;
					},

					get max_length() {
						return gradio.props.max_length;
					},

					get html_attributes() {
						return gradio.props.html_attributes;
					},

					get validation_error() {
						return get($0);
					},
					onchange: handle_change,
					oninput: handle_input,
					onsubmit: () => {
						gradio.shared.validation_error = null;
						gradio.dispatch("submit");
					},
					onblur: () => gradio.dispatch("blur"),
					onselect: (data) => gradio.dispatch("select", data),
					onfocus: () => gradio.dispatch("focus"),
					onstop: () => gradio.dispatch("stop"),
					oncopy: (data) => gradio.dispatch("copy", data),
					oncustombuttonclick: (id) => {
						gradio.dispatch("custom_button_click", { id });
					},

					get disabled() {
						return get($1);
					},

					get value() {
						return gradio.props.value;
					},

					set value($$value) {
						gradio.props.value = $$value;
					}
				});
			}

			append($$anchor, fragment_1);
		},
		$$slots: { default: true }
	});

	pop();
}

export { Textbox as BaseTextbox, Index as default };
//# sourceMappingURL=Index-A247jCMc.js.map
