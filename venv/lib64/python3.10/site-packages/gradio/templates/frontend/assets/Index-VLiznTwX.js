import { i as if_block, g as spread_props, b as set_class, f as set_style, r as rest_props } from './i18n-dpAHICcw.js';
import { R as push, u as state, v as proxy, y as user_effect, w as get, x as set, S as first_child, t as template_effect, a as append, X as sibling, V as child, T as pop, a5 as user_derived, av as prepare_files, W as from_html, Y as reset } from './index-CDZuCcOm.js';
import { G as Gradio, c as css_units } from './utils.svelte-CyWLYi-B.js';
import HTML from './HTML-B2dUNNxJ.js';
import { S as Static } from './index-DyDpuTN9.js';
import './StreamingBar.svelte_svelte_type_style_lang-BxBb9ZZb.js';
import { B as Block } from './Block-DntE23uJ.js';
import './MarkdownCode.svelte_svelte_type_style_lang-GeNUGzep.js';
import './ScrollFade.svelte_svelte_type_style_lang-DUvCSfRk.js';
import { B as BlockLabel } from './BlockLabel-D4yjUUAn.js';
import { C as Code } from './Code-DG4-IzG8.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-KjCt2Pl8.js';
export { default as BaseExample } from './Example-wUwiRzAM.js';
import './clone-dZfS06Ds.js';
import './snippet-DVkMfmSq.js';
import './Clear-tvJMRS4J.js';
import './html-h_YSgefI.js';
import './prism-python-C_fanlsZ.js';

var root_1 = from_html(`<!> <!> <!> <div><!></div>`, 1);

function Index($$anchor, $$props) {
	push($$props, true);

	let props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	let children = $$props.children;
	const gradio = new Gradio(props);

	let _props = user_derived(() => ({
		value: gradio.props.value ?? "",
		label: gradio.shared.label,
		visible: gradio.shared.visible,
		...gradio.props.props
	}));

	let old_value = state(proxy(gradio.props.value));

	user_effect(() => {
		if (JSON.stringify(get(old_value)) !== JSON.stringify(gradio.props.value)) {
			set(old_value, gradio.props.value, true);
			gradio.dispatch("change");
		}
	});

	let watch_entries = [];

	function watch(propOrProps, callback) {
		const prop_list = Array.isArray(propOrProps) ? propOrProps : [propOrProps];

		watch_entries.push({ props: prop_list, callback });
	}

	function fire_watchers(changed_keys) {
		const seen = new Set();

		for (const entry of watch_entries) {
			if (entry.props.some((k) => changed_keys.includes(k))) {
				seen.add(entry);
			}
		}

		for (const entry of seen) {
			try {
				entry.callback();
			} catch(e) {
				console.error("Error in watch callback:", e);
			}
		}
	}

	async function upload(file) {
		try {
			const file_data = await prepare_files([file]);
			const result = await gradio.shared.client.upload(file_data, gradio.shared.root, undefined, gradio.shared.max_file_size ?? undefined);

			if (result && result[0]) {
				return { path: result[0].path, url: result[0].url };
			}

			throw new Error("Upload failed");
		} catch(e) {
			gradio.dispatch("error", e instanceof Error ? e.message : String(e));

			throw e;
		}
	}

	{
		let $0 = user_derived(() => gradio.props.padding !== false);

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

			get container() {
				return gradio.shared.container;
			},

			get padding() {
				return get($0);
			},
			overflow_behavior: 'visible',
			children: ($$anchor, $$slotProps) => {
				var fragment_1 = root_1();
				var node = first_child(fragment_1);

				{
					var consequent = ($$anchor) => {
						IconButtonWrapper($$anchor, {
							get buttons() {
								return gradio.props.buttons;
							},

							on_custom_button_click: (id) => {
								gradio.dispatch("custom_button_click", { id });
							}
						});
					};

					if_block(node, ($$render) => {
						if (gradio.shared.show_label && gradio.props.buttons && gradio.props.buttons.length > 0) $$render(consequent);
					});
				}

				var node_1 = sibling(node, 2);

				{
					var consequent_1 = ($$anchor) => {
						BlockLabel($$anchor, {
							get Icon() {
								return Code;
							},

							get show_label() {
								return gradio.shared.show_label;
							},

							get label() {
								return gradio.shared.label;
							},
							float: true
						});
					};

					if_block(node_1, ($$render) => {
						if (gradio.shared.show_label) $$render(consequent_1);
					});
				}

				var node_2 = sibling(node_1, 2);

				Static(node_2, spread_props(
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
						variant: 'center',
						on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
					}
				));

				var div = sibling(node_2, 2);
				let classes;
				let styles;
				var node_3 = child(div);

				HTML(node_3, {
					get props() {
						return get(_props);
					},

					get html_template() {
						return gradio.props.html_template;
					},

					get css_template() {
						return gradio.props.css_template;
					},

					get js_on_load() {
						return gradio.props.js_on_load;
					},

					get elem_classes() {
						return gradio.shared.elem_classes;
					},

					get visible() {
						return gradio.shared.visible;
					},

					get autoscroll() {
						return gradio.shared.autoscroll;
					},

					get apply_default_css() {
						return gradio.props.apply_default_css;
					},

					get head() {
						return gradio.props.head;
					},

					get component_class_name() {
						return gradio.props.component_class_name;
					},
					upload,
					get server() {
						return gradio.shared.server;
					},
					watch_fn: watch,
					fire_watchers,
					$$events: {
						event: (e) => {
							gradio.dispatch(e.detail.type, e.detail.data);
						},

						update_value: (e) => {
							if (e.detail.property === "value") {
								gradio.props.value = e.detail.data;
							} else if (e.detail.property === "label") {
								gradio.shared.label = e.detail.data;
							} else if (e.detail.property === "visible") {
								gradio.shared.visible = e.detail.data;
							}
						}
					},

					children: ($$anchor, $$slotProps) => {
						children?.($$anchor);
					},
					$$slots: { default: true }
				});

				reset(div);

				template_effect(
					($0) => {
						classes = set_class(div, 1, 'html-container svelte-1jts93g', null, classes, {
							pending: gradio.shared.loading_status?.status === "pending" && gradio.shared.loading_status?.show_progress !== "hidden",
							'label-padding': gradio.shared.show_label ?? undefined
						});

						styles = set_style(div, '', styles, $0);
					},
					[
						() => ({
							'min-height': gradio.props.min_height && gradio.shared.loading_status?.status !== "pending" ? css_units(gradio.props.min_height) : undefined,
							'max-height': gradio.props.max_height ? css_units(gradio.props.max_height) : undefined,
							'overflow-y': gradio.props.max_height ? "auto" : undefined
						})
					]
				);

				append($$anchor, fragment_1);
			},
			$$slots: { default: true }
		});
	}

	pop();
}

export { HTML as BaseHTML, Index as default };
//# sourceMappingURL=Index-VLiznTwX.js.map
