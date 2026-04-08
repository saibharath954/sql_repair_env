import { p as prop, e as init, i as if_block, b as set_class } from './i18n-dpAHICcw.js';
import { R as push, S as first_child, X as sibling, w as get, x as set, ae as derived_safe_equal, I as deep_read_state, z as untrack, a as append, a7 as text, t as template_effect, a0 as set_text, V as child, T as pop, a3 as mutable_source, U as flushSync, W as from_html, Y as reset, a8 as next } from './index-CDZuCcOm.js';
import { h as html } from './html-h_YSgefI.js';
import { B as BaseForm } from './BaseForm-BSER6dDx.js';
import { T as Textbox } from './Textbox-D1wbJ-Bi.js';
import './StreamingBar.svelte_svelte_type_style_lang-BxBb9ZZb.js';
import { B as Block } from './Block-DntE23uJ.js';
import './MarkdownCode.svelte_svelte_type_style_lang-GeNUGzep.js';
import './ScrollFade.svelte_svelte_type_style_lang-DUvCSfRk.js';
/* empty css                                               */
import { B as Button } from './Button-DxE-syeF.js';
import { B as BaseColumn } from './Index.svelte_svelte_type_style_lang-CXhnGNdZ.js';
import './actions-BTh6ZJJ8.js';
import './input-UUW65DyE.js';
import './BlockTitle-Xgz-MKYS.js';
import './Info-CLoErKII.js';
import './MarkdownCode-Q694H4-C.js';
import './Check-4kogBHUX.js';
import './Copy-C8W4pNlO.js';
import './Send-DHvsoBjG.js';
import './Square-Bg2evxzG.js';
import './IconButtonWrapper-KjCt2Pl8.js';
import './snippet-DVkMfmSq.js';
import './prism-python-C_fanlsZ.js';
import './Image-CJziNDBt.js';
import './misc-C2MjMwBX.js';
/* empty css                                             */
/* empty css                                                    */
import './index-DyDpuTN9.js';
import './Clear-tvJMRS4J.js';

var root_3 = from_html(`<p class="auth svelte-14xt79u"><!></p>`);
var root_4 = from_html(`<p class="auth svelte-14xt79u"> </p>`);
var root_5 = from_html(`<p class="creds svelte-14xt79u"> </p>`);
var root_6 = from_html(`<!> <!>`, 1);
var root_2 = from_html(`<h2 class="svelte-14xt79u"> </h2> <!> <!> <!> <!> <!>`, 1);
var root_1 = from_html(`<div><!></div>`);

function Login($$anchor, $$props) {
	push($$props, false);

	let root = prop($$props, 'root', 12);
	let auth_message = prop($$props, 'auth_message', 12);
	let app_mode = prop($$props, 'app_mode', 12);
	let space_id = prop($$props, 'space_id', 12);
	let i18n = prop($$props, 'i18n', 12);
	let username = mutable_source("");
	let password = mutable_source("");
	let incorrect_credentials = mutable_source(false);

	const submit = async () => {
		const formData = new FormData();

		formData.append("username", get(username));
		formData.append("password", get(password));

		// Use URL constructor to properly join paths and avoid double slashes
		const login_url = new URL("login", root().endsWith("/") ? root() : root() + "/").href;

		let response = await fetch(login_url, { method: "POST", body: formData });

		if (response.status === 400) {
			set(incorrect_credentials, true);
			set(username, "");
			set(password, "");
		} else if (response.status == 200) {
			location.reload();
		}
	};

	var $$exports = {
		get root() {
			return root();
		},

		set root($$value) {
			root($$value);
			flushSync();
		},

		get auth_message() {
			return auth_message();
		},

		set auth_message($$value) {
			auth_message($$value);
			flushSync();
		},

		get app_mode() {
			return app_mode();
		},

		set app_mode($$value) {
			app_mode($$value);
			flushSync();
		},

		get space_id() {
			return space_id();
		},

		set space_id($$value) {
			space_id($$value);
			flushSync();
		},

		get i18n() {
			return i18n();
		},

		set i18n($$value) {
			i18n($$value);
			flushSync();
		}
	};

	init();

	var div = root_1();
	let classes;
	var node = child(div);

	BaseColumn(node, {
		variant: 'panel',
		min_width: 480,
		children: ($$anchor, $$slotProps) => {
			var fragment = root_2();
			var h2 = first_child(fragment);
			var text$1 = child(h2, true);

			reset(h2);

			var node_1 = sibling(h2, 2);

			{
				var consequent = ($$anchor) => {
					var p = root_3();
					var node_2 = child(p);

					html(node_2, auth_message);
					reset(p);
					append($$anchor, p);
				};

				if_block(node_1, ($$render) => {
					if (auth_message()) $$render(consequent);
				});
			}

			var node_3 = sibling(node_1, 2);

			{
				var consequent_1 = ($$anchor) => {
					var p_1 = root_4();
					var text_1 = child(p_1, true);

					reset(p_1);

					template_effect(($0) => set_text(text_1, $0), [
						() => (
							deep_read_state(i18n()),
							untrack(() => i18n()("login.enable_cookies"))
						)
					]);

					append($$anchor, p_1);
				};

				if_block(node_3, ($$render) => {
					if (space_id()) $$render(consequent_1);
				});
			}

			var node_4 = sibling(node_3, 2);

			{
				var consequent_2 = ($$anchor) => {
					var p_2 = root_5();
					var text_2 = child(p_2, true);

					reset(p_2);

					template_effect(($0) => set_text(text_2, $0), [
						() => (
							deep_read_state(i18n()),
							untrack(() => i18n()("login.incorrect_credentials"))
						)
					]);

					append($$anchor, p_2);
				};

				if_block(node_4, ($$render) => {
					if (get(incorrect_credentials)) $$render(consequent_2);
				});
			}

			var node_5 = sibling(node_4, 2);

			BaseForm(node_5, {
				children: ($$anchor, $$slotProps) => {
					var fragment_1 = root_6();
					var node_6 = first_child(fragment_1);

					Block(node_6, {
						children: ($$anchor, $$slotProps) => {
							{
								let $0 = derived_safe_equal(() => (
									deep_read_state(i18n()),
									untrack(() => i18n()("login.username"))
								));

								Textbox($$anchor, {
									get label() {
										return get($0);
									},
									lines: 1,
									show_label: true,
									max_lines: 1,
									onsubmit: submit,
									get value() {
										return get(username);
									},

									set value($$value) {
										set(username, $$value);
									},
									$$legacy: true
								});
							}
						},
						$$slots: { default: true }
					});

					var node_7 = sibling(node_6, 2);

					Block(node_7, {
						children: ($$anchor, $$slotProps) => {
							{
								let $0 = derived_safe_equal(() => (
									deep_read_state(i18n()),
									untrack(() => i18n()("login.password"))
								));

								Textbox($$anchor, {
									get label() {
										return get($0);
									},
									lines: 1,
									show_label: true,
									max_lines: 1,
									type: 'password',
									onsubmit: submit,
									get value() {
										return get(password);
									},

									set value($$value) {
										set(password, $$value);
									},
									$$legacy: true
								});
							}
						},
						$$slots: { default: true }
					});

					append($$anchor, fragment_1);
				},
				$$slots: { default: true }
			});

			var node_8 = sibling(node_5, 2);

			Button(node_8, {
				size: 'lg',
				variant: 'primary',
				onclick: submit,
				children: ($$anchor, $$slotProps) => {
					next();

					var text_3 = text();

					template_effect(($0) => set_text(text_3, $0), [
						() => (
							deep_read_state(i18n()),
							untrack(() => i18n()("login.login"))
						)
					]);

					append($$anchor, text_3);
				},
				$$slots: { default: true }
			});

			template_effect(($0) => set_text(text$1, $0), [
				() => (
					deep_read_state(i18n()),
					untrack(() => i18n()("login.login"))
				)
			]);

			append($$anchor, fragment);
		},
		$$slots: { default: true }
	});

	reset(div);
	template_effect(() => classes = set_class(div, 1, 'wrap svelte-14xt79u', null, classes, { 'min-h-screen': app_mode() }));
	append($$anchor, div);

	return pop($$exports);
}

export { Login as default };
//# sourceMappingURL=Login-DE0IVRzC.js.map
