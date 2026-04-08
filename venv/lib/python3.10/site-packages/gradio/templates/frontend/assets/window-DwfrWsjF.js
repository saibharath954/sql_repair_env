import { ac as listen, ad as without_reactive_context } from './index-CDZuCcOm.js';

/**
 * @param {'innerWidth' | 'innerHeight' | 'outerWidth' | 'outerHeight'} type
 * @param {(size: number) => void} set
 */
function bind_window_size(type, set) {
	listen(window, ['resize'], () => without_reactive_context(() => set(window[type])));
}

export { bind_window_size as b };
//# sourceMappingURL=window-DwfrWsjF.js.map
