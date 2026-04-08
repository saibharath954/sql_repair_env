import './async-D55cHugf.js';
import { a as attr } from './index-K3l_dLem.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import { o as onDestroy } from './index-server-BzRj6e_1.js';
import './2-DKaY_6dX.js';
import { f as format_time } from './utils.svelte-D1m0ck_w.js';

"function" == typeof SuppressedError && SuppressedError;
function MinimalAudioPlayer($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { value, label, loop = false } = $$props;
    let duration = 0;
    value.url;
    onDestroy(() => {
    });
    $$renderer2.push(`<div class="minimal-audio-player svelte-15unlcf"${attr("aria-label", label || "Audio")}${attr("data-testid", label && typeof label === "string" && label.trim() ? "waveform-" + label : "unlabelled-audio")}><button class="play-btn svelte-15unlcf"${attr("aria-label", "Play")}>`);
    {
      $$renderer2.push("<!--[!-->");
      $$renderer2.push(`<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" class="svelte-15unlcf"><path d="M8 5.74537C8 5.06444 8.77346 4.64713 9.35139 5.02248L18.0227 10.2771C18.5518 10.6219 18.5518 11.3781 18.0227 11.7229L9.35139 16.9775C8.77346 17.3529 8 16.9356 8 16.2546V5.74537Z" fill="currentColor"></path></svg>`);
    }
    $$renderer2.push(`<!--]--></button> <div class="waveform-wrapper svelte-15unlcf"></div> <div class="timestamp svelte-15unlcf">${escape_html(format_time(duration))}</div></div>`);
  });
}

export { MinimalAudioPlayer as M };
//# sourceMappingURL=MinimalAudioPlayer-C_fkMDT-.js.map
