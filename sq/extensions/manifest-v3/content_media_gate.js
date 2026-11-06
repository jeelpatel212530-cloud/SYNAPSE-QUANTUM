// ============================================================================
// SYNAPSE-QUANTUM: Node 9 Neuromorphic Media Gate Content Script
// Strips YouTube recommendations/comments and locks playback until the operator answers a
// freshly generated, shuffled optics / Heston question. Numbers are randomised per attempt so
// answers cannot be memorised; the correct option is in a random position.
// ============================================================================

(function () {
  const pick = (xs) => xs[Math.floor(Math.random() * xs.length)];
  const fmt = (x) => (x < 1e-3 || x >= 1e4 ? x.toExponential(2) : Number(x.toPrecision(3)).toString());

  function shuffle(options, correct) {
    const idx = options.map((_, i) => i);
    for (let i = idx.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    return { options: idx.map((i) => options[i]), correctIndex: idx.indexOf(correct) };
  }

  const GENERATORS = [
    function twoPhotonDepth() {
      const ls = pick([120, 150, 180, 220]);
      const z = pick([200, 300, 450, 600, 900]);
      const ok = Math.exp((-2 * z) / ls);
      const s = shuffle([
        `exp(-2z/l_s) = ${fmt(ok)}`,
        `exp(-z/l_s) = ${fmt(Math.exp(-z / ls))}`,
        `exp(-4z/l_s) = ${fmt(Math.exp((-4 * z) / ls))}`,
      ], 0);
      return { q: `[Wave Optics] Ballistic two-photon excitation at z = ${z} µm with scattering mean free path l_s = ${ls} µm. What is the attenuation factor?`, ...s };
    },
    function feller() {
      const kappa = pick([1.2, 1.5, 2, 2.5, 3]);
      const theta = pick([0.02, 0.04, 0.06, 0.09]);
      const sigma = pick([0.2, 0.35, 0.5, 0.7, 0.95]);
      const lhs = 2 * kappa * theta, rhs = sigma * sigma;
      const holds = lhs > rhs;
      const a = `Feller holds (2κθ = ${fmt(lhs)} > σ² = ${fmt(rhs)}): v(t) stays strictly positive`;
      const b = `Feller violated (2κθ = ${fmt(lhs)} < σ² = ${fmt(rhs)}): v(t) can reach zero`;
      const s = shuffle([holds ? a : b, holds ? b : a, 'Feller depends only on ρ, so v(t) is never degenerate'], 0);
      return { q: `[Stochastic Calculus] Heston with κ = ${kappa}, θ = ${theta}, σ = ${sigma}. What is the variance boundary behaviour?`, ...s };
    },
    function power() {
      const p = pick([20, 35, 50, 80]);
      const f = pick([2, 3]);
      const s = shuffle([
        `×${f * f} (two-photon excitation scales as P²)`,
        `×${f} (linear in P)`,
        `×${f * f * f} (cubic in P)`,
      ], 0);
      return { q: `[Wave Optics] Two-photon laser power is raised from ${p} mW to ${p * f} mW at fixed focus. By what factor does excitation probability rise?`, ...s };
    },
    function chr2() {
      const s = shuffle([
        'Inward depolarising cation current driving V_m toward threshold',
        'Hyperpolarising outward K+ current that silences firing',
        'Permanent Na+ inactivation (h = 0)',
      ], 0);
      return { q: '[Electrophysiology] When ChR2 enters the open state under two-photon irradiance, how does I_ChR2 enter the Hodgkin-Huxley membrane equation?', ...s };
    },
  ];

  let overlayEl = null;

  function stripExtraneousMediaDOM() {
    const styleId = 'synapse-quantum-media-gate-style';
    if (document.getElementById(styleId)) return;
    const style = document.createElement('style');
    style.id = styleId;
    style.textContent = `#related, #comments, ytd-watch-next-secondary-results-renderer, ytd-reel-shelf-renderer { display: none !important; }`;
    document.head.appendChild(style);
  }

  function pauseAllVideos() {
    document.querySelectorAll('video').forEach((v) => { if (!v.paused) v.pause(); });
  }

  function injectDerivationGateOverlay() {
    if (overlayEl) return;
    pauseAllVideos();
    overlayEl = document.createElement('div');
    overlayEl.id = 'synapse-quantum-gate-modal';
    overlayEl.style.cssText = `position: fixed; inset: 0; z-index: 2147483647; background: rgba(5, 8, 15, 0.96); color: #e2e8f0; font-family: 'JetBrains Mono', monospace; display: flex; align-items: center; justify-content: center; padding: 24px;`;

    const card = document.createElement('div');
    card.style.cssText = `max-width: 680px; width: 100%; background: #0c1322; border: 1px solid #00f5d4; border-radius: 8px; padding: 24px; box-shadow: 0 20px 50px rgba(0,0,0,0.8);`;

    const item = pick(GENERATORS)();

    const mk = (tag, css, text) => { const el = document.createElement(tag); el.style.cssText = css; el.textContent = text; return el; };
    card.appendChild(mk('div', 'color:#00f5d4;font-size:12px;letter-spacing:1px;margin-bottom:6px;', 'SYNAPSE-QUANTUM // NODE 9 NEUROMORPHIC MEDIA GATE'));
    card.appendChild(mk('h2', 'font-size:18px;margin-bottom:12px;color:#38bdf8;', 'Academic Playback Locked: answer to unlock 60 minutes'));
    card.appendChild(mk('div', 'background:#070c17;border:1px solid #1e293b;padding:14px;border-radius:6px;margin-bottom:16px;font-size:13px;line-height:1.5;', item.q));

    const optionsContainer = mk('div', 'display:flex;flex-direction:column;gap:10px;', '');
    const feedbackEl = mk('div', 'margin-top:12px;font-size:12px;color:#f87171;', '');
    card.appendChild(optionsContainer);
    card.appendChild(feedbackEl);

    item.options.forEach((optText, idx) => {
      const btn = mk('button', 'text-align:left;padding:10px 14px;background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:4px;cursor:pointer;font-family:inherit;font-size:12px;', optText);
      btn.addEventListener('click', () => {
        if (idx === item.correctIndex) {
          chrome.storage.local.set({ mediaGateUnlockedUntil: Date.now() + 60 * 60 * 1000 }, () => {
            overlayEl.remove();
            overlayEl = null;
            const v = document.querySelector('video');
            if (v) v.play();
          });
        } else {
          feedbackEl.textContent = 'Rejected. Re-check the scaling and boundary conditions; a new question loads on retry.';
          overlayEl.remove();
          overlayEl = null; // next tick injects a fresh randomised question
        }
      });
      optionsContainer.appendChild(btn);
    });

    overlayEl.appendChild(card);
    document.body.appendChild(overlayEl);
  }

  function checkGateState() {
    stripExtraneousMediaDOM();
    chrome.storage.local.get(['mediaGateUnlockedUntil'], (res) => {
      if (Date.now() > (res.mediaGateUnlockedUntil || 0)) {
        pauseAllVideos();
        injectDerivationGateOverlay();
      }
    });
  }

  setInterval(checkGateState, 2000);
})();
