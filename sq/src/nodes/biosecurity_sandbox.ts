// ============================================================================
// SYNAPSE-QUANTUM: Node 4 Biosecurity Protocol Sandbox + Node 16 Adversarial Viva Examiner
// Fixes: options are shuffled (the correct answer is no longer always #1), two questions are
// parameterised with computed answers, the timer actually locks the quiz, and the viva scorer
// checks claims instead of counting keywords.
// ============================================================================

export interface DiagnosticQuestion {
  id: string;
  category: 'PHOTOTHERMAL_CYTOTOXICITY' | 'HESTON_BOUNDARY' | 'METABOLIC_ATP_KINETICS' | 'HENYEY_GREENSTEIN_OPTICS';
  prompt: string;
  options: string[];
  correctIndex: number;
  explanation: string;
}

type Rng = () => number;

function shuffleWithAnswer(options: string[], correct: number, rnd: Rng): { options: string[]; correctIndex: number } {
  const idx = options.map((_, i) => i);
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  return { options: idx.map((i) => options[i]), correctIndex: idx.indexOf(correct) };
}

const pick = <T,>(rnd: Rng, xs: T[]): T => xs[Math.floor(rnd() * xs.length)];
const fmt = (x: number) => (x < 1e-3 || x >= 1e4 ? x.toExponential(2) : x.toPrecision(3));

function qPhotothermal(rnd: Rng): DiagnosticQuestion {
  const temp = pick(rnd, [38.8, 39.1, 39.4]);
  const power = pick(rnd, [85, 95, 110]);
  const s = shuffleWithAnswer([
    'Throttle SLM duty cycle (e.g. 75%) and lengthen the dark recovery interval while maintaining perfusion.',
    'Increase NA to 1.45 and double the repetition rate to 160 MHz to finish stimulation faster.',
    'Disable Na+/K+ ATPase pumps by removing extracellular glucose.',
  ], 0, rnd);
  return {
    id: 'NODE4-THERMAL', category: 'PHOTOTHERMAL_CYTOTOXICITY',
    prompt: `Telemetry reports parenchyma at ${temp} °C under 920 nm illumination (P_laser = ${power} mW), above the 38.5 °C limit. What is the correct protocol?`,
    ...s,
    explanation: 'Absorbed heating scales with average power (~P·exp(-z/l_s)). Reduce duty cycle / power and restore perfusion; raising rep-rate or removing glucose makes it worse.',
  };
}

function qFeller(rnd: Rng): DiagnosticQuestion {
  const kappa = pick(rnd, [1.2, 1.5, 2.0, 2.5, 3.0]);
  const theta = pick(rnd, [0.02, 0.04, 0.06, 0.09]);
  const sigma = pick(rnd, [0.2, 0.35, 0.5, 0.7, 0.95]);
  const lhs = 2 * kappa * theta, rhs = sigma * sigma;
  const ok = lhs > rhs;
  const s = shuffleWithAnswer([
    ok
      ? `Feller condition holds: 2κθ = ${fmt(lhs)} > σ² = ${fmt(rhs)}, so v(t) stays strictly positive.`
      : `Feller condition is violated: 2κθ = ${fmt(lhs)} < σ² = ${fmt(rhs)}, so v(t) can touch zero and needs boundary handling (full truncation / clamping).`,
    ok
      ? `Feller condition is violated: 2κθ = ${fmt(lhs)} < σ² = ${fmt(rhs)}, so v(t) can touch zero.`
      : `Feller condition holds: 2κθ = ${fmt(lhs)} > σ² = ${fmt(rhs)}, so v(t) stays strictly positive.`,
    'Feller depends only on ρ: with ρ < 0 the variance can never reach zero.',
  ], 0, rnd);
  return {
    id: 'NODE4-FELLER', category: 'HESTON_BOUNDARY',
    prompt: `Heston parameters: κ = ${kappa}, θ = ${theta}, σ = ${sigma}. What is the boundary behaviour of the variance process?`,
    ...s,
    explanation: `The CIR variance process never hits zero iff 2κθ ≥ σ². Here 2κθ = ${fmt(lhs)} and σ² = ${fmt(rhs)}.`,
  };
}

function qATP(rnd: Rng): DiagnosticQuestion {
  const s = shuffleWithAnswer([
    'Na+/K+ ATPase clearance degrades, collapsing the E_Na/E_K gradients and producing depolarisation block.',
    'Membrane capacitance C_m drops to zero, giving infinite action-potential velocity.',
    'ChR2 permanently freezes in the Closed_1 state.',
  ], 0, rnd);
  return {
    id: 'NODE4-ATP', category: 'METABOLIC_ATP_KINETICS',
    prompt: 'Sustained bursting (>45 Hz) drives intracellular [ATP] below 0.8 mM under Michaelis-Menten glucose kinetics. What happens to the Hodgkin-Huxley dynamics?',
    ...s,
    explanation: 'ATP depletion starves the Na+/K+ pump, the ionic gradients run down and the effective reversal potentials collapse.',
  };
}

function qAttenuation(rnd: Rng): DiagnosticQuestion {
  const ls = pick(rnd, [120, 150, 180, 220]);
  const z = pick(rnd, [200, 300, 450, 600, 900]);
  const correct = Math.exp((-2 * z) / ls);
  const wrong1 = Math.exp(-z / ls);        // forgets the two-photon factor of 2
  const wrong2 = Math.exp((-2 * z) / ls / 2);
  const s = shuffleWithAnswer([
    `exp(-2z/l_s) = ${fmt(correct)}`,
    `exp(-z/l_s) = ${fmt(wrong1)} (single-photon scaling)`,
    wrong2 === wrong1 ? 'Exactly 1.0: 920 nm photons do not scatter in grey matter' : `exp(-z/l_s)·exp(+z/l_s) = 1.0 (no scattering at 920 nm)`,
  ], 0, rnd);
  return {
    id: 'NODE4-ATTEN', category: 'HENYEY_GREENSTEIN_OPTICS',
    prompt: `Ballistic two-photon excitation at depth z = ${z} µm with scattering mean free path l_s = ${ls} µm. What is the excitation attenuation factor?`,
    ...s,
    explanation: `Two-photon excitation ∝ I² so the ballistic flux term is squared: exp(-2z/l_s) = exp(-${fmt((2 * z) / ls)}) = ${fmt(correct)}.`,
  };
}

export function buildDiagnosticSession(rnd: Rng = Math.random): DiagnosticQuestion[] {
  return [qPhotothermal(rnd), qFeller(rnd), qATP(rnd), qAttenuation(rnd), qFeller(rnd), qAttenuation(rnd)];
}

export class BiosecuritySandboxEngine {
  private questions: DiagnosticQuestion[] = buildDiagnosticSession();
  private currentIdx = 0;
  private score = 0;
  private answered = 0;
  private timeRemainingSec = 90;
  private timerHandle: number | null = null;
  private expired = false;

  public startSession(onTick: (sec: number) => void, onExpire: () => void) {
    this.questions = buildDiagnosticSession(); // fresh, reshuffled questions each attempt
    this.currentIdx = 0; this.score = 0; this.answered = 0; this.expired = false;
    this.timeRemainingSec = 90;
    if (this.timerHandle) window.clearInterval(this.timerHandle);
    onTick(this.timeRemainingSec);
    this.timerHandle = window.setInterval(() => {
      this.timeRemainingSec = Math.max(0, this.timeRemainingSec - 1);
      onTick(this.timeRemainingSec);
      if (this.timeRemainingSec <= 0) {
        this.expired = true;
        if (this.timerHandle) window.clearInterval(this.timerHandle);
        onExpire();
      }
    }, 1000);
  }

  public isExpired(): boolean { return this.expired; }
  public getCurrentQuestion(): DiagnosticQuestion { return this.questions[Math.min(this.currentIdx, this.questions.length - 1)]; }

  public submitAnswer(choiceIdx: number) {
    if (this.expired || this.answered >= this.questions.length) {
      return { correct: false, explanation: 'Session is closed. Reset to re-certify.', score: this.score, answered: this.answered, total: this.questions.length, completed: true };
    }
    const q = this.getCurrentQuestion();
    const correct = choiceIdx === q.correctIndex;
    if (correct) this.score++;
    this.answered++;
    this.currentIdx++;
    const completed = this.answered >= this.questions.length;
    if (completed && this.timerHandle) window.clearInterval(this.timerHandle);
    return { correct, explanation: q.explanation, score: this.score, answered: this.answered, total: this.questions.length, completed };
  }
}

// ----------------------------------------------------------------------------
// Node 16: claims-based viva evaluator. Each concept needs the term AND a supporting claim.
// Keyword stuffing is detected and capped.
// ----------------------------------------------------------------------------
interface Concept { name: string; weight: number; test: (t: string) => boolean; hint: string }

const has = (t: string, re: RegExp) => re.test(t);

const CONCEPTS: Concept[] = [
  {
    name: 'two-photon attenuation law', weight: 20,
    test: (t) => has(t, /exp\s*\(\s*-?\s*2\s*\*?\s*z\s*\/\s*(l_?s|ℓ_?s|l\s*s)\s*\)/) || (has(t, /two-photon|2p/) && has(t, /(squar|quadratic|p\^?2|p²)/) && has(t, /(exp|expon)/) && has(t, /(depth|\bz\b)/)),
    hint: 'state the attenuation law exp(-2z/l_s) and why the 2 appears (quadratic excitation)',
  },
  {
    name: 'Henyey-Greenstein scattering', weight: 14,
    test: (t) => has(t, /henyey/) && has(t, /(anisotrop|\bg\b\s*[=≈~]|phase function|mean free path|l_?s)/),
    hint: 'HG is the angular phase function (anisotropy g); l_s is the scattering mean free path',
  },
  {
    name: 'Gerchberg-Saxton compensation', weight: 18,
    test: (t) => has(t, /gerchberg/) && has(t, /(iterat|fourier|fft)/) && has(t, /(phase)/) && has(t, /(compensat|correct|weight|power|aberrat)/),
    hint: 'explain iterative Fourier phase retrieval and what it compensates (and its limits: it cannot undo scattering loss)',
  },
  {
    name: 'Feller condition', weight: 18,
    test: (t) => has(t, /2\s*\*?\s*(κ|kappa)\s*\*?\s*(θ|theta)\s*(>|≥|>=|\\ge)\s*(σ|sigma)\s*(\^|²)?\s*2?/) || (has(t, /feller/) && has(t, /(zero|positive|boundary)/)),
    hint: 'give the inequality 2κθ ≥ σ² and what happens at the zero boundary',
  },
  {
    name: 'ATP / pump limit', weight: 15,
    test: (t) => has(t, /\batp\b/) && has(t, /(pump|na\+?\/k\+?|depolari[sz]ation block|michaelis|gradient|reversal)/),
    hint: 'link ATP depletion to pump failure / reversal-potential collapse',
  },
  {
    name: 'photothermal limit', weight: 15,
    test: (t) => has(t, /(photothermal|heating|temperature)/) && has(t, /(38\.5|duty cycle|throttl|average power|perfusion|safe)/),
    hint: 'tie heating to average power and a concrete mitigation / threshold',
  },
];

export function evaluateAdversarialVivaResponse(responseText: string): { scoreOutOf100: number; verdict: string; critique: string } {
  const text = responseText.toLowerCase().replace(/\s+/g, ' ').trim();
  const words = text.split(/[^a-zα-ωθκσ0-9_+\/^.-]+/).filter(Boolean);
  const sentences = text.split(/[.!?;\n]+/).filter((s) => s.trim().split(' ').length >= 4);

  const matched: string[] = [];
  const missing: string[] = [];
  let score = 0;
  for (const c of CONCEPTS) {
    if (c.test(text)) { score += c.weight; matched.push(c.name); } else missing.push(`${c.name}: ${c.hint}`);
  }

  // keyword-stuffing detection: lots of jargon, almost no sentences
  const jargon = ['scattering', 'henyey', 'gerchberg', 'feller', 'atp', 'photothermal'];
  const jargonHits = words.filter((w) => jargon.some((j) => w.includes(j))).length;
  const empty = words.length === 0;
  const stuffed = !empty && (words.length < 15 || sentences.length < 2 || jargonHits / Math.max(words.length, 1) > 0.3);
  if (stuffed) score = Math.min(score, 35);
  score = Math.min(100, score);

  const verdict = empty
    ? 'NO RESPONSE SUBMITTED'
    : score >= 70
      ? 'VIVA DEFENSE PASSED'
      : stuffed ? 'VIVA DEFENSE REJECTED (keyword list, not an argument)' : 'VIVA DEFENSE CONDITIONAL';
  const parts = [`Score ${score}/100.`];
  if (matched.length) parts.push(`Supported claims: ${matched.join('; ')}.`);
  if (stuffed) parts.push('Write full sentences that make claims, not a list of terms.');
  if (missing.length && score < 100) parts.push(`Strengthen -> ${missing.slice(0, 3).join(' | ')}`);
  return { scoreOutOf100: score, verdict, critique: `[Node 16 Peer-Review Examiner] ${parts.join(' ')}` };
}
