(() => {
'use strict';

const canvas = document.getElementById('tank');
const ctx = canvas.getContext('2d');
const TAU = Math.PI * 2;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const lerp = (a, b, t) => a + (b - a) * t;
const rgb = (c, a = 1) => `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${a})`;

// Seeded so the scene layout is stable across resizes and reloads.
function rng(seed) {
  return () => {
    seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

let W = 0, H = 0, S = 1, DPR = 1;
let bedCanvas = null, rootCanvas = null, vignette = null;

/* ---------- lighting ---------- */

const MOODS = {
  day:   { ambient: [255, 255, 255], top: [150, 188, 138], mid: [72, 122, 100], deep: [26, 54, 46], sky: [238, 247, 216], ray: [255, 243, 200], rayA: 0.17, angle: 0.22, caustic: 0.6, motes: 0.55 },
  dusk:  { ambient: [255, 200, 156], top: [196, 150, 98], mid: [96, 102, 84], deep: [28, 42, 40], sky: [255, 206, 140], ray: [255, 168, 88], rayA: 0.16, angle: 0.62, caustic: 0.32, motes: 0.42 },
  night: { ambient: [96, 124, 182], top: [70, 102, 134], mid: [32, 58, 80], deep: [9, 22, 32], sky: [164, 192, 228], ray: [170, 205, 255], rayA: 0.09, angle: -0.18, caustic: 0.18, motes: 0.22 },
};
let mood = 'day';
const light = JSON.parse(JSON.stringify(MOODS.day));

function updateLight(dt) {
  const m = MOODS[mood], k = 1 - Math.exp(-dt * 1.4);
  for (const key in m) {
    if (Array.isArray(m[key])) for (let i = 0; i < 3; i++) light[key][i] = lerp(light[key][i], m[key][i], k);
    else light[key] = lerp(light[key], m[key], k);
  }
}

const surfY = (x, t) => H * 0.06 + (Math.sin(x * 0.012 + t * 1.1) * 3 + Math.sin(x * 0.031 - t * 1.7) * 1.5) * S;
const bedY = (x) => H * 0.87 + Math.sin(x * 0.004 + 1) * H * 0.012 + Math.sin(x * 0.011) * H * 0.006;

// Low-res animated caustic texture, stretched over the riverbed and fish backs.
const CW = 192, CH = 48;
const caustic = document.createElement('canvas');
caustic.width = CW; caustic.height = CH;
const cctx = caustic.getContext('2d');
const cimg = cctx.createImageData(CW, CH);
function updateCaustics(t) {
  const d = cimg.data;
  let i = 0;
  for (let y = 0; y < CH; y++) {
    const v = y * 0.3;
    for (let x = 0; x < CW; x++) {
      const u = x * 0.11;
      const a = Math.sin(u + t * 0.9 + Math.sin(v * 0.7 + t * 0.5) * 1.6);
      const b = Math.sin(u * 0.63 - t * 0.7 + v * 1.1 + Math.sin(u * 0.4 + t * 0.3) * 1.3);
      const c = Math.sin(v * 0.9 + u * 0.27 + t * 0.6);
      let val = 1 - Math.abs(a + b + c) / 1.6;
      val = val < 0 ? 0 : val * val * val;
      d[i] = 255; d[i + 1] = 250; d[i + 2] = 215; d[i + 3] = val * 255;
      i += 4;
    }
  }
  cctx.putImageData(cimg, 0, 0);
}

/* ---------- species ---------- */

const SPECIES = {
  pike: {
    name: 'Northern pike', latin: 'Esox lucius', L: 300, count: 1, band: [0.3, 0.72], speed: 46, pause: 9, tailK: 7,
    top: [[0, .004], [.06, .022], [.16, .045], [.3, .062], [.6, .066], [.8, .058], [.92, .035], [1, .028]],
    bot: [[0, .006], [.06, .03], [.16, .05], [.3, .066], [.6, .07], [.8, .058], [.92, .035], [1, .028]],
    head: .27, eye: { s: .15, v: -.35, r: .013, iris: [214, 190, 90] },
    back: [56, 70, 36], flank: [110, 126, 60], belly: [226, 222, 190],
    fins: [
      { side: 'top', s0: .7, s1: .86, h: .085, sweep: .5, color: [150, 104, 48] },
      { side: 'bot', s0: .72, s1: .86, h: .075, sweep: .5, color: [156, 100, 50] },
      { side: 'bot', s0: .47, s1: .54, h: .06, sweep: .9, color: [160, 110, 56] },
    ],
    tail: { len: .15, spread: .1, fork: .35, color: [140, 98, 46] }, pect: { len: .07, color: [168, 120, 60] },
    scales: { ds: .022, alpha: .16 }, marks: 'pike',
  },
  catfish: {
    name: 'Wels catfish', latin: 'Silurus glanis', L: 400, count: 1, band: [0.78, 0.9], speed: 24, pause: 12, tailK: 5, z: .85, nocturnal: true,
    top: [[0, .03], [.05, .05], [.15, .07], [.3, .075], [.6, .05], [.85, .03], [1, .018]],
    bot: [[0, .02], [.05, .045], [.15, .075], [.3, .085], [.6, .055], [.85, .03], [1, .018]],
    head: .2, eye: { s: .06, v: -.3, r: .007, iris: [150, 140, 90] },
    back: [38, 42, 32], flank: [82, 84, 58], belly: [212, 204, 174],
    fins: [
      { side: 'top', s0: .27, s1: .3, h: .04, sweep: .6, color: [50, 52, 40] },
      { side: 'bot', s0: .42, s1: .98, h: .05, sweep: .3, color: [60, 60, 44], long: true },
      { side: 'bot', s0: .36, s1: .41, h: .04, sweep: .9, color: [70, 68, 50] },
    ],
    tail: { len: .07, spread: .045, fork: 0, color: [52, 54, 40] }, pect: { len: .06, color: [70, 70, 50] },
    scales: null, marks: 'catfish', barbels: 'wels',
  },
  carp: {
    name: 'Common carp', latin: 'Cyprinus carpio', L: 240, count: 2, band: [0.5, 0.86], speed: 34, pause: 4, tailK: 7,
    top: [[0, .012], [.07, .07], [.2, .14], [.38, .175], [.6, .15], [.85, .065], [1, .05]],
    bot: [[0, .014], [.07, .06], [.2, .115], [.45, .14], [.7, .1], [.88, .058], [1, .05]],
    head: .25, eye: { s: .12, v: -.2, r: .017, iris: [224, 186, 80] },
    back: [72, 60, 28], flank: [176, 140, 58], belly: [236, 214, 150],
    fins: [
      { side: 'top', s0: .36, s1: .8, h: .085, sweep: .35, color: [112, 84, 46], long: true },
      { side: 'bot', s0: .7, s1: .8, h: .075, sweep: .6, color: [176, 96, 52] },
      { side: 'bot', s0: .45, s1: .54, h: .075, sweep: .9, color: [176, 100, 56] },
    ],
    tail: { len: .2, spread: .15, fork: .45, color: [150, 92, 50] }, pect: { len: .1, color: [186, 120, 64] },
    scales: { ds: .046, alpha: .34, shine: true }, barbels: 'carp',
  },
  trout: {
    name: 'Brown trout', latin: 'Salmo trutta', L: 175, count: 2, band: [0.16, 0.6], speed: 62, pause: 2, tailK: 8,
    top: [[0, .008], [.07, .045], [.2, .085], [.4, .105], [.65, .09], [.88, .045], [1, .04]],
    bot: [[0, .01], [.07, .045], [.2, .08], [.45, .1], [.7, .08], [.88, .045], [1, .04]],
    head: .23, eye: { s: .11, v: -.25, r: .02, iris: [216, 196, 110] },
    back: [84, 72, 40], flank: [178, 152, 84], belly: [240, 226, 170],
    fins: [
      { side: 'top', s0: .42, s1: .57, h: .09, sweep: .5, color: [150, 124, 66] },
      { side: 'top', s0: .76, s1: .81, h: .03, sweep: 1.2, color: [170, 110, 60] },
      { side: 'bot', s0: .68, s1: .78, h: .07, sweep: .6, color: [190, 150, 80] },
      { side: 'bot', s0: .5, s1: .57, h: .06, sweep: .9, color: [196, 156, 84] },
    ],
    tail: { len: .15, spread: .12, fork: .15, color: [150, 122, 66] }, pect: { len: .08, color: [200, 162, 90] },
    scales: { ds: .02, alpha: .12 }, marks: 'trout',
  },
  perch: {
    name: 'European perch', latin: 'Perca fluviatilis', L: 135, count: 3, band: [0.3, 0.8], speed: 44, pause: 3, tailK: 8,
    top: [[0, .01], [.08, .06], [.22, .125], [.35, .15], [.6, .12], [.85, .05], [1, .035]],
    bot: [[0, .012], [.08, .05], [.25, .1], [.5, .11], [.75, .07], [.9, .04], [1, .035]],
    head: .28, eye: { s: .13, v: -.25, r: .028, iris: [226, 180, 70] },
    back: [60, 76, 42], flank: [152, 162, 78], belly: [236, 232, 200],
    fins: [
      { side: 'top', s0: .3, s1: .57, h: .13, sweep: .25, color: [112, 118, 92], spiny: true, spot: true },
      { side: 'top', s0: .6, s1: .8, h: .09, sweep: .5, color: [132, 136, 100] },
      { side: 'bot', s0: .62, s1: .76, h: .08, sweep: .6, color: [214, 84, 40] },
      { side: 'bot', s0: .33, s1: .42, h: .09, sweep: .9, color: [220, 88, 40] },
    ],
    tail: { len: .17, spread: .12, fork: .3, color: [206, 96, 46] }, pect: { len: .09, color: [200, 170, 96] },
    scales: { ds: .026, alpha: .15 }, marks: 'perch',
  },
  roach: {
    name: 'Common roach', latin: 'Rutilus rutilus', L: 90, count: 9, band: [0.14, 0.55], speed: 54, pause: 0, tailK: 9, schooling: true,
    top: [[0, .01], [.08, .06], [.25, .12], [.42, .14], [.7, .1], [.88, .05], [1, .04]],
    bot: [[0, .012], [.08, .055], [.25, .11], [.45, .125], [.7, .085], [.88, .05], [1, .04]],
    head: .22, eye: { s: .1, v: -.2, r: .028, iris: [214, 70, 44] },
    back: [68, 88, 94], flank: [178, 192, 198], belly: [240, 242, 236],
    fins: [
      { side: 'top', s0: .45, s1: .6, h: .11, sweep: .5, color: [110, 104, 98] },
      { side: 'bot', s0: .66, s1: .78, h: .08, sweep: .6, color: [206, 84, 50] },
      { side: 'bot', s0: .45, s1: .53, h: .08, sweep: .9, color: [210, 88, 52] },
    ],
    tail: { len: .2, spread: .14, fork: .5, color: [150, 100, 84] }, pect: { len: .09, color: [200, 150, 120] },
    scales: { ds: .034, alpha: .2 },
  },
};

const N = 26; // body samples from nose to tail root
function profile(points, s) {
  for (let i = 1; i < points.length; i++) {
    if (s <= points[i][0]) {
      const [s0, h0] = points[i - 1], [s1, h1] = points[i];
      const u = (s - s0) / (s1 - s0);
      return lerp(h0, h1, u * u * (3 - 2 * u));
    }
  }
  return points[points.length - 1][1];
}
for (const key in SPECIES) {
  const sp = SPECIES[key];
  sp.key = key;
  sp.T0 = []; sp.B0 = [];
  sp.maxT = 0; sp.maxB = 0;
  for (let i = 0; i <= N; i++) {
    sp.T0[i] = profile(sp.top, i / N);
    sp.B0[i] = profile(sp.bot, i / N);
    sp.maxT = Math.max(sp.maxT, sp.T0[i]);
    sp.maxB = Math.max(sp.maxB, sp.B0[i]);
  }
}

function makeMarks(sp, r) {
  const m = [];
  const add = (n, s0, s1, v0, v1, r0, r1, kind) => {
    for (let i = 0; i < n; i++) m.push({ s: lerp(s0, s1, r()), v: lerp(v0, v1, r()), r: lerp(r0, r1, r()), kind });
  };
  if (sp.marks === 'pike') add(48, .24, .97, -.75, .55, .011, .022, 0);
  if (sp.marks === 'trout') { add(42, .08, .95, -.9, .2, .006, .011, 0); add(12, .25, .92, -.1, .4, .007, .009, 1); }
  if (sp.marks === 'catfish') { add(40, 0, 1, -1, .45, .012, .03, 0); add(30, 0, 1, -1, .5, .01, .024, 1); }
  return m;
}

/* ---------- fish rendering ---------- */

// Body geometry of the fish currently being drawn (local px, nose at +x).
const GX = new Float32Array(N + 1), GY = new Float32Array(N + 1), GT = new Float32Array(N + 1), GB = new Float32Array(N + 1);
let PX = 0, PY = 0;
// s runs nose→tail (0..1); v runs back→belly (-1..1).
function P(s, v) {
  const f = clamp(s, 0, 1) * N, i = Math.min(N - 1, f | 0), u = f - i;
  PX = lerp(GX[i], GX[i + 1], u);
  PY = lerp(GY[i], GY[i + 1], u) + v * (v < 0 ? lerp(GT[i], GT[i + 1], u) : lerp(GB[i], GB[i + 1], u));
}

function drawFin(f, L, phase) {
  const sign = f.side === 'top' ? -1 : 1, n = f.long ? 16 : 9;
  const bx = [], by = [], tx = [], ty = [];
  for (let j = 0; j <= n; j++) {
    const u = j / n;
    P(lerp(f.s0, f.s1, u), sign * 0.8);
    bx[j] = PX; by[j] = PY;
    let shape = f.long ? (0.6 + 0.4 * Math.pow(1 - u, 1.5)) * Math.min(1, u * 8 + 0.3) * Math.min(1, (1 - u) * 6 + 0.25)
      : Math.pow(1 - u, 0.6) * Math.min(1, u * 5 + 0.25);
    if (f.spiny && j % 2) shape *= 0.84;
    const h = f.h * L * shape;
    tx[j] = PX - f.sweep * h + Math.sin(phase * 1.3 + u * 3) * f.h * L * 0.1 * u;
    ty[j] = PY + sign * (h + f.h * L * 0.2);
  }
  ctx.beginPath();
  ctx.moveTo(bx[0], by[0]);
  for (let j = 1; j <= n; j++) ctx.lineTo(bx[j], by[j]);
  for (let j = n; j >= 0; j--) ctx.lineTo(tx[j], ty[j]);
  ctx.closePath();
  ctx.fillStyle = rgb(f.color, 0.8);
  ctx.fill();
  if (f.spot) {
    ctx.fillStyle = 'rgba(15,18,12,.75)';
    ctx.beginPath();
    ctx.ellipse(lerp(bx[n - 1], tx[n - 1], .45), lerp(by[n - 1], ty[n - 1], .45), f.h * L * .16, f.h * L * .12, 0, 0, TAU);
    ctx.fill();
  }
  ctx.beginPath();
  for (let j = 0; j <= n; j++) { ctx.moveTo(bx[j], by[j]); ctx.lineTo(tx[j], ty[j]); }
  ctx.strokeStyle = f.spiny ? 'rgba(30,34,24,.6)' : 'rgba(30,20,10,.3)';
  ctx.lineWidth = f.spiny ? 1.3 : 0.8;
  ctx.stroke();
}

function drawTail(sp, L, phase) {
  const tl = sp.tail;
  const dx = GX[N] - GX[N - 2], dy = GY[N] - GY[N - 2];
  const hT = GT[N], hB = GB[N];
  // Foreshortening fakes the side-to-side sweep of the tail.
  const len = tl.len * L * (0.78 + 0.22 * Math.cos(phase - 4.2)), spr = tl.spread * L, notch = len * (1 - tl.fork);
  ctx.save();
  ctx.translate(GX[N], GY[N]);
  ctx.rotate(Math.atan2(dy, dx) - Math.PI);
  ctx.beginPath();
  ctx.moveTo(3, -hT);
  ctx.quadraticCurveTo(-len * .45, -hT * 1.1, -len, -spr);
  ctx.quadraticCurveTo(-lerp(len, notch, .6), -spr * .35, -notch, 0);
  ctx.quadraticCurveTo(-lerp(len, notch, .6), spr * .35, -len, spr);
  ctx.quadraticCurveTo(-len * .45, hB * 1.1, 3, hB);
  ctx.closePath();
  const g = ctx.createLinearGradient(0, 0, -len, 0);
  g.addColorStop(0, rgb(sp.flank, .95));
  g.addColorStop(.45, rgb(tl.color, .88));
  g.addColorStop(1, rgb(tl.color, .7));
  ctx.fillStyle = g;
  ctx.fill();
  ctx.beginPath();
  for (let j = -4; j <= 4; j++) {
    const k = j / 4;
    ctx.moveTo(-len * .15, k * hT * .9);
    ctx.lineTo(-lerp(notch, len, Math.abs(k)) * .96, k * spr * .92);
  }
  ctx.strokeStyle = 'rgba(30,20,10,.28)';
  ctx.lineWidth = 0.8;
  ctx.stroke();
  ctx.restore();
}

function drawScales(sp, L) {
  const sc = sp.scales, r = sc.ds * L * 0.62, dv = sc.ds * 0.5 / ((sp.maxT + sp.maxB) / 2);
  const a0 = Math.PI * 0.62, a1 = Math.PI * 1.38;
  const pass = (off) => {
    ctx.beginPath();
    let row = 0;
    for (let v = -0.9; v <= 0.9; v += dv, row++) {
      for (let s = sp.head + 0.02 + (row % 2) * sc.ds * 0.5; s < 0.97; s += sc.ds) {
        P(s, v);
        ctx.moveTo(PX + off + r * Math.cos(a0), PY + r * Math.sin(a0));
        ctx.arc(PX + off, PY, r, a0, a1);
      }
    }
  };
  pass(0);
  ctx.strokeStyle = `rgba(20,16,8,${sc.alpha})`;
  ctx.lineWidth = sc.shine ? 1.2 : 0.7;
  ctx.stroke();
  if (sc.shine) {
    pass(1.6);
    ctx.strokeStyle = 'rgba(255,232,160,.22)';
    ctx.stroke();
  }
}

function drawMarks(fish, L) {
  const sp = fish.sp;
  if (sp.marks === 'perch') {
    ctx.fillStyle = 'rgba(28,38,20,.68)';
    for (let b = 0; b < 6; b++) {
      const s = 0.3 + b * 0.108, w = 0.024 - b * 0.0015;
      ctx.beginPath();
      P(s - w, -1.05); ctx.moveTo(PX, PY);
      P(s + w, -1.05); ctx.lineTo(PX, PY);
      P(s + w * .6 - .012, -.1); ctx.lineTo(PX, PY);
      P(s - .016, .5); ctx.lineTo(PX, PY);
      P(s - w * .8 - .01, -.1); ctx.lineTo(PX, PY);
      ctx.closePath();
      ctx.fill();
    }
    return;
  }
  for (const m of fish.marks) {
    P(m.s, m.v);
    const r = m.r * L;
    ctx.beginPath();
    if (sp.marks === 'pike') {
      ctx.fillStyle = 'rgba(218,216,140,.7)';
      ctx.ellipse(PX, PY, r, r * .5, 0, 0, TAU);
    } else if (sp.marks === 'trout') {
      if (m.kind) {
        ctx.fillStyle = 'rgba(238,226,190,.55)';
        ctx.arc(PX, PY, r * 1.9, 0, TAU);
        ctx.fill();
        ctx.beginPath();
        ctx.fillStyle = 'rgba(198,52,36,.92)';
      } else ctx.fillStyle = 'rgba(24,20,14,.8)';
      ctx.arc(PX, PY, r, 0, TAU);
    } else {
      ctx.fillStyle = m.kind ? 'rgba(156,154,108,.24)' : 'rgba(18,22,16,.36)';
      ctx.ellipse(PX, PY, r * 1.5, r * .8, m.s * 9, 0, TAU);
    }
    ctx.fill();
  }
}

function drawBarbels(sp, L, phase) {
  ctx.lineCap = 'round';
  if (sp.barbels === 'wels') {
    for (let k = 0; k < 2; k++) {
      const sw = Math.sin(phase * .5 + k * 1.7) * L * .02;
      P(.035, -.25);
      ctx.beginPath();
      ctx.moveTo(PX, PY);
      ctx.quadraticCurveTo(PX + L * .12, PY - L * (.035 - k * .03) + sw, PX + L * (.2 - k * .03), PY + L * (.05 + k * .03) + sw * 2);
      ctx.strokeStyle = rgb(k ? sp.flank : sp.back, .9);
      ctx.lineWidth = L * .007;
      ctx.stroke();
    }
    ctx.beginPath();
    for (let k = 0; k < 4; k++) {
      P(.04 + k * .022, .95);
      ctx.moveTo(PX, PY);
      ctx.quadraticCurveTo(PX + L * .01, PY + L * .03, PX - L * .012 + Math.sin(phase * .6 + k) * L * .008, PY + L * .055);
    }
    ctx.strokeStyle = rgb(sp.belly, .8);
    ctx.lineWidth = L * .004;
    ctx.stroke();
  } else {
    ctx.beginPath();
    for (let k = 0; k < 2; k++) {
      P(.03 + k * .03, .7);
      ctx.moveTo(PX, PY);
      ctx.quadraticCurveTo(PX + L * .005, PY + L * .025, PX - L * .012, PY + L * (.035 + k * .01));
    }
    ctx.strokeStyle = rgb(sp.flank, .95);
    ctx.lineWidth = L * .008;
    ctx.stroke();
  }
}

function drawFish(fish) {
  const sp = fish.sp, L = sp.L * S, ph = fish.phase, amp = fish.amp;
  const face = Math.abs(fish.face) < 0.12 ? (fish.face < 0 ? -0.12 : 0.12) : fish.face;
  for (let i = 0; i <= N; i++) {
    const s = i / N;
    GX[i] = L / 2 - s * L;
    GY[i] = L * 0.045 * amp * (s * s + 0.04) * Math.sin(ph - s * 4);
    GT[i] = sp.T0[i] * L;
    GB[i] = sp.B0[i] * L;
  }
  const maxT = sp.maxT * L, maxB = sp.maxB * L;

  ctx.save();
  ctx.translate(fish.x, fish.y);
  ctx.rotate(fish.pitch * Math.sign(face));
  ctx.scale(face * fish.sc, fish.sc);

  drawTail(sp, L, ph);
  for (const f of sp.fins) drawFin(f, L, ph);

  ctx.beginPath();
  ctx.moveTo(GX[0], GY[0] - GT[0]);
  for (let i = 1; i <= N; i++) ctx.lineTo(GX[i], GY[i] - GT[i]);
  for (let i = N; i >= 0; i--) ctx.lineTo(GX[i], GY[i] + GB[i]);
  ctx.closePath();
  const g = ctx.createLinearGradient(0, -maxT, 0, maxB);
  g.addColorStop(0, rgb(sp.back));
  g.addColorStop(0.3, rgb(sp.back));
  g.addColorStop(0.55, rgb(sp.flank));
  g.addColorStop(0.82, rgb(sp.belly));
  g.addColorStop(1, rgb(sp.belly));
  ctx.fillStyle = g;
  ctx.fill();

  ctx.save();
  ctx.clip();
  if (sp.marks) drawMarks(fish, L);
  if (sp.scales) drawScales(sp, L);
  // lateral line
  ctx.beginPath();
  for (let s = sp.head; s <= 0.98; s += 0.04) { P(s, 0.08); s === sp.head ? ctx.moveTo(PX, PY) : ctx.lineTo(PX, PY); }
  ctx.strokeStyle = 'rgba(20,16,8,.2)';
  ctx.lineWidth = 1;
  ctx.stroke();
  // top light on the flank, shadow under the belly
  const sh = ctx.createLinearGradient(0, -maxT, 0, maxB);
  sh.addColorStop(0, 'rgba(0,0,0,.3)');
  sh.addColorStop(0.2, 'rgba(255,250,220,0)');
  sh.addColorStop(0.36, 'rgba(255,250,220,.24)');
  sh.addColorStop(0.52, 'rgba(255,250,220,0)');
  sh.addColorStop(0.8, 'rgba(0,10,10,0)');
  sh.addColorStop(1, 'rgba(0,10,10,.4)');
  ctx.fillStyle = sh;
  ctx.fillRect(-L, -maxT * 2, L * 2, (maxT + maxB) * 2);
  ctx.globalCompositeOperation = 'lighter';
  ctx.globalAlpha = light.caustic * 0.55;
  ctx.drawImage(caustic, (fish.seed * 97) % 120, 0, 60, 22, -L / 2, -maxT * 1.2, L, maxT * 1.5);
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = 'source-over';
  // gill cover and mouth
  ctx.beginPath();
  P(sp.head - .01, -.7); ctx.moveTo(PX, PY);
  P(sp.head + .035, 0); const qx = PX, qy = PY;
  P(sp.head - .03, .8); ctx.quadraticCurveTo(qx, qy, PX, PY);
  ctx.strokeStyle = 'rgba(15,12,6,.45)';
  ctx.lineWidth = Math.max(1, L * .006);
  ctx.stroke();
  ctx.beginPath();
  P(0, .2); ctx.moveTo(PX, PY);
  P(sp.head * .3, .35); ctx.lineTo(PX, PY);
  ctx.lineWidth = Math.max(0.8, L * .004);
  ctx.stroke();
  ctx.restore();

  ctx.strokeStyle = 'rgba(12,16,10,.4)';
  ctx.lineWidth = 0.9;
  ctx.stroke();

  // eye
  const e = sp.eye, er = e.r * L;
  P(e.s, e.v);
  ctx.beginPath(); ctx.arc(PX, PY, er, 0, TAU); ctx.fillStyle = rgb(e.iris); ctx.fill();
  ctx.strokeStyle = 'rgba(10,10,6,.6)'; ctx.lineWidth = Math.max(0.6, er * .15); ctx.stroke();
  ctx.beginPath(); ctx.arc(PX + er * .08, PY, er * .56, 0, TAU); ctx.fillStyle = '#07090a'; ctx.fill();
  ctx.beginPath(); ctx.arc(PX + er * .3, PY - er * .32, er * .2, 0, TAU); ctx.fillStyle = 'rgba(255,255,255,.85)'; ctx.fill();

  // pectoral fin lies over the flank
  const pl = sp.pect.len * L;
  P(sp.head + .03, .45);
  ctx.save();
  ctx.translate(PX, PY);
  ctx.rotate(-(0.5 + Math.sin(ph * 0.8) * 0.28));
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.quadraticCurveTo(-pl * .5, -pl * .3, -pl, -pl * .05);
  ctx.quadraticCurveTo(-pl * .55, pl * .42, 0, 0);
  ctx.fillStyle = rgb(sp.pect.color, .72);
  ctx.fill();
  ctx.beginPath();
  for (let k = -1; k <= 2; k++) { ctx.moveTo(0, 0); ctx.lineTo(-pl * .88, k * pl * .09); }
  ctx.strokeStyle = 'rgba(30,20,10,.25)';
  ctx.lineWidth = 0.7;
  ctx.stroke();
  ctx.restore();

  if (sp.barbels) drawBarbels(sp, L, ph);
  ctx.restore();
}

/* ---------- behaviour ---------- */

const pellets = [];
const fishes = [];
const school = { x: 0, y: 0, tx: 0, ty: 0 };
const rnd = rng(20261110);

function pickTarget(sp) {
  const top = H * 0.06, water = bedY(W / 2) - top;
  return { x: lerp(0.06, 0.94, Math.random()) * W, y: top + lerp(sp.band[0], sp.band[1], Math.random()) * water };
}

function makeFish(sp, i) {
  const z = sp.z ?? rnd();
  const f = {
    sp, z, sc: 0.72 + 0.36 * z, seed: (rnd() * 1e4) | 0, marks: makeMarks(sp, rnd),
    x: 0, y: 0, vx: 0, vy: 0, face: rnd() < .5 ? -1 : 1, pitch: 0, phase: rnd() * TAU, amp: .5,
    target: null, idle: rnd() * 2, timer: 0,
    ox: (rnd() - .5) * 230, oy: (rnd() - .5) * 110,
  };
  if (sp.schooling) f.sc = 0.85 + 0.2 * rnd();
  return f;
}

function nearestPellet(f) {
  let best = null, bd = (f.sp.schooling ? 380 : 520) * S;
  for (const p of pellets) {
    const d = Math.hypot(p.x - f.x, p.y - f.y);
    if (d < bd) { bd = d; best = p; }
  }
  return best;
}

function updateFish(f, dt) {
  const sp = f.sp, L = sp.L * S * f.sc;
  let tx, ty, urgent = false;
  const pel = nearestPellet(f);
  if (pel) {
    tx = pel.x; ty = pel.y; urgent = true; f.idle = 0;
    if (Math.abs(pel.x - f.x) < L * 0.5 && Math.abs(pel.y - f.y) < L * 0.12 + 8 * S) pellets.splice(pellets.indexOf(pel), 1);
  } else if (sp.schooling) {
    tx = school.x + f.ox * S; ty = school.y + f.oy * S;
  } else {
    f.timer -= dt;
    if (!f.target || f.timer < 0 || Math.hypot(f.target.x - f.x, f.target.y - f.y) < 40 * S) {
      if (f.target) f.idle = Math.random() * sp.pause;
      f.target = pickTarget(sp);
      f.timer = 30;
    }
    tx = f.target.x; ty = f.target.y;
  }
  let speed = sp.speed * S * (urgent ? 1.9 : 1);
  if (sp.nocturnal) speed *= lerp(1, 2.2, clamp((255 - light.ambient[0]) / 150, 0, 1));
  let dx = tx - f.x, dy = ty - f.y;
  const d = Math.hypot(dx, dy) || 1;
  if (f.idle > 0) { f.idle -= dt; speed *= 0.04; }
  else speed *= clamp(d / (60 * S), 0.25, 1);
  const k = Math.min(1, dt * (urgent ? 2.4 : 1.1));
  f.vx += (dx / d * speed - f.vx) * k;
  f.vy += (dy / d * speed * 0.6 - f.vy) * k;
  f.x += f.vx * dt;
  f.y += f.vy * dt;
  f.y = clamp(f.y, H * 0.06 + sp.maxT * L + 14 * S, bedY(f.x) - sp.maxB * L - 2 * S);
  const v = Math.hypot(f.vx, f.vy);
  const want = f.vx > 3 ? 1 : f.vx < -3 ? -1 : Math.sign(f.face) || 1;
  f.face += (want - f.face) * Math.min(1, dt * 4);
  f.pitch += (clamp(Math.atan2(f.vy, Math.abs(f.vx) + 20 * S), -0.35, 0.35) - f.pitch) * Math.min(1, dt * 3);
  f.phase += dt * (2 + (v / L) * sp.tailK * 3);
  f.amp += (clamp(0.3 + v / (sp.speed * S), 0.3, 1.5) - f.amp) * Math.min(1, dt * 3);
}

function updateSchool(dt) {
  const sp = SPECIES.roach;
  if (Math.hypot(school.tx - school.x, school.ty - school.y) < 60 * S) {
    const t = pickTarget(sp);
    school.tx = t.x; school.ty = t.y;
  }
  const dx = school.tx - school.x, dy = school.ty - school.y, d = Math.hypot(dx, dy) || 1;
  school.x += dx / d * sp.speed * S * 0.85 * dt;
  school.y += dy / d * sp.speed * S * 0.5 * dt;
}

/* ---------- crayfish ---------- */

const cray = { x: 0, dir: 1, walk: 0, pause: 0, name: 'Noble crayfish', latin: 'Astacus astacus' };

function updateCray(dt) {
  if (cray.pause > 0) { cray.pause -= dt; return; }
  cray.x += cray.dir * 13 * S * dt;
  cray.walk += dt * 7;
  if (Math.random() < dt * 0.12) cray.pause = 1 + Math.random() * 4;
  if (cray.x > W * 0.9) cray.dir = -1;
  if (cray.x < W * 0.12) cray.dir = 1;
  else if (Math.random() < dt * 0.02) cray.dir *= -1;
}

function drawCray(t) {
  const u = S * 1.15, w = cray.walk, shell = [92, 58, 36], dark = [52, 32, 22];
  ctx.save();
  ctx.translate(cray.x, bedY(cray.x) + 6 * S);
  ctx.scale(cray.dir * u, u);
  ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  // legs
  ctx.beginPath();
  for (let i = 0; i < 4; i++) {
    const bx = 8 - i * 6.5, sw = Math.sin(w + i * 1.6);
    ctx.moveTo(bx, 1); ctx.lineTo(bx + sw * 3.5 + 2, 8); ctx.lineTo(bx + sw * 6, 14);
  }
  ctx.strokeStyle = rgb(dark); ctx.lineWidth = 2.1; ctx.stroke();
  // antennae
  ctx.beginPath();
  const as = Math.sin(t * 1.3) * 4;
  ctx.moveTo(22, -6); ctx.quadraticCurveTo(42, -20 + as, 64, -12 + as * 1.6);
  ctx.moveTo(22, -6); ctx.quadraticCurveTo(30, -26 - as, 6, -34 - as);
  ctx.moveTo(24, -5); ctx.lineTo(34, -8 + as * .4);
  ctx.strokeStyle = rgb([120, 76, 46]); ctx.lineWidth = 0.9; ctx.stroke();
  // abdomen and tail fan
  for (let i = 4; i >= 0; i--) {
    ctx.beginPath();
    ctx.ellipse(-15 - i * 6.2, -2 + i * i * 0.4, 5.6, 6.6 - i * 0.6, 0, 0, TAU);
    ctx.fillStyle = rgb(i % 2 ? shell : [80, 50, 32]); ctx.fill();
    ctx.strokeStyle = rgb(dark, .7); ctx.lineWidth = 0.7; ctx.stroke();
  }
  for (let k = -1; k <= 1; k++) {
    ctx.beginPath();
    ctx.ellipse(-48, 6 + k * 4.5, 7, 3.4, k * 0.5 - 0.2, 0, TAU);
    ctx.fillStyle = rgb([104, 64, 38]); ctx.fill(); ctx.stroke();
  }
  // claws
  for (let k = 0; k < 2; k++) {
    const lift = Math.sin(t * 0.9 + k * 2) * 2.5 - k * 5;
    ctx.beginPath();
    ctx.moveTo(15, 1); ctx.lineTo(26, 6 + lift * .5); ctx.lineTo(35, 1 + lift);
    ctx.strokeStyle = rgb(k ? dark : shell); ctx.lineWidth = 3.4; ctx.stroke();
    ctx.save();
    ctx.translate(35, 1 + lift);
    ctx.rotate(-0.25);
    ctx.fillStyle = rgb(k ? [70, 44, 28] : [112, 66, 38]);
    ctx.beginPath(); ctx.ellipse(6, 0, 8, 4.6, 0, 0, TAU); ctx.fill();
    ctx.beginPath(); ctx.moveTo(10, -4); ctx.quadraticCurveTo(22, -7, 25, -1.5); ctx.quadraticCurveTo(18, -1.5, 11, 0); ctx.fill();
    ctx.beginPath(); ctx.moveTo(10, 3.5); ctx.quadraticCurveTo(20, 5, 24, 0.5); ctx.quadraticCurveTo(17, 1, 11, 0); ctx.fillStyle = rgb([150, 74, 42]); ctx.fill();
    ctx.restore();
  }
  // carapace
  ctx.beginPath();
  ctx.moveTo(-12, -4);
  ctx.quadraticCurveTo(-8, -15, 8, -13);
  ctx.quadraticCurveTo(20, -11, 29, -5);
  ctx.quadraticCurveTo(20, 2, 8, 4);
  ctx.quadraticCurveTo(-8, 5, -12, -4);
  const g = ctx.createLinearGradient(0, -14, 0, 5);
  g.addColorStop(0, rgb([136, 88, 50])); g.addColorStop(.5, rgb(shell)); g.addColorStop(1, rgb(dark));
  ctx.fillStyle = g; ctx.fill();
  ctx.strokeStyle = rgb(dark, .8); ctx.lineWidth = 0.8; ctx.stroke();
  ctx.beginPath(); ctx.moveTo(6, -13); ctx.quadraticCurveTo(10, -5, 6, 3.5); ctx.stroke();
  ctx.beginPath(); ctx.arc(21, -8.5, 1.5, 0, TAU); ctx.fillStyle = '#0a0806'; ctx.fill();
  ctx.restore();
}

/* ---------- scenery ---------- */

let weeds = [], ribbons = [], reeds = [], pads = [], motes = [], bubbles = [], rays = [];

function buildScene() {
  const r = rng(4242);
  weeds = []; ribbons = []; reeds = []; pads = []; motes = []; rays = [];
  const clump = (cx, n, hMin, hMax, layer, col) => {
    for (let i = 0; i < n; i++) {
      const x = cx + (r() - .5) * 90 * S;
      weeds.push({ x, y: bedY(x) + 4 * S, h: lerp(hMin, hMax, r()) * H, n: 12, ph: r() * 9, sp: .6 + r() * .5, lean: (r() - .5) * .03, leaf: (6 + r() * 5) * S, layer, col });
    }
  };
  clump(W * .30, 6, .12, .26, 0, [44, 84, 58]);
  clump(W * .66, 7, .14, .3, 0, [48, 88, 60]);
  clump(W * .9, 5, .12, .24, 0, [40, 78, 54]);
  clump(W * .2, 7, .16, .34, 1, [74, 124, 52]);
  clump(W * .47, 6, .1, .22, 1, [88, 134, 56]);
  clump(W * .78, 8, .18, .38, 1, [66, 116, 50]);
  const ribbon = (cx, n, hMin, hMax, layer, col) => {
    for (let i = 0; i < n; i++) {
      const x = cx + (r() - .5) * 70 * S;
      ribbons.push({ x, y: bedY(x) + 6 * S, h: lerp(hMin, hMax, r()) * H, w: (5 + r() * 5) * S * (layer === 2 ? 2 : 1), ph: r() * 9, sp: .5 + r() * .5, layer, col: col.map(c => c * (.8 + r() * .4)) });
    }
  };
  ribbon(W * .56, 9, .12, .3, 1, [96, 142, 60]);
  ribbon(W * .36, 7, .1, .22, 1, [84, 130, 58]);
  ribbon(W * .04, 6, .25, .5, 2, [26, 50, 34]);
  ribbon(W * .97, 7, .3, .6, 2, [24, 46, 32]);
  for (let i = 0; i < 13; i++) {
    const right = i >= 5, x = right ? W * (.86 + r() * .13) : W * (.005 + r() * .1);
    reeds.push({ x, w: (4 + r() * 5) * S, lean: (r() - .5) * .08, ph: r() * 9, col: [lerp(110, 150, r()), lerp(130, 160, r()), lerp(60, 80, r())] });
  }
  for (const [fx, w] of [[.52, 78], [.6, 96], [.675, 70], [.74, 88], [.43, 60]]) {
    pads.push({ x: fx * W, w: w * S, base: fx * W + (r() - .5) * 160 * S, ph: r() * 9 });
  }
  for (let i = 0; i < 110; i++) motes.push({ x: r() * W, y: r() * H * .86, r: (.5 + r() * 1.5) * S, ph: r() * 9, v: (4 + r() * 8) * S });
  for (let i = 0; i < 11; i++) rays.push({ x: r() * 1.3, w: .02 + r() * .05, v: .004 + r() * .008, f: .25 + r() * .5, a: .5 + r() * .5, l: r() });
}

function buildBed() {
  const c = document.createElement('canvas');
  c.width = W * DPR; c.height = H * DPR;
  const b = c.getContext('2d');
  b.scale(DPR, DPR);
  const r = rng(77), top = H * 0.84;
  b.beginPath();
  b.moveTo(0, bedY(0));
  for (let x = 8; x <= W + 8; x += 8) b.lineTo(x, bedY(x));
  b.lineTo(W, H); b.lineTo(0, H); b.closePath();
  const g = b.createLinearGradient(0, top, 0, H);
  g.addColorStop(0, 'rgb(116,104,72)'); g.addColorStop(.4, 'rgb(96,84,58)'); g.addColorStop(1, 'rgb(46,40,30)');
  b.fillStyle = g; b.fill();
  for (let i = 0; i < 2600; i++) {
    const x = r() * W, y = bedY(x) + r() * (H - bedY(x));
    b.fillStyle = r() < .5 ? 'rgba(30,24,16,.25)' : 'rgba(210,196,150,.22)';
    b.fillRect(x, y, 1.4 * S, 1.1 * S);
  }
  const palette = [[128, 122, 108], [96, 90, 80], [150, 132, 98], [110, 96, 70], [84, 80, 74], [162, 150, 126], [120, 88, 62]];
  const stones = [];
  for (let i = 0; i < 520; i++) {
    const x = r() * W, d = r();
    stones.push({ x, y: bedY(x) + 3 * S + d * d * (H - bedY(x)), s: (3 + d * 13 + r() * 5) * S, c: palette[(r() * palette.length) | 0], rot: (r() - .5) * .8, k: .5 + r() * .25 });
  }
  for (const fx of [.12, .34, .5, .93]) stones.push({ x: fx * W, y: bedY(fx * W) + 10 * S, s: (34 + r() * 26) * S, c: [92, 90, 84], rot: (r() - .5) * .3, k: .55, moss: true });
  stones.sort((a, c2) => a.y - c2.y);
  for (const st of stones) {
    b.fillStyle = 'rgba(14,12,8,.4)';
    b.beginPath(); b.ellipse(st.x + st.s * .12, st.y + st.s * .3, st.s, st.s * st.k * .8, st.rot, 0, TAU); b.fill();
    const sg = b.createLinearGradient(st.x, st.y - st.s * st.k, st.x, st.y + st.s * st.k);
    sg.addColorStop(0, rgb(st.c.map(v => v * 1.2))); sg.addColorStop(1, rgb(st.c.map(v => v * .5)));
    b.fillStyle = sg;
    b.beginPath(); b.ellipse(st.x, st.y, st.s, st.s * st.k, st.rot, 0, TAU); b.fill();
    b.fillStyle = 'rgba(255,248,220,.2)';
    b.beginPath(); b.ellipse(st.x - st.s * .25, st.y - st.s * st.k * .4, st.s * .45, st.s * st.k * .28, st.rot, 0, TAU); b.fill();
    if (st.moss) {
      b.fillStyle = 'rgba(64,104,48,.6)';
      b.beginPath(); b.ellipse(st.x + st.s * .1, st.y - st.s * st.k * .55, st.s * .7, st.s * st.k * .4, st.rot, Math.PI, TAU); b.fill();
    }
  }
  // sunken branch
  const x0 = W * .56, y0 = bedY(x0) + 20 * S, x1 = W * .82, y1 = bedY(x1) - 26 * S;
  b.lineCap = 'round';
  const limb = (ax, ay, bx2, by2, w) => {
    b.strokeStyle = 'rgba(12,10,6,.4)'; b.lineWidth = w; b.beginPath(); b.moveTo(ax + 4 * S, ay + 9 * S); b.lineTo(bx2 + 4 * S, by2 + 9 * S); b.stroke();
    b.strokeStyle = 'rgb(58,43,28)'; b.beginPath(); b.moveTo(ax, ay); b.lineTo(bx2, by2); b.stroke();
    b.strokeStyle = 'rgba(122,96,62,.6)'; b.lineWidth = w * .3; b.beginPath(); b.moveTo(ax, ay - w * .25); b.lineTo(bx2, by2 - w * .25); b.stroke();
    b.lineWidth = w;
  };
  limb(lerp(x0, x1, .55), lerp(y0, y1, .55), lerp(x0, x1, .68), lerp(y0, y1, .55) - 78 * S, 11 * S);
  limb(x0, y0, x1, y1, 28 * S);
  b.strokeStyle = 'rgba(24,16,10,.45)'; b.lineWidth = 1.2 * S;
  for (let i = 0; i < 26; i++) {
    const u = r(), o = (r() - .5) * 20 * S, l = (.03 + r() * .08);
    b.beginPath(); b.moveTo(lerp(x0, x1, u), lerp(y0, y1, u) + o); b.lineTo(lerp(x0, x1, u + l), lerp(y0, y1, u + l) + o); b.stroke();
  }
  // swan mussels
  for (const fx of [.24, .27, .44]) {
    const x = fx * W, y = bedY(x) + 22 * S;
    b.save(); b.translate(x, y); b.rotate(-.3 + fx);
    b.fillStyle = 'rgb(54,58,40)'; b.beginPath(); b.ellipse(0, 0, 20 * S, 10 * S, 0, 0, TAU); b.fill();
    b.strokeStyle = 'rgba(150,150,100,.35)'; b.lineWidth = S;
    for (let k = 1; k < 4; k++) { b.beginPath(); b.ellipse(-k * 2 * S, k * S, (20 - k * 4.5) * S, (10 - k * 2.3) * S, 0, 0, TAU); b.stroke(); }
    b.restore();
  }
  bedCanvas = c;

  // Alder roots reaching in from the undercut bank, kept hazy behind the fish.
  const rc = document.createElement('canvas');
  rc.width = W * DPR; rc.height = H * DPR;
  const q = rc.getContext('2d');
  q.scale(DPR, DPR);
  q.lineCap = 'round';
  const rr = rng(909);
  for (let i = 0; i < 9; i++) {
    const ax = rr() * W * .13, ay = H * .03, ex = W * (.02 + rr() * .24), ey = bedY(ex) + 10 * S;
    const mx = lerp(ax, ex, .3) + (rr() - .3) * W * .1, my = lerp(ay, ey, .55);
    const w0 = (10 + rr() * 16) * S;
    for (let k = 0; k < 24; k++) {
      const u0 = k / 24, u1 = (k + 1) / 24;
      const pt = (u) => [(1 - u) * (1 - u) * ax + 2 * u * (1 - u) * mx + u * u * ex, (1 - u) * (1 - u) * ay + 2 * u * (1 - u) * my + u * u * ey];
      const [px0, py0] = pt(u0), [px1, py1] = pt(u1);
      q.strokeStyle = `rgba(${34 + i * 3},${30 + i * 2},22,.9)`;
      q.lineWidth = lerp(w0, w0 * .22, u0);
      q.beginPath(); q.moveTo(px0, py0); q.lineTo(px1, py1); q.stroke();
    }
  }
  rootCanvas = rc;
  vignette = ctx.createRadialGradient(W / 2, H * .45, Math.min(W, H) * .35, W / 2, H * .5, Math.max(W, H) * .75);
  vignette.addColorStop(0, 'rgba(0,0,0,0)'); vignette.addColorStop(1, 'rgba(0,6,8,.55)');
}

function drawWeed(p, t) {
  let x = p.x, y = p.y, a = -Math.PI / 2;
  const seg = p.h / p.n;
  ctx.beginPath();
  for (let i = 0; i < p.n; i++) {
    a += Math.sin(t * p.sp + i * 0.45 + p.ph) * 0.03 * (1 + i * 0.25) + p.lean;
    const nx = x + Math.cos(a) * seg, ny = y + Math.sin(a) * seg, lf = p.leaf * (1 - i / p.n * .45);
    ctx.moveTo(x, y); ctx.lineTo(nx, ny);
    for (const side of [-1, 1]) for (const fr of [.35, .85]) {
      const lx = lerp(x, nx, fr), ly = lerp(y, ny, fr), la = a + side * 1.05;
      ctx.moveTo(lx, ly); ctx.lineTo(lx + Math.cos(la) * lf, ly + Math.sin(la) * lf);
    }
    x = nx; y = ny;
  }
  ctx.strokeStyle = rgb(p.col);
  ctx.lineWidth = 2.2 * S;
  ctx.lineCap = 'round';
  ctx.stroke();
}

function drawRibbon(p, t) {
  const sway = Math.sin(t * p.sp + p.ph) * p.h * .13 + p.h * .1, tipX = p.x + sway, tipY = p.y - p.h;
  const cx = p.x + sway * .15 + Math.sin(t * p.sp * 1.3 + p.ph + 1) * p.h * .06, cy = p.y - p.h * .55;
  ctx.beginPath();
  ctx.moveTo(p.x - p.w, p.y);
  ctx.quadraticCurveTo(cx - p.w, cy, tipX, tipY);
  ctx.quadraticCurveTo(cx + p.w, cy, p.x + p.w, p.y);
  ctx.closePath();
  const g = ctx.createLinearGradient(0, p.y, 0, tipY);
  g.addColorStop(0, rgb(p.col.map(c => c * .55))); g.addColorStop(1, rgb(p.col));
  ctx.fillStyle = g;
  ctx.fill();
}

function drawReeds(t) {
  for (const r of reeds) {
    const by = bedY(r.x) + 8 * S, sway = Math.sin(t * .5 + r.ph) * 5 * S;
    const g = ctx.createLinearGradient(0, 0, 0, by);
    g.addColorStop(0, rgb(r.col)); g.addColorStop(1, rgb(r.col.map(c => c * .4)));
    ctx.strokeStyle = g; ctx.lineWidth = r.w; ctx.lineCap = 'butt';
    ctx.beginPath();
    ctx.moveTo(r.x, by);
    ctx.quadraticCurveTo(r.x + sway * .3, by * .5, r.x + r.lean * by + sway, -10);
    ctx.stroke();
  }
}

function drawPads(t) {
  for (const p of pads) {
    const y = surfY(p.x, t) + 3 * S, sway = Math.sin(t * .6 + p.ph) * 14 * S;
    ctx.strokeStyle = 'rgb(92,112,52)'; ctx.lineWidth = 3.6 * S; ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(p.base, bedY(p.base) + 8 * S);
    ctx.bezierCurveTo(p.base + sway, H * .6, p.x - sway * 2, H * .3, p.x, y);
    ctx.stroke();
  }
  for (const p of pads) {
    const y = surfY(p.x, t) + 3 * S, ry = p.w * .14;
    // Backlit from above: bright translucent centre, darker rim.
    const g = ctx.createRadialGradient(p.x, y, 0, p.x, y, p.w);
    g.addColorStop(0, 'rgb(150,196,70)'); g.addColorStop(.7, 'rgb(92,140,48)'); g.addColorStop(1, 'rgb(46,82,30)');
    ctx.save();
    ctx.translate(p.x, y); ctx.scale(1, ry / p.w); ctx.translate(-p.x, -y);
    ctx.beginPath(); ctx.arc(p.x, y, p.w, 0, TAU); ctx.fillStyle = g; ctx.fill();
    ctx.beginPath();
    for (let k = 0; k < 14; k++) { ctx.moveTo(p.x, y); ctx.lineTo(p.x + Math.cos(k / 14 * TAU) * p.w * .94, y + Math.sin(k / 14 * TAU) * p.w * .94); }
    ctx.strokeStyle = 'rgba(40,70,24,.5)'; ctx.lineWidth = 1.2; ctx.stroke();
    ctx.restore();
  }
}

function drawSurface(t) {
  ctx.beginPath();
  ctx.moveTo(0, 0); ctx.lineTo(W, 0);
  for (let x = W; x >= -10; x -= 10) ctx.lineTo(x, surfY(x, t));
  ctx.closePath();
  const g = ctx.createLinearGradient(0, 0, 0, H * .07);
  g.addColorStop(0, rgb(light.sky.map(c => c * .82))); g.addColorStop(1, rgb(light.sky));
  ctx.fillStyle = g; ctx.fill();
  ctx.beginPath();
  for (let x = -10; x <= W + 10; x += 10) ctx.lineTo(x, surfY(x, t));
  ctx.strokeStyle = 'rgba(255,255,255,.7)'; ctx.lineWidth = 1.6 * S; ctx.stroke();
  const glow = ctx.createLinearGradient(0, H * .06, 0, H * .4);
  glow.addColorStop(0, rgb(light.sky, .38)); glow.addColorStop(1, rgb(light.sky, 0));
  ctx.fillStyle = glow; ctx.fillRect(0, H * .05, W, H * .36);
}

function drawRays(t) {
  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  const ys = H * .055;
  for (let i = 0; i < rays.length; i++) {
    const r = rays[i];
    const x0 = (((r.x + t * r.v) % 1.3) - .15) * W, len = H * (.7 + r.l * .3), dx = Math.tan(light.angle) * len;
    const a = light.rayA * r.a * (.6 + .4 * Math.sin(t * r.f * 1.7 + i * 2)) / 3;
    const g = ctx.createLinearGradient(x0, ys, x0 + dx, ys + len);
    g.addColorStop(0, rgb(light.ray, a)); g.addColorStop(1, rgb(light.ray, 0));
    ctx.fillStyle = g;
    // nested quads soften the beam edges
    for (const k of [1, .62, .3]) {
      const w = r.w * W * (.75 + .25 * Math.sin(t * r.f + i)) * k, c = x0 + r.w * W * .5;
      ctx.beginPath();
      ctx.moveTo(c - w / 2, ys); ctx.lineTo(c + w / 2, ys);
      ctx.lineTo(c + w * 1.1 + dx, ys + len); ctx.lineTo(c - w * 1.1 + dx, ys + len);
      ctx.closePath(); ctx.fill();
    }
  }
  for (const m of motes) {
    const x = (m.x + t * m.v + Math.sin(t * .4 + m.ph) * 14) % W, y = m.y + Math.sin(t * .3 + m.ph * 2) * 12;
    ctx.fillStyle = rgb(light.ray, light.motes * (.25 + .75 * (.5 + .5 * Math.sin(t * 1.1 + m.ph * 5))));
    ctx.beginPath(); ctx.arc(x, y, m.r, 0, TAU); ctx.fill();
  }
  ctx.restore();
}

function updateBubbles(dt, t) {
  if (Math.random() < dt * 1.6) {
    const x = [W * .33, W * .62, W * .8][(Math.random() * 3) | 0] + (Math.random() - .5) * 16 * S;
    bubbles.push({ x, y: bedY(x), r: (1.5 + Math.random() * 3.5) * S, ph: Math.random() * 9 });
  }
  for (let i = bubbles.length - 1; i >= 0; i--) {
    const b = bubbles[i];
    b.y -= (36 + b.r * 9) * S * dt;
    b.x += Math.sin(t * 3 + b.ph) * 12 * S * dt;
    if (b.y < surfY(b.x, t)) bubbles.splice(i, 1);
  }
}

function drawBubbles() {
  for (const b of bubbles) {
    ctx.beginPath(); ctx.arc(b.x, b.y, b.r, 0, TAU);
    ctx.fillStyle = 'rgba(220,240,230,.1)'; ctx.fill();
    ctx.strokeStyle = 'rgba(235,250,240,.55)'; ctx.lineWidth = .8; ctx.stroke();
    ctx.beginPath(); ctx.arc(b.x - b.r * .35, b.y - b.r * .35, b.r * .25, 0, TAU);
    ctx.fillStyle = 'rgba(255,255,255,.8)'; ctx.fill();
  }
}

/* ---------- frame ---------- */

const mouse = { x: -1e4, y: -1e4 };

function drawShadow(f) {
  const by = bedY(f.x), h = by - f.y, L = f.sp.L * S * f.sc;
  const a = 0.3 * (light.caustic / 0.6) * clamp(1 - h / (H * .5), 0, 1);
  if (a <= 0.01) return;
  const x = f.x + Math.tan(light.angle) * h;
  ctx.fillStyle = `rgba(10,10,6,${a})`;
  ctx.beginPath();
  ctx.ellipse(x, bedY(x) + 14 * S, L * .42 * Math.max(.3, Math.abs(f.face)), (6 + h * .02) * S, 0, 0, TAU);
  ctx.fill();
}

function drawLabel() {
  let hit = null, hz = -1;
  for (const f of fishes) {
    const L = f.sp.L * S * f.sc;
    if (f.z > hz && Math.abs(mouse.x - f.x) < L / 2 && Math.abs(mouse.y - f.y) < Math.max(f.sp.maxT, f.sp.maxB) * L + 8) hz = f.z, hit = { x: f.x, y: f.y - f.sp.maxT * L, name: f.sp.name, latin: f.sp.latin };
  }
  if (!hit && Math.abs(mouse.x - cray.x) < 60 * S && Math.abs(mouse.y - bedY(cray.x)) < 40 * S) hit = { x: cray.x, y: bedY(cray.x) - 34 * S, name: cray.name, latin: cray.latin };
  canvas.dataset.hover = hit ? hit.name : '';
  if (!hit) return;
  ctx.font = '600 14px system-ui, sans-serif';
  const w1 = ctx.measureText(hit.name).width;
  ctx.font = 'italic 12px system-ui, sans-serif';
  const w = Math.max(w1, ctx.measureText(hit.latin).width) + 24;
  const x = clamp(hit.x - w / 2, 8, W - w - 8), y = Math.max(8, hit.y - 62);
  ctx.fillStyle = 'rgba(8,22,20,.78)';
  ctx.beginPath(); ctx.roundRect(x, y, w, 46, 9); ctx.fill();
  ctx.strokeStyle = 'rgba(255,255,255,.25)'; ctx.lineWidth = 1; ctx.stroke();
  ctx.fillStyle = '#f2f6e8'; ctx.textBaseline = 'top';
  ctx.font = '600 14px system-ui, sans-serif'; ctx.fillText(hit.name, x + 12, y + 8);
  ctx.fillStyle = 'rgba(232,240,214,.8)';
  ctx.font = 'italic 12px system-ui, sans-serif'; ctx.fillText(hit.latin, x + 12, y + 27);
}

let last = 0, frame = 0;
function tick(ms) {
  const t = ms / 1000, dt = Math.min(0.05, t - last || 0.016);
  last = t;
  updateLight(dt);
  if (frame++ % 2 === 0) updateCaustics(t);
  updateSchool(dt);
  for (const f of fishes) updateFish(f, dt);
  updateCray(dt);
  updateBubbles(dt, t);
  for (let i = pellets.length - 1; i >= 0; i--) {
    const p = pellets[i], floor = bedY(p.x) + 6 * S;
    if (p.y < floor) { p.y += 30 * S * dt; p.x += Math.sin(t * 2 + p.ph) * 8 * S * dt; }
    else if ((p.rest += dt) > 25) pellets.splice(i, 1);
  }

  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  const bg = ctx.createLinearGradient(0, 0, 0, H);
  bg.addColorStop(0, rgb(light.top)); bg.addColorStop(.45, rgb(light.mid)); bg.addColorStop(1, rgb(light.deep));
  ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H);
  drawSurface(t);
  ctx.drawImage(rootCanvas, 0, 0, W, H);
  for (const p of weeds) if (p.layer === 0) drawWeed(p, t);
  const far = fishes.filter(f => f.z < 0.35), near = fishes.filter(f => f.z >= 0.35).sort((a, b) => a.z - b.z);
  for (const f of far) drawFish(f);
  ctx.fillStyle = rgb(light.mid, 0.34);
  ctx.fillRect(0, H * .06, W, H);

  ctx.drawImage(bedCanvas, 0, 0, W, H);
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(0, bedY(0));
  for (let x = 16; x <= W + 16; x += 16) ctx.lineTo(x, bedY(x));
  ctx.lineTo(W, H); ctx.lineTo(0, H);
  ctx.clip();
  for (const f of near) drawShadow(f);
  ctx.globalCompositeOperation = 'lighter';
  ctx.globalAlpha = light.caustic;
  ctx.drawImage(caustic, 0, H * .84, W, H * .16);
  ctx.restore();

  drawReeds(t);
  for (const p of ribbons) if (p.layer === 1) drawRibbon(p, t);
  for (const p of weeds) if (p.layer === 1) drawWeed(p, t);
  drawCray(t);
  for (const f of near) drawFish(f);
  ctx.fillStyle = 'rgb(150,104,58)';
  for (const p of pellets) { ctx.beginPath(); ctx.arc(p.x, p.y, 2.6 * S, 0, TAU); ctx.fill(); }
  drawBubbles();
  drawPads(t);
  for (const p of ribbons) if (p.layer === 2) drawRibbon(p, t);

  ctx.globalCompositeOperation = 'multiply';
  ctx.fillStyle = rgb(light.ambient);
  ctx.fillRect(0, 0, W, H);
  ctx.globalCompositeOperation = 'source-over';
  drawRays(t);
  ctx.fillStyle = vignette; ctx.fillRect(0, 0, W, H);
  drawLabel();
  requestAnimationFrame(tick);
}

function resize() {
  const ow = W, oh = H;
  DPR = Math.min(2, window.devicePixelRatio || 1);
  W = window.innerWidth; H = window.innerHeight;
  S = clamp(Math.min(H / 800, W / 1100), 0.45, 1.6);
  canvas.width = W * DPR; canvas.height = H * DPR;
  buildScene();
  buildBed();
  for (const f of fishes) {
    if (ow) { f.x *= W / ow; f.y *= H / oh; }
    f.target = null;
  }
  if (ow) { cray.x *= W / ow; school.x *= W / ow; school.y *= H / oh; }
  school.tx = school.x; school.ty = school.y;
}

resize();
for (const key in SPECIES) {
  const sp = SPECIES[key];
  for (let i = 0; i < sp.count; i++) {
    const f = makeFish(sp, i), p = pickTarget(sp);
    f.x = p.x; f.y = p.y;
    fishes.push(f);
  }
}
{ const p = pickTarget(SPECIES.roach); school.x = school.tx = p.x; school.y = school.ty = p.y; }
for (const f of fishes) if (f.sp.schooling) { f.x = school.x + f.ox * S; f.y = school.y + f.oy * S; }
cray.x = W * 0.4;

window.addEventListener('resize', resize);
canvas.addEventListener('pointermove', (e) => { mouse.x = e.clientX; mouse.y = e.clientY; });
canvas.addEventListener('pointerleave', () => { mouse.x = mouse.y = -1e4; });
canvas.addEventListener('pointerdown', (e) => {
  if (e.clientY > bedY(e.clientX)) return;
  for (let i = 0; i < 5 && pellets.length < 40; i++) {
    pellets.push({ x: e.clientX + (Math.random() - .5) * 40 * S, y: Math.max(e.clientY, H * .07) + (Math.random() - .5) * 20 * S, ph: Math.random() * 9, rest: 0 });
  }
});
for (const btn of document.querySelectorAll('[data-mood]')) {
  btn.addEventListener('click', () => {
    mood = btn.dataset.mood;
    for (const o of document.querySelectorAll('[data-mood]')) o.setAttribute('aria-pressed', String(o === btn));
  });
}
// Read-only handle for tests and console poking.
window.aquarium = { fishes, pellets, cray, light };
requestAnimationFrame(tick);
})();
