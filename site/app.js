// Aurora Odds: turns the TabPFN nowcast (data/nowcast.json) into an answer for one sky.

const $ = (id) => document.getElementById(id);
const rad = Math.PI / 180;
const POLE = { n: { lat: 80.8, lon: -72.6 }, s: { lat: -80.8, lon: 107.4 } }; // IGRF dipole poles, epoch 2025
// The auroral oval's equatorward edge in corrected (AACGM) magnetic latitude: about 66.5 deg at Kp 0 and about
// 2 deg lower per Kp step (Gussenhoven et al. 1983; the same rule NOAA's Kp maps use). Low on the poleward horizon
// it is seen from roughly 3.5 deg further away, and a phone's night mode picks it up about 5.5 deg away, values
// checked against what observers report for Tromso, Edinburgh, Minneapolis, London and the May 2024 storm.
const EDGE0 = 66.5, SLOPE = 2.04;
const MARGIN = { overhead: 0, eyes: 3.5, camera: 5.5 };
let AACGM = null;
const EARTH_KM = 6371;

const CITIES = [
  ["Tromsø", 69.65, 18.96], ["Fairbanks", 64.84, -147.72], ["Reykjavík", 64.15, -21.94], ["Yellowknife", 62.45, -114.37],
  ["Anchorage", 61.22, -149.9], ["Helsinki", 60.17, 24.94], ["Oslo", 59.91, 10.75], ["Stockholm", 59.33, 18.07],
  ["Edinburgh", 55.95, -3.19], ["Copenhagen", 55.68, 12.57], ["Edmonton", 53.55, -113.49], ["Berlin", 52.52, 13.41],
  ["Amsterdam", 52.37, 4.9], ["London", 51.51, -0.13], ["Winnipeg", 49.9, -97.14], ["Paris", 48.86, 2.35],
  ["Seattle", 47.61, -122.33], ["Minneapolis", 44.98, -93.27], ["Toronto", 43.65, -79.38], ["Sapporo", 43.06, 141.35],
  ["Boston", 42.36, -71.06], ["Chicago", 41.88, -87.63], ["New York", 40.71, -74.01], ["Denver", 39.74, -104.99],
  ["Ushuaia", -54.8, -68.3], ["Invercargill", -46.41, 168.35], ["Dunedin", -45.87, 170.5], ["Hobart", -42.88, 147.33],
  ["Melbourne", -37.81, 144.96],
];

const state = { now: null, place: null, cloud: null, cloudAt: 0, replayAt: null };
const params = new URLSearchParams(location.search);
const REPLAY = /^[a-z0-9-]+$/.test(params.get("replay") || "") ? params.get("replay") : null;
const nowTime = () => state.replayAt || new Date();
const store = {
  get(k) { try { return JSON.parse(localStorage.getItem(k)); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode */ } },
};

// ---------- geometry ----------
function magLat(lat, lon) {
  if (AACGM) {
    const { step, lat0, lon0, mlat } = AACGM;
    const fy = (lat - lat0) / step, fx = (((lon - lon0) % 360) + 360) % 360 / step;
    const y0 = Math.min(mlat.length - 2, Math.max(0, Math.floor(fy))), x0 = Math.min(mlat[0].length - 2, Math.floor(fx));
    const ty = fy - y0, tx = fx - x0;
    const c = [[y0, x0, (1 - ty) * (1 - tx)], [y0, x0 + 1, (1 - ty) * tx], [y0 + 1, x0, ty * (1 - tx)], [y0 + 1, x0 + 1, ty * tx]]
      .map(([i, j, w]) => [mlat[i][j], w]).filter(([v]) => v != null);
    const wsum = c.reduce((a, [, w]) => a + w, 0);
    if (c.length && wsum > 0.2) return c.reduce((a, [v, w]) => a + v * w, 0) / wsum;
  }
  return dipoleLat(lat, lon);
}
function dipoleLat(lat, lon) {
  const p = POLE.n;
  const s = Math.sin(lat * rad) * Math.sin(p.lat * rad) + Math.cos(lat * rad) * Math.cos(p.lat * rad) * Math.cos((lon - p.lon) * rad);
  return Math.asin(Math.max(-1, Math.min(1, s))) / rad;
}
function bearing(lat1, lon1, lat2, lon2) {
  const y = Math.sin((lon2 - lon1) * rad) * Math.cos(lat2 * rad);
  const x = Math.cos(lat1 * rad) * Math.sin(lat2 * rad) - Math.sin(lat1 * rad) * Math.cos(lat2 * rad) * Math.cos((lon2 - lon1) * rad);
  return (Math.atan2(y, x) / rad + 360) % 360;
}
const compass = (deg) => ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"][Math.round(deg / 22.5) % 16];
const needHp = (mlat, kind) => Math.max(0, (EDGE0 - MARGIN[kind] - Math.abs(mlat)) / SLOPE);
const edgeLat = (hp) => EDGE0 - SLOPE * hp;
function elevation(distKm, altKm) {
  const th = distKm / EARTH_KM;
  return Math.atan2((EARTH_KM + altKm) * Math.cos(th) - EARTH_KM, (EARTH_KM + altKm) * Math.sin(th)) / rad;
}

// ---------- sun ----------
function sunAltitude(date, lat, lon) {
  const d = date.getTime() / 86400000 - 10957.5; // days from J2000.0
  const g = (357.529 + 0.98560028 * d) * rad;
  const q = 280.459 + 0.98564736 * d;
  const L = (q + 1.915 * Math.sin(g) + 0.02 * Math.sin(2 * g)) * rad;
  const e = (23.439 - 0.00000036 * d) * rad;
  const ra = Math.atan2(Math.cos(e) * Math.sin(L), Math.cos(L)) / rad;
  const dec = Math.asin(Math.sin(e) * Math.sin(L));
  const gmst = (18.697374558 + 24.06570982441908 * d) % 24;
  const ha = ((gmst * 15 + lon - ra) % 360) * rad;
  return Math.asin(Math.sin(lat * rad) * Math.sin(dec) + Math.cos(lat * rad) * Math.cos(dec) * Math.cos(ha)) / rad;
}
function nextDark(lat, lon, from = new Date(), limit = -12) {
  for (let m = 5; m <= 24 * 60; m += 5) {
    const t = new Date(from.getTime() + m * 60000);
    if (sunAltitude(t, lat, lon) < limit) return t;
  }
  return null;
}
const hhmm = (d) => d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
const placeTime = (d) => {
  const off = state.cloud?.offset;
  if (off == null) return hhmm(d);
  const t = new Date(d.getTime() + off * 1000);
  return `${String(t.getUTCHours()).padStart(2, "0")}:${String(t.getUTCMinutes()).padStart(2, "0")}`;
};

// ---------- forecast ----------
function pAtLeast(exceed, levels, hp) {
  if (hp <= levels[0]) return exceed[0] + (1 - exceed[0]) * Math.max(0, (levels[0] - hp) / levels[0]);
  if (hp >= levels[levels.length - 1]) return exceed[exceed.length - 1];
  const i = levels.findIndex((l) => l >= hp);
  const f = (hp - levels[i - 1]) / (levels[i] - levels[i - 1]);
  return exceed[i - 1] + f * (exceed[i] - exceed[i - 1]);
}
function quantileHp(exceed, levels, p) { // level where P(Hp30 >= level) falls to p
  for (let i = 1; i < levels.length; i++) {
    if (exceed[i] <= p) {
      const f = (exceed[i - 1] - p) / Math.max(1e-9, exceed[i - 1] - exceed[i]);
      return levels[i - 1] + f * (levels[i] - levels[i - 1]);
    }
  }
  return levels[levels.length - 1];
}
const pct = (p) => (p < 0.01 ? "<1%" : p > 0.99 ? ">99%" : `${Math.round(p * 100)}%`);

// ---------- data ----------
async function loadNowcast() {
  const file = REPLAY ? `data/replay-${REPLAY}.json` : `data/nowcast.json?t=${Date.now()}`;
  const r = await fetch(file, { cache: "no-store" });
  if (!r.ok) throw new Error(`nowcast ${r.status}`);
  state.now = await r.json();
  if (state.now.replay) {
    const rp = state.now.replay;
    state.replayAt = new Date(rp.time);
    const when = state.replayAt.toISOString().slice(0, 16).replace("T", " ");
    $("replay").hidden = false;
    $("replay").innerHTML = `<b>Replay</b> · ${rp.label}, ${when} UTC, with that night's solar wind and clouds. In the following hour Hp30 actually reached <b>${rp.actual_hp30.toFixed(1)}</b>. <a href="./">Back to tonight</a>`;
    $("updated").textContent = `Replay of ${when} UTC.`;
    return;
  }
  const gen = new Date(state.now.generated);
  $("updated").textContent = `Model run ${hhmm(gen)} (${Math.max(0, Math.round((Date.now() - gen) / 60000))} min ago).`;
}
async function loadCloud() {
  if (!state.place || Date.now() - state.cloudAt < 20 * 60000) return;
  const { lat, lon } = state.place;
  const ll = `latitude=${lat.toFixed(2)}&longitude=${lon.toFixed(2)}`;
  try {
    if (state.replayAt) {
      const day = state.replayAt.toISOString().slice(0, 10);
      const r = await fetch(`https://archive-api.open-meteo.com/v1/archive?${ll}&start_date=${day}&end_date=${day}&hourly=cloud_cover&timezone=GMT`);
      const j = await r.json();
      const tz = await (await fetch(`https://api.open-meteo.com/v1/forecast?${ll}&current=cloud_cover&timezone=auto`)).json();
      state.cloud = { now: j.hourly?.cloud_cover?.[state.replayAt.getUTCHours()] ?? null, next: [], offset: tz.utc_offset_seconds ?? null };
    } else {
      const r = await fetch(`https://api.open-meteo.com/v1/forecast?${ll}&current=cloud_cover&hourly=cloud_cover&forecast_hours=6&timezone=auto`);
      const j = await r.json();
      state.cloud = { now: j.current?.cloud_cover ?? null, next: j.hourly?.cloud_cover ?? [], offset: j.utc_offset_seconds ?? null };
    }
    state.cloudAt = Date.now();
  } catch { state.cloud = null; }
}

// ---------- render ----------
function setFact(id, value, sub, tone) {
  const el = $(id);
  el.querySelector(".v").textContent = value;
  el.querySelector(".s").textContent = sub;
  el.classList.toggle("good", tone === "good");
  el.classList.toggle("bad", tone === "bad");
}

function render() {
  const nc = state.now;
  if (!nc) return;
  renderWind(nc);
  renderTrend(nc);
  if (!state.place) return renderWorld(nc);
  const { lat, lon, name } = state.place;
  const mlat = magLat(lat, lon);
  const south = mlat < 0;
  const need = { camera: needHp(mlat, "camera"), eyes: needHp(mlat, "eyes"), overhead: needHp(mlat, "overhead") };
  const P = Object.fromEntries(Object.entries(need).map(([k, h]) => [k, pAtLeast(nc.now.exceed, nc.levels, h)]));
  const now = nowTime();
  const alt = sunAltitude(now, lat, lon);
  const dark = alt < -12;
  const darkAt = dark ? null : nextDark(lat, lon, now);
  const cloud = state.cloud?.now;

  $("where").textContent = `${name} · magnetic latitude ${Math.abs(mlat).toFixed(1)}°${south ? " S" : " N"}`;
  let verdict, reason;
  const odds = `<b>${pct(P.eyes)}</b> chance it is bright enough to see with your eyes in the next hour, <b>${pct(P.camera)}</b> for a phone camera`;
  if (alt > -6) {
    verdict = "Not yet. It's still light out.";
    reason = `${darkAt ? `It gets dark enough at <b>${placeTime(darkAt)}</b> local time. ` : "It doesn't get dark enough here today. "}Right now: ${odds}. This is a one-hour nowcast, so check again after dark.`;
  } else if (cloud != null && cloud >= 85) {
    verdict = "Clouds are in the way.";
    reason = `The sky is ${cloud}% cloud. Above it: ${odds}. Watch for gaps.`;
  } else if (P.eyes >= 0.5) {
    verdict = "Go outside now.";
    reason = `${odds}. Face ${south ? "south" : "north"}, away from lights.`;
  } else if (P.eyes >= 0.25) {
    verdict = "Worth stepping outside.";
    reason = `${odds}. Give your eyes ten minutes to adjust.`;
  } else if (P.camera >= 0.35) {
    verdict = "Bring your phone.";
    reason = `Your eyes may miss it; a night-mode photo facing ${south ? "south" : "north"} may not. ${odds}.`;
  } else if (P.camera >= 0.1) {
    verdict = "Maybe later.";
    reason = `${odds}. The Sun is fairly quiet; this page updates every ten minutes.`;
  } else {
    verdict = "Stay in. The sky is quiet.";
    reason = `${odds}. Nothing strong is on its way from the Sun yet.`;
  }
  $("verdict").textContent = verdict;
  $("reason").innerHTML = reason;
  state.intensity = dark ? Math.min(1, P.camera * 0.6 + P.eyes) : Math.min(0.35, P.eyes);

  $("facts").hidden = false;
  setFact("f-odds", pct(P.eyes), `by eye · ${pct(P.camera)} on camera`, P.eyes >= 0.25 ? "good" : null);
  setFact("f-dark", dark ? "Dark" : alt > -6 ? "Daylight" : "Twilight",
    dark ? `sun ${Math.round(-alt)}° below the horizon` : darkAt ? `dark enough at ${placeTime(darkAt)}` : "no real night today", dark ? "good" : "bad");
  if (cloud == null) setFact("f-cloud", "–", "forecast unavailable", null);
  else setFact("f-cloud", `${cloud}%`, cloud < 30 ? "mostly clear" : cloud < 70 ? "broken cloud" : "overcast", cloud < 50 ? "good" : "bad");
  if (need.eyes < 0.5) setFact("f-need", "Any", "you live under the auroral oval", "good");
  else setFact("f-need", `Kp ${need.eyes.toFixed(1)}`, `for your eyes · ${need.camera.toFixed(1)} for a camera`, null);

  renderLevels(P, need);
  renderLook(nc, mlat, lat, lon);
}

function renderWorld(nc) {
  const med = quantileHp(nc.now.exceed, nc.levels, 0.5);
  const reach = edgeLat(med) - MARGIN.camera;
  const lit = CITIES.filter(([, la, lo]) => Math.abs(magLat(la, lo)) >= reach).map(([n]) => n);
  $("where").textContent = "Choose where you are";
  $("verdict").textContent = "Should you go outside and look up?";
  $("reason").innerHTML = `Right now a phone camera would most likely catch the aurora down to about <b>${reach.toFixed(0)}° magnetic latitude</b>${lit.length ? `: ${lit.slice(0, 5).join(", ")}${lit.length > 5 ? " and further poleward" : ""}` : ""}. Pick your place for the full answer.`;
  state.intensity = Math.min(0.7, med / 9);
}

function renderLevels(P, need) {
  const rows = [
    ["camera", "Phone camera", `Kp ${need.camera.toFixed(1)}+: a night-mode photo shows it`],
    ["eyes", "Your eyes", `Kp ${need.eyes.toFixed(1)}+: a glow over the horizon`],
    ["overhead", "Overhead", `Kp ${need.overhead.toFixed(1)}+: curtains above you`],
  ];
  $("bars").innerHTML = rows.map(([k, t, s]) => `<li><div class="lbl"><b>${t}</b><span>${s}</span></div><div class="track"><div class="fill" style="width:${(P[k] * 100).toFixed(1)}%"></div></div><div class="pct">${pct(P[k])}</div></li>`).join("");
  $("levels").hidden = false;
}

function renderLook(nc, mlat, lat, lon) {
  const south = mlat < 0;
  const pole = south ? POLE.s : POLE.n;
  const brg = bearing(lat, lon, pole.lat, pole.lon);
  const hpLikely = quantileHp(nc.now.exceed, nc.levels, 0.5);
  const hpHigh = quantileHp(nc.now.exceed, nc.levels, 0.1);
  const band = (hp) => {
    const d = (edgeLat(hp) - Math.abs(mlat)) * 111.2;
    if (d <= 0) return { overhead: true, lo: 90, hi: 90 };
    return { overhead: false, lo: elevation(d, 110), hi: elevation(d, 300) };
  };
  const likely = band(hpLikely), high = band(hpHigh);
  likely.dist = (edgeLat(hpLikely) - Math.abs(mlat)) * 111.2;
  high.dist = (edgeLat(hpHigh) - Math.abs(mlat)) * 111.2;
  const dir = `${compass(brg)} (${Math.round(brg)}°)`;
  let text;
  if (likely.overhead) text = `Face <b>${dir}</b> and look up: at this activity level the oval sits over you, so the aurora can fill the whole sky.`;
  else if (likely.hi > 3) text = `Face <b>${dir}</b>, toward the magnetic pole. Expect the aurora's lower edge about <b>${Math.max(0, Math.round(likely.lo))}°</b> above the horizon and its top near ${Math.round(likely.hi)}°: ${Math.max(1, Math.round(likely.hi / 10))} fist${likely.hi >= 15 ? "s" : ""} at arm's length.`;
  else if (high.hi > 3) text = `Face <b>${dir}</b>. At the most likely level it stays below your horizon; if activity reaches the upper end of the forecast, it rises to about <b>${Math.round(high.hi)}°</b>.`;
  else text = `Face <b>${dir}</b> when it happens. For the next hour the aurora is expected to stay below your horizon.`;
  $("look-text").innerHTML = text;
  $("look").hidden = false;
  drawHorizon(likely, high, brg);
}

const widthOf = (id, min = 300) => Math.max(min, Math.round($(id).getBoundingClientRect().width) || 640);
function sized(id, W, H) { $(id).setAttribute("viewBox", `0 0 ${W} ${H}`); }

function drawHorizon(likely, high, brg) {
  // x is azimuth from 90 deg left to 90 deg right of the magnetic pole. A band of aurora at a fixed magnetic
  // latitude lies further away toward the sides, so it appears as an arch, highest straight toward the pole.
  const W = widthOf("horizon"), H = W < 480 ? 230 : 260, base = H - 46, L = 40;
  sized("horizon", W, H);
  const y = (e) => base - (Math.min(70, Math.max(0, e)) / 70) * (base - 16);
  const az = (x) => ((x - L) / (W - L)) * 180 - 90;
  let s = `<defs><linearGradient id="ag" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="var(--accent)" stop-opacity=".62"/><stop offset=".5" stop-color="var(--accent)" stop-opacity=".22"/><stop offset="1" stop-color="var(--accent)" stop-opacity="0"/></linearGradient>
  <linearGradient id="ah" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="var(--accent)" stop-opacity=".2"/><stop offset="1" stop-color="var(--accent)" stop-opacity="0"/></linearGradient></defs>`;
  for (const e of [10, 20, 30, 45, 60]) s += `<line class="gridline" x1="${L}" x2="${W}" y1="${y(e)}" y2="${y(e)}" stroke-dasharray="2 6"/><text class="axis" x="0" y="${y(e) + 4}">${e}°</text>`;
  const arch = (b, fill) => {
    if (b.overhead) {
      let d = `M${L} ${y(0)}`;
      for (let x = L; x <= W; x += 6) d += ` L${x} ${(y(70) + Math.sin(x / 37) * 5).toFixed(1)}`;
      return `<path d="${d} L${W} ${y(0)} Z" fill="url(#${fill})"/>`;
    }
    const top = [], bot = [];
    for (let x = L; x <= W; x += 6) {
      const c = Math.cos(Math.min(80, Math.abs(az(x))) * rad);
      const dist = b.dist / Math.max(0.17, c);
      const lo = elevation(dist, 110), hi = elevation(dist, 300) + Math.sin(x / 29) * 1.2;
      top.push(`${x} ${y(Math.max(0, hi)).toFixed(1)}`);
      bot.push(`${x} ${y(Math.max(0, lo)).toFixed(1)}`);
    }
    if (b.hi <= 0) return "";
    return `<path d="M${top.join(" L")} L${bot.reverse().join(" L")} Z" fill="url(#${fill})"/>`;
  };
  s += arch(high, "ah") + arch(likely, "ag");
  const hill = (x) => base - 6 - Math.sin(x / 61) * 5 - Math.sin(x / 23 + 1) * 2.5;
  let ground = `M${L} ${hill(L)}`;
  for (let x = L; x <= W; x += 8) ground += ` L${x} ${hill(x).toFixed(1)}`;
  s += `<path d="${ground} L${W} ${H} L${L} ${H} Z" fill="var(--bg)"/><path d="${ground}" fill="none" stroke="var(--faint)" stroke-width="1"/>`;
  if (likely.overhead) s += `<text class="axis" x="${W - 8}" y="${y(64)}" text-anchor="end">overhead</text>`;
  for (const a of [-90, -45, 0, 45, 90]) {
    const x = L + ((a + 90) / 180) * (W - L);
    const label = compass((brg + a + 360) % 360);
    s += `<text class="axis" x="${x}" y="${H - 12}" text-anchor="${a === -90 ? "start" : a === 90 ? "end" : "middle"}"${a === 0 ? ' style="fill:var(--text)"' : ""}>${a === 0 ? `${label} ${Math.round(brg)}°` : label}</text>`;
  }
  $("horizon").innerHTML = s;
}

function renderWind(nc) {
  const sw = nc.solar_wind || [];
  const t0 = Date.parse(nc.now.time);
  const pts = sw.map((r) => ({ m: (Date.parse(r.time) - t0) / 60000, bz: r.bz, v: r.v })).filter((p) => p.bz != null && p.m >= -180);
  const ahead = Math.max(0, ...pts.map((p) => p.m));
  const W = widthOf("bzchart"), H = 200, L = 40, R = W - 6, top = 10, bot = 170;
  sized("bzchart", W, H);
  const xmax = Math.max(30, Math.ceil(ahead / 10) * 10);
  const x = (m) => L + ((m + 180) / (180 + xmax)) * (R - L);
  const lim = Math.max(10, ...pts.map((p) => Math.abs(p.bz))) * 1.1;
  const y = (bz) => top + ((lim - bz) / (2 * lim)) * (bot - top);
  let s = `<rect x="${x(0)}" y="${top}" width="${x(xmax) - x(0)}" height="${bot - top}" fill="var(--ahead)" rx="6"/>`;
  s += `<line class="gridline" x1="${L}" x2="${R}" y1="${y(0)}" y2="${y(0)}"/>`;
  for (const v of [-lim * 0.8, lim * 0.8]) s += `<text class="axis" x="0" y="${y(v) + 4}">${v > 0 ? "+" : ""}${Math.round(v)}</text>`;
  if (pts.length > 1) {
    const line = pts.map((p, i) => `${i ? "L" : "M"}${x(p.m).toFixed(1)} ${y(p.bz).toFixed(1)}`).join(" ");
    const area = `M${x(pts[0].m)} ${y(0)} ` + pts.map((p) => `L${x(p.m).toFixed(1)} ${y(Math.min(0, p.bz)).toFixed(1)}`).join(" ") + ` L${x(pts[pts.length - 1].m)} ${y(0)} Z`;
    s += `<path d="${area}" fill="var(--south)" opacity=".35"/><path d="${line}" fill="none" stroke="var(--text)" stroke-width="1.5" opacity=".85"/>`;
  }
  for (const m of [-180, -120, -60, 0]) s += `<text class="axis" x="${x(m)}" y="${H - 6}" text-anchor="${m === -180 ? "start" : "middle"}">${m === 0 ? "now" : `${m / 60}h`}</text>`;
  if (xmax >= 20) s += `<text class="axis" x="${x(xmax)}" y="${H - 6}" text-anchor="end">+${Math.round(ahead)} min</text>`;
  s += `<text class="axis" x="${L + 4}" y="${top + 12}">Bz, nT</text>`;
  $("bzchart").innerHTML = s;

  const n = nc.now;
  const bits = [];
  if (n.bz != null) bits.push(n.bz < -5 ? `The field is pointing <b>south at ${Math.abs(n.bz).toFixed(1)} nT</b>, the orientation that lets solar wind energy into the magnetosphere` : n.bz < 0 ? `The field is tilted slightly south (${n.bz.toFixed(1)} nT)` : `The field is pointing north (+${n.bz.toFixed(1)} nT), which mostly keeps the door shut`);
  if (n.speed != null) bits.push(`the wind is blowing at <b>${Math.round(n.speed)} km/s</b>`);
  let tail = "";
  if (n.bz_min_ahead != null && n.ahead_minutes > 5) tail = ` Already measured upstream: the next ${Math.round(n.ahead_minutes)} minutes, with the field reaching ${n.bz_min_ahead.toFixed(1)} nT.`;
  $("wind-text").innerHTML = `${bits.join(" and ")}.${tail} Source: ${nc.source.spacecraft} at L1.`;
}

function renderTrend(nc) {
  const h = nc.history || [];
  if (h.length < 2) return;
  const W = widthOf("spark"), H = 140, L = 40, R = W - 6, top = 10, bot = 112;
  sized("spark", W, H);
  const t0 = Date.parse(h[0].time), t1 = Date.parse(h[h.length - 1].time);
  const x = (t) => L + ((Date.parse(t) - t0) / Math.max(1, t1 - t0)) * (R - L);
  const vmax = Math.max(6, ...h.map((r) => quantileHp(r.exceed, nc.levels, 0.1))) + 0.5;
  const y = (v) => bot - (v / vmax) * (bot - top);
  let s = "";
  for (const v of [0, 3, 6, 9].filter((v) => v <= vmax)) s += `<line class="gridline" x1="${L}" x2="${R}" y1="${y(v)}" y2="${y(v)}" stroke-dasharray="2 6"/><text class="axis" x="0" y="${y(v) + 4}">${v}</text>`;
  const up = h.map((r) => [x(r.time), y(quantileHp(r.exceed, nc.levels, 0.1))]);
  const lo = h.map((r) => [x(r.time), y(quantileHp(r.exceed, nc.levels, 0.9))]);
  s += `<path d="M${up.map((p) => p.join(" ")).join(" L")} L${lo.reverse().map((p) => p.join(" ")).join(" L")} Z" fill="var(--accent)" opacity=".16"/>`;
  s += `<path d="M${h.map((r) => `${x(r.time).toFixed(1)} ${y(r.median).toFixed(1)}`).join(" L")}" fill="none" stroke="var(--accent)" stroke-width="2"/>`;
  if (state.place) {
    const need = needHp(magLat(state.place.lat, state.place.lon), "eyes");
    if (need < vmax) s += `<line x1="${L}" x2="${R}" y1="${y(need)}" y2="${y(need)}" stroke="var(--text)" stroke-dasharray="5 4" opacity=".6"/><text class="axis" x="${L + 4}" y="${y(need) - 6}">needed for your eyes</text>`;
  }
  s += `<text class="axis" x="${L}" y="${H - 4}">${hhmm(new Date(t0))}</text><text class="axis" x="${R}" y="${H - 4}" text-anchor="end">${hhmm(new Date(t1))}</text>`;
  $("spark").innerHTML = s;
  $("trend-note").textContent = "Forecast Hp30 for the hour after each half-hour mark: the line is the median, the band spans the 10th to 90th percentile.";
}

// ---------- sky canvas ----------
// A procedural curtain: thin vertical rays whose height and brightness drift with layered sine noise, softened
// by a blur, standing above a dark ridge. Its strength follows the odds for the chosen sky.
function startSky() {
  const cv = $("sky"), ctx = cv.getContext("2d");
  const off = document.createElement("canvas"), octx = off.getContext("2d");
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const K = 4; // rays are drawn at quarter resolution and scaled up, which softens them for free
  let stars = [], w = 0, h = 0, shown = 0, last = 0;
  const resize = () => {
    const dpr = Math.min(2, devicePixelRatio || 1);
    w = cv.clientWidth; h = cv.clientHeight;
    cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    off.width = Math.max(1, Math.round(w / K)); off.height = Math.max(1, Math.round(h / K));
    stars = Array.from({ length: Math.round((w * h) / 2400) }, () => ({ x: Math.random() * w, y: Math.random() * h * 0.82, r: Math.random() * 1.2 + 0.25, p: Math.random() * 6.3 }));
  };
  const css = (name, fallback) => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
  const ridge = (x) => h * 0.86 - Math.sin(x / 140) * 10 - Math.sin(x / 47 + 2) * 4 - Math.sin(x / 13) * 1.2;
  const frame = (t) => {
    if (!reduce && !document.hidden) requestAnimationFrame(frame);
    if (t - last < 33 && !reduce) return;
    last = t;
    shown += ((state.intensity ?? 0.2) - shown) * 0.05;
    const c = css("--accent", "#7fe0ab");
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#cfd8e3";
    for (const st of stars) {
      ctx.globalAlpha = 0.25 + 0.3 * (0.5 + 0.5 * Math.sin(t / 1500 + st.p));
      ctx.fillRect(st.x, st.y, st.r, st.r);
    }
    const ow = off.width, oh = off.height, wide = w > 720;
    const x0 = wide ? ow * 0.3 : 0, span = ow - x0;
    octx.clearRect(0, 0, ow, oh);
    octx.globalCompositeOperation = "lighter";
    for (let i = 0; i < span; i++) {
      const u = i / span;
      const edgeFade = Math.min(1, u / 0.25) * Math.min(1, (1 - u) / 0.06);
      const fold = Math.sin(u * 5.3 + t / 6100) * oh * 0.05 + Math.sin(u * 13.7 - t / 3900) * oh * 0.018;
      const base = oh * 0.66 + fold;
      const pulse = 0.5 + 0.5 * Math.sin(u * 31 + t / 1300) * Math.sin(u * 9 - t / 2700);
      const tall = oh * (0.16 + 0.4 * Math.pow(0.5 + 0.5 * Math.sin(u * 8.1 + t / 4700), 1.6)) * (0.55 + 0.45 * shown);
      const g = octx.createLinearGradient(0, base, 0, base - tall);
      g.addColorStop(0, c); g.addColorStop(0.2, c); g.addColorStop(1, "rgba(0,0,0,0)");
      octx.globalAlpha = edgeFade * (0.12 + 0.3 * shown) * (0.45 + 0.55 * pulse);
      octx.fillStyle = g;
      octx.fillRect(x0 + i, base - tall, 1.4, tall + 1);
    }
    ctx.globalAlpha = 1;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(off, 0, 0, w, h);
    ctx.beginPath();
    ctx.moveTo(0, h);
    for (let x = 0; x <= w + 8; x += 8) ctx.lineTo(x, ridge(x));
    ctx.lineTo(w, h); ctx.closePath();
    ctx.fillStyle = css("--bg", "#070b12");
    ctx.fill();
  };
  addEventListener("resize", resize);
  document.addEventListener("visibilitychange", () => { if (!document.hidden && !reduce) requestAnimationFrame(frame); });
  resize();
  requestAnimationFrame(frame);
}

// ---------- place ----------
function setPlace(place, save = true) {
  state.place = place;
  state.cloudAt = 0;
  if (save) store.set("place", place);
  $("status").textContent = "";
  loadCloud().then(render);
  render();
}
function setupPlace() {
  const sel = $("city");
  sel.innerHTML += CITIES.map(([n], i) => `<option value="${i}">${n}</option>`).join("");
  sel.addEventListener("change", () => {
    if (sel.value === "") return;
    const [name, lat, lon] = CITIES[+sel.value];
    setPlace({ name, lat, lon });
  });
  $("geo").addEventListener("click", () => {
    if (!navigator.geolocation) { $("status").textContent = "Your browser can't share a location; pick a city instead."; return; }
    $("status").textContent = "Finding you…";
    navigator.geolocation.getCurrentPosition(
      (p) => setPlace({ name: "Your location", lat: p.coords.latitude, lon: p.coords.longitude }),
      () => { $("status").textContent = "Location was not shared. Pick a city instead."; },
      { enableHighAccuracy: false, timeout: 15000, maximumAge: 600000 },
    );
  });
  const named = CITIES.find(([n]) => n.toLowerCase() === (params.get("place") || "").toLowerCase());
  const saved = store.get("place");
  if (named) {
    state.place = { name: named[0], lat: named[1], lon: named[2] };
    sel.value = String(CITIES.indexOf(named));
  } else if (saved && typeof saved.lat === "number") state.place = saved;
}
function setupRed() {
  const btn = $("redmode");
  const apply = (on) => { document.documentElement.classList.toggle("red", on); btn.setAttribute("aria-pressed", String(on)); store.set("red", on); };
  btn.addEventListener("click", () => apply(!document.documentElement.classList.contains("red")));
  if (store.get("red")) apply(true);
}

async function tick() {
  try {
    await loadNowcast();
    await loadCloud();
    render();
  } catch (e) {
    $("status").textContent = "The latest nowcast could not be loaded. Retrying in a minute.";
    setTimeout(tick, 60000);
    return;
  }
  if (!REPLAY) setTimeout(tick, 5 * 60000);
}

let resizeTimer = 0;
addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(render, 200); });

async function boot() {
  try { AACGM = await (await fetch("aacgm.json")).json(); } catch { AACGM = null; }
  setupRed();
  setupPlace();
  startSky();
  tick();
}
boot();
