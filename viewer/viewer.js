// WPGT Mirante viewer (WikiProject GeoTwin): browsing a georeferenced SfM reconstruction of Commons photos.
// Photos are projected (with their real intrinsics and radial distortion) onto a proxy plane;
// moving between two photos animates the camera and cross-fades both projections.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const DATA = (window.MIRANTE_DATA || 'data').replace(/\/$/, '');
const $ = (id) => document.getElementById(id);
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

// ------------------------------------------------------------------ load
const toast = (msg) => { const t = $('toast'); t.textContent = msg; t.hidden = !msg; };
toast('Loading reconstruction…');
let S, xyz, rgb, visAll;
try {
  S = await fetch(`${DATA}/scene.json`).then((r) => { if (!r.ok) throw new Error(`scene.json: HTTP ${r.status}`); return r.json(); });
  // Binary buffers live next to scene.json, or inline as base64 for hosts that only serve text types.
  const b64 = (s) => Uint8Array.from(atob(s), (ch) => ch.charCodeAt(0)).buffer;
  const [pts, vis] = S.inline ? [b64(S.inline.points), b64(S.inline.vis)] : await Promise.all([
    fetch(`${DATA}/points.bin`).then((r) => r.arrayBuffer()),
    fetch(`${DATA}/vis.bin`).then((r) => r.arrayBuffer()),
  ]);
  const N = S.num_points;
  xyz = new Float32Array(pts, 0, N * 3);
  rgb = new Uint8Array(pts, N * 12, N * 3);
  visAll = new Uint32Array(vis);
} catch (e) {
  toast(`Could not load the scene data (${e.message}). Check that the data/ folder sits next to this page.`);
  throw e;
}
toast('');

const center = new THREE.Vector3(...S.center);
document.title = `${S.title} · WPGT Mirante`;
$('title').textContent = S.title;
$('stats').textContent = `${S.images.length} of ${S.num_images_total ?? S.images.length} photos placed · ${S.num_points.toLocaleString()} points` +
  (S.aligned && S.georef?.residual_m ? ` · georef median ${S.georef.residual_m.median.toFixed(1)} m` : '');

// ------------------------------------------------------------------ renderer & scene
const canvas = $('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.LinearSRGBColorSpace; // pass-through: photos & point colours are already sRGB
renderer.setClearColor(0x0e1113, 1);
const scene = new THREE.Scene();
let needsRender = true;
const invalidate = () => { needsRender = true; };
const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 5000);
camera.up.set(0, 0, 1);

// Points (recentred on scene centre to keep float precision)
const N = S.num_points;
const pos = new Float32Array(N * 3);
const col = new Float32Array(N * 3);
for (let i = 0; i < N; i++) {
  pos[3 * i] = xyz[3 * i] - center.x; pos[3 * i + 1] = xyz[3 * i + 1] - center.y; pos[3 * i + 2] = xyz[3 * i + 2] - center.z;
  col[3 * i] = rgb[3 * i] / 255; col[3 * i + 1] = rgb[3 * i + 1] / 255; col[3 * i + 2] = rgb[3 * i + 2] / 255;
}
const pgeo = new THREE.BufferGeometry();
pgeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
pgeo.setAttribute('color', new THREE.BufferAttribute(col, 3));
const pmat = new THREE.PointsMaterial({ size: 2, sizeAttenuation: false, vertexColors: true, transparent: true, opacity: 0.35, depthWrite: false });
const points = new THREE.Points(pgeo, pmat);
points.renderOrder = 0;
scene.add(points);

// ------------------------------------------------------------------ cameras
const FWD = new THREE.Vector3(0, 0, -1);
const cams = S.images.map((im, i) => {
  const C = new THREE.Vector3(...im.position).sub(center);
  const q = new THREE.Quaternion(...im.quaternion);
  const fwd = FWD.clone().applyQuaternion(q);
  const [o, n] = im.vis;
  const vis = visAll.subarray(o, o + n);
  // median depth of visible points along the optical axis
  const depths = [];
  for (let k = 0; k < vis.length; k += Math.max(1, Math.floor(vis.length / 800))) {
    const p = vis[k];
    depths.push((pos[3 * p] - C.x) * fwd.x + (pos[3 * p + 1] - C.y) * fwd.y + (pos[3 * p + 2] - C.z) * fwd.z);
  }
  depths.sort((a, b) => a - b);
  const depth = depths.length ? Math.max(0.5, depths[Math.floor(depths.length / 2)]) : 10;
  const view = new THREE.Matrix4().compose(C, q, new THREE.Vector3(1, 1, 1)).invert();
  const vfov = 2 * Math.atan(im.height / 2 / im.f);
  const hfov = 2 * Math.atan(im.width / 2 / im.f);
  const src = im.src_local || im.src_remote;
  return { i, im, C, q, fwd, vis, visSet: null, depth, view, vfov, hfov, src,
    intr: new THREE.Vector4(im.f / im.width, im.f / im.height, im.cx / im.width, im.cy / im.height) };
});
// robust: one badly placed camera must not push the overview out to infinity
const sceneRadius = (() => {
  const d = cams.map((c) => c.C.length()).filter(Number.isFinite).sort((a, b) => a - b);
  return Math.max(10, d.length ? d[Math.min(d.length - 1, Math.floor(0.95 * d.length))] : 10);
})();

// ------------------------------------------------------------------ textures
// Full-resolution photos cost ~15 MB of GPU memory each (1920 px + mipmaps), so only the most recently used
// ones are kept; older ones are disposed unless they are on screen. The overview uses small thumbnails.
const TEX_KEEP = 16;
const loader = new THREE.TextureLoader();
loader.setCrossOrigin('anonymous');
const texCache = new Map(); // cam index -> { p: Promise<Texture|null>, t: Texture|null|undefined }, oldest first
function loadTex(c) {
  let e = texCache.get(c.i);
  if (e) { texCache.delete(c.i); texCache.set(c.i, e); return e.p; } // mark as recently used
  e = { t: undefined };
  e.p = new Promise((resolve) => {
    const load = (url, fallback) => loader.load(url, (t) => {
      t.colorSpace = THREE.NoColorSpace; t.minFilter = THREE.LinearMipmapLinearFilter; t.anisotropy = 4;
      e.t = t; resolve(t);
    }, undefined, () => (fallback ? load(fallback, null) : (e.t = null, resolve(null))));
    load(c.src, c.im.src_local && c.im.src_remote ? c.im.src_remote : null);
  });
  texCache.set(c.i, e);
  trimTextures();
  return e.p;
}
function texInUse(t) {
  const u = photoMat.uniforms;
  return t === u.texA.value || t === u.texB.value || (anim && anim.texB === t);
}
function trimTextures() {
  for (const [i, e] of texCache) {
    if (texCache.size <= TEX_KEEP) return;
    if (e.t === undefined || texInUse(e.t)) continue; // still loading, or on screen
    if (e.t) e.t.dispose();
    texCache.delete(i);
  }
}
const THUMB_PX = 256;
// Commons serves only standard thumbnail widths (currently 120, 250, 330, 500, 960, 1280, 1920, 3840; others -> HTTP 400)
const smallUrl = (c) => c.im.src_local || (c.im.src_remote || '').replace(/\/(\d+)px-([^/]+)$/, '/330px-$2');
function loadThumbTex(c) { // small texture for the overview frusta, whatever size the source is
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const k = Math.min(1, THUMB_PX / Math.max(img.naturalWidth, img.naturalHeight));
      const cv = document.createElement('canvas');
      cv.width = Math.max(1, Math.round(img.naturalWidth * k)); cv.height = Math.max(1, Math.round(img.naturalHeight * k));
      cv.getContext('2d').drawImage(img, 0, 0, cv.width, cv.height);
      const t = new THREE.CanvasTexture(cv);
      t.colorSpace = THREE.NoColorSpace;
      resolve(t);
    };
    img.onerror = () => resolve(null);
    img.src = smallUrl(c);
  });
}
const blank = new THREE.DataTexture(new Uint8Array([0, 0, 0, 0]), 1, 1);
blank.needsUpdate = true;

// ------------------------------------------------------------------ projective photo material on a proxy plane
const photoMat = new THREE.ShaderMaterial({
  transparent: true, depthTest: false, depthWrite: false, side: THREE.DoubleSide,
  uniforms: {
    texA: { value: blank }, texB: { value: blank },
    viewA: { value: new THREE.Matrix4() }, viewB: { value: new THREE.Matrix4() },
    intrA: { value: new THREE.Vector4(1, 1, 0.5, 0.5) }, intrB: { value: new THREE.Vector4(1, 1, 0.5, 0.5) },
    kA: { value: 0 }, kB: { value: 0 }, onA: { value: 0 }, onB: { value: 0 }, t: { value: 0 },
  },
  vertexShader: /* glsl */`
    varying vec3 vW;
    void main() { vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`,
  fragmentShader: /* glsl */`
    uniform sampler2D texA, texB; uniform mat4 viewA, viewB; uniform vec4 intrA, intrB;
    uniform float kA, kB, onA, onB, t; varying vec3 vW;
    vec4 proj(sampler2D tex, mat4 V, vec4 K, float k1, out float cov) {
      vec4 pc = V * vec4(vW, 1.0);
      float z = -pc.z;                       // depth in front of the photo's camera
      if (z < 1e-3) { cov = 0.0; return vec4(0.0); }
      float xn = pc.x / z, yn = -pc.y / z;   // COLMAP-normalised coordinates (y down)
      float d = 1.0 + k1 * (xn * xn + yn * yn);
      vec2 uv = vec2(K.z + K.x * xn * d, 1.0 - (K.w + K.y * yn * d));
      vec2 e = min(uv, 1.0 - uv);
      cov = smoothstep(0.0, 0.004, min(e.x, e.y));
      return texture2D(tex, clamp(uv, 0.0, 1.0));
    }
    void main() {
      float cA, cB;
      vec4 a = proj(texA, viewA, intrA, kA, cA);
      vec4 b = proj(texB, viewB, intrB, kB, cB);
      float wA = cA * onA * (1.0 - t), wB = cB * onB * t;
      float w = wA + wB;
      if (w < 1e-3) discard;
      gl_FragColor = vec4((a.rgb * wA + b.rgb * wB) / w, clamp(w, 0.0, 1.0));
    }`,
});
const proxy = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), photoMat);
proxy.renderOrder = 2;
proxy.frustumCulled = false;
scene.add(proxy);

function placeProxy(point, normal, size) {
  proxy.position.copy(point);
  proxy.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), normal.clone().normalize());
  proxy.scale.set(size, size, 1);
}
function restPlane(c) { // fronto-parallel plane at median depth: covers the whole frustum
  const p = c.C.clone().addScaledVector(c.fwd, c.depth);
  placeProxy(p, c.fwd.clone().negate(), 6 * c.depth * Math.tan(Math.max(c.vfov, c.hfov) / 2) + 1);
}
function visSet(c) { return c.visSet || (c.visSet = new Set(c.vis)); }
function median(arr) { const s = Float32Array.from(arr).sort(); return s[Math.floor(s.length / 2)]; }
function pairPlane(a, b) {
  // Normal = mean viewing direction; position = median of points both photos see (Photo Tourism style)
  let n = a.fwd.clone().add(b.fwd);
  if (n.lengthSq() < 0.1) n = a.fwd.clone();
  n.normalize();
  const sb = visSet(b);
  const xs = [], ys = [], zs = [];
  for (const p of a.vis) if (sb.has(p)) { xs.push(pos[3 * p]); ys.push(pos[3 * p + 1]); zs.push(pos[3 * p + 2]); }
  let P;
  if (xs.length >= 8) P = new THREE.Vector3(median(xs), median(ys), median(zs));
  else P = a.C.clone().addScaledVector(a.fwd, a.depth).add(b.C.clone().addScaledVector(b.fwd, b.depth)).multiplyScalar(0.5);
  const reach = Math.max(P.distanceTo(a.C), P.distanceTo(b.C));
  placeProxy(P, n.negate(), 12 * reach + 2);
}

// ------------------------------------------------------------------ frusta for the overview
// One pyramid + photo quad per camera, built at unit depth and rescaled every overview frame to a constant
// on-screen size (FRUSTUM_PX), capped in world size by a tenth of the camera's viewing depth and by the spacing
// to its nearest cameras: from afar they stay findable, close up they get out of the way of the scene.
// F cycles: thumbnails -> wireframe -> hidden.
const FRUSTUM_PX = 28;
const frustumGroup = new THREE.Group();
frustumGroup.visible = false;
scene.add(frustumGroup);
const edgeMat = new THREE.LineBasicMaterial({ color: 0x7cc4b8, transparent: true, opacity: 0.6, depthWrite: false });
const edgeMatActive = new THREE.LineBasicMaterial({ color: 0xffffff, depthTest: false });
const quadMeshes = [];
const frusta = cams.map((c) => {
  const w = Math.tan(c.hfov / 2), h = Math.tan(c.vfov / 2);
  const apex = new THREE.Vector3();
  const k = [[-w, -h], [w, -h], [w, h], [-w, h]].map(([x, y]) => new THREE.Vector3(x, y, -1));
  const lines = [];
  for (let j = 0; j < 4; j++) lines.push(apex, k[j], k[j], k[(j + 1) % 4]);
  const edges = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(lines), edgeMat);
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2 * w, 2 * h),
    new THREE.MeshBasicMaterial({ color: 0x7cc4b8, side: THREE.DoubleSide, transparent: true, opacity: 0.08, depthWrite: false }));
  quad.position.z = -1;
  quad.userData.cam = c.i;
  quadMeshes.push(quad);
  const g = new THREE.Group();
  g.add(edges, quad);
  g.position.copy(c.C);
  g.quaternion.copy(c.q);
  g.userData = { edges, quad, cap: 1 };
  frustumGroup.add(g);
  return g;
});
for (let i = 0; i < cams.length; i++) { // world-size cap (O(n^2), fine for a few thousand cameras)
  const d = [];
  for (let j = 0; j < cams.length; j++) if (j !== i) d.push(cams[i].C.distanceTo(cams[j].C));
  d.sort((x, y) => x - y);
  const spacing = d.length ? d[Math.min(2, d.length - 1)] : Infinity; // 3rd nearest: tolerates clusters
  frusta[i].userData.cap = Math.max(0.25, Math.min(0.1 * cams[i].depth, 0.5 * spacing));
}
function sizeFrusta() {
  const k = 2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2) * FRUSTUM_PX / Math.max(1, canvas.clientHeight);
  for (const g of frusta) g.scale.setScalar(Math.min(g.userData.cap, camera.position.distanceTo(g.position) * k));
}
let frustumStyle = 0; // 0 thumbnails, 1 wireframe, 2 hidden
function applyFrustumStyle() {
  frustumGroup.visible = mode === 'overview' && frustumStyle < 2;
  for (const q of quadMeshes) q.visible = frustumStyle === 0;
  invalidate();
}
let activeFrustum = null;
function markActive(c) {
  invalidate();
  if (activeFrustum) {
    activeFrustum.userData.edges.material = edgeMat;
    activeFrustum.userData.quad.material.opacity = activeFrustum.userData.quad.material.map ? 0.7 : 0.08;
    activeFrustum.renderOrder = 0;
  }
  activeFrustum = c ? frusta[c.i] : null;
  if (activeFrustum) {
    activeFrustum.userData.edges.material = edgeMatActive;
    activeFrustum.userData.quad.material.opacity = 1;
    activeFrustum.renderOrder = 3;
  }
}
let quadsTextured = false;
function textureQuads() { // lazy: only when the overview is first opened; nearest to the current photo first
  if (quadsTextured) return;
  quadsTextured = true;
  const queue = cams.slice().sort((a, b) => (cur ? a.C.distanceTo(cur.C) - b.C.distanceTo(cur.C) : 0));
  let next = 0;
  const worker = async () => {
    while (next < queue.length) {
      const c = queue[next++];
      const t = await loadThumbTex(c);
      if (!t) continue;
      const m = quadMeshes[c.i].material;
      m.map = t; m.color.set(0xffffff); m.opacity = frusta[c.i] === activeFrustum ? 1 : 0.7; m.needsUpdate = true;
      invalidate();
    }
  };
  for (let k = 0; k < 6; k++) worker();
}

// ------------------------------------------------------------------ view fitting & state
function fitFov(c) { // vertical FOV that fits the whole photo inside the viewport (+3% margin)
  const aspect = canvas.clientWidth / Math.max(1, canvas.clientHeight);
  const photoAspect = c.im.width / c.im.height;
  const v = aspect >= photoAspect ? c.vfov : 2 * Math.atan(Math.tan(c.hfov / 2) / aspect);
  return THREE.MathUtils.radToDeg(v) * 1.03;
}
let mode = 'photo';
let cur = null;          // current camera record
let anim = null;         // running transition
const controls = new OrbitControls(camera, canvas);
controls.enabled = false;
controls.enableDamping = true;
controls.target.set(0, 0, 0);
controls.addEventListener('change', invalidate);

function setPhotoUniforms(slot, c, tex) {
  const u = photoMat.uniforms;
  u[`tex${slot}`].value = tex || blank;
  u[`view${slot}`].value.copy(c.view);
  u[`intr${slot}`].value.copy(c.intr);
  u[`k${slot}`].value = c.im.k1 || 0;
  u[`on${slot}`].value = tex ? 1 : 0;
}
function snapTo(c, tex) {
  camera.position.copy(c.C); camera.quaternion.copy(c.q);
  camera.fov = fitFov(c); camera.updateProjectionMatrix();
  setPhotoUniforms('A', c, tex); photoMat.uniforms.onB.value = 0; photoMat.uniforms.t.value = 0;
  restPlane(c);
  proxy.visible = true;
  invalidate();
  trimTextures(); // the previous photo is no longer on screen
}
const ease = (x) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);

async function goTo(j, { instant = false } = {}) {
  const b = cams[j];
  if (!b || (cur === b && mode === 'photo')) return;
  const fromOverview = mode !== 'photo';
  if (fromOverview) setMode('photo', { silent: true });
  toast(texCache.get(j)?.t ? '' : 'Loading photo…');
  const texB = await Promise.race([loadTex(b), new Promise((r) => setTimeout(() => r(null), 4000))]);
  toast('');
  for (const nb of b.im.neighbors.slice(0, 4)) loadTex(cams[nb[0]]); // prefetch
  const a = fromOverview ? null : cur;
  const from = { p: camera.position.clone(), q: camera.quaternion.clone(), fov: camera.fov };
  cur = b;
  updateUI();
  if (instant || reduceMotion) { snapTo(b, texB); anim = null; return; }
  if (a) {
    const texA = photoMat.uniforms.texA.value;
    setPhotoUniforms('A', a, photoMat.uniforms.onA.value ? texA : null);
    pairPlane(a, b);
  } else {
    photoMat.uniforms.onA.value = 0;
    restPlane(b);
  }
  setPhotoUniforms('B', b, texB);
  proxy.visible = true;
  const dist = from.p.distanceTo(b.C);
  const ang = from.q.angleTo(b.q);
  const dur = 700 + 900 * Math.min(1, dist / Math.max(2, b.depth)) + 500 * Math.min(1, ang / Math.PI * 2);
  anim = { a, b, from, to: { p: b.C, q: b.q, fov: fitFov(b) }, t0: performance.now(), dur, texB };
  invalidate();
}
function stepAnim(now) {
  if (!anim) return;
  const s = Math.min(1, (now - anim.t0) / anim.dur);
  const e = ease(s);
  camera.position.lerpVectors(anim.from.p, anim.to.p, e);
  camera.quaternion.slerpQuaternions(anim.from.q, anim.to.q, e);
  camera.fov = THREE.MathUtils.lerp(anim.from.fov, anim.to.fov, e);
  camera.updateProjectionMatrix();
  photoMat.uniforms.t.value = anim.a ? THREE.MathUtils.smoothstep(e, 0.15, 0.85) : THREE.MathUtils.smoothstep(e, 0.55, 1);
  pmat.opacity = pointsOn ? 0.35 + 0.35 * Math.sin(Math.PI * s) : 0;
  if (s >= 1) { const { b, texB } = anim; anim = null; snapTo(b, texB); pmat.opacity = pointsOn ? 0.35 : 0; }
}

// ------------------------------------------------------------------ navigation logic
const _inv = new THREE.Quaternion();
const pairDepthCache = new Map();
function pairDepth(a, b) { // median depth (along a's axis) of the points both photos see
  const key = a.i * 100000 + b.i;
  if (pairDepthCache.has(key)) return pairDepthCache.get(key);
  const sb = visSet(b), ds = [];
  for (const p of a.vis) if (sb.has(p)) ds.push((pos[3 * p] - a.C.x) * a.fwd.x + (pos[3 * p + 1] - a.C.y) * a.fwd.y + (pos[3 * p + 2] - a.C.z) * a.fwd.z);
  const d = ds.length >= 5 ? Math.max(0.5, median(ds)) : a.depth;
  pairDepthCache.set(key, d);
  return d;
}
function relative(a, b) {
  _inv.copy(a.q).invert();
  // b's position in a's camera frame, in units of the depth of what they both see
  const d = b.C.clone().sub(a.C).applyQuaternion(_inv).divideScalar(pairDepth(a, b));
  const f = b.fwd.clone().applyQuaternion(_inv);
  const yaw = Math.atan2(f.x, -f.z);
  // Moved (orbit/walk): direction of travel decides. Standing still (pan/zoom): rotation decides.
  const moved = Math.hypot(d.x, d.z) > 0.05;
  const lateral = moved ? d.x : yaw;
  const forward = moved ? -d.z : Math.log(b.im.f / b.im.width / (a.im.f / a.im.width)); // zoom-in counts as forward
  return { lateral, forward };
}
function candidates(a) {
  const byCovis = new Map(a.im.neighbors.map(([j, n]) => [j, n]));
  const pool = byCovis.size ? [...byCovis.keys()] : cams.filter((c) => c !== a && c.C.distanceTo(a.C) < 3 * a.depth).map((c) => c.i);
  const maxN = Math.max(1, ...byCovis.values());
  return pool.map((j) => ({ j, covis: (byCovis.get(j) || 0) / maxN, ...relative(a, cams[j]) }));
}
function pick(a, dir) {
  if (!a) return null;
  let best = null, bestCost = Infinity;
  for (const c of candidates(a)) {
    const main = dir === 'right' ? c.lateral : dir === 'left' ? -c.lateral : dir === 'fwd' ? c.forward : -c.forward;
    const other = dir === 'right' || dir === 'left' ? c.forward : c.lateral;
    if (main < 0.06 || Math.abs(other) > main * 1.2) continue;
    const cost = Math.abs(main - 0.3) + 0.5 * Math.abs(other) + 0.4 * (1 - c.covis);
    if (cost < bestCost) { bestCost = cost; best = c.j; }
  }
  return best;
}
const dirs = { right: 'nav-right', left: 'nav-left', fwd: 'nav-fwd', back: 'nav-back' };
function updateArrows() {
  for (const [d, id] of Object.entries(dirs)) {
    const j = mode === 'photo' ? pick(cur, d) : null;
    $(id).disabled = j === null;
    $(id).dataset.target = j ?? '';
  }
}
for (const [d, id] of Object.entries(dirs)) $(id).addEventListener('click', () => { const j = pick(cur, d); if (j !== null) goTo(j); });

function bestViewOf(X, exclude) { // photo that shows world point X nearest its centre, preferring closer views
  let best = null, bestCost = Infinity;
  const p = new THREE.Vector3();
  const dCur = exclude ? exclude.C.distanceTo(X) : 1;
  for (const c of cams) {
    if (c === exclude) continue;
    p.copy(X).applyMatrix4(c.view);
    const z = -p.z;
    if (z <= 0.1) continue;
    const u = (c.im.cx + c.im.f * p.x / z) / c.im.width, v = (c.im.cy - c.im.f * p.y / z) / c.im.height;
    if (u < 0.05 || u > 0.95 || v < 0.05 || v > 0.95) continue;
    const off = Math.hypot(u - 0.5, v - 0.5);
    const cost = off + 0.35 * Math.log(z / dCur) + (visSet(c).size ? 0 : 1);
    if (cost < bestCost) { bestCost = cost; best = c.i; }
  }
  return best;
}

// ------------------------------------------------------------------ pointer: tap to jump, swipe to move, click frusta in overview
const ray = new THREE.Raycaster();
let down = null;
canvas.addEventListener('pointerdown', (e) => { down = { x: e.clientX, y: e.clientY, t: performance.now() }; });
canvas.addEventListener('pointerup', (e) => {
  if (!down) return;
  const dx = e.clientX - down.x, dy = e.clientY - down.y;
  const quick = performance.now() - down.t < 600;
  down = null;
  if (mode === 'photo' && quick && Math.abs(dx) > 50 && Math.abs(dx) > 1.5 * Math.abs(dy)) {
    const j = pick(cur, dx < 0 ? 'right' : 'left'); if (j !== null) goTo(j); return;
  }
  if (Math.hypot(dx, dy) > 6) return;
  const r = canvas.getBoundingClientRect();
  const ndc = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  if (mode === 'overview') {
    if (frustumStyle === 2) return;
    const hit = ray.intersectObjects(quadMeshes, false)[0];
    if (hit) goTo(hit.object.userData.cam);
    return;
  }
  if (anim || !cur) return;
  const hit = ray.intersectObject(proxy, false)[0];
  if (!hit) return;
  const j = bestViewOf(hit.point, cur);
  if (j !== null) goTo(j);
});

// ------------------------------------------------------------------ modes, keyboard, UI
let pointsOn = true;
function setMode(m, { silent = false } = {}) {
  mode = m;
  $('mode-photo').setAttribute('aria-pressed', m === 'photo');
  $('mode-overview').setAttribute('aria-pressed', m === 'overview');
  applyFrustumStyle();
  controls.enabled = m === 'overview';
  canvas.classList.toggle('orbit', m === 'overview');
  pmat.size = m === 'overview' ? 2.5 : 2;
  pmat.opacity = pointsOn ? (m === 'overview' ? 0.9 : 0.35) : 0;
  invalidate();
  if (m === 'overview') {
    anim = null;
    proxy.visible = false;
    textureQuads();
    markActive(cur);
    const back = cur ? cur.fwd.clone().multiplyScalar(-1) : new THREE.Vector3(0, -1, 0);
    back.z = 0; back.normalize();
    camera.position.set(0, 0, 0).addScaledVector(back, sceneRadius * 1.6).add(new THREE.Vector3(0, 0, sceneRadius * 1.1));
    camera.fov = 50; camera.updateProjectionMatrix();
    controls.target.set(0, 0, 0);
    camera.lookAt(controls.target);
    controls.update();
  } else if (!silent && cur) {
    const c = cur; cur = null; goTo(c.i);
  }
  updateArrows();
}
$('mode-photo').addEventListener('click', () => setMode('photo'));
$('mode-overview').addEventListener('click', () => setMode('overview'));
$('toggle-points').addEventListener('click', () => {
  pointsOn = !pointsOn;
  $('toggle-points').setAttribute('aria-pressed', pointsOn);
  pmat.opacity = pointsOn ? (mode === 'overview' ? 0.9 : 0.35) : 0;
  invalidate();
});
{ // About dialog: scene-specific line
  const nAnchor = S.images.filter((im) => im.role === 'anchor').length;
  const g = S.georef || {};
  $('about-scene').textContent = `${S.title}: ${S.images.length} of ${S.num_images_total ?? S.images.length} photos placed.` +
    (nAnchor ? ` Positions are anchored to ${nAnchor} accurately positioned (RTK) survey photos` +
      (g.residual_m ? `, median fit ${g.residual_m.median.toFixed(1)} m.` : '.') :
      S.aligned ? ' Positions come from the photos\' own geotags.' : '');
  $('about-open').addEventListener('click', () => $('about').showModal());
}
$('info-toggle').addEventListener('click', () => {
  const box = $('info'); const collapsed = box.classList.toggle('collapsed');
  $('info-toggle').setAttribute('aria-expanded', !collapsed);
});
addEventListener('keydown', (e) => {
  if (e.target.closest('input, textarea')) return;
  const map = { ArrowRight: 'right', ArrowLeft: 'left', ArrowUp: 'fwd', ArrowDown: 'back' };
  if (map[e.key] && mode === 'photo') { e.preventDefault(); const j = pick(cur, map[e.key]); if (j !== null) goTo(j); }
  if (e.key === 'o' || e.key === 'O') setMode(mode === 'photo' ? 'overview' : 'photo');
  if (e.key === 'p' || e.key === 'P') $('toggle-points').click();
  if ((e.key === 'f' || e.key === 'F') && mode === 'overview') { frustumStyle = (frustumStyle + 1) % 3; applyFrustumStyle(); }
});

// thumbnail strip, ordered by azimuth of the camera around the scene centre
const order = cams.slice().sort((a, b) => Math.atan2(a.C.y, a.C.x) - Math.atan2(b.C.y, b.C.x));
const strip = $('strip');
const thumbBtns = new Map();
for (const c of order) {
  const btn = document.createElement('button');
  btn.setAttribute('role', 'option');
  btn.title = c.im.title.replace(/^File:/, '');
  btn.setAttribute('aria-label', btn.title);
  btn.addEventListener('click', () => goTo(c.i));
  strip.appendChild(btn);
  thumbBtns.set(c.i, btn);
}
const thumbObserver = new IntersectionObserver((entries) => {
  for (const en of entries) if (en.isIntersecting) {
    const i = [...thumbBtns].find(([, b]) => b === en.target)[0];
    en.target.style.backgroundImage = `url("${smallUrl(cams[i])}")`;
    thumbObserver.unobserve(en.target);
  }
}, { root: strip, rootMargin: '0px 400px' });
thumbBtns.forEach((b) => thumbObserver.observe(b));

const fmt = (x, d) => (x == null ? '—' : x.toFixed(d));
function updateUI() {
  const c = cur;
  $('info').hidden = !c;
  if (!c) return;
  const im = c.im;
  const a = $('info-title');
  a.textContent = im.title.replace(/^File:/, '');
  if (im.file_page) a.href = im.file_page; else a.removeAttribute('href');
  const credit = $('info-credit');
  credit.textContent = `${im.author || 'Unknown author'}${im.date ? ` · ${im.date}` : ''} · `;
  const lic = document.createElement(im.license_url && /^https?:\/\//.test(im.license_url) ? 'a' : 'span');
  lic.textContent = im.license || 'licence unknown';
  if (lic.tagName === 'A') { lic.href = im.license_url; lic.target = '_blank'; lic.rel = 'noopener'; }
  credit.appendChild(lic);
  $('info-index').textContent = `${order.indexOf(c) + 1} / ${cams.length}`;
  $('info-latlon').textContent = im.sfm_lat == null ? '—' : `${fmt(im.sfm_lat, 6)}, ${fmt(im.sfm_lon, 6)}`;
  $('info-heading').textContent = `${fmt(im.sfm_heading_deg, 0)}°` + (im.commons_heading_deg != null ? ` (Commons ${fmt(im.commons_heading_deg, 0)}°)` : '');
  const extra = im.role === 'extra';
  const anchorsExist = cams.some((c) => c.im.role === 'anchor');
  $('info-role').textContent = extra ? 'matched to anchors' : im.role === 'anchor' ? 'anchor (geotag in BA)'
    : im.role === 'grouped' ? 'separate ground model, placed by geotags'
    : anchorsExist ? 'SfM with the anchors (geotag not used)' : 'SfM, georeferenced by geotags';
  $('info-resid-label').textContent = extra || (im.role === 'photo' && anchorsExist) ? 'geotag offset' : 'GPS residual';
  const r = im.gps_residual_m;
  const cls = r == null ? 'none' : r < 3 ? 'ok' : r < 15 ? 'warn' : 'bad';
  $('info-resid').innerHTML = `<span class="chip ${cls}">${r == null ? 'no geotag' : `${r.toFixed(1)} m`}</span>`;
  $('info-points').textContent = im.num_points.toLocaleString();
  thumbBtns.forEach((btn, i) => {
    btn.setAttribute('aria-current', i === c.i);
    btn.classList.toggle('near', im.neighbors.slice(0, 6).some(([j]) => j === i));
  });
  thumbBtns.get(c.i).scrollIntoView({ block: 'nearest', inline: 'center', behavior: reduceMotion ? 'auto' : 'smooth' });
  if (mode === 'overview') markActive(c);
  try { history.replaceState(null, '', `#img-${c.i}`); } catch { /* sandboxed */ }
  updateArrows();
}

// ------------------------------------------------------------------ resize & loop
function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / Math.max(1, h);
  if (mode === 'photo' && cur && !anim) camera.fov = fitFov(cur);
  camera.updateProjectionMatrix();
  invalidate();
}
addEventListener('resize', resize);
resize();
renderer.setAnimationLoop((now) => {
  // Render only when something changed: a transition, orbiting (incl. damping), or an invalidate() call.
  const moving = anim !== null;
  stepAnim(now);
  const orbiting = mode === 'overview' && controls.update();
  if (!(moving || orbiting || needsRender)) return;
  needsRender = false;
  if (mode === 'overview') sizeFrusta();
  renderer.render(scene, camera);
});

if (matchMedia('(max-width: 640px)').matches) { $('info').classList.add('collapsed'); $('info-toggle').setAttribute('aria-expanded', 'false'); }
const fromHash =/^#img-(\d+)$/.exec(location.hash);
const start = fromHash && cams[+fromHash[1]] ? +fromHash[1] : cams.reduce((b, c) => (c.vis.length > cams[b].vis.length ? c.i : b), 0);
await goTo(start, { instant: true });
window.mirante = { goTo, setMode, cams, pick, renderer, camera, controls, texCache, get cur() { return cur; } }; // for debugging / tests
