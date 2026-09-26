# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Adam Sturrock and Shards of Stone Contributors
"""Fit a many-legged skeleton (arachnid, crustacean, scorpion) to the mesh.

python3 scripts/critter/fit_arthropod.py <critter> [--write]

Reads <work>/<critter>/prep/{verts,faces}.npy (prep_critter.py; rig space:
head toward -Y, left = +X, feet at z = 0) and the profile's `fit` hints, and
derives every joint from the mesh itself:

  feet        the lowest-vertex clusters (fit_landmarks.foot_contacts); the
              2 x legs_per_side largest, split by side (x sign) and ordered
              front to back. `fit.feet` overrides them.
  appendages  each leg, claw, eye stalk, fang and tail is TRACED from its tip
              by geodesic level sets: Dijkstra over the welded surface from
              the tip vertex, then the vertices at geodesic distance s..s+h
              form a ring around the limb, and the ring centroid is the
              limb's centreline at arc length s. The trace stops where the
              ring radius blows up (the ring has spilled onto the body): that
              is the limb root. A limb is its tip plus a centreline, so a
              curved or arched leg is followed wherever it goes.
  legs        4 bones per leg (coxa, femur, tibia, tarsus): the knee is the
              centreline point farthest from the root-tip chord (the arch),
              the coxa ends a quarter of the way from root to knee, the
              tibia/tarsus joint halfway from knee to tip.
  body        the vertices no limb claimed. Its centroid is `body_0`, its
              front `body_1`; an arachnid's abdomen is split off at the
              narrowest cross-section (the pedicel), a scorpion's mesosoma
              runs from behind the last leg root to the tail root.
Hints (profile `fit`), all rig-space points read off the fit_landmarks grid:
  legs_per_side   expected count (fails loudly on a mismatch)
  claw            {"tip": fixed-finger tip, "dactyl": moving-finger tip,
                   "fracs": [gape, wrist, elbow] arc fractions from the tip}
  eye / fang / palp  {"tip": [...]}
  tail            {"tip": stinger tip, "segments": 6}
  spinneret       [x, y, z]  (arachnid; default: abdomen rear)
  pedicel_y       overrides the automatic abdomen split
  leg_roots       {"leg4.L": [x, y, z]}: where a leg FUSED to the body along
                  its length (Meshy often glues a spider's rear legs to the
                  abdomen) really joins it; the trace follows the leg inside
                  a ball around its own axis up to that point
Only '.L' hints are needed; '.R' hints are their mirror (x -> -x) unless
given. Each hint is snapped to the nearest mesh vertex before tracing.

Writes (with --write) the joints into profiles.json (`joint_overrides` win)
and <work>/<critter>/prep/labels.npy: per vertex, the appendage that owns it
(index into labels.json's names; -1 = body). rig_critter.py uses the labels
to keep one leg's skin off its neighbour's bones.
"""
import json, os, sys
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fit_landmarks import foot_contacts  # noqa: E402
from paths import work_dir, profiles_path  # noqa: E402

PROFILES = profiles_path()


class Surface:
    """The prep mesh welded at coincident positions (the glTF import splits
    vertices along UV seams, which a geodesic walk would read as cuts)."""

    def __init__(self, verts, faces, sdf=None):
        self.v = verts.astype(np.float64)
        q = np.round(self.v / 1e-4).astype(np.int64)
        _, inv = np.unique(q, axis=0, return_inverse=True)
        self.inv = inv.ravel()
        n = int(self.inv.max()) + 1
        self.n = n
        # welded positions: mean of the members
        self.p = np.zeros((n, 3))
        np.add.at(self.p, self.inv, self.v)
        self.p /= np.bincount(self.inv, minlength=n)[:, None]
        e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        e = self.inv[e]
        e = e[e[:, 0] != e[:, 1]]
        w = np.linalg.norm(self.p[e[:, 0]] - self.p[e[:, 1]], axis=1) + 1e-9
        self.g = coo_matrix((w, (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
        self.span = float(np.ptp(self.v, axis=0).max())
        self.sdf = None
        if sdf is not None:
            self.sdf = np.zeros(n)
            np.add.at(self.sdf, self.inv, sdf)
            self.sdf /= np.bincount(self.inv, minlength=n)

    def core_dist(self, thresh):
        """Geodesic distance from every node to the body core: the nodes
        whose shape diameter exceeds `thresh`. Cached per threshold."""
        self._core = getattr(self, '_core', {})
        key = round(float(thresh), 5)
        if key not in self._core:
            core = np.flatnonzero(self.sdf > thresh)
            self._core[key] = dijkstra(self.g, directed=False, indices=core, min_only=True) if len(core) \
                else np.full(self.n, np.inf)
        return self._core[key]

    def nearest(self, pt):
        return int(np.argmin(np.linalg.norm(self.p - np.asarray(pt, float), axis=1)))

    def geodesic(self, node):
        return dijkstra(self.g, directed=False, indices=node)


def trace(S, tip, h, sdf_stop=None, grow=2.1, min_rings=3, root=None, ball=None, core_frac=0.25, core_reach=1.5):
    """Centreline of the limb whose tip is welded node `tip`.

    Geodesic level sets from the tip give rings around the limb. With `ball`
    (set for a limb given a `root` hint), each ring is kept to a ball around
    the previous ring's centre (`ball` x the limb's recent radius), so where
    a limb lies fused along the body (a spider's rear leg against its
    abdomen) the ring follows the limb instead of spreading over the body.
    The trace stops where the ring reaches the body core: over `core_frac`
    of it within `core_reach` ring steps of the body-thick surface (shape
    diameter > sdf_stop). Without a shape-diameter
    field: a jump in ring radius (`grow` x the limb's median). `root`, a
    rig-space point, overrides the stop: the trace ends at the ring whose
    centre is nearest to it (use it for a limb fused to the body).
    Returns (points tip->root [(s, centroid, radius)], s_root, d) where d is
    the geodesic distance from the tip, +inf outside the kept rings, so
    `d < s_root` is exactly the limb's vertices."""
    d = S.geodesic(tip)
    fin = np.isfinite(d)
    k = np.full(S.n, -1)
    k[fin] = (d[fin] / h).astype(int)
    kmax = int(k.max())
    order = np.argsort(k)
    ks = k[order]
    starts = np.searchsorted(ks, np.arange(kmax + 2))
    pts, radii = [], []
    kept = np.zeros(S.n, bool)
    prev = S.p[tip]
    dc = S.core_dist(sdf_stop) if (S.sdf is not None and sdf_stop) else None
    for i in range(kmax + 1):
        members = order[starts[i]:starts[i + 1]]
        if len(members) < 3:
            continue
        if ball:
            # the ball follows the limb's own girth, capped at twice its
            # girth near the tip so a ring spilling onto the body cannot
            # inflate the ball and run away over it
            rad = float(np.median(radii[-4:])) if radii else 0.0
            if len(radii) >= 6:
                rad = min(rad, 2.0 * float(np.median(radii[:6])))
            rad = max(rad, h)
            members = members[np.linalg.norm(S.p[members] - prev, axis=1) < ball * rad + h]
            if len(members) < 3:
                break
        ring = S.p[members]
        c = ring.mean(0)
        r = float(np.median(np.linalg.norm(ring - c, axis=1)))
        if root is None and len(radii) >= min_rings:
            if dc is not None:
                if float(np.mean(dc[members] < core_reach * h)) > core_frac:
                    break
            elif r > grow * float(np.median(radii[1:])):
                break
        kept[members] = True
        radii.append(r)
        pts.append(((i + 0.5) * h, c, r))
        prev = c
    if root is not None and pts:
        j = int(np.argmin([np.linalg.norm(c - np.asarray(root, float)) for _, c, _ in pts]))
        pts = pts[:j + 1]
    s_root = pts[-1][0] + 0.5 * h if pts else 0.0
    kept[tip] = True
    d = np.where(kept, d, np.inf)
    if pts:  # the tip itself, not the first ring's centroid
        pts[0] = (0.0, S.p[tip], pts[0][2])
    return pts, s_root, d


def smooth_line(pts, n=1):
    P = np.array([p for _, p, _ in pts])
    for _ in range(n):
        Q = P.copy()
        Q[1:-1] = (P[:-2] + P[1:-1] + P[2:]) / 3
        P = Q
    return P


def arc(P):
    L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    return L


def at_arc(P, L, s):
    s = float(np.clip(s, 0, L[-1]))
    i = int(np.searchsorted(L, s))
    if i <= 0:
        return P[0]
    if i >= len(L):
        return P[-1]
    t = (s - L[i - 1]) / max(L[i] - L[i - 1], 1e-9)
    return P[i - 1] + t * (P[i] - P[i - 1])


def corner(P, L, lo=0.2, hi=0.85):
    """Arc position of the centreline point farthest from the root-tip chord."""
    a, b = P[0], P[-1]
    ab = b - a
    ab /= max(np.linalg.norm(ab), 1e-9)
    dist = np.linalg.norm(np.cross(P - a, ab), axis=1)
    ok = (L >= lo * L[-1]) & (L <= hi * L[-1])
    if not ok.any():
        return 0.5 * L[-1]
    i = int(np.argmax(np.where(ok, dist, -1)))
    return float(L[i])


def push_in(p, toward, amount):
    d = np.asarray(toward) - p
    n = np.linalg.norm(d)
    return p + d / n * min(amount, 0.5 * n) if n > 1e-9 else p


def mirror(pt):
    return [-pt[0], pt[1], pt[2]]


def hint_for(fit, key, side):
    """fit[key] for side 'L', or its '.R' variant / mirror for 'R'."""
    if key not in fit:
        return None
    h = dict(fit[key])
    if side == 'R':
        if f'{key}.R' in fit:
            return dict(fit[f'{key}.R'])
        for k in ('tip', 'dactyl', 'root'):
            if k in h:
                h[k] = mirror(h[k])
    return h


def extremities(S, wlabel, ring=None):
    """Welded nodes at the tip of an unclaimed protrusion: local maxima of the
    geodesic distance from the body's middle over a geodesic neighbourhood
    (grown by repeated 1-ring max filters), not already inside a limb label,
    strongest first."""
    body = np.flatnonzero(wlabel == -1)
    c = S.p[body].mean(0) if len(body) else S.p.mean(0)
    src = int(np.argmin(np.linalg.norm(S.p - c, axis=1)))
    d = S.geodesic(src)
    d[~np.isfinite(d)] = -1
    g = S.g.tocoo()
    m = d.copy()
    if ring is None:
        # a neighbourhood of ~4% of the creature, in rings of mean edge length
        el = float(np.mean(S.g.data))
        ring = int(np.clip(np.ceil(0.04 * S.span / max(el, 1e-9)), 2, 40))
    for _ in range(ring):
        nm = m.copy()
        np.maximum.at(nm, g.row, m[g.col])
        m = nm
    tips = np.flatnonzero((d >= m - 1e-9) & (wlabel == -1) & (d > 0))
    return sorted(tips.tolist(), key=lambda n: -d[n])


def equalise_pairs(tr, min_ratio=0.6):
    """Mirror limbs are one design: when a pair's traces reach different
    depths into the body (one ring touched the core a step early), cut the
    longer back to the shorter's root arc, so both limbs own the same
    stretch. A pair whose short side is under `min_ratio` of the long one is
    left alone: that is a broken trace, and the label checks will say so."""
    done = {}
    for k in list(tr):
        if not k.endswith('.L') or k[:-2] + '.R' not in tr:
            continue
        a, b = tr[k], tr[k[:-2] + '.R']
        sa, sb = a[1], b[1]
        lo, hi = min(sa, sb), max(sa, sb)
        if hi <= 0 or lo / hi < min_ratio or lo / hi > 0.97:
            continue
        for key, t in ((k, a), (k[:-2] + '.R', b)):
            if t[1] > lo:
                pts = [p for p in t[0] if p[0] <= lo] or t[0][:1]
                tr[key] = (pts, lo) + tuple(t[2:])
        done[k[:-2]] = [round(sa, 3), round(sb, 3)]
    return done


def _pair_ok(x, y):
    """A claw and its own moving finger may touch (the dactyl is carved out
    of the claw label by design); every other pair of limbs meets only
    through the body."""
    return x.split('.')[0] in ('claw', 'dactyl') and y.split('.')[0] in ('claw', 'dactyl') \
        and x.split('.')[-1] == y.split('.')[-1]


def clean_labels(S, wlabel, names, rounds=2):
    """Where two limbs' claims touch (neighbouring leg bases, a fang against a
    palp) return the contact band to the body, so limbs meet only through
    it; then give tiny body islands enclosed by ONE limb back to that limb.
    An island touching two limbs stays body and fails check (e): that is a
    genuine fusion in the mesh and needs a hint, not a silent merge."""
    from scipy.sparse.csgraph import connected_components
    g = S.g.tocoo()
    ea, eb = g.row, g.col
    eroded = 0
    for _ in range(rounds):
        la, lb = wlabel[ea], wlabel[eb]
        bad = (la >= 0) & (lb >= 0) & (la != lb)
        if bad.any():
            ok = np.array([_pair_ok(names[x], names[y]) for x, y in zip(la[bad], lb[bad])], bool)
            idx = np.flatnonzero(bad)[~ok]
            nodes = np.unique(np.concatenate([ea[idx], eb[idx]]))
            eroded += len(nodes)
            wlabel[nodes] = -1
    # each limb keeps only its largest piece; stray fragments (a node or
    # two a ring picked up across a crease) go back to the body
    frag = 0
    for i in range(len(names)):
        idx = np.flatnonzero(wlabel == i)
        if len(idx) < 2:
            continue
        pos = -np.ones(S.n, int)
        pos[idx] = np.arange(len(idx))
        keep = (wlabel[ea] == i) & (wlabel[eb] == i)
        sub = coo_matrix((np.ones(keep.sum()), (pos[ea[keep]], pos[eb[keep]])), shape=(len(idx), len(idx)))
        n, lab = connected_components(sub, directed=False)
        if n > 1:
            big = int(np.argmax(np.bincount(lab)))
            wlabel[idx[lab != big]] = -1
            frag += int((lab != big).sum())
    body = wlabel == -1
    pos = -np.ones(S.n, int)
    bidx = np.flatnonzero(body)
    pos[bidx] = np.arange(len(bidx))
    keep = body[ea] & body[eb]
    sub = coo_matrix((np.ones(keep.sum()), (pos[ea[keep]], pos[eb[keep]])), shape=(len(bidx), len(bidx)))
    n, lab = connected_components(sub, directed=False)
    sizes = np.bincount(lab)
    main_c = int(np.argmax(sizes))
    absorbed = 0
    for c in range(n):
        if c == main_c or sizes[c] > 0.1 * sizes[main_c]:   # (a transplanted limb brings its own root rings)
            continue
        nodes = bidx[lab == c]
        inn = np.isin(ea, nodes) & ~np.isin(eb, nodes)
        neigh = set(wlabel[eb[inn]].tolist()) - {-1}
        if len(neigh) == 1:
            wlabel[nodes] = neigh.pop()
            absorbed += len(nodes)
    # A body island between two limbs (their bases fused, the contact band
    # returned to the body above) is joined to the main body by the shortest
    # corridor through those limbs' nodes, so every limb still meets the
    # others only through the body and the body stays one piece.
    corridors = 0
    body = wlabel == -1
    bidx = np.flatnonzero(body)
    pos = -np.ones(S.n, int)
    pos[bidx] = np.arange(len(bidx))
    keep = body[ea] & body[eb]
    sub = coo_matrix((np.ones(keep.sum()), (pos[ea[keep]], pos[eb[keep]])), shape=(len(bidx), len(bidx)))
    n, lab = connected_components(sub, directed=False)
    if n > 1:
        main_c = int(np.argmax(np.bincount(lab)))
        main_nodes = bidx[lab == main_c]
        for c in range(n):
            if c == main_c:
                continue
            nodes = bidx[lab == c]
            inn = np.isin(ea, nodes) & ~np.isin(eb, nodes)
            neigh = set(wlabel[eb[inn]].tolist()) - {-1}
            allowed = body | np.isin(wlabel, list(neigh))
            m = allowed[ea] & allowed[eb]
            G = coo_matrix((S.g.tocoo().data[m], (ea[m], eb[m])), shape=(S.n, S.n)).tocsr()
            dist, pred, _src = dijkstra(G, directed=False, indices=main_nodes, min_only=True, return_predecessors=True)
            end = nodes[int(np.argmin(dist[nodes]))]
            if not np.isfinite(dist[end]):
                continue
            k = int(pred[end])
            while k >= 0 and wlabel[k] != -1:
                wlabel[k] = -1
                corridors += 1
                k = int(pred[k])
    return {'contact_nodes_to_body': int(eroded), 'fragment_nodes_to_body': frag,
            'body_island_nodes_absorbed': int(absorbed), 'corridor_nodes': int(corridors)}


def extend_to_core(S, wlabel, names, core_thresh, max_extend):
    """Carry every limb label all the way to the body wall. The CORE is the
    body-thick surface (shape diameter > core_thresh: a carapace, a thorax,
    an abdomen). Every other body node reachable from a limb WITHOUT
    crossing the core, within `max_extend`, joins the geodesically nearest
    limb: a crab claw's merus and carpus, a leg's coxa and femur. Where the
    trace stopped early (a stout segment read as body) this is what gets
    the rest of the limb. Returns the number of nodes moved per limb."""
    if S.sdf is None:
        return {}
    core = S.sdf > core_thresh
    g = S.g.tocoo()
    ok = ~core[g.row] & ~core[g.col]
    G = coo_matrix((g.data[ok], (g.row[ok], g.col[ok])), shape=(S.n, S.n)).tocsr()
    src_nodes = np.flatnonzero(wlabel >= 0)
    if not len(src_nodes):
        return {}
    dl, _, src = dijkstra(G, directed=False, indices=src_nodes, min_only=True, return_predecessors=True)
    take = (wlabel == -1) & ~core & np.isfinite(dl) & (dl < max_extend) & (src >= 0)
    moved = {}
    lab_src = wlabel[np.where(src >= 0, src, 0)]
    for i, n in enumerate(names):
        m = take & (lab_src == i)
        if n.split('.')[0] in ('fang', 'palp', 'eye'):
            # small head appendages sit ON the head: only a short reach, or
            # they spread over the face
            m &= dl < 0.3 * max_extend
        wlabel[m] = i
        if m.any():
            moved[n] = int(m.sum())
    return moved


def check_labels(S, wlabel, names, tips, mirror_tol=0.25):
    """The limb segmentation's own acceptance checks, on the welded surface:
      (a) every appendage label is ONE connected component
      (b) mirror pairs (X.L / X.R) have vertex counts within `mirror_tol`
      (c) no two appendage labels share an edge: limbs meet only through the
          body (a claw and its own moving finger excepted, by design)
      (d) every label contains its traced tip
      (e) the body is one connected component
    Returns {'pass', 'failures', per-check detail}."""
    from scipy.sparse.csgraph import connected_components
    g = S.g.tocoo()
    ea, eb = g.row, g.col
    fails, detail = [], {}

    def components(mask):
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return 0, []
        pos = -np.ones(S.n, int)
        pos[idx] = np.arange(len(idx))
        keep = mask[ea] & mask[eb]
        sub = coo_matrix((np.ones(keep.sum()), (pos[ea[keep]], pos[eb[keep]])), shape=(len(idx), len(idx)))
        n, lab = connected_components(sub, directed=False)
        return n, sorted(np.bincount(lab).tolist(), reverse=True)

    comp = {}
    for i, n in enumerate(names):
        k, sizes = components(wlabel == i)
        comp[n] = sizes[:4]
        if k != 1:
            fails.append(f'(a) {n} is {k} pieces {sizes[:4]}')
    detail['a_components'] = comp
    counts = {n: int((wlabel == i).sum()) for i, n in enumerate(names)}
    mir = {}
    for n in names:
        if n.endswith('.L') and n[:-2] + '.R' in counts:
            a, b = counts[n], counts[n[:-2] + '.R']
            r = abs(a - b) / max(a, b, 1)
            mir[n[:-2]] = [a, b, round(r, 3)]
            if r > mirror_tol:
                fails.append(f'(b) {n[:-2]} L/R counts {a}/{b} differ by {r:.0%}')
    detail['b_mirror'] = mir
    la, lb = wlabel[ea], wlabel[eb]
    cross = (la >= 0) & (lb >= 0) & (la != lb)
    pairs = {}
    for x, y in zip(la[cross], lb[cross]):
        p = tuple(sorted((names[x], names[y])))
        pairs[p] = pairs.get(p, 0) + 1
    shared = {}
    for (x, y), c in pairs.items():
        if not _pair_ok(x, y):
            shared[f'{x}|{y}'] = c // 2
            fails.append(f'(c) {x} and {y} share {c // 2} edges')
    detail['c_shared_edges'] = shared
    miss = [n for i, n in enumerate(names) if n in tips and wlabel[tips[n]] != i]
    for n in miss:
        fails.append(f'(d) {n} does not contain its traced tip')
    detail['d_tip_missing'] = miss
    kb, sizes = components(wlabel == -1)
    detail['e_body_components'] = sizes[:4]
    if kb != 1:
        fails.append(f'(e) body is {kb} pieces {sizes[:4]}')
    detail['pass'] = not fails
    detail['failures'] = fails
    return detail


def main():
    name = sys.argv[1]
    profs = json.load(open(PROFILES))
    prof = profs[name]
    plan = prof['plan']
    fit = json.loads(json.dumps(prof.get('fit', {})))   # a copy: auto hints are not saved
    prep = work_dir(name, 'prep')
    verts = np.load(os.path.join(prep, 'verts.npy'))
    faces = np.load(os.path.join(prep, 'faces.npy'))
    sdf_path = os.path.join(prep, 'sdf.npy')
    S = Surface(verts, faces, np.load(sdf_path) if os.path.exists(sdf_path) else None)
    # the thickest part is the body (legs outnumber it in vertices, so a
    # median or 95th percentile would read a leg): its 99th percentile
    body_sdf = float(np.percentile(S.sdf, 99)) if S.sdf is not None else None
    h = fit.get('ring_step', 0.012) * S.span
    J = {}
    names = []                       # appendage names, label index order
    wlabel = np.full(S.n, -1)        # welded-node label
    claimed = np.full(S.n, np.inf)   # geodesic distance of the claiming trace

    tips = {}

    def claim(label_name, d, s_root):
        idx = len(names)
        names.append(label_name)
        tips[label_name] = int(np.argmin(d))
        m = (d < s_root) & (d < claimed)
        wlabel[m] = idx
        claimed[m] = d[m]

    report = {'critter': name, 'plan': plan, 'ring_step': round(h, 4), 'limbs': {}}

    # ── legs ────────────────────────────────────────────────────────────
    # ── legs ────────────────────────────────────────────────────────────
    # Feet, in order of trust:
    #   1. ground contacts (lowest-vertex clusters) away from the midline,
    #      specks dropped: a resting belly or abdomen is never a foot
    #   2. low free protrusion tips (geodesic extremities under a quarter of
    #      the height): a foot that stops a hair above the ground
    #   3. the other side's mirror image: a foot fused to the abdomen or to a
    #      neighbour has neither. Every filled foot is named in the report.
    # Each foot is then snapped to a free tip within 5% of the creature, so
    # the trace starts at the true end of the leg.
    lps = fit.get('legs_per_side', 4)
    # a dict {"L": 3, "R": 4} fits a creature Meshy gave uneven legs, as the
    # DONOR for add_limb.py (the rig itself wants the same count a side)
    NPS = {'L': int(lps['L']), 'R': int(lps['R'])} if isinstance(lps, dict) else {'L': int(lps), 'R': int(lps)}
    nps = max(NPS.values())
    H_ = float(verts[:, 2].max())
    xmin = fit.get('foot_min_x', 0.05 if plan in ('drake', 'sprawl') else 0.3) * float(np.abs(verts[:, 0]).max())
    ext_all = extremities(S, np.full(S.n, -1))
    ext = [n for n in ext_all
           if S.p[n][2] < fit.get('foot_max_z', 0.25) * H_ and abs(S.p[n][0]) >= xmin
           and S.p[n][1] >= fit.get('foot_min_y', -1e9)]
    far = lambda p, lst: all(np.linalg.norm(np.asarray(p)[:2] - np.asarray(o)[:2]) > 0.08 * S.span for o in lst)  # noqa: E731
    if 'feet' in fit:
        feet = [np.array(p, float) for p in fit['feet']]
    else:
        ymin = fit.get('foot_min_y', -1e9)   # e.g. behind a scorpion's claws resting on the ground
        cl = [(n, c) for n, c in foot_contacts(verts, fit.get('foot_frac', 0.08), fit.get('foot_cluster', 0.03)) if abs(c[0]) >= xmin and c[1] >= ymin]
        if cl:
            med = float(np.median([n for n, _ in cl[:NPS['L'] + NPS['R']]]))
            cl = [(n, c) for n, c in cl if n >= 0.4 * med]
        sides = {'L': [], 'R': []}
        for sd, sg in (('L', 1), ('R', -1)):
            for _, c in cl:
                if len(sides[sd]) < NPS[sd] and c[0] * sg > 0 and far(c, sides[sd]):
                    sides[sd].append(c)
            for n in ext:
                if len(sides[sd]) < NPS[sd] and S.p[n][0] * sg > 0 and far(S.p[n], sides[sd]):
                    sides[sd].append(S.p[n])
                    report.setdefault('feet_from_tips', []).append([round(float(x), 3) for x in S.p[n]])
        for a_, b_ in (('L', 'R'), ('R', 'L')):
            for c in list(sides[b_]):
                m = np.array([-c[0], c[1], c[2]])
                if len(sides[a_]) < NPS[a_] and far(m, sides[a_]):
                    sides[a_].append(m)
                    report.setdefault('mirrored_feet', []).append([round(float(x), 3) for x in m])
        feet = sides['L'] + sides['R']
    snapped = []
    for f_ in feet:
        near = [n for n in ext if np.linalg.norm(S.p[n] - f_) < 0.05 * S.span]
        snapped.append(S.p[min(near, key=lambda n: np.linalg.norm(S.p[n] - f_))] if near else f_)
    feet = snapped
    left = sorted([f for f in feet if f[0] > 0], key=lambda f: f[1])
    right = sorted([f for f in feet if f[0] <= 0], key=lambda f: f[1])
    if len(left) != NPS['L'] or len(right) != NPS['R']:
        raise SystemExit(f'expected {NPS} feet, found L={len(left)} R={len(right)}: '
                         'set fit.legs_per_side or give fit.feet')
    leg_roots = []
    leg_tr = {}
    for side, fl in (('L', left), ('R', right)):
        for i, f in enumerate(fl, 1):
            # the lowest vertex of the contact cluster is the tip
            tip = S.nearest(f)
            if np.linalg.norm(S.p[tip] - f) > 1e-6:      # a contact centroid:
                near = np.linalg.norm(S.p[:, :2] - f[:2], axis=1) < 0.03 * S.span
                cand = np.flatnonzero(near) if near.any() else [tip]
                tip = int(min(cand, key=lambda n: S.p[n, 2]))   # its lowest vertex
            lr = fit.get('leg_roots', {}).get(f'leg{i}.{side}')
            if lr is None and side == 'R' and f'leg{i}.L' in fit.get('leg_roots', {}):
                lr = mirror(fit['leg_roots'][f'leg{i}.L'])
            leg_tr[f'leg{i}.{side}'] = trace(S, tip, h, sdf_stop=fit.get('leg_stop', 0.45) * (body_sdf or 0),
                                             grow=fit.get('leg_grow', 2.6), root=lr, ball=2.6 if lr is not None else None)
    # A leg traced to under 60% of its mirror twin's depth stopped on the
    # body early (it lies against it): re-trace it along a tube up to the
    # twin's root, mirrored. Then equalise the pairs.
    leg_tips = {}
    for side, fl in (('L', left), ('R', right)):
        for i, f in enumerate(fl, 1):
            leg_tips[f'leg{i}.{side}'] = f
    for k in list(leg_tr):
        tw = k[:-1] + ('R' if k.endswith('L') else 'L')
        if tw in leg_tr and leg_tr[k][1] < fit.get('pair_min_ratio', 0.6) * leg_tr[tw][1] and \
                f'{k}' not in fit.get('leg_roots', {}):
            twin_root = smooth_line(leg_tr[tw][0])[-1]
            lr = mirror(twin_root)
            tip_n = int(np.argmin(np.where(np.isfinite(leg_tr[k][2]), leg_tr[k][2], np.inf)))
            leg_tr[k] = trace(S, tip_n, h, sdf_stop=None, root=lr, ball=2.6)
            # and the twin the same way (to its own root), so both legs of
            # the pair are claimed by the same rule
            tw_tip = int(np.argmin(np.where(np.isfinite(leg_tr[tw][2]), leg_tr[tw][2], np.inf)))
            leg_tr[tw] = trace(S, tw_tip, h, sdf_stop=None, root=twin_root, ball=2.6)
            report.setdefault('retraced_to_twin', []).append(k)
    report['pair_equalised'] = equalise_pairs(leg_tr, fit.get('pair_min_ratio', 0.6))
    for side, fl in (('L', left), ('R', right)):
        for i, f in enumerate(fl, 1):
            pts, s_root, d = leg_tr[f'leg{i}.{side}']
            P = smooth_line(pts)
            L = arc(P)
            root = P[-1]
            leg_roots.append(root)
            key = f'leg{i}.{side}'
            claim(key, d, s_root)
            report['limbs'][key] = {'rings': len(pts), 'length': round(float(L[-1]), 3),
                                    'radius': round(float(np.median([r for _, _, r in pts])), 4)}
            J[f'leg{i}_4.{side}'] = P[0]
            # root to tip, reversed for fractions measured from the root
            Pr, Lr = P[::-1], arc(P[::-1])
            s_knee = corner(Pr, Lr)
            J[f'leg{i}_2.{side}'] = at_arc(Pr, Lr, s_knee)
            J[f'leg{i}_1.{side}'] = at_arc(Pr, Lr, fit.get('coxa_frac', 0.28) * s_knee)
            J[f'leg{i}_3.{side}'] = at_arc(Pr, Lr, s_knee + fit.get('tibia_frac', 0.5) * (Lr[-1] - s_knee))
            J[f'leg{i}_0.{side}'] = root
    leg_roots = np.array(leg_roots)
    body_r = float(np.median(np.linalg.norm(leg_roots[:, :2] - leg_roots[:, :2].mean(0), axis=1)))

    # ── other appendages: claws, eyes, fangs, palps, tail ───────────────
    limb_tr = {}

    def pre(key, tip_pt, hint):
        # claws and tails carry thick parts (a palm, a sting bulb) that a
        # leg's threshold would read as body: they stop later by default
        stop = hint.get('stop', 0.65 if key.startswith(('claw', 'tail')) else 0.45)
        tip = S.nearest(tip_pt)
        limb_tr[key] = trace(S, tip, h, sdf_stop=stop * (body_sdf or 0),
                             grow=hint.get('grow', 4.0 if key.startswith(('claw', 'tail')) else 3.0),
                             root=hint.get('root'), ball=hint.get('ball')) + (tip,)

    def limb(key, tip_pt, hint):
        if key not in limb_tr:
            pre(key, tip_pt, hint)
        pts, s_root, d, tip = limb_tr[key]
        P = smooth_line(pts)
        claim(key, d, s_root)
        report['limbs'][key] = {'rings': len(pts), 'length': round(float(arc(P)[-1]), 3)}
        return P, d, tip

    # Unclaimed protrusion tips (fangs, palps, a spinneret, a tail...):
    # listed in the report so an operator can copy one into a hint, and used
    # directly by `auto_front` (arachnids: the two frontmost tips a side, the
    # lower is the fang, the other the palp).
    free_tips = extremities(S, wlabel)
    report['free_tips'] = [[round(float(x), 3) for x in S.p[n]] for n in free_tips[:16]]
    if fit.get('auto_front') and plan == 'arachnid':
        front_y = float(leg_roots[:, 1].min())
        for side, sg in (('L', 1), ('R', -1)):
            cand = [n for n in free_tips if S.p[n][0] * sg > 0.01 and S.p[n][1] < front_y]
            cand = sorted(cand, key=lambda n: S.p[n][1])[:2]
            if len(cand) == 2:
                # chelicerae (fangs) sit medial, pedipalps lateral
                fang, palp = sorted(cand, key=lambda n: abs(S.p[n][0]))
                key = '' if side == 'L' else '.R'
                fit.setdefault('fang' + key, {'tip': S.p[fang].tolist()})
                fit.setdefault('palp' + key, {'tip': S.p[palp].tolist()})
            else:
                report.setdefault('auto_front_missing', []).append(side)
        if report.get('auto_front_missing'):
            # half a pair is worse than none: never guess the other side
            for k in ('fang', 'fang.R', 'palp', 'palp.R'):
                if k not in prof.get('fit', {}):
                    fit.pop(k, None)
        report['auto_front'] = {k: fit[k] for k in ('fang', 'fang.R', 'palp', 'palp.R') if k in fit}
    if fit.get('auto_claw') and 'claw' not in fit:
        # the two frontmost free tips a side are the pincer's fingers; the
        # moving finger (dactyl) is the lateral one
        for side, sg in (('L', 1), ('R', -1)):
            front = min(S.p[n][1] for n in ext_all)
            cand = sorted([n for n in ext_all if S.p[n][0] * sg > 0.1 * S.span and S.p[n][1] < front + 0.12 * S.span],
                          key=lambda n: S.p[n][1])[:2]
            if len(cand) == 2:
                fixed, dac = sorted(cand, key=lambda n: abs(S.p[n][0]))
                fit['claw' if side == 'L' else 'claw.R'] = dict(fit.get('claw_extra', {}), tip=S.p[fixed].tolist(),
                                                               dactyl=S.p[dac].tolist())
        report['auto_claw'] = {k: fit[k] for k in ('claw', 'claw.R') if k in fit}
    if fit.get('auto_tail') and 'tail' not in fit:
        # the sting: the highest free tip near the midline
        cand = [n for n in ext_all if abs(S.p[n][0]) < 0.1 * S.span and S.p[n][2] > 0.4 * H_]
        if cand:
            st = cand[0]   # extremities come farthest-first: the sting ends the longest limb
            fit['tail'] = dict(fit.get('tail_extra', {}), tip=S.p[st].tolist())
            report['auto_tail'] = fit['tail']
    for kind in ('claw', 'eye', 'fang', 'palp'):
        for side in ('L', 'R'):
            hk = hint_for(fit, kind, side)
            if hk:
                pre(f'{kind}.{side}', hk['tip'], hk)
    report['pair_equalised'].update(equalise_pairs(limb_tr, fit.get('pair_min_ratio', 0.6)))
    for side in ('L', 'R'):
        hc = hint_for(fit, 'claw', side)
        if hc:
            P, d, tip = limb(f'claw.{side}', hc['tip'], hc)
            L = arc(P)
            g, w, e = hc.get('fracs', [0.3, 0.55, 0.78])
            J[f'claw_4.{side}'] = P[0]
            J[f'claw_3.{side}'] = at_arc(P, L, g * L[-1])
            J[f'claw_2.{side}'] = at_arc(P, L, w * L[-1])
            J[f'claw_1.{side}'] = at_arc(P, L, e * L[-1])
            J[f'claw_0.{side}'] = P[-1]
            if 'dactyl' in hc:
                dt = S.nearest(hc['dactyl'])
                J[f'dactyl_1.{side}'] = S.p[dt]
                # The gape: the geodesic path from the moving finger's tip to
                # the fixed finger's tip runs down one finger, round the gape
                # and up the other; its point lowest along the claw's axis
                # (wrist -> fixed tip) is the bottom of the gape, the hinge.
                dd, pred = dijkstra(S.g, directed=False, indices=dt, return_predecessors=True)
                path, n = [], tip
                while n >= 0 and n != dt:
                    path.append(n); n = pred[n]
                path = np.array(path + [dt])
                axis = P[0] - J[f'claw_2.{side}']
                axis /= np.linalg.norm(axis)
                ax_of = lambda q: (q - J[f'claw_2.{side}']) @ axis  # noqa: E731
                gape = S.p[path[int(np.argmin(ax_of(S.p[path])))]]
                J[f'dactyl_0.{side}'] = gape
                J[f'claw_3.{side}'] = gape
                # the moving finger's own vertices: geodesically nearer its
                # tip than the fixed tip's, and past the gape along the axis
                m = np.isfinite(d) & (dd < d) & (ax_of(S.p) > ax_of(gape) - hc.get('hinge_margin', 0.02) * S.span) & \
                    (dd < 1.5 * np.linalg.norm(S.p[dt] - gape))
                idx = len(names); names.append(f'dactyl.{side}')
                wlabel[m] = idx
                tips[f'dactyl.{side}'] = dt
        for kind in ('eye', 'fang', 'palp'):
            hk = hint_for(fit, kind, side)
            if hk:
                P, d, tip = limb(f'{kind}.{side}', hk['tip'], hk)
                L = arc(P)
                J[f'{kind}_0.{side}'] = P[-1]
                if kind == 'palp':
                    J[f'{kind}_1.{side}'] = at_arc(P, L, 0.5 * L[-1])
                    J[f'{kind}_2.{side}'] = P[0]
                else:
                    J[f'{kind}_1.{side}'] = P[0]
    tail_P = None
    if 'tail' in fit:
        ht = fit['tail']
        tail_P, d, tip = limb('tail', ht['tip'], ht)
        L = arc(tail_P)
        n = int(ht.get('segments', 6))
        Pr, Lr = tail_P[::-1], L[-1] - L[::-1]
        # a scorpion's sting (last segment) is a fixed fraction; else even
        st = ht.get('stinger_frac', 0.16 if plan == 'scorpion' else 0.0)
        for k in range(n):
            J[f'tail_{k}'] = at_arc(Pr, Lr, ((1 - st) * Lr[-1] * k / (n - 1)) if st > 0 else Lr[-1] * k / n)
        J[f'tail_{n}'] = tail_P[0]
    if 'head' in fit:
        # neck + head, traced from the snout: the head is the first
        # `head_frac` of the arc, the neck the rest in equal segments
        hh = fit['head']
        P, d, tip = limb('head', hh['tip'], hh)
        L = arc(P)
        Pr, Lr = P[::-1], L[-1] - L[::-1]
        n = int(hh.get('neck_segments', 2))
        s_head = (1 - hh.get('head_frac', 0.3)) * Lr[-1]
        for k in range(n + 1):
            J[f'neck_{k}'] = at_arc(Pr, Lr, s_head * k / n)
        J['head_tip'] = P[0]
    for side in ('L', 'R'):
        hw = hint_for(fit, 'wing', side)
        if hw:
            # a wing spar (or a membrane's centre line) from tip to shoulder;
            # a real wing needs its `root` (shoulder) hint and fitted
            # elbow/wrist fractions (fit each wing)
            P, d, tip = limb(f'wing.{side}', hw['tip'], hw)
            L = arc(P)
            Pr, Lr = P[::-1], L[-1] - L[::-1]
            e, w = hw.get('fracs', [0.35, 0.68])
            J[f'wing_0.{side}'] = Pr[0]
            J[f'wing_1.{side}'] = at_arc(Pr, Lr, e * Lr[-1])
            J[f'wing_2.{side}'] = at_arc(Pr, Lr, w * Lr[-1])
            J[f'wing_3.{side}'] = P[0]

    # ── body ────────────────────────────────────────────────────────────
    body = S.p[wlabel == -1]
    lab_v = wlabel[S.inv]            # per prep vertex
    c = body.mean(0)
    if plan == 'arachnid':
        ys = body[:, 1]
        if 'pedicel_y' in fit:
            py = float(fit['pedicel_y'])
        else:
            # The waist between cephalothorax and abdomen is the deepest
            # notch in the dorsal outline (midline slab top z against its
            # upper convex hull), searched just behind the leg roots (the
            # legs all hang off the cephalothorax). Width is useless here:
            # the coxae make the cephalothorax look as wide as the abdomen.
            # dense surface samples (a low-poly abdomen has few vertices on
            # its midline): 10 fixed barycentric points per body face
            bf = faces[(lab_v[faces] == -1).all(1)]
            bw = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2],
                           [2, 2, 1], [2, 1, 2], [1, 2, 2]], float)
            bw /= bw.sum(1, keepdims=True)
            dense = np.einsum('kj,fjc->fkc', bw, verts[bf].astype(float)).reshape(-1, 3)
            slab = dense[np.abs(dense[:, 0]) < 0.05 * S.span]
            lo = float(np.mean(leg_roots[:, 1]))
            hi = float(leg_roots[:, 1].max()) + 0.25 * S.span
            grid = np.linspace(slab[:, 1].min(), slab[:, 1].max(), 80)
            top = np.array([slab[np.abs(slab[:, 1] - y) < 0.015 * S.span][:, 2].max()
                            if (np.abs(slab[:, 1] - y) < 0.015 * S.span).any() else np.nan for y in grid])
            ok = ~np.isnan(top)
            gy, gz = grid[ok], top[ok]
            hull = []
            for p_ in zip(gy, gz):
                while len(hull) >= 2 and (hull[-1][0] - hull[-2][0]) * (p_[1] - hull[-2][1]) - \
                        (hull[-1][1] - hull[-2][1]) * (p_[0] - hull[-2][0]) >= 0:
                    hull.pop()
                hull.append(p_)
            hy, hz = zip(*hull)
            gap = np.interp(gy, hy, hz) - gz
            win = (gy >= lo) & (gy <= hi)
            py = float(gy[int(np.argmax(np.where(win, gap, -1)))])
            report['pedicel_notch'] = round(float(gap[win].max()), 3) if win.any() else None
        ceph, abd = body[ys < py], body[ys >= py]
        cc = ceph.mean(0)
        J['body_0'] = cc
        J['body_1'] = np.array([cc[0], ceph[:, 1].min() + 0.15 * (cc[1] - ceph[:, 1].min()), cc[2]])
        ped = body[np.abs(ys - py) < 0.03 * S.span]
        J['abdomen_0'] = np.array([cc[0], py, ped[:, 2].mean()])
        ac = abd.mean(0)
        rear = abd[abd[:, 1] > np.percentile(abd[:, 1], 97)].mean(0)
        J['abdomen_1'] = rear + 0.25 * (ac - rear)
        sp = fit.get('spinneret')
        J['spinneret_1'] = np.array(sp, float) if sp else rear + np.array([0, 0.04 * S.span, -0.02 * S.span])
        report['pedicel_y'] = round(py, 3)
    elif plan == 'scorpion':
        lr_y = leg_roots[:, 1].max()
        pro = body[body[:, 1] <= lr_y]
        pc = pro.mean(0)
        J['body_0'] = pc
        J['body_1'] = np.array([pc[0], pro[:, 1].min() + 0.15 * (pc[1] - pro[:, 1].min()), pc[2]])
        meso = body[body[:, 1] > lr_y]
        J['abdomen_0'] = np.array([pc[0], lr_y, meso[:, 2].mean() if len(meso) else pc[2]])
    elif plan in ('drake', 'sprawl'):
        # a vertebrate spine from the hind-leg roots (hips) to the front-leg
        # roots (chest), each joint at the centroid of the body slab there
        roots = {k: np.asarray(J[f'{k}_0.L']) * 0.5 + np.asarray(J[f'{k}_0.R']) * 0.5
                 for k in ('leg1', f'leg{nps}')}
        yf, yh = roots['leg1'][1], roots[f'leg{nps}'][1]
        ns = int(fit.get('spine_segments', 3))
        for k in range(ns + 1):
            y = yh + (yf - yh) * k / ns
            slab = body[np.abs(body[:, 1] - y) < 0.04 * S.span]
            J[f'spine_{k}'] = slab.mean(0) if len(slab) else np.array([0.0, y, roots['leg1'][2]])
            if not fit.get('spine_follow_x', plan == 'sprawl'):
                J[f'spine_{k}'][0] = 0.0      # symmetric (drake); a sprawler may lie curled
        J['body_0'] = J[f'spine_{ns // 2}']
        if 'neck_0' in J:
            J[f'spine_{ns}'] = 0.5 * (J[f'spine_{ns}'] + J['neck_0'])
    else:
        J['body_0'] = c
        front = body[body[:, 1] < np.percentile(body[:, 1], 3)].mean(0)
        J['body_1'] = np.array([c[0], c[1] + 0.6 * (front[1] - c[1]), c[2]])
    J['root'] = np.array([J['body_0'][0], J['body_0'][1], 0.0])
    J['root_tip'] = J['root'] + np.array([0, -0.15 * S.span, 0])

    # limb roots go a little INSIDE the body so the first bone sits in it
    for k in list(J):
        if k.endswith(('_0.L', '_0.R')) and not k.startswith('dactyl'):
            J[k] = push_in(np.asarray(J[k]), J['body_0'], fit.get('root_push', 0.35) * body_r * 0.25)
    if 'tail_0' in J and 'abdomen_0' in J:
        J['tail_0'] = push_in(np.asarray(J['tail_0']), J['abdomen_0'], 0.05 * body_r)

    if fit.get('extend_to_core', True) and body_sdf:
        report['extended_to_core'] = extend_to_core(
            S, wlabel, names, fit.get('core_frac', 0.45) * body_sdf, fit.get('max_extend', 0.25) * S.span)
        # A limb that grew moves its ROOT joint to where the label now meets
        # the body (the centroid of its boundary with the body), pushed a
        # little inside; the old root becomes the first joint out from it
        # when it is far enough along, so the new segment gets a bone.
        g = S.g.tocoo()
        roots_moved = {}
        for n, cnt in report['extended_to_core'].items():
            i = names.index(n)
            kind, _, sd = n.partition('.')
            key = {'claw': 'claw_{}.' + sd}.get(kind) if kind == 'claw' else \
                (f'{kind}_{{}}.{sd}' if kind.startswith('leg') else None)
            if key is None or key.format(0) not in J:
                continue
            edge = (wlabel[g.row] == i) & (wlabel[g.col] == -1)
            if not edge.any():
                continue
            new0 = S.p[np.unique(g.row[edge])].mean(0)
            old0, old1 = np.asarray(J[key.format(0)]), np.asarray(J[key.format(1)])
            tipj = np.asarray(J[key.format(4)])
            if np.linalg.norm(new0 - old0) < 0.05 * np.linalg.norm(tipj - old0):
                continue
            if np.linalg.norm(old0 - new0) > 0.25 * np.linalg.norm(old1 - new0):
                J[key.format(1)] = old0 if np.linalg.norm(old0 - new0) < np.linalg.norm(old1 - new0) else old1
            J[key.format(0)] = push_in(new0, J['body_0'], fit.get('root_push', 0.35) * body_r * 0.25)
            roots_moved[n] = [round(float(x), 3) for x in J[key.format(0)]]
        report['roots_moved'] = roots_moved
    report['label_cleanup'] = clean_labels(S, wlabel, names)

    J = {k: [round(float(x), 4) for x in v] for k, v in J.items() if v is not None}
    for k, p in prof.get('joint_overrides', {}).items():
        J[k] = p
    report['body_r'] = round(body_r, 3)
    report['labels'] = {n: int((wlabel == i).sum()) for i, n in enumerate(names)}
    report['labels']['body'] = int((wlabel == -1).sum())
    if '--control' in sys.argv:
        # POSITIVE CONTROL: deliberately mislabel the root half of one leg as
        # its neighbour, and show the checks catch it (nothing is written)
        victim = sys.argv[sys.argv.index('--control') + 1]
        side = victim.split('.')[1]
        i = int(victim[3:].split('.')[0])
        other = f'leg{i + 1 if f"leg{i + 1}.{side}" in names else i - 1}.{side}'
        vi, oi = names.index(victim), names.index(other)
        m = np.flatnonzero(wlabel == vi)
        d = S.geodesic(tips[victim])
        far = m[d[m] > np.median(d[m])]
        wlabel[far] = oi
        print(f'CONTROL: relabelled the root half of {victim} ({len(far)} nodes) as {other}')
    checks = check_labels(S, wlabel, names, tips, fit.get('mirror_tolerance', 0.25))
    # An operator may ACCEPT a specific failure with a written reason
    # (profile fit.accept {"(b) leg4": "why"}): it stays in the report and on
    # labels.png, marked accepted, but no longer stops the pipeline.
    acc = fit.get('accept', {})
    checks['accepted'] = [f_ + '  [ACCEPTED: ' + acc[k] + ']' for f_ in checks['failures'] for k in acc if f_.startswith(k)]
    checks['failures'] = [f_ for f_ in checks['failures'] if not any(f_.startswith(k) for k in acc)]
    checks['pass'] = not checks['failures']
    report['checks'] = checks
    print(json.dumps(report, indent=1))
    if '--write' in sys.argv:
        prof['joints'] = J
        profs[name] = prof
        json.dump(profs, open(PROFILES, 'w'), indent=1)
        lab = wlabel[S.inv].astype(np.int16)   # back to prep (unwelded) vertices
        np.save(os.path.join(prep, 'labels.npy'), lab)
        json.dump({'names': names, 'verts': int(len(verts))}, open(os.path.join(prep, 'labels.json'), 'w'), indent=1)
        json.dump(report, open(os.path.join(prep, 'fit-report.json'), 'w'), indent=1)
        print('wrote joints + labels for', name)
    print('LABEL CHECKS ' + ('PASS' if checks['pass'] else 'FAIL: ' + '; '.join(checks['failures']))
          + (' | accepted: ' + '; '.join(checks['accepted']) if checks['accepted'] else ''))
    if not checks['pass'] and '--allow-fail' not in sys.argv:
        raise SystemExit(3)


if __name__ == '__main__':
    main()
