
"""

Phase 3.2C/3.3 (v06): micro-optimizations in range word-fills to reduce kernel overhead.

gpu_pixelize_cones_moclike_v1.py

GPU pixelization of spherical cones (discs on the sky) designed to align with
mocpy.MOC.from_cones(..., union_strategy="small_cones", delta_depth=2) behavior,
i.e. cdshealpix::nested::cone_coverage_approx_custom.

Goal
----
Return a flat list of NESTED HEALPix pixels at a fixed output order, suitable to
OR-paint into a HealSparse mask.

Design (v1)
-----------
- Uses the *same* acceptance / overlap / reject decision rule as cdshealpix:
    shs = sin^2(ad/2) = (1 - dot(u_cell, u_cone))/2
    compare against per-depth Min/Max computed from
    largest_center_to_vertex_distances_with_radius(depth_start..depth_eff+1, lon,lat,radius).

- Uses healpy on CPU for:
    * ang2pix(nest=True) to compute the starting pixel (hash) at depth_start
    * get_all_neighbours(nest=True) to seed the 9-neighbour set for small cones

  This matches standard HEALPix; remaining mismatches (if any) tend to come from
  floating boundary cases and from the Rust "BMOC packing" lowering. v1 outputs
  pixels at output order directly, avoiding naive leaf->shift lowering.

- GPU traversal:
    One warp per cone; iterative stack; emits:
      * FULL nodes -> directly expanded / mapped to output order
      * leaf overlap nodes at depth_eff -> mapped to output order

Notes
-----
- Selection semantics: ALL (any overlap sets pixel).
- This module is intended as a drop-in replacement for gpu_pixelize_discs_exact_v2.py
  from Skykatana wrappers: exports is_gpu_available, GpuBitpackDictAccumulator, accumulate_discs_bitpack.

Limitations (explicit)
----------------------
- The per-depth bound function is ported from cdshealpix/src/lib.rs, but computed on CPU.
  Minor numerical differences vs Rust can still occur in rare boundary cases.
- Large cones (radius >= ~48 deg): falls back to seeding the 12 base cells at depth 0.
- FULL-node expansion to output order can generate many pixels for very large cones;
  star masks (arcsec–arcmin) are safe.

"""

import math
import time
import numpy as np


# --- numba import (robust to some coverage/numba packaging mismatches) ---
try:
    from numba import cuda  # type: ignore
except Exception as _e:  # pragma: no cover
    # Some environments have an incompatible 'coverage' package that breaks numba import.
    # We patch minimal attributes to let numba import.
    try:
        import types as _types
        import coverage as _coverage  # type: ignore
        if not hasattr(_coverage, "types"):
            _coverage.types = _types.SimpleNamespace()
        _missing = [
            'Tracer',
            'TShouldTraceFn',
            'TShouldStartContextFn',
            'TTraceData',
            'TFileDisposition',
        ]
        for _n in _missing:
            if not hasattr(_coverage.types, _n):
                setattr(_coverage.types, _n, object)
        from numba import cuda  # type: ignore
    except Exception as _e2:
        raise


try:
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None

try:
    import healpy as hp
    _HAVE_HEALPY = True
except Exception:
    _HAVE_HEALPY = False

def is_gpu_available() -> bool:
    """Return True if CUDA GPU stack (numba.cuda + cupy) is available."""
    try:
        if cp is None:
            return False
        # numba.cuda imported as `cuda` above
        from numba import cuda as _cuda  # type: ignore
        return _cuda.is_available()
    except Exception:
        return False



# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Constants ported from cdshealpix/src/lib.rs
# ----------------------------------------------------------------------

# MAX ORDER supported by moc / cdshealpix
MAX_ORDER_SUPPORTED = 29

# HEALPix constants used in cdshealpix
PI = math.pi
FRAC_PI_2 = 0.5 * PI
FRAC_PI_4 = 0.25 * PI
PI_OVER_FOUR = 0.25 * PI

# cdshealpix constants (exact literals)
TRANSITION_Z = 2.0 / 3.0
TRANSITION_LATITUDE = 0.729_727_656_226_966_3
LAT_OF_SQUARE_CELL = 0.399_340_199_478_977_75
COS_LAT_OF_SQUARE_CELL = 0.921_317_731_923_561_3

# Table SMALLER_EDGE2OPEDGE_DIST from cdshealpix/src/lib.rs
SMALLER_EDGE2OPEDGE_DIST = np.array([
  8.410686705679302e-1,
  3.7723631722170065e-1,
  1.8203364957037313e-1,
  8.91145416330163e-2,
  4.3989734509169175e-2,
  2.1817362566054977e-2,
  1.0854009694242892e-2,
  5.409888140793663e-3,
  2.6995833266547898e-3,
  1.3481074874673246e-3,
  6.735240905806414e-4,
  3.365953703015157e-4,
  1.682452196838741e-4,
  8.410609042173736e-5,
  4.204784317861652e-5,
  2.1022283297961136e-5,
  1.0510625670060442e-5,
  5.255150320257332e-6,
  2.6275239729465538e-6,
  1.3137458638808036e-6,
  6.568678535571394e-7,
  3.284323270983175e-7,
  1.642156595517884e-7,
  8.21076709163242e-8,
  4.105378528139296e-8,
  2.0526876713226626e-8,
  1.0263433216329513e-8,
  5.131714858175969e-9,
  2.5658567623093986e-9,
  1.2829280665188905e-9,
], dtype=np.float64)

# ----------------------------------------------------------------------
# CPU ports of cdshealpix bound helpers
# ----------------------------------------------------------------------

def has_best_starting_depth(d_max_rad: float) -> bool:
    # cdshealpix::has_best_starting_depth
    return d_max_rad < float(SMALLER_EDGE2OPEDGE_DIST[0])

def best_starting_depth(d_max_rad: float) -> int:
    # Equivalent to cdshealpix::best_starting_depth, but using search on the table.
    # It returns the smallest depth where d_max_rad < SMALLER_EDGE2OPEDGE_DIST[depth],
    # except it saturates at 29 if below depth 29 threshold.
    if not has_best_starting_depth(d_max_rad):
        raise ValueError("Too large value, use has_best_starting_depth first")
    # If smaller than depth 29 threshold -> 29
    if d_max_rad < float(SMALLER_EDGE2OPEDGE_DIST[29]):
        return 29
    # Find smallest depth where d < table[depth] scanning coarse-to-fine
    # Table decreases with depth; we want depth such that d < table[depth] but d >= table[depth+1].
    lo, hi = 0, 29
    while lo < hi:
        mid = (lo + hi) // 2
        if d_max_rad < float(SMALLER_EDGE2OPEDGE_DIST[mid]):
            # can be at least this deep
            lo = mid + 1
        else:
            hi = mid
    # lo is first index where d >= table[lo]; answer is lo-1
    ans = max(0, lo - 1)
    return int(ans)

def _pow2(x: float) -> float:
    return x * x

def _to_squared_half_segment(spherical_distance: float) -> float:
    # sin^2(d/2)
    s = math.sin(0.5 * spherical_distance)
    return s * s

def _linear_approx(x: float, slope: float, intercept: float) -> float:
    return slope * x + intercept

class _ConstantsC2V:
    __slots__ = ("slope_npc","intercept_npc","slope_eqr","intercept_eqr","coeff_x2_eqr","coeff_cst_eqr")
    def __init__(self, depth: int):
        # Port of ConstantsC2V::new(depth) from cdshealpix/src/lib.rs
        nside = 1 << int(depth)
        dist_cw = 1.0 / float(nside)
        one_min_dist_cw = 1.0 - dist_cw

        lat_north = math.asin(1.0 - (_pow2(one_min_dist_cw) / 3.0))
        d_min_npc = lat_north - TRANSITION_LATITUDE

        # d_max_npc computed using haversine shs then sphe_dist
        # shs = sin^2(dlat/2) + cos1*cos2*sin^2(dlon/2)
        dlon = FRAC_PI_4 * dist_cw
        dlat = d_min_npc
        cos1 = math.cos(lat_north)
        cos2 = math.cos(TRANSITION_LATITUDE)
        shs = (math.sin(0.5*dlat)**2) + cos1*cos2*(math.sin(0.5*dlon)**2)
        d_max_npc = 2.0 * math.asin(math.sqrt(shs))

        slope_npc = (d_max_npc - d_min_npc) / (FRAC_PI_4 * one_min_dist_cw)
        intercept_npc = d_min_npc

        d_min_eqr_top = PI_OVER_FOUR * dist_cw * COS_LAT_OF_SQUARE_CELL
        d_max_eqr_top = TRANSITION_LATITUDE - math.asin(one_min_dist_cw * TRANSITION_Z)
        slope_eqr = (d_max_eqr_top - d_min_eqr_top) / (TRANSITION_LATITUDE - LAT_OF_SQUARE_CELL)
        intercept_eqr = d_min_eqr_top - slope_eqr * LAT_OF_SQUARE_CELL

        d_max = PI_OVER_FOUR * dist_cw
        coeff_cst_eqr = d_max
        coeff_x2_eqr = (d_min_eqr_top - d_max) / _pow2(LAT_OF_SQUARE_CELL)

        self.slope_npc = slope_npc
        self.intercept_npc = intercept_npc
        self.slope_eqr = slope_eqr
        self.intercept_eqr = intercept_eqr
        self.coeff_x2_eqr = coeff_x2_eqr
        self.coeff_cst_eqr = coeff_cst_eqr

# cache constants per depth
_C2V_CACHE = {}

def _get_c2v(depth: int) -> _ConstantsC2V:
    c = _C2V_CACHE.get(depth)
    if c is None:
        c = _ConstantsC2V(depth)
        _C2V_CACHE[depth] = c
    return c

def _largest_c2v_dist_in_npc_with_radius(lon: float, radius: float, csts: _ConstantsC2V) -> float:
    # lon' = |pi/4 - (lon mod pi/2)|, then min(lon'+radius, pi/4)
    lonm = abs(FRAC_PI_4 - (lon % (FRAC_PI_2)))
    lonm = min(lonm + radius, FRAC_PI_4)
    return _linear_approx(lonm, csts.slope_npc, csts.intercept_npc)

def _largest_c2v_dist_in_eqr_top(lat_abs: float, csts: _ConstantsC2V) -> float:
    return _linear_approx(lat_abs, csts.slope_eqr, csts.intercept_eqr)

def _largest_c2v_dist_in_eqr_bottom(lat_abs: float, csts: _ConstantsC2V) -> float:
    return csts.coeff_x2_eqr * _pow2(lat_abs) + csts.coeff_cst_eqr

def largest_center_to_vertex_distance_with_radius(depth: int, lon: float, lat: float, radius: float) -> float:
    # Port of cdshealpix::largest_center_to_vertex_distance_with_radius
    if depth == 0:
        return FRAC_PI_2 - TRANSITION_LATITUDE

    csts = _get_c2v(depth)
    lat_abs = abs(lat)
    lat_max = lat_abs + radius
    lat_min = max(lat_abs - radius, 0.0)

    if lat_max >= TRANSITION_LATITUDE:
        return _largest_c2v_dist_in_npc_with_radius(lon, radius, csts)
    elif lat_min >= LAT_OF_SQUARE_CELL:
        return _largest_c2v_dist_in_eqr_top(lat_max, csts)
    elif lat_max <= LAT_OF_SQUARE_CELL:
        return _largest_c2v_dist_in_eqr_bottom(lat_min, csts)
    else:
        # straddles LAT_OF_SQUARE_CELL
        return max(_largest_c2v_dist_in_eqr_top(lat_max, csts),
                   _largest_c2v_dist_in_eqr_bottom(lat_min, csts))

def largest_center_to_vertex_distances_with_radius(from_depth: int, to_depth: int, lon: float, lat: float, radius: float) -> np.ndarray:
    # Port of cdshealpix::largest_center_to_vertex_distances_with_radius
    # returns len = to_depth - from_depth values for depths [from_depth, to_depth)
    vec = []
    if from_depth == 0:
        vec.append(FRAC_PI_2 - TRANSITION_LATITUDE)
        from_depth = 1

    lat_abs = abs(lat)
    lat_max = lat_abs + radius
    lat_min = lat_abs - radius

    if lat_max >= TRANSITION_LATITUDE:
        lonm = abs(FRAC_PI_4 - (lon % FRAC_PI_2))
        lonm = min(lonm + radius, FRAC_PI_4)
        for depth in range(from_depth, to_depth):
            csts = _get_c2v(depth)
            vec.append(_linear_approx(lonm, csts.slope_npc, csts.intercept_npc))
    elif lat_min >= LAT_OF_SQUARE_CELL:
        for depth in range(from_depth, to_depth):
            vec.append(_largest_c2v_dist_in_eqr_top(lat_max, _get_c2v(depth)))
    elif lat_max <= LAT_OF_SQUARE_CELL:
        val_min = max(lat_min, 0.0)
        for depth in range(from_depth, to_depth):
            vec.append(_largest_c2v_dist_in_eqr_bottom(val_min, _get_c2v(depth)))
    else:
        val_max = min(lat_max, TRANSITION_LATITUDE)
        val_min = max(lat_min, 0.0)
        for depth in range(from_depth, to_depth):
            csts = _get_c2v(depth)
            vec.append(max(_largest_c2v_dist_in_eqr_top(val_max, csts),
                           _largest_c2v_dist_in_eqr_bottom(val_min, csts)))
    return np.asarray(vec, dtype=np.float64)

def build_shs_minmax_abs(depth_start: int, depth_eff: int, lon: float, lat: float, radius: float):
    """Return arrays min_abs[0..depth_eff], max_abs[0..depth_eff] (float64),
    with entries < depth_start set to -1.
    """
    n = depth_eff + 1
    min_abs = np.full(n, -1.0, dtype=np.float64)
    max_abs = np.full(n, -1.0, dtype=np.float64)
    dists = largest_center_to_vertex_distances_with_radius(depth_start, depth_eff + 1, lon, lat, radius)
    # dists[k] corresponds to depth = depth_start + k
    for k, d in enumerate(dists):
        depth = depth_start + k
        # min
        if radius < d:
            minv = 0.0
        else:
            minv = _to_squared_half_segment(radius - d)
        maxv = _to_squared_half_segment(radius + d)
        min_abs[depth] = minv
        max_abs[depth] = maxv
    return min_abs, max_abs


# ----------------------------------------------------------------------
# Phase 3.2D: GPU min/max (per-cone, per-depth) computation
# ----------------------------------------------------------------------

# Precompute the cdshealpix per-depth C2V constants once on CPU and expose as
# CUDA constant arrays for device-side min/max construction.
_C2V_SLOPE_NPC = np.zeros(MAX_ORDER_SUPPORTED + 1, dtype=np.float64)
_C2V_INTERCEPT_NPC = np.zeros(MAX_ORDER_SUPPORTED + 1, dtype=np.float64)
_C2V_SLOPE_EQR = np.zeros(MAX_ORDER_SUPPORTED + 1, dtype=np.float64)
_C2V_INTERCEPT_EQR = np.zeros(MAX_ORDER_SUPPORTED + 1, dtype=np.float64)
_C2V_COEFF_X2_EQR = np.zeros(MAX_ORDER_SUPPORTED + 1, dtype=np.float64)
_C2V_COEFF_CST_EQR = np.zeros(MAX_ORDER_SUPPORTED + 1, dtype=np.float64)
for _d in range(1, MAX_ORDER_SUPPORTED + 1):
    _c = _get_c2v(_d)
    _C2V_SLOPE_NPC[_d] = float(_c.slope_npc)
    _C2V_INTERCEPT_NPC[_d] = float(_c.intercept_npc)
    _C2V_SLOPE_EQR[_d] = float(_c.slope_eqr)
    _C2V_INTERCEPT_EQR[_d] = float(_c.intercept_eqr)
    _C2V_COEFF_X2_EQR[_d] = float(_c.coeff_x2_eqr)
    _C2V_COEFF_CST_EQR[_d] = float(_c.coeff_cst_eqr)

#+
# NOTE: Do NOT use cuda.const.array_like() at import time.
# In environments where Numba's CUDA support is not initialized, calling
# cuda.const.array_like from host code raises NotImplementedError.
# We only need read-only per-depth constants in device code, so we expose
# them as Python tuples (compile-time constants for Numba).
_C2V_SLOPE_NPC_C = tuple(_C2V_SLOPE_NPC.tolist())
_C2V_INTERCEPT_NPC_C = tuple(_C2V_INTERCEPT_NPC.tolist())
_C2V_SLOPE_EQR_C = tuple(_C2V_SLOPE_EQR.tolist())
_C2V_INTERCEPT_EQR_C = tuple(_C2V_INTERCEPT_EQR.tolist())
_C2V_COEFF_X2_EQR_C = tuple(_C2V_COEFF_X2_EQR.tolist())
_C2V_COEFF_CST_EQR_C = tuple(_C2V_COEFF_CST_EQR.tolist())


@cuda.jit(device=True, inline=True)
def _shs_from_dist(d):
    # sin^2(d/2)
    s = math.sin(0.5 * d)
    return s * s


@cuda.jit(device=True, inline=True)
def _largest_c2v_dist_with_radius_dev(depth, lon, lat, radius):
    # Device-side port of largest_center_to_vertex_distance_with_radius
    if depth == 0:
        return FRAC_PI_2 - TRANSITION_LATITUDE

    lat_abs = abs(lat)
    lat_max = lat_abs + radius
    lat_min = lat_abs - radius

    if lat_max >= TRANSITION_LATITUDE:
        lonm = abs(FRAC_PI_4 - (lon % FRAC_PI_2))
        lonm = min(lonm + radius, FRAC_PI_4)
        return lonm * _C2V_SLOPE_NPC_C[depth] + _C2V_INTERCEPT_NPC_C[depth]

    if lat_min >= LAT_OF_SQUARE_CELL:
        return lat_max * _C2V_SLOPE_EQR_C[depth] + _C2V_INTERCEPT_EQR_C[depth]

    if lat_max <= LAT_OF_SQUARE_CELL:
        val_min = max(lat_min, 0.0)
        return _C2V_COEFF_X2_EQR_C[depth] * (val_min * val_min) + _C2V_COEFF_CST_EQR_C[depth]

    # straddles LAT_OF_SQUARE_CELL
    val_max = min(lat_max, TRANSITION_LATITUDE)
    val_min = max(lat_min, 0.0)
    d1 = val_max * _C2V_SLOPE_EQR_C[depth] + _C2V_INTERCEPT_EQR_C[depth]
    d2 = _C2V_COEFF_X2_EQR_C[depth] * (val_min * val_min) + _C2V_COEFF_CST_EQR_C[depth]
    return d1 if d1 > d2 else d2


@cuda.jit
def build_shs_minmax_abs_kernel(depth_start, depth_eff, lon, lat, radius, min_abs, max_abs):
    i = cuda.grid(1)
    if i >= lon.size:
        return

    ds = int(depth_start[i])
    de = int(depth_eff)
    r = float(radius[i])
    lo = float(lon[i])
    la = float(lat[i])

    # Initialize (depths < ds stay -1)
    for d in range(de + 1):
        min_abs[i, d] = -1.0
        max_abs[i, d] = -1.0

    for depth in range(ds, de + 1):
        dist = _largest_c2v_dist_with_radius_dev(depth, lo, la, r)
        if r < dist:
            minv = 0.0
        else:
            minv = _shs_from_dist(r - dist)
        maxv = _shs_from_dist(r + dist)
        min_abs[i, depth] = minv
        max_abs[i, depth] = maxv


# ----------------------------------------------------------------------
# GPU device helpers: NESTED ipix@order -> unit vector center
# (Copied from gpu_pixelize_discs_exact_v2; keep local for compilation stability.)
# ----------------------------------------------------------------------

@cuda.jit(device=True, inline=True)
def _nest_to_face_xy(ipix, order):
    """
    Deinterleave the NESTED pixel index (within face) into x,y at given order,
    and compute base face (0..11).

    ipix is the *global* NESTED index at this order: [0, 12*nside^2).
    """
    nside = 1 << order
    ns2 = nside * nside
    face = ipix // ns2
    idx = ipix - face * ns2

    x = 0
    y = 0
    # Deinterleave: x gets bit0, y gets bit1, then repeat
    for i in range(order):
        x |= (idx & 1) << i
        idx >>= 1
        y |= (idx & 1) << i
        idx >>= 1

    return face, x, y


@cuda.jit(device=True, inline=True)
def _isnorth(face):
    return face <= 3


@cuda.jit(device=True, inline=True)
def _issouth(face):
    return face >= 8


# ============================================================
# Device: (face, x, y, order, dx, dy) -> unit vector (x,y,z)
# This is a direct-port style implementation of the standard HEALPix base mapping,
# using the pixel center dx=dy=0.5.
# ============================================================
@cuda.jit(device=True, inline=True)
def _face_xy_to_xyz(face, xp, yp, order, dx, dy):
    """
    Map HEALPix face + (x,y) to unit vector at subpixel offset (dx,dy),
    where dx=dy=0.5 gives pixel center.

    This uses the classic HEALPix face parameterization (equatorial vs polar parts).
    """
    nside = 1 << order
    pi = math.pi

    # "fine" coordinates within face
    x = float(xp) + dx
    y = float(yp) + dy

    equatorial = True
    zfactor = 1.0

    if _isnorth(face):
        if (x + y) > nside:
            equatorial = False
            zfactor = 1.0
    if _issouth(face):
        if (x + y) < nside:
            equatorial = False
            zfactor = -1.0

    if equatorial:
        zoff = 0.0
        phioff = 0.0
        xf = x / nside
        yf = y / nside
        chp = face

        if chp <= 3:
            # north equatorial band faces
            phioff = 1.0
        elif chp <= 7:
            # equator faces
            zoff = -1.0
            chp -= 4
        else:
            # south equatorial band faces
            phioff = 1.0
            zoff = -2.0
            chp -= 8

        z = (2.0 / 3.0) * (xf + yf + zoff)
        phi = (pi / 4.0) * (xf - yf + phioff + 2.0 * chp)
        rad = math.sqrt(max(0.0, 1.0 - z * z))

    else:
        # polar region
        xx = x
        yy = y

        if zfactor == -1.0:
            # swap and flip (south polar handling)
            tmp = xx
            xx = yy
            yy = tmp
            xx = (nside - xx)
            yy = (nside - yy)

        # phi_t in [0, pi/2]
        if (yy == nside) and (xx == nside):
            phi_t = 0.0
        else:
            denom = 2.0 * ((nside - xx) + (nside - yy))
            phi_t = pi * (nside - yy) / denom

        root3 = 1.7320508075688772  # sqrt(3)
        if phi_t < (pi / 4.0):
            # note (2*phi_t - pi) is negative here; abs handles it
            vv = abs(pi * (nside - xx) / ((2.0 * phi_t - pi) * nside) / root3)
        else:
            vv = abs(pi * (nside - yy) / (2.0 * phi_t * nside) / root3)

        z = (1.0 - vv) * (1.0 + vv)
        rad = math.sqrt(max(0.0, 1.0 + z)) * vv
        z *= zfactor

        if _issouth(face):
            phi = (pi / 2.0) * (face - 8) + phi_t
        else:
            phi = (pi / 2.0) * face + phi_t

    # wrap phi into [0, 2pi)
    if phi < 0.0:
        phi += 2.0 * pi

    vx = rad * math.cos(phi)
    vy = rad * math.sin(phi)
    vz = z
    return vx, vy, vz

@cuda.jit(device=True, inline=True)
def pix2vec_nest_center(ipix, order):
    face, x, y = _nest_to_face_xy(ipix, order)
    return _face_xy_to_xyz(face, x, y, order, 0.5, 0.5)

@cuda.jit(device=True, inline=True)
def _shs_from_dot(dot):
    # squared half segment = (1 - dot)/2 for unit vectors
    if dot > 1.0:
        dot = 1.0
    if dot < -1.0:
        dot = -1.0
    return 0.5 * (1.0 - dot)


# ----------------------------------------------------------------------
# GPU traversal kernels (warp-per-cone)
# ----------------------------------------------------------------------

MAX_STACK_WARP = 256  # conservative; increase if you see overflow
MAX_SEEDS = 12        # 12 base faces, or 9 neighbours

# Heuristic to avoid pathological DFS expansion on GPU:
# If depth_eff - depth_start is large, the per-cone tree can explode and overflow the fixed stack.
# For selection=ALL masks, we safely fall back to exact mocpy for those cones (vectorized, unioned).
DEPTH_SPAN_CPU_FALLBACK = 6  # if (depth_eff - depth_start) > this, do CPU mocpy for that cone batch

@cuda.jit
def count_cone_pixels_kernel_warp(
    cone_x, cone_y, cone_z,
    depth_start_arr, depth_eff,
    seeds, n_seeds,
    min_abs, max_abs,
    out_order,
    counts,
    overflow
):
    """
    Count number of output pixels at out_order produced by moc-like traversal.
    One warp per cone.
    """
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    lane = cuda.threadIdx.x & 31
    warp = tid >> 5  # warp id global
    n_cones = cone_x.size
    if warp >= n_cones:
        return

    # Only lane0 executes the stack control for simplicity (warp-coop write not needed in count)
    if lane != 0:
        return

    cx = cone_x[warp]; cy = cone_y[warp]; cz = cone_z[warp]
    dstart = depth_start_arr[warp]

    # local stack arrays
    stack_h = cuda.local.array(shape=MAX_STACK_WARP, dtype=np.uint64)
    stack_d = cuda.local.array(shape=MAX_STACK_WARP, dtype=np.int16)
    sp = 0

    ns = n_seeds[warp]
    for i in range(ns):
        h = seeds[warp, i]
        if h >= 0:
            stack_h[sp] = np.uint64(h)
            stack_d[sp] = np.int16(dstart)
            sp += 1
            if sp >= MAX_STACK_WARP:
                overflow[warp] = 1
                counts[warp] = 0
                return

    c = 0
    while sp > 0:
        sp -= 1
        h = stack_h[sp]
        d = int(stack_d[sp])

        vx, vy, vz = pix2vec_nest_center(h, d)
        dot = vx*cx + vy*cy + vz*cz
        shs = _shs_from_dot(dot)

        mn = min_abs[warp, d]
        mx = max_abs[warp, d]

        if shs <= mn:
            # FULL node
            if d >= out_order:
                c += 1
            else:
                dd = out_order - d
                c += 1 << (2 * dd)  # 4^(dd)
        elif shs <= mx:
            if d == depth_eff:

                # leaf overlap mapped to out_order; re-test at out_order to avoid false positives

                if depth_eff == out_order:

                    c += 1

                else:

                    pix = h >> (2 * (depth_eff - out_order))

                    vx2, vy2, vz2 = pix2vec_nest_center(pix, out_order)

                    dot2 = vx2*cx + vy2*cy + vz2*cz

                    shs2 = _shs_from_dot(dot2)

                    mx2 = max_abs[warp, out_order]

                    if shs2 <= mx2:

                        c += 1
            else:
                # split
                if sp + 4 >= MAX_STACK_WARP:
                    overflow[warp] = 1
                    counts[warp] = 0
                    return
                base = h << 2
                nd = np.int16(d + 1)
                stack_h[sp] = base + 0; stack_d[sp] = nd; sp += 1
                stack_h[sp] = base + 1; stack_d[sp] = nd; sp += 1
                stack_h[sp] = base + 2; stack_d[sp] = nd; sp += 1
                stack_h[sp] = base + 3; stack_d[sp] = nd; sp += 1

    counts[warp] = c


@cuda.jit
def fill_cone_pixels_kernel_warp(
    cone_x, cone_y, cone_z,
    depth_start_arr, depth_eff,
    seeds, n_seeds,
    min_abs, max_abs,
    out_order,
    offsets,
    out_pix,
    overflow
):
    """
    Fill output pixels at out_order using moc-like traversal.
    One warp per cone; only lane0 controls stack and writes sequentially.
    """
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    lane = cuda.threadIdx.x & 31
    warp = tid >> 5
    n_cones = cone_x.size
    if warp >= n_cones:
        return
    if lane != 0:
        return

    if overflow[warp] != 0:
        return

    cx = cone_x[warp]; cy = cone_y[warp]; cz = cone_z[warp]
    dstart = depth_start_arr[warp]

    stack_h = cuda.local.array(shape=MAX_STACK_WARP, dtype=np.uint64)
    stack_d = cuda.local.array(shape=MAX_STACK_WARP, dtype=np.int16)
    sp = 0

    ns = n_seeds[warp]
    for i in range(ns):
        h = seeds[warp, i]
        if h >= 0:
            stack_h[sp] = np.uint64(h)
            stack_d[sp] = np.int16(dstart)
            sp += 1
            if sp >= MAX_STACK_WARP:
                overflow[warp] = 1
                return

    write = offsets[warp]

    while sp > 0:
        sp -= 1
        h = stack_h[sp]
        d = int(stack_d[sp])

        vx, vy, vz = pix2vec_nest_center(h, d)
        dot = vx*cx + vy*cy + vz*cz
        shs = _shs_from_dot(dot)

        mn = min_abs[warp, d]
        mx = max_abs[warp, d]

        if shs <= mn:
            # FULL node: map/expand to out_order
            if d >= out_order:
                pix = h >> (2 * (d - out_order))
                out_pix[write] = np.uint64(pix)
                write += 1
            else:
                dd = out_order - d
                shift = 2 * dd
                start = h << shift
                n = 1 << (2 * dd)
                # write contiguous range
                for t in range(n):
                    out_pix[write + t] = np.uint64(start + t)
                write += n
        elif shs <= mx:
            if d == depth_eff:
                # leaf overlap; map to out_order (depth_eff >= out_order)
                pix = h >> (2 * (depth_eff - out_order))
                out_pix[write] = np.uint64(pix)
                write += 1
            else:
                if sp + 4 >= MAX_STACK_WARP:
                    overflow[warp] = 1
                    return
                base = h << 2
                nd = np.int16(d + 1)
                stack_h[sp] = base + 0; stack_d[sp] = nd; sp += 1
                stack_h[sp] = base + 1; stack_d[sp] = nd; sp += 1
                stack_h[sp] = base + 2; stack_d[sp] = nd; sp += 1
                stack_h[sp] = base + 3; stack_d[sp] = nd; sp += 1

    # done


# ----------------------------------------------------------------------
# Host API: query + accumulate
# ----------------------------------------------------------------------


@cuda.jit
def paint_cone_pixels_kernel_warp(
    cone_x, cone_y, cone_z,
    depth_start_arr, depth_eff,
    seeds, n_seeds,
    min_abs, max_abs,
    out_order,
    out_pix,
    counter,
    out_cap,
    overflow
):
    """Single-pass pixel emission (no per-cone offsets).

    One warp per cone; lane0 traverses and appends pixels into a global output buffer
    using an atomic reservation counter.

    This is meant to reduce the (count -> prefix -> fill) pipeline overhead.
    """
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    lane = cuda.threadIdx.x & 31
    warp = tid >> 5
    n_cones = cone_x.size
    if warp >= n_cones:
        return
    if lane != 0:
        return

    if overflow[warp] != 0:
        return

    cx = cone_x[warp]; cy = cone_y[warp]; cz = cone_z[warp]
    dstart = depth_start_arr[warp]

    stack_h = cuda.local.array(shape=MAX_STACK_WARP, dtype=np.uint64)
    stack_d = cuda.local.array(shape=MAX_STACK_WARP, dtype=np.int16)
    sp = 0

    ns = n_seeds[warp]
    for i in range(ns):
        h = seeds[warp, i]
        if h >= 0:
            stack_h[sp] = np.uint64(h)
            stack_d[sp] = np.int16(dstart)
            sp += 1
            if sp >= MAX_STACK_WARP:
                overflow[warp] = 1
                return

    while sp > 0:
        sp -= 1
        h = stack_h[sp]
        d = int(stack_d[sp])

        vx, vy, vz = pix2vec_nest_center(h, d)
        dot = vx*cx + vy*cy + vz*cz
        shs = _shs_from_dot(dot)

        mn = min_abs[warp, d]
        mx = max_abs[warp, d]

        if shs <= mn:
            # FULL node: map/expand to out_order
            if d >= out_order:
                pix = h >> (2 * (d - out_order))
                idx = cuda.atomic.add(counter, 0, 1)
                if idx >= out_cap:
                    overflow[warp] = 1
                    return
                out_pix[idx] = np.uint64(pix)
            else:
                dd = out_order - d
                shift = 2 * dd
                start = h << shift
                n = 1 << (2 * dd)
                idx0 = cuda.atomic.add(counter, 0, n)
                if idx0 + n > out_cap:
                    overflow[warp] = 1
                    return
                for t in range(n):
                    out_pix[idx0 + t] = np.uint64(start + t)

        elif shs <= mx:
            # OVERLAP node
            if d == depth_eff:
                # accept leaf (mapped to out_order)
                if d >= out_order:
                    pix = h >> (2 * (d - out_order))
                    # parent-level re-test at out_order to avoid false positives
                    if d > out_order:
                        vx2, vy2, vz2 = pix2vec_nest_center(pix, out_order)
                        dot2 = vx2*cx + vy2*cy + vz2*cz
                        shs2 = _shs_from_dot(dot2)
                        mx2 = max_abs[warp, out_order]
                        if shs2 > mx2:
                            continue
                    idx = cuda.atomic.add(counter, 0, 1)
                    if idx >= out_cap:
                        overflow[warp] = 1
                        return
                    out_pix[idx] = np.uint64(pix)
                else:
                    dd = out_order - d
                    shift = 2 * dd
                    start = h << shift
                    n = 1 << (2 * dd)
                    idx0 = cuda.atomic.add(counter, 0, n)
                    if idx0 + n > out_cap:
                        overflow[warp] = 1
                        return
                    for t in range(n):
                        out_pix[idx0 + t] = np.uint64(start + t)
            else:
                # push 4 children
                child = h << 2
                nd = d + 1
                if sp + 4 >= MAX_STACK_WARP:
                    overflow[warp] = 1
                    return
                stack_h[sp] = np.uint64(child + 0); stack_d[sp] = np.int16(nd); sp += 1
                stack_h[sp] = np.uint64(child + 1); stack_d[sp] = np.int16(nd); sp += 1
                stack_h[sp] = np.uint64(child + 2); stack_d[sp] = np.int16(nd); sp += 1
                stack_h[sp] = np.uint64(child + 3); stack_d[sp] = np.int16(nd); sp += 1

        # else reject


# ----------------------------------------------------------------------
# GPU bitpack painter: atomic OR directly into per-coverage-pixel bitsets
# ----------------------------------------------------------------------

@cuda.jit

@cuda.jit(device=True, inline=True)
def _u64_low_mask(nbits):
    """Return low nbits bits set (uint64). nbits in [0, 64]."""
    if nbits <= 0:
        return np.uint64(0)
    if nbits >= 64:
        return np.uint64(0xFFFFFFFFFFFFFFFF)
    return (np.uint64(1) << np.uint64(nbits)) - np.uint64(1)

@cuda.jit(device=True, inline=True)
def _bitpack_or_range(bitpack_u64, row, a_off, b_off):
    """Atomic-OR set bits in [a_off, b_off] inclusive within a single covpix row."""
    # a_off, b_off: uint64 offsets within [0, fine_mask]
    w0 = int(a_off >> np.uint64(6))
    w1 = int(b_off >> np.uint64(6))
    b0 = int(a_off & np.uint64(63))
    b1 = int(b_off & np.uint64(63))

    if w0 == w1:
        m = _u64_low_mask(b1 + 1) & (~_u64_low_mask(b0))
        cuda.atomic.or_(bitpack_u64, (row, w0), m)
        return

    # first word
    m0 = ~_u64_low_mask(b0)
    cuda.atomic.or_(bitpack_u64, (row, w0), m0)

    # full words in between
    # Unrolled loop to reduce Numba loop overhead
    w = w0 + 1
    while w + 3 < w1:
        cuda.atomic.or_(bitpack_u64, (row, w    ), np.uint64(0xFFFFFFFFFFFFFFFF))
        cuda.atomic.or_(bitpack_u64, (row, w + 1), np.uint64(0xFFFFFFFFFFFFFFFF))
        cuda.atomic.or_(bitpack_u64, (row, w + 2), np.uint64(0xFFFFFFFFFFFFFFFF))
        cuda.atomic.or_(bitpack_u64, (row, w + 3), np.uint64(0xFFFFFFFFFFFFFFFF))
        w += 4
    while w < w1:
        cuda.atomic.or_(bitpack_u64, (row, w), np.uint64(0xFFFFFFFFFFFFFFFF))
        w += 1

    # last word
    m1 = _u64_low_mask(b1 + 1)
    cuda.atomic.or_(bitpack_u64, (row, w1), m1)


@cuda.jit
def paint_cone_pixels_kernel_bitpack_warp(
    cone_x, cone_y, cone_z,
    depth_start_arr, depth_eff,
    seeds, n_seeds,
    min_abs, max_abs,
    out_order,
    cov2row,               # int32 lookup table: covpix -> row index in bitpack (or -1)
    bitpack_u64,           # uint64 (n_cov_rows, words_per_cov)
    cov_order,
    delta_cov,             # out_order - cov_order
    missing_covpix,        # uint32 (n_cones,) count of pixels that mapped to missing covpix
    overflow               # uint8 (n_cones,) overflow / stack overflow flag
):
    """Single-pass traversal that *paints* into a GPU-resident bitpack buffer.

    - One warp per cone; lane0 traverses and emits pixels.
    - For each emitted pixel at out_order, compute (covpix, finepix) and atomicOr
      the corresponding bit in bitpack_u64[row, word].

    The bit order is little-endian within each 64-bit word (bit 0 is LSB), matching
    BITORD='L' conventions used by SkyMaskPipe's bitpack I/O.
    """
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    lane = cuda.threadIdx.x & 31
    warp = tid >> 5
    n_cones = cone_x.size
    if warp >= n_cones:
        return
    if lane != 0:
        return

    if overflow[warp] != 0:
        return

    cx = cone_x[warp]; cy = cone_y[warp]; cz = cone_z[warp]
    dstart = depth_start_arr[warp]

    stack_h = cuda.local.array(shape=MAX_STACK_WARP, dtype=np.uint64)
    stack_d = cuda.local.array(shape=MAX_STACK_WARP, dtype=np.int16)
    sp = 0

    ns = n_seeds[warp]
    for i in range(ns):
        h = seeds[warp, i]
        if h >= 0:
            stack_h[sp] = np.uint64(h)
            stack_d[sp] = np.int16(dstart)
            sp += 1
            if sp >= MAX_STACK_WARP:
                overflow[warp] = 1
                return

    # Precompute for mapping to coverage pixel bitpack
    # delta_cov = out_order - cov_order >= 0
    shift_cov = 2 * delta_cov
    # mask for fine-pixel offset within coverage pixel at out_order
    # NOTE: delta_cov can be up to ~14 in your use-cases; 1<<(2*delta_cov) fits in uint64 for delta_cov<=16
    fine_mask = (np.uint64(1) << np.uint64(shift_cov)) - np.uint64(1)

    while sp > 0:
        sp -= 1
        h = stack_h[sp]
        d = int(stack_d[sp])

        vx, vy, vz = pix2vec_nest_center(h, d)
        dot = vx*cx + vy*cy + vz*cz
        shs = _shs_from_dot(dot)

        mn = min_abs[warp, d]
        mx = max_abs[warp, d]

        if shs <= mn:
            # FULL node: map/expand to out_order and paint
            if d >= out_order:
                pix = h >> (2 * (d - out_order))
                covpix = np.uint64(pix) >> np.uint64(shift_cov)
                row = cov2row[np.int32(covpix)]
                if row < 0:
                    missing_covpix[warp] += 1
                else:
                    off = np.uint64(pix) & fine_mask
                    word = np.int64(off >> np.uint64(6))
                    bit  = np.uint64(off & np.uint64(63))
                    cuda.atomic.or_(bitpack_u64, (row, word), (np.uint64(1) << bit))
            else:
                # Phase 3: FULL node range-paint into uint64 word blocks instead of per-pixel atomics
                dd = out_order - d
                shift = 2 * dd
                start = np.uint64(h) << np.uint64(shift)
                n = np.uint64(1) << np.uint64(shift)   # 4^dd = 1<<(2*dd) = 1<<shift
                end = start + n - np.uint64(1)

                cov_start = start >> np.uint64(shift_cov)
                cov_end   = end   >> np.uint64(shift_cov)

                c0 = int(cov_start)
                c1 = int(cov_end)

                for covpix_i in range(c0, c1 + 1):
                    row = cov2row[np.int32(covpix_i)]
                    if covpix_i == c0:
                        a = start & fine_mask
                    else:
                        a = np.uint64(0)
                    if covpix_i == c1:
                        b = end & fine_mask
                    else:
                        b = fine_mask

                    if row < 0:
                        # Preserve previous semantics: count *pixels* dropped
                        missing_covpix[warp] += np.uint32(b - a + np.uint64(1))
                        continue

                    _bitpack_or_range(bitpack_u64, row, a, b)

        elif shs <= mx:
            # OVERLAP node
            if d == depth_eff:
                # accept leaf (mapped to out_order)
                if d >= out_order:
                    pix = h >> (2 * (d - out_order))
                    covpix = np.uint64(pix) >> np.uint64(shift_cov)
                    row = cov2row[np.int32(covpix)]
                    if row < 0:
                        missing_covpix[warp] += 1
                    else:
                        off = np.uint64(pix) & fine_mask
                        word = np.int64(off >> np.uint64(6))
                        bit  = np.uint64(off & np.uint64(63))
                        cuda.atomic.or_(bitpack_u64, (row, word), (np.uint64(1) << bit))
                else:
                    # Accepted leaf at depth_eff covers the full node area.
                    # When expanding to out_order, paint it as a contiguous range (word-fills)
                    # rather than per-pixel atomics to reduce atomic pressure.
                    dd = out_order - d
                    shift = 2 * dd
                    start = np.uint64(h) << np.uint64(shift)
                    n = np.uint64(1) << np.uint64(shift)   # 4^dd
                    end = start + n - np.uint64(1)

                    cov_start = start >> np.uint64(shift_cov)
                    cov_end   = end   >> np.uint64(shift_cov)

                    c0 = int(cov_start)
                    c1 = int(cov_end)

                    for covpix_i in range(c0, c1 + 1):
                        row = cov2row[np.int32(covpix_i)]
                        if covpix_i == c0:
                            a = start & fine_mask
                        else:
                            a = np.uint64(0)
                        if covpix_i == c1:
                            b = end & fine_mask
                        else:
                            b = fine_mask

                        if row < 0:
                            # Preserve previous semantics: count *pixels* dropped
                            missing_covpix[warp] += np.uint32(b - a + np.uint64(1))
                            continue

                        _bitpack_or_range(bitpack_u64, row, a, b)
            else:
                # push 4 children
                child = h << 2
                nd = d + 1
                if sp + 4 >= MAX_STACK_WARP:
                    overflow[warp] = 1
                    return
                stack_h[sp] = np.uint64(child + 0); stack_d[sp] = np.int16(nd); sp += 1
                stack_h[sp] = np.uint64(child + 1); stack_d[sp] = np.int16(nd); sp += 1
                stack_h[sp] = np.uint64(child + 2); stack_d[sp] = np.int16(nd); sp += 1
                stack_h[sp] = np.uint64(child + 3); stack_d[sp] = np.int16(nd); sp += 1

        # else reject


class GpuBitpackDictAccumulator:
    """Accumulate painted bitpack blocks on the CPU as a dict[covpix] -> uint64 words.

    This is a *bridge* accumulator:
      - painting is done on GPU into (n_cov_rows, words_per_cov) uint64 blocks
      - blocks are copied back and OR'ed into this dict
      - later you can serialize dict into SkyMaskPipe BITPACK FITS without expanding pixels

    Notes
    -----
    - cov_order is the HealSparse coverage order.
    - order_sparse is the target pixelization (out_order).
    """
    def __init__(self, order_sparse: int, cov_order: int):
        self.order_sparse = int(order_sparse)
        self.cov_order = int(cov_order)
        if self.order_sparse < self.cov_order:
            raise ValueError("order_sparse must be >= cov_order for bitpack painting.")
        self.delta_cov = self.order_sparse - self.cov_order
        self.words_per_cov = (1 << (2 * self.delta_cov)) // 64
        if (1 << (2 * self.delta_cov)) % 64 != 0:
            self.words_per_cov += 1
        self.blocks: dict[int, np.ndarray] = {}
        # Benchmark stats (optional)
        self.kernel_ms_total: float = 0.0
        self.kernel_batches: int = 0
        # Detailed timing splits (optional; populated when timing_split=True)
        self.timing: dict[str, float] = {}
        self.timing_counts: dict[str, int] = {}

    def or_blocks(self, covpix_list: np.ndarray, bitpack_u64_host: np.ndarray):
        """OR host bitpack blocks into accumulator dict."""
        covpix_list = np.asarray(covpix_list, dtype=np.int64)
        if bitpack_u64_host.ndim != 2 or bitpack_u64_host.shape[0] != covpix_list.size:
            raise ValueError("bitpack_u64_host must have shape (n_covpix, words_per_cov).")
        for i, cv in enumerate(covpix_list):
            cv = int(cv)
            row = bitpack_u64_host[i]
            if cv in self.blocks:
                self.blocks[cv] |= row
            else:
                self.blocks[cv] = row.copy()


def accumulate_discs_bitpack(
    acc: GpuBitpackDictAccumulator,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    radius_deg: np.ndarray,
    *,
    order: int,
    delta_depth: int = 2,
    stream_batch: int = 20000,
    covpix_list: np.ndarray,
    threads_per_block: int = 128,
    group_by_covpix: bool = True,
    keep_on_gpu: bool = True,
    benchmark_mode: bool = False,
    timing_split: bool = False,
):
    """Pixelize cones and paint them into a GPU bitpack buffer, then OR into `acc`.

    Phase 3.2A: reduce per-batch CPU work and host<->device transfers by precomputing
    per-cone metadata once per call and uploading it once, then slicing device arrays
    for each batch.

    Parameters
    ----------
    acc : GpuBitpackDictAccumulator
        The CPU-side dict accumulator.
    ra_deg, dec_deg, radius_deg : arrays
        Cone centers and radii (degrees).
    order : int
        Output order (must equal acc.order_sparse for consistent geometry).
    delta_depth : int
        Extra traversal depth (as in mocpy delta_depth).
    stream_batch : int
        Cones processed per GPU batch.
    covpix_list : array[int64]
        Coverage pixels (at acc.cov_order) that are allowed / preallocated for this paint call.
        MUST be a superset of all coverage pixels touched by these cones, otherwise pixels
        will be dropped (and missing_covpix will be >0).
    """
    if cp is None or cuda is None:
        raise ImportError("cupy and numba.cuda required.")
    if not _HAVE_HEALPY:
        raise ImportError("healpy required for seeding.")
    order = int(order)
    if order != int(acc.order_sparse):
        raise ValueError("accumulate_discs_bitpack requires order == acc.order_sparse.")
    cov_order = int(acc.cov_order)
    delta_cov = int(acc.delta_cov)

    ra_deg = np.asarray(ra_deg, dtype=np.float64)
    dec_deg = np.asarray(dec_deg, dtype=np.float64)
    radius_deg = np.asarray(radius_deg, dtype=np.float64)
    n = int(ra_deg.size)
    if n == 0:
        return

    covpix_list = np.asarray(covpix_list, dtype=np.int64)
    if covpix_list.size == 0:
        return

    # Optional detailed timing splits for diagnostics (no algorithm changes).
    _do_timing = bool(timing_split)
    if _do_timing:
        def _tadd(k: str, dt: float):
            acc.timing[k] = float(acc.timing.get(k, 0.0)) + float(dt)
        def _cadd(k: str, dv: int = 1):
            acc.timing_counts[k] = int(acc.timing_counts.get(k, 0)) + int(dv)
    else:
        def _tadd(k: str, dt: float):
            return
        def _cadd(k: str, dv: int = 1):
            return

    # Build covpix->row lookup (dense for speed; size is npix at cov_order)
    nside_cov = 1 << cov_order
    npix_cov = 12 * nside_cov * nside_cov

    # Optional: group cones by their center coverage pixel to improve locality.
    if group_by_covpix and _HAVE_HEALPY:
        _t0 = time.perf_counter() if _do_timing else 0.0
        theta_all = np.deg2rad(90.0 - dec_deg)
        phi_all = np.deg2rad(np.mod(ra_deg, 360.0))
        cov_center = hp.ang2pix(nside_cov, theta_all, phi_all, nest=True).astype(np.int64)
        sort_idx = np.argsort(cov_center, kind="mergesort")
        ra_deg = ra_deg[sort_idx]
        dec_deg = dec_deg[sort_idx]
        radius_deg = radius_deg[sort_idx]
        if _do_timing:
            _tadd('t_group_sort', time.perf_counter() - _t0)

    _t0 = time.perf_counter() if _do_timing else 0.0
    cov2row_host = np.full(npix_cov, -1, dtype=np.int32)
    for i, cv in enumerate(covpix_list):
        cvi = int(cv)
        if cvi < 0 or cvi >= npix_cov:
            raise ValueError(f"covpix {cvi} out of range for cov_order={cov_order}")
        cov2row_host[cvi] = np.int32(i)
    cov2row_dev = cp.asarray(cov2row_host)
    if _do_timing:
        _tadd('t_cov2row_h2d', time.perf_counter() - _t0)

    words_per_cov = int(acc.words_per_cov)

    # Keep a single global bitpack buffer on GPU across batches to avoid repeated allocations/transfers.
    use_global_bitpack = bool(keep_on_gpu)
    max_words_keep = int(100_000_000)  # ~800 MB of uint64; safety cap
    if use_global_bitpack and (int(covpix_list.size) * int(words_per_cov) > max_words_keep):
        use_global_bitpack = False

    if benchmark_mode:
        acc.kernel_ms_total = 0.0
        acc.kernel_batches = 0
        _bench_have_cp = (cp is not None)
    else:
        _bench_have_cp = False

    bitpack_global = None
    if use_global_bitpack:
        _t0 = time.perf_counter() if _do_timing else 0.0
        bitpack_global = cp.zeros((covpix_list.size, words_per_cov), dtype=cp.uint64)
        if _do_timing:
            _tadd('t_alloc_bitpack', time.perf_counter() - _t0)

    # ---- Phase 3.2A: precompute per-cone metadata ONCE (CPU) and upload ONCE (GPU) ----
    depth_eff = min(MAX_ORDER_SUPPORTED, order + int(delta_depth))
    radius_rad_all = np.deg2rad(radius_deg)

    # Seeds + depth_start on CPU
    _t0 = time.perf_counter() if _do_timing else 0.0
    depth_start, seeds, n_seeds = _build_seeds_fixed9(ra_deg, dec_deg, radius_rad_all, depth_eff)
    if _do_timing:
        _tadd('t_seed_cpu', time.perf_counter() - _t0)

    # Cone unit vectors on CPU
    _t0 = time.perf_counter() if _do_timing else 0.0
    cone_x, cone_y, cone_z = _radec_to_vec(ra_deg, dec_deg)
    if _do_timing:
        _tadd('t_vec_cpu', time.perf_counter() - _t0)

    # Min/max arrays on GPU (per cone)
    # Shape (n, depth_eff+1) to allow direct indexing by depth.
    lon_rad_all = np.deg2rad(np.mod(ra_deg, 360.0)).astype(np.float64)
    lat_rad_all = np.deg2rad(dec_deg).astype(np.float64)

    # Upload once
    _t0 = time.perf_counter() if _do_timing else 0.0
    cone_x_d = cp.asarray(cone_x)
    cone_y_d = cp.asarray(cone_y)
    cone_z_d = cp.asarray(cone_z)
    depth_start_d = cp.asarray(depth_start.astype(np.int16))
    seeds_d = cp.asarray(seeds.astype(np.int64))
    n_seeds_d = cp.asarray(n_seeds.astype(np.int8))
    lon_d = cp.asarray(lon_rad_all)
    lat_d = cp.asarray(lat_rad_all)
    radius_rad_d = cp.asarray(radius_rad_all.astype(np.float64))
    if _do_timing:
        _tadd('t_h2d_meta', time.perf_counter() - _t0)

    min_abs_d = cp.empty((n, depth_eff + 1), dtype=cp.float64)
    max_abs_d = cp.empty((n, depth_eff + 1), dtype=cp.float64)

    _t0 = time.perf_counter() if _do_timing else 0.0
    threads_mm = 256
    blocks_mm = (n + threads_mm - 1) // threads_mm
    if _do_timing:
        _ev_mm0 = cp.cuda.Event(); _ev_mm1 = cp.cuda.Event()
        _ev_mm0.record()
    build_shs_minmax_abs_kernel[blocks_mm, threads_mm](
        depth_start_d,
        np.int32(depth_eff),
        lon_d,
        lat_d,
        radius_rad_d,
        min_abs_d,
        max_abs_d,
    )
    if _do_timing:
        _ev_mm1.record(); _ev_mm1.synchronize()
        _tadd('t_minmax_gpu', 1e-3 * float(cp.cuda.get_elapsed_time(_ev_mm0, _ev_mm1)))

    # Allocate missing/overflow arrays ONCE and check ONCE at the end (avoid per-batch sync).
    _t0 = time.perf_counter() if _do_timing else 0.0
    missing_all_d = cp.zeros(n, dtype=cp.uint32)
    overflow_all_d = cp.zeros(n, dtype=cp.uint8)
    if _do_timing:
        _tadd('t_alloc_checks_buf', time.perf_counter() - _t0)

    # Process in batches (slicing GPU arrays only)
    for b0 in range(0, n, int(stream_batch)):
        b1 = min(n, b0 + int(stream_batch))
        n_cones = int(b1 - b0)

        # Select bitpack target: global (persistent) or local (per-batch)
        if bitpack_global is None:
            _t0 = time.perf_counter() if _do_timing else 0.0
            bitpack_d = cp.zeros((covpix_list.size, words_per_cov), dtype=cp.uint64)
            if _do_timing:
                _tadd('t_alloc_bitpack_local', time.perf_counter() - _t0)
        else:
            bitpack_d = bitpack_global

        # Slices (device views, no host transfers)
        cone_x_b = cone_x_d[b0:b1]
        cone_y_b = cone_y_d[b0:b1]
        cone_z_b = cone_z_d[b0:b1]
        depth_start_b = depth_start_d[b0:b1]
        seeds_b = seeds_d[b0:b1]
        n_seeds_b = n_seeds_d[b0:b1]
        min_abs_b = min_abs_d[b0:b1]
        max_abs_b = max_abs_d[b0:b1]
        missing_b = missing_all_d[b0:b1]
        overflow_b = overflow_all_d[b0:b1]

        # Launch kernel: one warp per cone
        warps_per_block = threads_per_block // 32
        blocks = (n_cones + warps_per_block - 1) // warps_per_block
        if _bench_have_cp or _do_timing:
            _ev0 = cp.cuda.Event(); _ev1 = cp.cuda.Event()
            _ev0.record()

        paint_cone_pixels_kernel_bitpack_warp[blocks, threads_per_block](
            cone_x_b, cone_y_b, cone_z_b,
            depth_start_b, np.int32(depth_eff),
            seeds_b, n_seeds_b,
            min_abs_b, max_abs_b,
            np.int32(order),
            cov2row_dev,
            bitpack_d,
            np.int32(cov_order),
            np.int32(delta_cov),
            missing_b,
            overflow_b,
        )

        if _bench_have_cp or _do_timing:
            _ev1.record(); _ev1.synchronize()
            _k_ms = float(cp.cuda.get_elapsed_time(_ev0, _ev1))
            if _bench_have_cp:
                acc.kernel_ms_total += _k_ms
                acc.kernel_batches += 1
            if _do_timing:
                _tadd('t_kernel', 1e-3 * _k_ms)
                _cadd('kernel_batches', 1)

        # Bring painted blocks back and OR into accumulator if not using persistent GPU bitpack
        if bitpack_global is None:
            _t0 = time.perf_counter() if _do_timing else 0.0
            bitpack_host = bitpack_d.get()
            if _do_timing:
                _tadd('t_d2h_bitpack', time.perf_counter() - _t0)
            _t0 = time.perf_counter() if _do_timing else 0.0
            acc.or_blocks(covpix_list, bitpack_host)
            if _do_timing:
                _tadd('t_or_cpu', time.perf_counter() - _t0)

    # One-time correctness check (avoid per-batch synchronization)
    _t0 = time.perf_counter() if _do_timing else 0.0
    miss = int(cp.sum(missing_all_d).get())
    of = int(cp.sum(overflow_all_d).get())
    if _do_timing:
        _tadd('t_checks', time.perf_counter() - _t0)
    if miss != 0:
        raise RuntimeError(
            f"[bitpack] missing covpix mappings: {miss} pixels dropped "
            f"(covpix_list too small / not a superset)."
        )
    if of != 0:
        raise RuntimeError(f"[bitpack] overflow/stack overflow in {of} cones")

    # Finalize: if using a persistent GPU bitpack, copy back once and OR into accumulator.
    if bitpack_global is not None:
        _t0 = time.perf_counter() if _do_timing else 0.0
        bitpack_host = bitpack_global.get()
        if _do_timing:
            _tadd('t_d2h_bitpack', time.perf_counter() - _t0)
        _t0 = time.perf_counter() if _do_timing else 0.0
        acc.or_blocks(covpix_list, bitpack_host)
        if _do_timing:
            _tadd('t_or_cpu', time.perf_counter() - _t0)

    if _do_timing:
        _cadd('cones_total', n)
        _cadd('covpix_total', int(covpix_list.size))



def _radec_to_vec(ra_deg: np.ndarray, dec_deg: np.ndarray):
    ra = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
    cdec = np.cos(dec)
    x = cdec * np.cos(ra)
    y = cdec * np.sin(ra)
    z = np.sin(dec)
    return x, y, z

def _build_seeds_healpy(ra_deg, dec_deg, radius_rad, depth_eff):
    """Build per-cone depth_start and seed set following cdshealpix fast-start logic.

    Vectorized version:
      - compute depth_start for all cones using the SMALLER_EDGE2OPEDGE_DIST table
      - group by depth_start and call healpy ang2pix/get_all_neighbours on arrays

    Returns
    -------
    depth_start : int array (n,)
    seeds : int64 array (n, MAX_SEEDS)
    n_seeds : int8 array (n,)
    """
    if not _HAVE_HEALPY:
        raise ImportError("healpy is required for v1 seeding (ang2pix/get_all_neighbours).")

    ra_deg = np.asarray(ra_deg, dtype=np.float64)
    dec_deg = np.asarray(dec_deg, dtype=np.float64)
    radius_rad = np.asarray(radius_rad, dtype=np.float64)

    n = ra_deg.size
    depth_start = np.empty(n, dtype=np.int16)
    seeds = np.full((n, MAX_SEEDS), -1, dtype=np.int64)
    n_seeds = np.zeros(n, dtype=np.int8)

    # Large cones: cdshealpix seeds 12 base faces (depth 0)
    # Small cones: pick best_starting_depth from the table.
    # Table is decreasing with depth; cdshealpix picks the deepest depth such that d_max < table[depth].
    table = SMALLER_EDGE2OPEDGE_DIST
    small = radius_rad < float(table[0])

    if np.any(small):
        rev = table[::-1]  # increasing
        # idx = number of elements <= d (in rev); depth = 29 - idx
        idx = np.searchsorted(rev, radius_rad[small], side="right").astype(np.int32)
        ds = (29 - idx).astype(np.int16)
        ds = np.clip(ds, 0, 29).astype(np.int16)
        depth_start[small] = ds

    if np.any(~small):
        depth_start[~small] = 0
        # seed 12 base faces
        ii = np.where(~small)[0]
        # Fill first 12 slots with 0..11
        seeds[ii, :12] = np.arange(12, dtype=np.int64)[None, :]
        n_seeds[ii] = 12

    # Now seed center + neighbours for small cones, grouped by depth_start
    if np.any(small):
        theta_all = np.deg2rad(90.0 - dec_deg)         # colat
        phi_all = np.deg2rad(np.mod(ra_deg, 360.0))    # lon

        small_idx = np.where(small)[0]
        ds_small = depth_start[small_idx].astype(np.int32)

        for ds in np.unique(ds_small):
            idx_ds = small_idx[ds_small == ds]
            if idx_ds.size == 0:
                continue
            nside = 1 << int(ds)
            ip = hp.ang2pix(nside, theta_all[idx_ds], phi_all[idx_ds], nest=True)  # (m,)
            neigh = hp.get_all_neighbours(nside, ip, nest=True)                    # (8, m)
            neigh = np.asarray(neigh).T                                            # (m, 8)

            seeds[idx_ds, 0] = ip.astype(np.int64)
            # Fill neighbours; keep -1 as-is for missing
            seeds[idx_ds, 1:9] = neigh.astype(np.int64)

            # n_seeds = 1 + count(neigh >= 0)
            n_seeds[idx_ds] = (1 + np.sum(neigh >= 0, axis=1)).astype(np.int8)

    return depth_start, seeds, n_seeds


# ---- Phase 3.2B: fixed 9-neighbor seeding (no SHS pre-filter) ----
# The seed builder below is equivalent to the cdshealpix "center + 8 neighbours" start,
# but we intentionally skip the expensive Rust-style SHS max pre-filter to reduce CPU overhead.
_build_seeds_fixed9 = _build_seeds_healpy


def _filter_seeds_by_shsmax(ra_deg, dec_deg, radius_rad, depth_start, seeds, n_seeds):
    """Apply the Rust pre-filter on the seed neighborhood:
       keep seed if shs(center(seed), cone_center) <= shs(radius + d_start)

    Vectorized version:
      - group by depth_start to call healpy.pix2vec on arrays (reduces Python overhead)
      - still computes per-cone shs_max on CPU (cheap vs many pix2vec calls)
    """
    if not _HAVE_HEALPY:
        raise ImportError("healpy is required for v1 seeding (pix2vec).")

    ra_deg = np.asarray(ra_deg, dtype=np.float64)
    dec_deg = np.asarray(dec_deg, dtype=np.float64)
    radius_rad = np.asarray(radius_rad, dtype=np.float64)

    n = ra_deg.size
    if n == 0:
        return

    # Compute cone vectors once
    cx, cy, cz = _radec_to_vec(ra_deg, dec_deg)

    # Precompute shs_max per cone (still CPU; modest cost)
    shs_max_arr = np.empty(n, dtype=np.float64)
    skip_filter = np.zeros(n, dtype=bool)
    for i in range(n):
        ds = int(depth_start[i])
        # Large cones seeded with 12 faces: skip filter (matches previous behavior)
        if int(n_seeds[i]) == 12 and ds == 0 and not has_best_starting_depth(float(radius_rad[i])):
            skip_filter[i] = True
            shs_max_arr[i] = 1.0  # unused
            continue
        lon = math.radians(float(ra_deg[i]) % 360.0)
        lat = math.radians(float(dec_deg[i]))
        r = float(radius_rad[i])
        d = largest_center_to_vertex_distance_with_radius(ds, lon, lat, r)
        shs_max_arr[i] = _to_squared_half_segment(r + d)

    # Process by depth_start groups for efficient pix2vec
    ds_all = depth_start.astype(np.int32)
    for ds in np.unique(ds_all):
        idx = np.where(ds_all == ds)[0]
        if idx.size == 0:
            continue

        # Split into those we skip and those we filter
        idx_f = idx[~skip_filter[idx]]
        if idx_f.size == 0:
            continue

        # Extract up to 9 seeds (center + 8 neighbours) for these cones
        sblock = seeds[idx_f, :9].astype(np.int64)  # (m, 9)
        valid = sblock >= 0
        if not np.any(valid):
            # all empty
            seeds[idx_f, :] = -1
            n_seeds[idx_f] = 0
            continue

        # Flatten valid pixels and parent mapping
        pix_flat = sblock[valid]
        # parent indices repeated by positions
        parent_flat = np.repeat(idx_f, 9)[valid.ravel()]

        nside = 1 << int(ds)
        vx, vy, vz = hp.pix2vec(nside, pix_flat, nest=True)  # arrays (k,)
        dot = vx * cx[parent_flat] + vy * cy[parent_flat] + vz * cz[parent_flat]
        dot = np.clip(dot, -1.0, 1.0)
        shs = 0.5 * (1.0 - dot)
        keep_flat = shs <= shs_max_arr[parent_flat]

        # Build keep matrix aligned to sblock positions
        keep_mat = np.zeros_like(sblock, dtype=bool)
        keep_mat[valid] = keep_flat

        # Compact per row (m <= 20000, width=9 -> cheap)
        seeds[idx_f, :] = -1
        new_n = np.zeros(idx_f.size, dtype=np.int8)
        for row in range(idx_f.size):
            kept = sblock[row, keep_mat[row]]
            nk = min(kept.size, MAX_SEEDS)
            if nk:
                seeds[idx_f[row], :nk] = kept[:nk]
            new_n[row] = nk
        n_seeds[idx_f] = new_n

__all__ = ['is_gpu_available', 'GpuBitpackDictAccumulator', 'accumulate_discs_bitpack']


