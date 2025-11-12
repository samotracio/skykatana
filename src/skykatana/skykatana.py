import numpy as np, matplotlib.pyplot as plt
import healsparse as hsp, healpy as hp, pandas as pd
from astropy.table import Table, join
from astropy.coordinates import Angle, Latitude, Longitude, SkyCoord
import astropy.units as u
import lsdb
from mocpy import MOC, WCS
from tqdm import tqdm
import re, json, os, shutil, tempfile, fitsio, gc, math, threading
from pathlib import Path, PosixPath
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Tuple, Union, Optional, Any, Dict, Sequence, List
from numpy.typing import NDArray
from matplotlib.axes import Axes
from healsparse import HealSparseMap
from matplotlib.figure import Figure
from numpy.random import RandomState
from reproject import reproject_from_healpix
from scipy.ndimage import gaussian_filter
# Supress info msgs from dask -> distributed.core INFO: Event loop was unresponsive in Nanny ...
# which repeats a lot during pixelization of circles
import logging
logging.getLogger('distributed.core').setLevel(logging.ERROR)




# Numba auxiliary kernels (compiled)
try:
    from numba import njit
except Exception:  # no numba available
    def njit(*args, **kwargs):
        def wrap(f): return f
        return wrap

# popcount lookup (0..255)
_POPCNT_U8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

@njit(cache=True)
def _sum_popcount_u8(arr_u8):
    s = 0
    for i in range(arr_u8.size):
        s += _POPCNT_U8[arr_u8[i]]
    return s

@njit(cache=True)
def _expand_bitpack_row(base, packed_u8, out_pix, pos):
    """
    Expand one BITPACK row into out_pix starting at pos.
    Returns new pos. (Little-endian within each byte.)
    """
    for bi in range(packed_u8.size):
        b = packed_u8[bi]
        if b == 0:
            continue
        off = base + bi * 8
        # unrolled bit tests are faster under numba
        if b & 1:   out_pix[pos]; out_pix[pos] = off + 0; pos += 1
        if b & 2:   out_pix[pos]; out_pix[pos] = off + 1; pos += 1
        if b & 4:   out_pix[pos]; out_pix[pos] = off + 2; pos += 1
        if b & 8:   out_pix[pos]; out_pix[pos] = off + 3; pos += 1
        if b & 16:  out_pix[pos]; out_pix[pos] = off + 4; pos += 1
        if b & 32:  out_pix[pos]; out_pix[pos] = off + 5; pos += 1
        if b & 64:  out_pix[pos]; out_pix[pos] = off + 6; pos += 1
        if b & 128: out_pix[pos]; out_pix[pos] = off + 7; pos += 1
    return pos


def _iter_covpix(hspmap):
    # Auxiliary iterator over coverage pixels
    cov_mask = hspmap.coverage_mask
    return np.where(cov_mask)[0]

def _iter_valid_by_covpix(hspmap):
    """
    Iterator over valid pixels per coverage pixel
    """
    for covpix in _iter_covpix(hspmap):
        arr = hspmap.valid_pixels_single_covpix(int(covpix))
        if arr.size > 1 and (arr[1:] < arr[:-1]).any():
            arr.sort()
        yield int(covpix), arr


def _read_stage_fits_bitpack_fast(stage_path: str | os.PathLike, *,
        io_block_rows: int = 200_000, per_worker_buffer_cap: int = 12_000_000,
        verbose: bool = False, print_lock=None,
        stage_name: str = "") -> hsp.HealSparseMap:
    """
    Load a SkyMaskPipe stage from a bit-packed FITS table using fast streaming I/O.

    This function reconstructs a `HealSparseMap` from a stage FITS file. Data are
    read in row blocks, unpacked from the bit-packed representation, and assembled
    into a boolean map.

    Parameters
    ----------
    stage_path : str or os.PathLike
        Path to the FITS stage file to read.
    io_block_rows : int, default=200_000
        Number of FITS table rows to read per block. Larger values improve
        throughput but increase memory usage.
    per_worker_buffer_cap : int, default=12_000_000
        Maximum number of pixels buffered per worker during unpacking.
    verbose : bool, default=False
        If True, prints progress messages during load. Normally False;
    print_lock : threading.Lock, optional
        Shared lock to synchronize console output when called in parallel contexts.
        Only relevant if `verbose=True`.
    stage_name : str, optional
        Name of the stage being loaded (for logging purposes only).

    Returns
    -------
    hspmap : healsparse.HealSparseMap
        The reconstructed HealSparse map with boolean or bit-packed data.

    Notes
    -----
    - Pixel data are decoded from the `PB` column and expanded to full
      boolean arrays, preserving geometry from FITS headers.
    - This function is optimized for speed and is designed to be run
      concurrently from multiple worker threads.
    """
    with fitsio.FITS(str(stage_path), mode='r') as f:
        tble = f[1]
        hdr  = tble.read_header()

        if str(hdr.get('ENCOD', 'BITPACK')).upper() != 'BITPACK':
            raise ValueError(f"{stage_path} is not BITPACK-encoded.")
        if str(hdr.get('BITORD', 'L')).upper() != 'L':
            raise ValueError("Only BITORD='L' supported.")

        nside_cov = int(hdr['NSIDE_COV'])
        nside_spa = int(hdr['NSIDE_SPA'])
        nfine     = int(hdr['NFINE'])

        out = hsp.HealSparseMap.make_empty(
            nside_coverage=nside_cov,
            nside_sparse=nside_spa,
            dtype=np.bool_,
            bit_packed=True,
        )

        nrows = tble.get_nrows()
        if nrows == 0:
            return out

        # large work buffer (int64 of child pixel ids)
        buf  = np.empty(per_worker_buffer_cap, dtype=np.int64)
        bpos = 0

        def _update_map(pix):
            # healsparse API differences: accept both scalar-bool and array-bool
            try:
                out.update_values_pix(pix, True, operation="replace")
            except TypeError:
                if not hasattr(_update_map, "_ones") or _update_map._ones.size < pix.size:
                    _update_map._ones = np.ones(pix.size, dtype=np.bool_)
                out.update_values_pix(pix, _update_map._ones[:pix.size], operation="replace")

        def flush():
            nonlocal bpos
            if bpos:
                _update_map(buf[:bpos])
                bpos = 0

        block = int(io_block_rows)
        for start in range(0, nrows, block):
            stop = min(start + block, nrows)
            rows_idx = np.arange(start, stop, dtype='i8')
            blk = tble.read(columns=['COVPIX', 'PACKED'], rows=rows_idx)

            covpix = blk['COVPIX'].astype(np.int64, copy=False)
            packed = blk['PACKED']  # object array of uint8 arrays

            # Pre-size check for this block (to reduce flush calls)
            est = 0
            for i in range(covpix.size):
                pk = packed[i]
                if pk.size:
                    est += int(_sum_popcount_u8(pk))
            if est and (bpos + est > buf.size):
                flush()

            # Expand rows into buffer
            for i in range(covpix.size):
                pk = packed[i]
                if pk.size == 0:
                    continue
                base = int(covpix[i]) * nfine
                need = int(_sum_popcount_u8(pk))
                if need == 0:
                    continue

                # If a single row is larger than our buffer, stream it in place
                if need > buf.size:
                    w = 0
                    w = _expand_bitpack_row(base, pk, buf, w)
                    _update_map(buf[:w])
                    continue

                # Ensure capacity
                if bpos + need > buf.size:
                    flush()
                bpos = _expand_bitpack_row(base, pk, buf, bpos)

        flush()

    if verbose:
        msg = f"[{stage_name}] Loaded {stage_path.name}"
        if print_lock:
            with print_lock:
                print(msg, flush=True)
        else:
            print(msg, flush=True)

    return out



def _write_stage_fits_bitpack(stage_path: str | os.PathLike, hspmap,
                              rows_per_batch: int = 32768) -> None:
    """
    Write a single SkyMaskPipe stage to disk as a bit-packed FITS table.

    This function serializes the pixel data of a `HealSparseMap` into a streaming
    FITS file, using a bit-packed boolean representation to reduce disk space.
    Data are written in batches of rows to balance throughput and memory footprint.

    Parameters
    ----------
    stage_path : str or os.PathLike
        Output file path for the FITS stage.
    hspmap : healsparse.HealSparseMap
        The stage map to be serialized. Must be a boolean or bit-packed boolean map
    rows_per_batch : int, default=8192
        Number of FITS table rows written per batch. Higher values may improve
        I/O performance at the cost of larger memory usage.

    Returns
    -------
    None
        Writes the stage to `stage_path`

    Notes
    -----
    - The output FITS file contains:
        * Coverage pixel index (COVPIX)
        * Encoded occupancy (ENC = 1 if any children are set)
        * Bit-packed pixel data (PB column)
    - The header stores geometry information, including `NSIDE_COV`,
      `NSIDE_SPA`, `NFINE`, and encoding flags.
    - Bit order is fixed (`BITORD = 'L'`) for reproducibility.
    """
    # --- sanity
    if not np.issubdtype(hspmap.dtype, np.bool_):
        raise TypeError(f"Expected boolean HealSparseMap; got dtype={hspmap.dtype!r}")

    nside_cov = int(hspmap.nside_coverage)
    nside_spa = int(hspmap.nside_sparse)
    if nside_spa % nside_cov != 0:
        raise ValueError("nside_sparse must be a multiple of nside_coverage (NESTED).")
    ratio = nside_spa // nside_cov
    nfine = ratio * ratio  # fine pixels per coverage pixel

    # Choose row iterator. The default _iter_valid_by_covpix() is fast, but provide
    # a fallback case if doesn't work as planned
    it = globals().get("_iter_valid_by_covpix")
    if callable(it):
        def _iter_rows():
            # yields (covpix:int64, fine_pix_sorted: np.ndarray[int64])
            yield from it(hspmap)
    else:
        def _iter_rows():
            vp = hspmap.valid_pixels
            if vp.size == 0:
                return
            covpix = vp // nfine
            order = np.argsort(covpix, kind="mergesort")
            vp = vp[order]
            covpix = covpix[order]
            i, N = 0, vp.size
            while i < N:
                j = i
                cv = int(covpix[i])
                while j < N and covpix[j] == cv:
                    j += 1
                yield cv, vp[i:j]
                i = j

    # Quick empty check (without buffering a batch)
    has_any = False
    for _peek in _iter_rows():
        has_any = True
        break
    if not has_any:
        with fitsio.FITS(str(stage_path), mode="rw", clobber=True) as f:
            f.create_table_hdu(
                names=['COVPIX', 'ENC', 'PACKED'],
                formats=['K', 'B', 'PB()'],
            )
            f[-1].write_keys({
                'NSIDE_COV': nside_cov,
                'NSIDE_SPA': nside_spa,
                'DTYPE':     'bool',
                'ENCOD':     'BITPACK',
                'NFINE':     int(nfine),
                'BITORD':    'L',
            })
        return

    # Recreate iterator after peek
    if callable(it):
        def _iter_rows():
            yield from it(hspmap)
    else:
        def _iter_rows():
            vp = hspmap.valid_pixels
            covpix = vp // nfine
            order = np.argsort(covpix, kind="mergesort")
            vp = vp[order]
            covpix = covpix[order]
            i, N = 0, vp.size
            while i < N:
                j = i
                cv = int(covpix[i])
                while j < N and covpix[j] == cv:
                    j += 1
                yield cv, vp[i:j]
                i = j

    def _pack_offsets_to_bytes(off_sorted: np.ndarray) -> np.ndarray:
        """Pack 1-bit offsets into little-endian bytes without allocating width-sized bit arrays."""
        if off_sorted.size == 0:
            return np.empty(0, dtype=np.uint8)
        width = int(off_sorted[-1]) + 1
        byte_len = (width + 7) // 8
        byte_idx = (off_sorted >> 3).astype(np.int64)   # which byte each bit goes in
        bit_pos  = (off_sorted & 7).astype(np.int64)    # 0..7 within byte
        contrib  = (1 << bit_pos).astype(np.uint16)     # ≤255; use u16 to avoid overflow in add.at
        out_u16 = np.zeros(byte_len, dtype=np.uint16)
        np.add.at(out_u16, byte_idx, contrib)           # sums act like OR since no duplicates
        return out_u16.astype(np.uint8, copy=False)

    # Create fits table (explicit formats => no dtype inference)
    with fitsio.FITS(str(stage_path), mode="rw", clobber=True) as f:
        f.create_table_hdu(
            names=['COVPIX', 'ENC', 'PACKED'],
            formats=['K', 'B', 'PB()'],
        )
        f[-1].write_keys({
            'NSIDE_COV': nside_cov,
            'NSIDE_SPA': nside_spa,
            'DTYPE':     'bool',
            'ENCOD':     'BITPACK',
            'NFINE':     int(nfine),
            'BITORD':    'L',   # little-endian bit order
        })

        # batch buffers
        batch_cov, batch_enc, batch_packed = [], [], []

        def _flush():
            if not batch_cov:
                return
            rec = {
                'COVPIX': np.asarray(batch_cov, dtype='i8'),
                'ENC':    np.asarray(batch_enc, dtype='u1'),     # all 1s
                'PACKED': np.array(batch_packed, dtype=object), # elems: np.ndarray(uint8)
            }
            f[-1].append(rec)
            batch_cov.clear(); batch_enc.clear(); batch_packed.clear()

        # Finally, stream rows into disk ---------------------------------
        in_batch = 0
        for covpix, fine_pix in _iter_rows():
            if fine_pix.size == 0:
                continue
            base = covpix * nfine
            off = (fine_pix - base).astype(np.int64)  # sorted by iterator
            packed = _pack_offsets_to_bytes(off)      # minimal byte array
            # enqueue
            batch_cov.append(int(covpix))
            batch_enc.append(1)
            batch_packed.append(packed)               # already uint8
            in_batch += 1
            if in_batch >= rows_per_batch:
                _flush()
                in_batch = 0

        _flush()



def getarea_moc(moc: "MOC") -> float:
    """
    Get the area of a MOC in deg^2
    """
    ipix = moc.flatten()
    return ipix.size * hp.nside2pixarea(1<<moc.max_order, degrees=True)

def pixel_area_deg2(order: int) -> float:
    """
    Get the area of a pixel at a given order
    """
    nside = 1<<order
    return hp.nside2pixarea(nside, degrees=True)

def pick_coarse_order(target_deg2: float, pixels_per_chunk: int = 6) -> int:
    """
    Choose a coarse order k so that the area of a single pixel is ~(target_deg2/pixels_per_chunk).
    This yields ~pixels_per_chunk, i.e. the number of pixels per chunk to pick, keeping chunks compact.
    """
    FULL_SKY_DEG2 = 41252.96125
    desired = target_deg2 / float(pixels_per_chunk)
    rhs = FULL_SKY_DEG2 / (12.0 * desired)  # 4^k ~ FULL_SKY / (12*desired)
    if rhs <= 1:
        k = 0
    else:
        k = int(max(0, round(math.log(rhs, 4))))

    while pixel_area_deg2(k) > desired and k < 29:
        k += 1
    while k > 0 and pixel_area_deg2(k - 1) <= desired:
        k -= 1
    return k

def hp_neighbors_present(nside: int, ipix: np.ndarray, present: set) -> list[list[int]]:
    """
    For each ipix (NESTED scheme), return neighbor list filtered to those in 'present'.
    Uses healpy.get_all_neighbours with nest=True and the pixel-ID overload (positional).
    """
    ipix = np.asarray(ipix, dtype=np.int64)
    # NOTE: pass pixel IDs as 2nd positional arg; set nest=True (NESTED indexing).
    neigh = hp.get_all_neighbours(nside, ipix, nest=True)  # shape (8, N)
    out = []
    for j in range(ipix.size):
        ns = neigh[:, j]
        ns = ns[ns >= 0]  # drop -1 sentinels if any
        out.append([int(q) for q in ns if int(q) in present])
    return out

def contiguous_chunks_from_flat(ipix: np.ndarray,
                                order: int,
                                target_deg2: float,
                                min_chunk_frac: float = 0.6):
    """
    Partition uniform-order HEALPix pixels (NEST) into contiguous chunks of
    area roughly <= target_deg2 using greedy BFS, i.e. breadth-first
    region grower that greedily stop adding neighbors when target area is reached
    """
    ipix = np.asarray(ipix, dtype=np.int64)
    if ipix.size == 0:
        return []

    nside = 1<<order
    px_area = pixel_area_deg2(order)
    present = set(ipix.tolist())

    neighs = hp_neighbors_present(nside, ipix, present)
    index_of = {int(p): i for i, p in enumerate(ipix)}

    # Connected components
    comp_id = -np.ones(ipix.size, dtype=np.int64)
    comp_list = []
    cid = 0
    for i in range(ipix.size):
        if comp_id[i] >= 0:
            continue
        stack = [i]
        comp = [i]
        comp_id[i] = cid
        while stack:
            j = stack.pop()
            for q in neighs[j]:
                qj = index_of[q]
                if comp_id[qj] < 0:
                    comp_id[qj] = cid
                    stack.append(qj)
                    comp.append(qj)
        comp_list.append(np.array(comp, dtype=np.int64))
        cid += 1

    chunks = []

    def bfs_seed(seed_idx, visited_mask):
        frontier = [seed_idx]
        visited_mask[seed_idx] = True
        current_pix = [ipix[seed_idx]]
        cur_area = px_area
        target = target_deg2

        while frontier:
            i = frontier.pop()
            for q in neighs[i]:
                qi = index_of[q]
                if not visited_mask[qi]:
                    if cur_area + px_area > target and cur_area >= min_chunk_frac * target:
                        continue
                    visited_mask[qi] = True
                    frontier.append(qi)
                    current_pix.append(q)
                    cur_area += px_area
        return np.asarray(current_pix, dtype=np.int64), cur_area

    for comp in comp_list:
        visited = np.zeros(ipix.size, dtype=bool)
        remain = set(comp.tolist())
        while remain:
            seed_idx = next(iter(remain))
            chunk_pixels, _ = bfs_seed(seed_idx, visited)
            for p in chunk_pixels:
                remain.discard(index_of[int(p)])
            chunks.append((order, chunk_pixels))

        # Merge tiny tail into previous
        if len(chunks) >= 2:
            last_order, last_pixels = chunks[-1]
            if last_pixels.size * px_area < min_chunk_frac * target_deg2:
                prev_order, prev_pixels = chunks[-2]
                if prev_order == last_order:
                    merged = np.unique(np.concatenate([prev_pixels, last_pixels]))
                    chunks[-2] = (prev_order, merged)
                    chunks.pop()

    return chunks

# min_chunk_frac was 0.6
def split_moc_into_chunks(moc: MOC,
                          target_deg2: float = 1000.0,
                          coarse_order: int | None = None,
                          pixels_per_chunk: int = 6,
                          min_chunk_frac: float = 0.8) -> list[MOC]:
    """
    Split a MOC into  aprox. contiguous chunks of ≲ target_deg2 each, by degrading
    to a coarse order and add neighbor pixels to seed locations.
    """
    if coarse_order is None:
        coarse_order = pick_coarse_order(target_deg2, pixels_per_chunk=pixels_per_chunk)

    moc_coarse = moc.degrade_to_order(coarse_order)
    ipix = np.asarray(moc_coarse.flatten(), dtype=np.int64)

    if ipix.size == 0:
        return []

    chunks = contiguous_chunks_from_flat(ipix, order=coarse_order,
                                         target_deg2=target_deg2,
                                         min_chunk_frac=min_chunk_frac)

    out = []
    for ord_k, pix in chunks:
        out.append(
            MOC.from_healpix_cells(
                ipix=pix.astype(np.uint64),
                depth=np.uint8(ord_k),
                max_depth=ord_k
            )
        )
    return out


#####################################################################
########################  CLASS DEFINITION  #########################
#####################################################################
class SkyMaskPipe:
    """
    SkyMaskPipe is a class to work with boolean sky masks in a pipeline way

    Attributes
    ----------
    order_cov : int
        Default coverage order of (all) healsparse maps.
    order_out : int
        Default sparse order of combining healsparse maps.
    """

    # Class scalar attributes to be saved in JSON file
    _SCALAR_ATTRS = ["order_cov", "order_out"]
    _BITPACK_FITS_VERSION = 1   # File format version in case we update

    
    def __init__(self, **kwargs):
        # Default values for orders, when not provided
        self.order_out = kwargs.get('order_out', 15)
        self.order_cov = kwargs.get('order_cov', 4)
        self.nside_out       = 1<<self.order_out
        self.nside_cov       = 1<<self.order_cov
        self.footmask        = None
        self.propmask        = None
        self.starmask        = None
        self.circmask        = None
        self.boxmask         = None
        self.ellipmask       = None
        self.polymask        = None
        self.zonemask        = None
        self.mwmask          = None
        self.mask            = None
        self._params: Dict[str, Dict[str, Any]] = {}  # all stage params live here


    def _is_healsparse_map(self, obj) -> bool:
        """
        Return True if `obj` is a real HealSparseMap or a duck-typed equivalent.
        """
        if obj is None:
            return False
        try:
            import healsparse as hsp
        except Exception:
            hsp = None
        if hsp is not None and isinstance(obj, hsp.HealSparseMap):
            return True
        # Duck-typing: minimal surface area used elsewhere in the class
        required = ("nside_coverage", "nside_sparse", "get_valid_area", "n_valid")
        return all(hasattr(obj, attr) for attr in required)


    def _discover_stage_items(self):
        """
        Return a sorted list of (name, map) for attributes that look like stages:
        names ending with 'mask' and passing _is_healsparse_map().
        """
        found = []
        for name in dir(self):
            if name.startswith("_"):
                continue
            if not (name.endswith("mask")):    # or name.endswith("map")
                continue
            obj = getattr(self, name, None)
            if self._is_healsparse_map(obj):
                found.append((name, obj))

        # Stable order: preferred names first, then alphabetical for customs
        preferred = (
            "footmask", "patchmask", "propmask", "starmask",
            "circmask", "boxmask", "ellipmask", "polymask",
            "zonemask","mwmask", "mask",
        )
        rank = {n: i for i, n in enumerate(preferred)}
        found.sort(key=lambda kv: (rank.get(kv[0], 999), kv[0]))
        return found


    def _summarize_stage(self, name, hsmap) -> str:
        """
        Aux function that builds a string with useful info for a stage of a given
        name and healsparse map. Used by __str__ and __repr__.
        """
        nside_cov = int(hsmap.nside_coverage)
        nside_sparse = int(hsmap.nside_sparse)
        order_cov = int(np.log2(max(1, nside_cov)))
        order_sparse = int(np.log2(max(1, nside_sparse)))
        npix = int(hsmap.n_valid)
        area = float(hsmap.get_valid_area(degrees=True))
        pix_area_deg2 = hp.nside2pixarea(nside_sparse, degrees=True)
        pix_size_arcsec = (pix_area_deg2 ** 0.5) * 3600.0  # ~side length

        return (f"{name:<15}: (ord/nside)cov={order_cov:<1}/{nside_cov:<4} "
                f"(ord/nside)sparse={order_sparse:<2}/{nside_sparse:<5} "
                f"valid_pix={npix:<7}  area={area:6.2f} deg² "
                f"pix_size={pix_size_arcsec:6.1f}\"")


    def __str__(self):
        """
        Pretty summary showing ONLY stored stages discovered dynamically.
        """
        discovered = self._discover_stage_items()
        if not discovered:
            return "SkyMaskPipe: no stage maps defined"
        return "\n".join(self._summarize_stage(n, m) for n, m in discovered)

    # #########################
    # Uncomment this if you want to display the summary just by typing the name of the object
    __repr__ = __str__
    ###########################

    
    def _finalize_stage(self, hspmap, *, default_name: str, output_stage: Optional[str] = None, 
                        extra_meta: Optional[dict] = None):
        """Attach a  map as a stage and record standardized metadata. Overwrite canonical when output_stage is None."""
        # extra_meta : pass here the dict associated to the map that you want to store/updata
        import math
        name = (output_stage or default_name).strip()
        if not name.endswith("mask"):
            raise ValueError("Stage names must end with 'mask'.")
        if not name.isidentifier():
            raise ValueError(f"Invalid Python identifier for stage name: {name!r}")
        setattr(self, name, hspmap)
        if not hasattr(self, "_params") or self._params is None:
            self._params = {}
        nside_sparse = getattr(hspmap, "nside_sparse", None)
        nside_cov    = getattr(hspmap, "nside_coverage", None)
        order_sparse   = int(math.log2(nside_sparse)) if nside_sparse else None
        order_coverage = int(math.log2(nside_cov))    if nside_cov   else None
        meta = {
            "nside_sparse": nside_sparse,
            "nside_coverage": nside_cov,
            "order_sparse": order_sparse,
            "order_coverage": order_coverage,
        }
        if extra_meta:
            try:
                meta.update(extra_meta)
            except Exception:
                meta["extra"] = repr(extra_meta)
        self._params[name] = meta
        return hspmap

    
    def stage_meta(self, name: str) -> dict:
        """Return the metadata associated to a given stage"""
        return dict(self._params.get(name, {}))

    
    def stage_orders(self, name: str):
        """Return the (order_sparse, order_coverage) of a given stage"""
        import math
        h = getattr(self, name, None)
        if h is not None:
            os_ = int(math.log2(getattr(h, "nside_sparse", 1))) if getattr(h, "nside_sparse", None) else None
            oc_ = int(math.log2(getattr(h, "nside_coverage", 1))) if getattr(h, "nside_coverage", None) else None
            return os_, oc_
        m = self._params.get(name, {})
        return m.get("order_sparse"), m.get("order_coverage")


    def write(self, outdir: str | os.PathLike, overwrite: bool = True,
              rows_per_batch: int = 8192) -> None:
        """
        Save a SkyMaskPipe instance to disk. The output is a directoy containing one FITS
        file per stage (with per-row Bitpack encoding), which hold the corresponding valid pixels.
        It also ouputs a JSON file for metadada comprising scalars, parameter dictionaries,
        and stage filenames.

        Parameters
        ----------
        outdir : str or os.PathLike
            Destination directory where the pipeline will be saved.
        overwrite : bool, default=True
            If True, replaces any existing directory at `outdir`. If False and the
            directory exists, raises a `FileExistsError`.
        rows_per_batch : int, default=8192
            Number of rows to write per batch when streaming stage FITS files.
        """
        outdir = Path(outdir)
        tmpdir = Path(tempfile.mkdtemp(prefix="skymaskpipe_write_", dir=outdir.parent))

        try:
            tmpdir.mkdir(exist_ok=True, parents=True)

            # ---- metadata skeleton
            meta = {
                "format": "skymaskpipe-bitpack-fits-stream",
                "version": getattr(self, "_BITPACK_FITS_VERSION", 1),
                "class": self.__class__.__name__,
                "stages": {},     # name -> {filename: ...}
                "scalars": {},    # from _SCALAR_ATTRS if present
                "params": {},     # JSON-safe _params
            }

            # ---- scalars: use your existing list, independent of __str__()
            if hasattr(self, "_SCALAR_ATTRS"):
                for k in self._SCALAR_ATTRS:
                    if hasattr(self, k):
                        meta["scalars"][k] = getattr(self, k)

            # ---- discover stages (NO fallback to _STAGE_ATTRS)
            discovered = list(self._discover_stage_items())
            if not discovered:
                raise RuntimeError("No stages present to write.")

            # ---- write each stage as FITS (bit-packed streaming)
            wrote = 0
            for name, hspmap in discovered:
                fn = f"{name}.fits"
                _write_stage_fits_bitpack(tmpdir / fn, hspmap, rows_per_batch=rows_per_batch)
                meta["stages"][name] = {"filename": fn}
                wrote += 1

            if wrote == 0:
                raise RuntimeError("No stages present to write.")

            # ---- JSON-safe copy of _params
            def _to_jsonable(x):
                import numpy as _np
                from pathlib import Path as _Path
                if isinstance(x, (str, int, float, bool)) or x is None:
                    return x
                if isinstance(x, (list, tuple)):
                    return [_to_jsonable(v) for v in x]
                if isinstance(x, dict):
                    return {str(k): _to_jsonable(v) for k, v in x.items()}
                if isinstance(x, _np.generic):
                    return x.item()
                if isinstance(x, _np.ndarray):
                    return x.tolist()
                if isinstance(x, _Path):
                    return str(x)
                return str(x)

            if getattr(self, "_params", None):
                meta["params"] = _to_jsonable(self._params)

            # ---- write metadata.json
            with open(tmpdir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, sort_keys=True)

            # ---- move into place atomically-ish
            if outdir.exists():
                if not overwrite:
                    raise FileExistsError(f"{outdir} exists and overwrite=False")
                shutil.rmtree(outdir)
            shutil.move(str(tmpdir), str(outdir))

        finally:
            # best-effort cleanup if tmpdir still around (e.g., move failed)
            if tmpdir.exists() and tmpdir.parent != outdir:
                shutil.rmtree(tmpdir, ignore_errors=True)


    @staticmethod
    def _band_polygon_lb(b0_deg: float, n_long_samples: int = 720) -> SkyCoord:
        """
        Build a single closed polygon (in Galactic coords) that traces the boundary
        of the galactica plane band, |b| <= b0. The polygon runs along b=+b0 from ℓ=0→360,
        then back along b=-b0 from ℓ=360→0
        """
        b0 = float(b0_deg)
        l_up = np.linspace(0.0, 360.0, n_long_samples, endpoint=True)
        l_dn = l_up[::-1]
        b_up = np.full_like(l_up,  +b0)
        b_dn = np.full_like(l_dn,  -b0)

        l_poly = np.concatenate([l_up, l_dn, l_up[:1]])  # close polygon
        b_poly = np.concatenate([b_up, b_dn, b_up[:1]])

        return SkyCoord(l=l_poly * u.deg, b=b_poly * u.deg, frame="galactic")


    @staticmethod
    def _ellipse_polygon_lb(a_l_deg: float, b_b_deg: float,
                            l0_deg: float = 0.0, b0_deg: float = 0.0,
                            n_theta: int = 720) -> SkyCoord:
        """
        Build an ellipse polygon in Galactic coords centered at (l0, b0) with
        semi-axes a_l (in longitude) and b_b (in latitude), both in degrees.
        Parametric form: l = l0 + a_l cosθ, b = b0 + b_b sinθ
        """
        th = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=True)
        l = l0_deg + a_l_deg * np.cos(th)
        b = b0_deg + b_b_deg * np.sin(th)

        # Wrap longitudes to [0, 360)
        l = np.mod(l, 360.0)

        # Close polygon explicitly
        if l[0] != l[-1] or b[0] != b[-1]:
            l = np.concatenate([l, l[:1]])
            b = np.concatenate([b, b[:1]])

        return SkyCoord(l=l * u.deg, b=b * u.deg, frame="galactic")


    @staticmethod
    def gal_plane_bulge_moc(max_depth: int = 8, b0_deg: float = 15.,
        bulge_a_deg: float = 25., bulge_b_deg: float = 20.,
        bulge_center_l_deg: float = 0., bulge_center_b_deg: float = 0.,
        band_longitude_samples: int = 1440, ellipse_samples: int = 720) -> MOC:
        """
        Create a MOC of the Galactic plane (|b| <= b0_deg) in union with an elliptical bulge.

        Parameters
        ----------
        b0_deg : float
            Half-thickness of the Galactic plane band (|b| <= b0_deg).
        bulge_a_deg : float
            Semi-axis along Galactic longitude for the bulge ellipse (degrees).
        bulge_b_deg : float
            Semi-axis along Galactic latitude for the bulge ellipse (degrees).
        bulge_center_l_deg : float
            Bulge ellipse center longitude (degrees, Galactic).
        bulge_center_b_deg : float
            Bulge ellipse center latitude (degrees, Galactic).
        band_longitude_samples : int
            Number of samples along longitude to trace the band boundary.
        ellipse_samples : int
            Number of parametric samples for the bulge ellipse.
        max_depth : int
            MOC max depth (HEALPix order). 8 ≈ 0.228° pixels (~13.7 arcmin).

        Returns
        -------
        moc : mocpy.MOC
            Union of the band and bulge MOCs, expressed in ICRS.
        """
        # Build polygons in Galactic coords
        poly_band_gal   = SkyMaskPipe._band_polygon_lb(b0_deg, n_long_samples=band_longitude_samples)
        poly_bulge_gal  = SkyMaskPipe._ellipse_polygon_lb(bulge_a_deg, bulge_b_deg,
                                           l0_deg=bulge_center_l_deg, b0_deg=bulge_center_b_deg,
                                           n_theta=ellipse_samples)

        # Convert polygon vertices to ICRS
        poly_band_icrs  = poly_band_gal.icrs
        poly_bulge_icrs = poly_bulge_gal.icrs

        # Build MOCs from polygons at requested resolution
        moc_band  = MOC.from_polygon_skycoord(poly_band_icrs,  max_depth=max_depth)
        moc_bulge = MOC.from_polygon_skycoord(poly_bulge_icrs, max_depth=max_depth)

        # Make union of plane ∪ bulge
        moc = moc_band.union(moc_bulge)
        return moc


    @staticmethod
    def readQApatches(qafile : str):
        """
        Read contents of HSC QA patch list.

        See https://hsc-release.mtk.nao.ac.jp/schema/#pdr3.pdr3_wide.patch_qa

        Parameters
        ----------
        qafile : string
            File path

        Returns
        -------
        astropy table
            Table with QA patches
        """
        pqa = Table.read(qafile)
        return pqa



    @staticmethod
    def parse_condition(condition_str : str):
        """
        Parses a condition string, converting column names into a format usable for direct evaluation over an astropy table.
        For example, converts "(ra>20) and (dec<30)" into "(table['ra']>20) & (table['dec']<30)".

        Parameters
        ----------
        condition_str : str
            Condition string, e.g., "(gmag>20.4) and (imag>19.8)"

        Returns
        -------
        str
            Condition in new format
        """
        # Regex to match alphanumeric + underscores after a '(' and before a comparison operator (<, >, =, !=)
        pattern = r'\(([\w_]+)(?=[<>!=])'

        # Substitute the matched column names with table['column_name']
        condition_str = re.sub(pattern, r"(table['\1']", condition_str)

        # Use regex to replace 'and' and 'or' as standalone words, not as substrings within column names
        condition_str = re.sub(r'\band\b', '&', condition_str)
        condition_str = re.sub(r'\bor\b', '|', condition_str)

        return condition_str


    @staticmethod
    def filter_and_pixelate_patches(file: str | os.PathLike[str], qatable : Table, 
                                    filt: Optional[str] = None, order : int = 13) -> NDArray[np.int64]:
        """
        Reads a file with HSC patches, matches it against the QA table containing quality measures
        of each patch, filter those that meet some depth/seeing/etc critera, and return their
        pixels at a given order. Patches are pixelized as quadrangle polygons.

        Modify the patch aceptance criteria to suit your purposes, e.g. accepting only those above
        a minimum depth threshold in a given band, etc.

        Parameters
        ----------
        file : str
            HSC patch file (defult parquet, but anything astropy can read)
        qatable : astropy table
            Table of patches with QA measurements
        filt : str
            Contition(s) to apply to patches, e.g filt='(ra>20) and (dec<10)'. If None,
            all patches will be considered
        order : int
            Pixelization order

        Returns
        -------
        ndarray
            Array of pixels inside patches fulfilling the patch selection criteria
        """

        # Read patch file
        print('--- Processing', str(file))
        print('    Order ::',order)
        table = Table.read(file)
        # Construct skymap_id to merge the patch table and the QA table
        table['skymap_id'] = table['Tract']*10000 + table['Patch1']*100 + table['Patch2']
        table = join(table, qatable, keys='skymap_id')
        print('    Patches with QA                       :', len(table))

        if filt:
            # Apply uniformity criteria to patches based on minimum depth. This gets rids of border zones.
            filt_goodstr = __class__.parse_condition(filt)
            idx = eval(filt_goodstr)
            print('    Patches with QA fulfilling conditions :', len(table[idx]))
        else:
            idx = np.full(len(table), fill_value=True, dtype=np.bool_)
            print('    Patches adopted (no filters applied) :', len(table[idx]))

        # Change to float64 because moc complains otherise
        table['ra0']=table['ra0'].astype(np.float64)
        table['ra1']=table['ra1'].astype(np.float64)
        table['ra2']=table['ra2'].astype(np.float64)
        table['ra3']=table['ra3'].astype(np.float64)
        table['dec0']=table['dec0'].astype(np.float64)
        table['dec1']=table['dec1'].astype(np.float64)
        table['dec2']=table['dec2'].astype(np.float64)
        table['dec3']=table['dec3'].astype(np.float64)

        # Create list of skycoords for each patch
        sks = [ SkyCoord([p['ra0'], p['ra1'], p['ra2'], p['ra3']], [p['dec0'], p['dec1'], p['dec2'], p['dec3']],
                unit='deg') for p in table[idx]]
        # Generate moc from the polygon of the four vertices of each patch.
        moc_ptchs = MOC.from_polygons(sks, max_depth=order)
        # Return flat and unique pixels at max_depth
        hp_index = np.concatenate([p.flatten() for p in moc_ptchs])
        hp_index = np.unique(hp_index).astype(np.int64)
        print('    Surviving patch pixels                :', hp_index.shape[0])

        return hp_index


    @staticmethod
    def remove_isopixels(hsmap):
        """
        Remove empty isolated pixels (i.e. one False pixel surrounded by 8 True pixels),
        by settiing it them True. This can help for example when pixelating sources
        with just the right order so that a few artificial empty pixels appear

        Parameters
        ----------
        hsmap
            Healsparse boolean map

        Returns
        -------
        hsmap
            Healsparse boolean map
        """
        print('    ...removing isolated pixels...')
        from collections import Counter
        active_pixels = hsmap.valid_pixels
        neighbors_deact = []
        for pix in tqdm(active_pixels):
            neighbors = hp.get_all_neighbours(hsmap.nside_sparse, pix, nest=True)
            neighbors_deact.append(neighbors[~hsmap[neighbors]])

        counts = Counter(np.hstack(neighbors_deact))
        val8times = [key for key, count in counts.items() if count == 8]
        val8times = np.hstack(val8times)
        hsmap.update_values_pix(val8times, True, operation='replace')  #np.full_like(val8times, True, dtype=np.bool_)
        return hsmap


    @staticmethod
    def erode_borders(hsmap):
        """
        Remove the borders of holes in the mask, i.e. detect the pixels that delineate
        zones set to False (completely surrounded by pixels set to True) as well as
        the external border of regions, and set those border pixels off. This can
        help to remove jagged boundaries around empty regions, when pixelated
        at relatively coarse resolutions.

        Parameters
        ----------
        hsmap
            Healsparse boolean map

        Returns
        -------
        hsmap
            Healsparse boolean map
        """
        print('    ...eroding borders...')
        nborders = 8
        active_pixels = hsmap.valid_pixels
        active_pixel_set = set(active_pixels)
        filtered_pixels = []
        for pix in tqdm(active_pixels):
            neighbors = hp.get_all_neighbours(hsmap.nside_sparse, pix, nest=True)
            active_neighbors_count = 0
            for neighbor in neighbors:
                if neighbor in active_pixel_set:
                    active_neighbors_count += 1
            if active_neighbors_count < nborders:
                filtered_pixels.append(pix)

        filtered_pixels = np.array(filtered_pixels)
        hsmap.update_values_pix(filtered_pixels, False, operation='replace')  #np.full_like(filtered_pixels, False, dtype=np.bool_)
        return hsmap



    @staticmethod
    def pixelate_circles(data: Union[pd.DataFrame, str, os.PathLike[str]], stage : HealSparseMap, fmt: str = "ascii", 
                         columns: Sequence[str] = ("ra", "dec", "radius"), order: int = 15, delta_depth: int = 2, 
                         n_threads: int = 1, chunk_size: int = 600_000) -> None:
        """
        Pixelize many circles into HEALPix indices and update the corresponding 
        pixels of a given input stage.
        """
        colra, coldec, colrad = columns
        if isinstance(data, pd.DataFrame):
            table = data
            print('--- Pixelating circles from DataFrame')
        elif isinstance(data, (str, PosixPath)):
            print('--- Pixelating circles from', data)
            table = Table.read(data, format=fmt)
        else:
            raise TypeError("data must be a DataFrame or path")

        print(f'    Order :: {order} | pixelization_threads={n_threads}')
        n_total = len(table)
        print(f'    Circles to pixelate: {n_total} in batches of {chunk_size}')

        for start in range(0, n_total, chunk_size):
            end = min(start + chunk_size, n_total)
            print(f'        processing circles {start}:{end} ...')

            mocs = MOC.from_cones(
                lon=Longitude(table[colra][start:end], unit='deg'),
                lat=Latitude(table[coldec][start:end], unit='deg'),
                radius=Angle(table[colrad][start:end], unit='deg'),
                max_depth=order, delta_depth=delta_depth, n_threads=n_threads)

            #if not mocs: continue
            hp_idx = np.concatenate([moc.flatten() for moc in mocs])
            #if not hp_idx: continue
            hp_idx = np.unique(hp_idx).astype(np.int64)

            # Stream directly into the sparse map
            stage.update_values_pix(hp_idx, True)

            # be nice with memory
            del hp_idx, mocs
            gc.collect()

        return

    
    @staticmethod
    def pixelate_ellipses(data: Union[pd.DataFrame, str, os.PathLike[str]], fmt: str = "ascii", 
                          columns: Sequence[str] = ("ra", "dec", "a", "b", "pa"), 
                          order: int = 15, delta_depth: int = 2) -> NDArray[np.int64]:
        """
        Read elliptical regions around extended sources, pixelize them and return the (unique)
        pixels inside. Coordinates and distances should be in degrees.

        Parameters
        ----------
        data : pd.Dataframe or str or Path
            Pandas dataFrame or path to file
        fmt : str
            Format of file, e.g. 'ascii', 'parquet', or any accepted by astropy.table
        columns : list of str
            Columns for ra, dec, a, b and pa (position angle) of ellipses
        delta_depth : int
            Delta to higher orders to improve pixelization
        order : int
            Pixelization order

        Returns
        -------
        ndarray
            Array of pixels
        """
        colra, coldec, cola, colb, colpa = columns
        if isinstance(data, pd.DataFrame):
            print('--- Pixelating ellipses from DataFrame')
            table = data
        elif isinstance(data, (str, PosixPath)):
            print('--- Pixelating ellipses from', data)
            table = Table.read(data, format=fmt)
        print('    Order ::',order)

        mocs = []
        for ra, dec, a_axis, b_axis, pa_angle in zip(table[colra], table[coldec], table[cola], table[colb], table[colpa]):
            moc = MOC.from_elliptical_cone(
            lon=Longitude(ra, unit='deg'), lat=Latitude(dec, unit='deg'), a=Angle(a_axis, unit='deg'), b=Angle(b_axis, unit='deg'),
            pa=Angle(pa_angle, unit='deg'), max_depth=order, delta_depth=delta_depth )
            pixels = moc.flatten().astype(np.int64)
            mocs.append(pixels)

        hp_index = np.hstack(mocs)
        print('    done')
        return np.unique(hp_index).astype(np.int64)

 
    @staticmethod
    def pixelate_boxes(data: Union[pd.DataFrame, str, os.PathLike[str]], fmt: str = "ascii", 
                       columns: Sequence[str] = ("ra_c", "dec_c", "width", "height"), 
                       order: int = 15, n_threads: int = 4):
        """
        Read box regions around bright stars, pixelize them and return the (unique) pixels inside.
        Coordinates and distances should be in degrees.

        Parameters
        ----------
        data : pd.Dataframe or str or Path
            Pandas dataFrame or path to file
        fmt : str
            Format of file, e.g. 'ascii', 'parquet', or any accepted by astropy.table
        columns : list of str
            Columns for ra_center, dec_center, width and height of boxes
        order : int
            Pixelization order
        n_threads : int
            Number of threads. Set to None to use all available threads

        Returns
        -------
        ndarray
            Array of pixels
        """
        colra, coldec, colw, colh = columns
        if isinstance(data, pd.DataFrame):
            print('--- Pixelating boxes from DataFrame')
            table = data
        elif isinstance(data, (str, PosixPath)):
            print('--- Pixelating boxes from', data)
            table = Table.read(data, format=fmt)
        print('    Order ::',order)

        ra_center = Longitude(table[colra], unit='deg')
        dec_center = Latitude(table[coldec], unit='deg')
        width = Angle(table[colw], unit='deg')
        height = Angle(table[colh], unit='deg')

        width_larger = width > height
        a = np.where(width_larger, 0.5 * width, 0.5 * height)
        b = np.where(width_larger, 0.5 * height, 0.5 * width)
        angle = np.where(width_larger, Angle(90, 'deg'), 0)

        # Boxes strech at high declination. For now, multiply by cos(dec) #######
        a = a*np.cos(table[coldec].value*np.pi/180.)

        mocs = MOC.from_boxes(lon=ra_center, lat=dec_center, a=a, b=b, angle=angle,
                              max_depth=order, n_threads=n_threads)

        hp_index = np.concatenate([moc.flatten() for moc in mocs])
        print('    done')
        return np.unique(hp_index).astype(np.int64)
    

    @staticmethod
    def pixelate_zones(data: Union[pd.DataFrame, str, os.PathLike[str]], fmt: str = "ascii", 
                       columns: Sequence[str] = ("ra1", "dec1", "ra2", "dec2"), 
                       order: int = 15) -> NDArray[np.int64]:
        """
        Read zone regions, pixelize them and return the (unique) pixels inside. Zones are
        regions delimited by ra/dec boundaries that follow great circles along ra and
        minor circles along dec. Coordinates and distances should be in degrees.

        Parameters
        ----------
        data : pd.Dataframe or str or Path
            Pandas dataFrame or path to file
        fmt : str
            Format of file, e.g. 'ascii', 'parquet', or any accepted by astropy.table
        columns : list of str
            Columns for ra1, dec1, ra2, dec2, coordinates for the left-low and right-top points
            that define the zone
        order : int
            Pixelization order

        Returns
        -------
        ndarray
            Array of pixels
        """
        colra1, coldec1, colra2, coldec2 = columns
        if isinstance(data, pd.DataFrame):
            print('--- Pixelating zones from DataFrame')
            table = data
        elif isinstance(data, (str, PosixPath)):
            print('--- Pixelating zones from', data)
            table = Table.read(data, format=fmt)
        print('    Order ::',order)

        # Change to float64 because otherwise moc complains
        table[colra1]=table[colra1].astype(np.float64)
        table[coldec1]=table[coldec1].astype(np.float64)
        table[colra2]=table[colra2].astype(np.float64)
        table[coldec2]=table[coldec2].astype(np.float64)

        # Create list of skycoords for each zone
        zns = [ SkyCoord([[p[colra1], p[coldec1]], [p[colra2], p[coldec2]]], unit='deg') for p in table]
        # Generate moc from the zones defined by lower-left and top-right coordinates
        mocs = []
        for z in zns:
            moc = MOC.from_zone(z, max_depth=order)
            pixels = moc.flatten().astype(np.int64)
            mocs.append(pixels)

        hp_index = np.unique(np.hstack(mocs))
        print('    done')
        return hp_index

    

    @staticmethod
    def pixelate_polys(data: Union[pd.DataFrame, str, os.PathLike[str]], fmt: str = "ascii", 
                       columns: Sequence[str] = ("ra0", "ra1", "ra2", "ra3", "dec0", "dec1", "dec2", "dec3"),
                       n_threads: int = 4, order: int = 15) -> NDArray[np.int64]:
        """
        Read quadrangular polygons, pixelize them and return the (unique) pixels inside.
        Input data must have 8 columns for the coordinates of the 4 vertices.
        Coordinates and distances should be in degrees.

        Parameters
        ----------
        data : pd.Dataframe or str or Path
            Pandas dataFrame or path to file
        fmt : str
            Format of file, e.g. 'ascii', 'parquet', or any accepted by astropy.table
        columns : list of str
            Columns for ra, dec for each of the four vertexs
        n_threads : int
            Number of threads. Set to None to use all available threads
        order : int
            Pixelization order

        Returns
        -------
        ndarray
            Array of pixels
        """
        cr0, cr1, cr2, cr3, cd0, cd1, cd2, cd3 = columns
        if isinstance(data, pd.DataFrame):
            print('--- Pixelating polys from DataFrame')
            table = data
        elif isinstance(data, (str, PosixPath)):
            print('--- Pixelating polys from', data)
            table = Table.read(data, format=fmt)
        print('    Order ::',order)

        # Change to float64 because otherwise moc complains
        table[cr0]=table[cr0].astype(np.float64)
        table[cr1]=table[cr1].astype(np.float64)
        table[cr2]=table[cr2].astype(np.float64)
        table[cr3]=table[cr3].astype(np.float64)
        table[cd0]=table[cd0].astype(np.float64)
        table[cd1]=table[cd1].astype(np.float64)
        table[cd2]=table[cd2].astype(np.float64)
        table[cd3]=table[cd3].astype(np.float64)

        # Create list of skycoords for each poly
        sks = [ SkyCoord([p[cr0], p[cr1], p[cr2], p[cr3]], [p[cd0], p[cd1], p[cd2], p[cd3]], unit='deg') for p in table]
        # Generate moc from the polygon of the four vertices of each patch.
        moc_plys = MOC.from_polygons(sks, max_depth=order, n_threads=n_threads)
        # Return flat and unique pixels at max_depth
        hp_index = np.concatenate([p.flatten() for p in moc_plys])
        hp_index = np.unique(hp_index).astype(np.int64)
        print('    done')
        return hp_index




    def build_star_mask_online(self, starq, order_sparse: int = 15, order_cov: Optional[int] = None,
                               columns: Optional[Sequence[str]] = ['ra','dec','radius'], save_stars: bool = False,
                               bit_packed: bool = True, n_threads: int = 4, chunk_size: int = 600_000, 
                               output_stage: Optional[str] = None, mwmask_output_stage: Optional[str] = None):
        """
        Build a bright-star mask on the fly by querying a remote catalog.

        This method constructs a `HealSparseMap` mask of bright stars by:
        (1) defining a search region from a user-supplied `search_stage`,
        (2) querying the Gaia catalog in chunks of sky defined by a MOC,
        (3) applying a custom radius function to compute star exclusion radii,
        (4) pixelizing the stars into HEALPix pixels at the requested order,
        and (5) streaming results directly into a sparse mask. By default, the 
        final mask is stored in the `starmask` attribute.

        Parameters
        ----------
        starq : dict
            Dictionary of parameters controlling the star mask construction.
            Required keys:
            
              - ``search_stage`` : `HealSparseMap` defining the search region.
              - ``cat`` : catalog identifier to open with `lsdb.open_catalog`.
              - ``columns`` : list of columns to load from the Gaia catalog.
              - ``gaia_gmag_lims`` : tuple ``(gmin, gmax)`` magnitude limits.
              - ``radfunction`` : callable that takes a DataFrame and assigns radii to stars.
              
            Optional keys:
            
              - ``max_area_single`` : maximum deg² before splitting (default 500).
              - ``target_chunk_area`` : target deg² per MOC chunk (default 300).
              - ``coarse_order_bfs`` : order for initial chunk splitting (default 5).
              
        order_sparse : int, optional
            Sparse order for pixelizing circles due to stars.
        order_cov : int, optional
            Coverage order. Defaults to `self.order_cov`.
        columns : sequence of str, optional
            Column names expected in the star DataFrame. Defaults to
            ``['ra', 'dec', 'radius']``.
        save_stars : bool, default=False
            If True, for each chunk piece save the retrieved stars (with radius column) in parquet format
        bit_packed : bool, optional
            If True, convert the final mask to bit-packed format.
        n_threads : int, default=4
            Number of threads used during circle pixelization.
        chunk_size : int, default=600000
            Number circles pixelized at once. Watch out memory if chunk_size and n_threads are both large
        output_stage : str, optional
            Name for the star ouput stage. If None, defaults to then canonical name 'starmask'
        mwmask_output_stage : str, optional
            Name for the Milky Way disc+bulge stage. If None, defaults to the canonical name "mkmask"

        Returns
        -------
        starmask : healsparse.HealSparseMap
            The star mask as a `HealSparseMap`, also stored as `self.starmask` when `output_stage` is None
        """
        # from lsdb.core.search.moc_search import MOCSearch   # this was for lsdb 0.6.4
        from lsdb.core.search.region_search import MOCSearch  # important import!

        print('BUILDING STAR MASK >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        if not(isinstance(starq, dict)): raise Exception("starq must be a valid dictionary")

        # Check if user want different coverage order, otherwise get from defaults
        ord_cov = order_cov if order_cov is not None else self.order_cov

        nside_sparse = 1<<order_sparse
        nside_cov = 1<<ord_cov

        # Create the empty boolean map up front
        mask = hsp.HealSparseMap.make_empty(nside_cov, nside_sparse, dtype=np.bool_, bit_packed=True)

        MAX_DEPTH = 8     # The moc of the search_stage will be degraded to this order.
                          # 8 means 822" pixels, so is fine for including stars with up to ~411" radii
        search_stage = starq['search_stage']    # This is the area for searching Gaia stars
        if not isinstance(search_stage, hsp.HealSparseMap):
            raise TypeError('search_stage must be a valid HealSparseMap')

        # Build search MOC but degraded to MAX_DEPTH
        order_search_stg = int(np.log2(search_stage.nside_sparse))
        moc = MOC.from_healpix_cells(ipix=search_stage.valid_pixels, depth=order_search_stg, max_depth=MAX_DEPTH)

        # Create moc for the Milky Way at MAX_DEPTH and subtract
        avoid_mw = starq['avoid_mw']
        b0_deg = starq['b0_deg']
        bulge_a_deg = starq['bulge_a_deg']
        bulge_b_deg = starq['bulge_b_deg']
        mwmessage = ''
        if avoid_mw:
            print('--- Avoid Milky Way is ON')
            moc_pb = SkyMaskPipe.gal_plane_bulge_moc(max_depth=MAX_DEPTH, b0_deg=b0_deg,
                                                     bulge_a_deg=bulge_a_deg, bulge_b_deg=bulge_b_deg)
            moc = moc.difference(moc_pb)
            mwmessage = '(MW subtracted)'

        # Open Gaia (already filtered by columns/mag/lat outside MW plane)
        gaia = lsdb.open_catalog(starq['cat'],
                                columns=starq['columns'],
                                search_filter=MOCSearch(moc),
                                filters=[["phot_g_mean_mag", ">", starq['gaia_gmag_lims'][0]],
                                        ["phot_g_mean_mag", "<=", starq['gaia_gmag_lims'][1]]] )

        # Split moc parameters
        max_area_sing     = starq.get('max_area_single',   800.0)
        target_chunk_area = starq.get('target_chunk_area', 800.0)
        coarse_order_bfs  = starq.get('coarse_order_bfs',  5)

        # Get chunks (or a single piece)
        moc_area = getarea_moc(moc)
        if moc_area < max_area_sing:
            print(f'Area of search_stage {mwmessage} is {moc_area:.2f} deg2 -> no splitting')
            chunk_mocs = [moc.add_neighbours()]  # 1px border at max depth
        else:
            print(f'Area of search_stage {mwmessage} is {moc_area:.2f} deg2 -> splitting...')
            chunk_mocs = split_moc_into_chunks(moc, target_deg2=target_chunk_area, coarse_order=coarse_order_bfs)
            # refine to original depth + add border
            chunk_mocs = [c.intersection(moc).add_neighbours() for c in chunk_mocs]
            chunk_areas = [getarea_moc(mi) for mi in chunk_mocs]
            print(f"Got {len(chunk_mocs)} chunks of areas [{' '.join(f'{a:6.2f}' for a in chunk_areas)}] deg²")


        # Star radii function
        radfunction = starq['radfunction']
        if not callable(radfunction):
            raise TypeError("radfunction must be a callable that accepts a DataFrame with optional kwargs")

        # MOC STREAMING PIPELINE ---------------
        for i, mi in enumerate(chunk_mocs, 1):
            print(f'Chunk {i}/{len(chunk_mocs)}: querying catalog ...')
            # Get stars in this chunk only
            s = gaia.moc_search(moc=mi).compute()
            print(f'--- Stars found : {len(s)}')
            # Add radii in-place
            radfunction(s)
            # Save stars to disk if requested
            if save_stars:
                fn = f"stars_chunk_{i:02d}.parquet"
                s.to_parquet(fn)
                print('---',fn, 'written to disk')
            # Pixelate this moc_chunk
            self.pixelate_circles(s, mask, order=order_sparse, columns=columns,
                                  n_threads=n_threads, chunk_size=chunk_size)
            # Free memory promptly
            del s
            gc.collect()

        # Save mask for Milky Way is case the user might need to subtract it later
        if avoid_mw:
            mwmask = self._empty_like_geometry(nside_cov=nside_cov, nside_sparse=1<<MAX_DEPTH, bit_packed=False)
            pixels = moc_pb.flatten().astype(np.int64)
            mwmask.update_values_pix(pixels, True)
            #print(f'Milky Way mask stored in stage -> mwmask')       # fix this message

        # Force packing if desired
        if bit_packed:
            mask = mask.as_bit_packed_map()
            mwmask = mwmask.as_bit_packed_map()

        # Store calling/useful info in its own dictionary
        area_deg2 = mask.get_valid_area(degrees=True)
        npix = mask.n_valid
        starq['search_stage'] = '<dummy>'  # for now just set a dummy string for the search_stage to avoid
                                           # storing the actual map. We should fix this later

        if 'mwmask' in locals() and mwmask is not None:
            extra_meta = { 'b0_deg': b0_deg, 'bulge_a_deg': bulge_a_deg, 'buge_b_deg': bulge_b_deg, 
                           'pixels': npix, 'area_deg2': area_deg2 }
            self._finalize_stage(mwmask, default_name='mwmask', output_stage=mwmask_output_stage, extra_meta=extra_meta);

        extra_meta = { 'starq': starq, 'columns': columns, 'bit_packed': bit_packed, 'n_threads': n_threads, 
                       'chunk_size': chunk_size, 'avoid_mw': avoid_mw, 'b0_deg': b0_deg, 'bulge_a_deg': bulge_a_deg, 
                       'bulge_b_deg': bulge_b_deg, 'pixels': npix, 'area_deg2': area_deg2 }
        return self._finalize_stage(mask, default_name='starmask', output_stage=output_stage, extra_meta=extra_meta)


        

    @staticmethod
    def reproject_nside_coverage(hspmap, newcov):
        """
        OLDER METHOD TO CHANGE COVERAGE -> TO BE DECREPATED. Use change_cov_order() instead
        """
        oldcov = hspmap.nside_coverage
        # Get all sparse pixels with valid data
        ipix = hspmap.valid_pixels
        values = hspmap.get_values_pix(ipix)

        # Create new map with same nside_sparse, but new coverage resolution
        new_map = hsp.HealSparseMap.make_empty(
            nside_coverage=newcov,
            nside_sparse=hspmap.nside_sparse,
            dtype=hspmap.dtype
        )

        # Insert data into the new map
        new_map.update_values_pix(ipix, values)

        print(f'    Warning: nside_coverage changed from {oldcov} to {newcov}')
        return new_map


    def build_prop_mask(self, prop_maps, thresholds, comparisons, order_sparse: int = 15, order_cov=None, 
                        bit_packed=False, output_stage: Optional[str] = None):
        """
        Build a HealSparse boolean mask based on pixels meeting multiple property map thresholds

        Parameters
        ----------
        prop_maps : list of HealSparseMap or list of str
            One or more HealSparse maps to threshold, either as objects or file paths
        thresholds : float or list of float
            Threshold value(s) for each property map
        comparisons : str or list of {'gt', 'lt', 'ge', 'le'}
            Comparison operator(s) for each threshold
        bit_packed : bool, optional
            If True, return the ouput as bit-packed boolean map
        output_stage : str, optional
            Name for the ouput stage. If None, defaults to then canonical name 'propmask'
            
        Returns
        -------
        mask_map : HealSparseMap
            The boolean (or bit-packed) mask is stored at `self.propmask` and also
            returned to prompt
        """

        print('BUILDING PROPERTY MAP >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # Check if user wants specific cov corder, otherwise get from pipeline default
        ord_cov = order_cov if order_cov is not None else self.order_cov

        nside_sparse = 1<<order_sparse
        nside_cov = 1<<ord_cov

        # Normalize inputs to lists
        if not isinstance(prop_maps, (list, tuple)):
            prop_maps = [prop_maps]
        if not isinstance(thresholds, (list, tuple)):
            thresholds = [thresholds]
        if not isinstance(comparisons, (list, tuple)):
            comparisons = [comparisons]

        if not (len(prop_maps) == len(thresholds) == len(comparisons)):
            raise ValueError("prop_maps, thresholds, and comparisons must be of the same length")

        if isinstance(prop_maps[0], (str, os.PathLike)):
            pmstring = prop_maps   # keep the string list to save in the parameter dict at the end
        else:
            pmstring = "<mem>"     # bogus string if maps are passed from memory

        # Convert file paths to HealSparseMap objects
        resolved_maps = []
        for pm in prop_maps:
            if isinstance(pm, (str, os.PathLike)):               # if isinstance(pm, str):
                print('--- Processing', str(pm))
                resolved_maps.append(hsp.HealSparseMap.read(str(pm)))    #resolved_maps.append(hsp.HealSparseMap.read(pm))
            elif isinstance(pm, hsp.HealSparseMap):
                resolved_maps.append(pm)
            else:
                raise TypeError(f"Each prop_map must be a str (path) or a HealSparseMap instance, not {type(pm)}")
        prop_maps = resolved_maps

        # Verify that all maps have the same resolution
        cov_res_set = {pm.nside_coverage for pm in prop_maps}
        sparse_res_set = {pm.nside_sparse for pm in prop_maps}

        if len(cov_res_set) != 1 or len(sparse_res_set) != 1:
            raise ValueError("All input property maps must have the same nside_coverage and nside_sparse")

        ops = {'gt': lambda v, t: v > t,
               'ge': lambda v, t: v >= t,
               'lt': lambda v, t: v < t,
               'le': lambda v, t: v <= t }

        # Start with the pixel set from the first map
        pixels = prop_maps[0].valid_pixels.copy()

        for i, (prop_map, threshold, comparison) in enumerate(zip(prop_maps, thresholds, comparisons)):
            if comparison not in ops:
                raise ValueError(f"Invalid comparison: {comparison}")

            this_pixels = prop_map.valid_pixels
            this_values = prop_map.get_values_pix(this_pixels)

            # Apply comparison and filter
            selected = this_pixels[ops[comparison](this_values, threshold)]

            # Intersect with current valid pixel set
            pixels = np.intersect1d(pixels, selected, assume_unique=True)

            if len(pixels) == 0:
                raise ValueError(f"0 pixels remaining after condition {i} ({comparison} {threshold})")
        
        # Build combined mask using geometry from the first map, i.e. prop_maps[0].nside_coverage
        mask = hsp.HealSparseMap.make_empty(prop_maps[0].nside_coverage, prop_maps[0].nside_sparse, dtype=np.bool_)
        mask[pixels] = True
        
        # Change to desired coverage order. This honors the input keyword parameter, which itself defaults
        # to the pipeline value (order_cov) when not specified
        if nside_cov != prop_maps[0].nside_coverage:
            print(f'--- Propertymap coverage order changed to {ord_cov}')
            mask = self.change_cov_order(mask, ord_cov, inplace=True, verbose=False)
            #mask = self.reproject_nside_coverage(mask, self.nside_cov)

        # Change to desired sparse order.This honors the input keyword parameter, which itself defaults
        # to the pipeline value (order_prop) when not specified
        if nside_sparse != prop_maps[0].nside_sparse:
            print(f'--- Propertymap sparse order changed to {order_sparse}')
            mask = self.change_sparse_order(mask, order_sparse, inplace=True, verbose=False)
        
        # Force packing if desired
        if bit_packed: mask = mask.as_bit_packed_map()

        # Store calling/useful info in its own dictionary
        area_deg2 = mask.get_valid_area(degrees=True)
        npix = mask.n_valid
        print('--- Propertymap mask area                       :', area_deg2)
        extra_meta = { 'prop_maps': pmstring, 'thresholds': thresholds, 'comparisons': comparisons, 
                       'bit_packed': bit_packed, 'pixels': npix, 'area_deg2': area_deg2 }
        return self._finalize_stage(mask, default_name='propmask', output_stage=output_stage, extra_meta=extra_meta)



    def build_patch_mask(self, patchfile=None, qafile=None, order_sparse=13, order_cov=None,
                         filt=None, bit_packed=False, output_stage: Optional[str] = None):
        """
        For a series of HSC patches, matches them against the QA table containing
        quality measurements, filter those that meet some depth/seeing/etc critera,
        and returns a pixelated map of all accepted patches.

        Parameters
        ----------
        patchfile : list of str
            HSC patch files (e.g. for hectomap, sping, autumn, aegis)
        qafile : str
            File with the table of patches with QA measurements
        order_sparse : int
            Pixelization order of patches
        filt : string
            Contition(s) to apply to patches. If None, all patches will be considered
        bit_packed : bool, optional
            If True, return the ouput as bit-packed boolean map
        output_stage : str, optional
            Name for the ouput stage. If None, defaults to then canonical name 'patchmask'
            
        Returns
        -------
        hsp_map
            Healsparse boolean map

        The filt keyword
        ----------------
        The filt keyword can be a string to filter which patches will be pixelized later. For example:
        filt='(imag_psf_depth>26) and (rmag_psf_depth>26.1)'. If filt=None, no filtering will be applied
        and all patches will be used.
        """

        print('BUILDING PATCH MAP >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # Check if user wants specific orders, otherwise get from defaults
        ord_cov = order_cov if order_cov is not None else self.order_cov

        nside_sparse = 1<<order_sparse
        nside_cov = 1<<ord_cov

        
        # Read patch qa list
        qatable = self.readQApatches(qafile)

        # Create the empty boolean map *up front*
        mask = hsp.HealSparseMap.make_empty(nside_cov, nside_sparse, dtype=np.bool_)

        for p in patchfile:
            fpx = self.filter_and_pixelate_patches(p, qatable, order=order_sparse, filt=filt)
            mask[fpx] = True

        # Force packing if desired
        if bit_packed: mask = mask.as_bit_packed_map()

        # Store calling/useful info in its own dictionary
        area_deg2 = mask.get_valid_area(degrees=True)
        npix = mask.n_valid
        print('--- Patch mask area                           :', area_deg2)
        extra_meta = { 'patchfile': patchfile, 'qafile': qafile, 'filt': filt, 'bit_packed': bit_packed, 
                       'pixels': npix, 'area_deg2': area_deg2 }
        return self._finalize_stage(mask, default_name='patchmask', output_stage=output_stage, extra_meta=extra_meta)
    


    @staticmethod
    def _is_lsdb_catalog(obj) -> bool:
        # Duck-typing: works without importing lsdb to check if object is a HATS catalog
        return hasattr(obj, "map_partitions") and hasattr(obj, "npartitions")

    @staticmethod
    def _extract_ra_dec(sources: Union[str, Table, pd.DataFrame], columns: Tuple[str, str]) -> Tuple[np.ndarray, np.ndarray]:
        """Return (ra, dec) numpy arrays from common source types."""
        colra, coldec = columns
        if isinstance(sources, str):
            tab = Table.read(sources)
            return np.asarray(tab[colra]), np.asarray(tab[coldec])
        if isinstance(sources, Table): return np.asarray(sources[colra]), np.asarray(sources[coldec])
        if isinstance(sources, pd.DataFrame): return sources[colra].to_numpy(), sources[coldec].to_numpy()
        raise TypeError("Unsupported source type for RA/Dec extraction.")

    @staticmethod
    def _footpartition(df: pd.DataFrame, pixel, *, order_foot: int, order_cov: int, columns: Tuple[str, str]):
        """
        Auxliliary for HATS/lsdb map_partitions. Given a partition (as pandas DataFrame),
        return a DataFrame with a 'pxs' column holding the pixelated sources.
        """
        nside_foot = 1 << order_foot
        nside_cov  = 1 << order_cov

        foot = hsp.HealSparseMap.make_empty(nside_cov, nside_foot, dtype=np.bool_)
        ra = df[columns[0]].to_numpy()
        dec = df[columns[1]].to_numpy()
        if ra.size:
            pixels = hp.ang2pix(nside_foot, ra, dec, nest=True, lonlat=True)
            foot.update_values_pix(pixels, True, operation="or")
        return pd.DataFrame({"pxs": foot.valid_pixels})


    @staticmethod
    def _pixels_from_sources(sources, nside_foot: int, columns: Tuple[str, str], *,
                             mapping: bool, order_foot: int, order_cov: int) -> np.ndarray:
        """
        For an input set of discrete sources, returns an array of pixel indices (nest scheme).
        Special-case is for HATS/lsdb with mapping=True to distribute the pixelization.
        """
        colra, coldec = columns

        # HATS/lsdb catalog case
        if SkyMaskPipe._is_lsdb_catalog(sources):
            print('--- Pixelating HATS catalog')
            srcs = sources  # keep original ref
            if mapping:
                print(f"    Partitions for mapping: {srcs.npartitions:<7}")
                meta = pd.DataFrame([{"pxs": 0}])
                # NOTE: include_pixel=True passes a 'pixel' column; we don't use it here but keeps parity
                pixdf = srcs.map_partitions(SkyMaskPipe._footpartition, include_pixel=True, meta=meta,
                        order_foot=order_foot, order_cov=order_cov, columns=columns).compute()
                return pixdf["pxs"].to_numpy(dtype=np.int64)
            else:
                # Local compute then plain ang2pix
                df = srcs[[colra, coldec]].compute()
                ra = df[colra].to_numpy()  ;  dec = df[coldec].to_numpy()
                return hp.ang2pix(nside_foot, ra, dec, nest=True, lonlat=True)

        # Table / DataFrame / file path case
        if isinstance(sources, (Table, pd.DataFrame, str)):
            print('--- Pixelating sources' + (f' from: {sources}' if isinstance(sources, str) else ''))
            ra, dec = SkyMaskPipe._extract_ra_dec(sources, columns)
            return hp.ang2pix(nside_foot, ra, dec, nest=True, lonlat=True)

        raise ValueError("sources must be a HATS/lsdb Catalog, str, astropy.table.Table, or pandas.DataFrame")


    def build_foot_mask(self, sources, *, order_sparse: int = 13, order_cov: Optional[int] = None,
                        columns: Iterable[str] = ("ra", "dec"), remove_isopixels: bool = False,
                        erode_borders: bool = False, mapping: bool = False, bit_packed: bool = False, 
                        output_stage: Optional[str] = None):
        """
        Create a footprint mask of a source catalog (from any astropy-supported table or HATS catalog),
        pixelated at a given order. Optionally remove isolated empty pixels and erode borders
        around empty zones. For details see remove_isopixels() and erode_borders().

        Parameters
        ----------
        sources : str | astropy.table.Table | pandas.DataFrame | HATS/lsdb Catalog
            Input catalog or path (any format supported by astropy, or a HATS catalog).
        order_sparse : int, optional
            Pixelization order of sources
        columns : (str, str)
            Names of the RA and Dec columns (degrees).
        remove_isopixels : bool
            Remove isolated (empty) pixels surrounded by 8 non-empty pixels.
        erode_borders : bool
            Detect and remove border pixels around holes.
        mapping : bool
            If True, distribute pixelization across HATS partitions (requires a running Dask cluster).
        bit_packed : bool
            If True, return the ouput as bit-packed boolean map
        output_stage : str, optional
            Name for the ouput stage. If None, defaults to then canonical name 'footmask'
            
        Returns
        -------
        healsparse.HealSparseMap
            The boolean (or bit-packed) mask is stored at `self.footmask` and also
            returned to prompt
        """
        columns = tuple(columns)
        # Check if user wants specific orders, otherwise get from defaults
        ord_cov = order_cov if order_cov is not None else self.order_cov

        nside_sparse = 1 << order_sparse
        nside_cov = 1 << ord_cov

        print("BUILDING FOOT MASK >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("    Order ::", order_sparse)

        # Prepare empty output footprint map
        mask = hsp.HealSparseMap.make_empty(nside_cov, nside_sparse, dtype=np.bool_)

        # Compute pixels exactly once (except for HATS+mapping which is distributed)
        pixels = self._pixels_from_sources(sources, nside_sparse, columns, mapping=mapping,
                                           order_foot=order_sparse, order_cov=ord_cov)

        # Update map values for pixels that have objects
        if pixels.size: mask.update_values_pix(pixels, True, operation="or")

        # Optional post-processing
        if remove_isopixels: mask = self.remove_isopixels(mask)
        if erode_borders: mask = self.erode_borders(mask)

        # Force packing if desired
        if bit_packed: mask = mask.as_bit_packed_map()

        # Store calling/useful info in its own dictionary
        area_deg2 = mask.get_valid_area(degrees=True)
        npix = mask.n_valid
        if hasattr(sources, "hc_structure") and hasattr(sources.hc_structure, "catalog_path"):
            sources_string = sources.hc_structure.catalog_path  # extract lsdb path
        elif isinstance(sources, (str, Path)):
            sources_string = str(sources)                        # get simply the string
        else:
            sources_string = "<mem>"                             # fallback when input from mem
        print('--- Foot mask area                         :', area_deg2)
        extra_meta = { 'sources': sources_string, 'columns': columns, 'remove_isopixels': remove_isopixels, 
                       'erode_borders': erode_borders, 'bit_packed': bit_packed, 'pixels': npix, 'area_deg2': area_deg2 }
        return self._finalize_stage(mask, default_name='footmask', output_stage=output_stage, extra_meta=extra_meta)



    @staticmethod
    def intersect_boolmask(mask1 : HealSparseMap, mask2 : HealSparseMap, 
                           bit_packed: Optional[bool] = None) -> HealSparseMap:
        """
        Intersect two arbitrary boolean masks in healsparse format.

        Parameters
        ----------
        mask1 : hsp_map
            Healsparse boolean map 1
        mask2 : hsp_map
            Healsparse boolean map 2
        bit_packed : bool
            If True, returns ouput as a bit-packed boolean map

        Returns
        -------
        hsp_map
            Healsparse boolean map
        """

        if mask1.nside_sparse != mask2.nside_sparse:
            raise Exception('Maps have different nside_sparse')

        if mask1.nside_coverage != mask2.nside_coverage:
            raise Exception('Maps have different nside_coverage')

        tmp = mask1 & mask2
        msk = mask2 & tmp

        # Preserve original packing
        if bit_packed:
            return msk.as_bit_packed_map()
        else:
            return msk


    @staticmethod
    def subtract_boolmask(mask1 : HealSparseMap, mask2 : HealSparseMap, 
                          bit_packed: Optional[bool] = None) -> HealSparseMap:
        """
        Subtract two arbitrary boolean masks in healsparse format.

        Parameters
        ----------
        mask1 : hsp_map
            Healsparse boolean map 1
        mask2 : hsp_map
            Healsparse boolean map 2
        bit_packed : bool
            If True, returns ouput as a bit-packed boolean map

        Returns
        -------
        hsp_map
            Healsparse boolean map
        """

        if mask1.nside_sparse != mask2.nside_sparse:
            raise Exception('Maps have different nside_sparse')

        if mask1.nside_coverage != mask2.nside_coverage:
            raise Exception('Maps have different nside_coverage')

        msk = (mask1 & (~mask2))

        # Preserve original packing
        if bit_packed:
            return msk.as_bit_packed_map()
        else:
            return msk


    def plot(self, stage: str = "mask", nr: int = 100_000, s: float = 0.5,
             figsize: Union[Tuple[float, float], List[float]] = [12, 6],
             clipra: Optional[Tuple[float, float]] = None, clipdec: Optional[Tuple[float, float]] = None,
             plot_circles: Optional[Dict[str, Any]] = False, plot_boxes: Optional[Dict[str, Any]] = False,
             ax: Optional[Axes] = None, **kwargs) -> Tuple[plt.Figure, Axes]:
        """
        Quickly visualize a mask stage by means of randoms points in an x-y plot (no WCS projection).
        Optionally plot circles and boxes to inspect areas masked by stars. If you need more precise
        sky plots, use plot_moc() and plot_srcs()

        Note boxes shoud not cross the 360/0 boundary, as this is a straight xy plot intended for speed.

        Parameters
        ----------
        stage : string
            Mask stage to plot, e.g. 'mask', 'foot', 'holemap', etc.
        nr : integer
            Number of randoms
        s : float
            Point size
        figsize : list of floats
            Figure size
        clipra : list of floats
            Plot limits in ra, e.g. clipra=[226.5,227.5]
        clipdec : list of floats
            Plot limits in dec, e.g. clipdec=[10.,11.]
        plot_circles : dict
            Overlay circles due to bright stars if set to a dictionary as explained below
        plot_boxes : bool
            Overlay boxes due to bright stars if set to a dictionary as explained below
        ax : axes
            If given, plot will be added to the axes object provided
        kwargs : kwargs
            Adittional keyword arguments passed to mataplolib.scatter()

        Circles and boxes dictionaries
        ------------------------------
        Below are examples of dictionaries to specify the circles/boxes to overplot:
         - plot_circles = {'data':'path/to/circles.fits', 'fmt':'fits', 'columns':['ra','dec','radius']}
         - plot_boxes = {'data':'path/to/boxes.csv', 'fmt':'csv', 'columns':['ra_c','dec_c','width', 'height']}
        """

        # Choose stage based on its name in a pipeline
        mk = getattr(self, stage)

        # Use randoms for scatter plot
        radec = self.makerans(stage=mk, nr=nr, file=None, rng=None)
        xx=radec['ra']
        yy=radec['dec']
        #xx, yy = hsp.make_uniform_randoms_fast(mk, nr)

        # Do plot ------------------------------------------
        if not(ax): fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(xx, yy, s=s, **kwargs)
        ax.set_title(stage)
        if clipra: ax.set_xlim(clipra)
        if clipdec: ax.set_ylim(clipdec)
        clipra=ax.get_xlim()  ;  clipdec=ax.get_ylim()

        if plot_circles:
            # Extract from dictionary
            dataloc = plot_circles['data']
            fmt = plot_circles['fmt']
            colra, coldec, colrad = plot_circles['columns']
            # Read stars and find those inside window
            stars = Table.read(dataloc, format=fmt)
            idx = (stars[colra]>clipra[0]) & (stars[colra]<clipra[1]) & (stars[coldec]>clipdec[0]) & (stars[coldec]<clipdec[1])
            ts = stars[idx]
            for i in range(len(ts)):
                star_ra, star_dec, star_rad = ts[colra][i], ts[coldec][i], ts[colrad][i]
                circ = plt.Circle((star_ra, star_dec), star_rad, color='r', fill=False, linewidth=0.4)
                ax.add_artist(circ)
                #print(i, (ts[colra][i], ts[coldec][i]), ts[colrad][i])

        if plot_boxes:
            # Extract from dictionary
            dataloc = plot_boxes['data']
            fmt = plot_boxes['fmt']
            ra_c, dec_c, width, height = plot_boxes['columns']
            # Read boxes and find boxes inside window
            boxes = Table.read(dataloc, format=fmt)
            boxes['corner_ra']=boxes[ra_c]-0.5*boxes[width]   # assume no box crosses 360 boundary
            boxes['corner_dec']=boxes[dec_c]-0.5*boxes[height]
            idxb = (boxes[ra_c]>clipra[0]) & (boxes[ra_c]<clipra[1]) & (boxes[dec_c]>clipdec[0]) & (boxes[dec_c]<clipdec[1])
            tsb = boxes[idxb]
            for i in range(len(tsb)):
                box_ra, box_dec, box_sx, box_sy = tsb['corner_ra'][i], tsb['corner_dec'][i], tsb[width][i], tsb[height][i]
                rec = plt.Rectangle((box_ra, box_dec), box_sx, box_sy, color='r', fill=False, linewidth=0.4)
                ax.add_artist(rec)
                #print(i, box_ra, box_dec, box_sx, box_sy)

        plt.tight_layout()
        #if not(ax): plt.show()


    def plot_srcs(self, ra: Union[Sequence[float], Any], dec: Union[Sequence[float], Any],
                  center: Optional[SkyCoord] = None, fov: Optional[Angle] = None, frame: str = "icrs",
                  projection: str = "SIN", figsize: Tuple[float, float] = (10, 5),
                  ax: Optional[Axes] = None, wcs: Optional[WCS] = None,
                  show: bool = False, marker: str = ".", s: float = 0.5, color: str = "k",
                  edgecolor: str = "none", alpha: float = 0.5, zorder: int = 8,
                  label: Optional[str] = None, **scatter_kwargs: Any) -> Tuple[plt.Figure, Axes, WCS] :
        """
        Overlay sources on the current figure with WCS axes (or create one if needed).

        Parameters
        ----------
        ra, dec : array-like in degrees
            RA/Dec of sources
        center : SkyCoord
            Center of plot (required on first call when ax/wcs are not provided)
        fov    : Angle
            Field of view (required on first call when ax/wcs are not provided)
        frame : string
            Coordinate frame. 'icrs' | 'galactic' | ...
        projection : string
            Projection type for WCS. 'SIN', 'AIT', 'TAN', etc.
        figsize : tuple
            Figure size
        ax, wcs : axes type, wcs type
            Axes and WCS objects. Pass these from a previous call to layer plots
        show : bool
            Call plt.show() if True
        marker, s, color, edgecolor : string, float, color, color
            Marker symbol, size, color and edge color
        alpha, zorder, label : float, integer, string
            Transparency, zorder and label for the set of points
        scatter_kwargs : scatter_kwargs
            Extra arguments passed to ax.scatter

        Returns
        -------
        fig, ax, wcs
            The figure, the axes and the WCS objects. Useful to build layered plots
        """

        lon = np.asanyarray(ra, dtype=float)
        lat = np.asanyarray(dec, dtype=float)

        # Create axes/WCS if not provided (first call)
        created = False
        if ax is None or wcs is None:
            if center is None or fov is None:
                raise ValueError("When ax/wcs are not provided, pass 'center' and 'fov'.")
            fig = plt.figure(figsize=figsize)
            with WCS(fig, fov=fov, center=center,
                    coordsys=frame, projection=projection,
                    rotation=Angle(0, u.deg)) as _wcs:
                ax = fig.add_subplot(1, 1, 1, projection=_wcs)
                # Basic formatting only once
                lon_c = ax.coords['ra']; lat_c = ax.coords['dec']
                lon_c.set_format_unit(u.deg, decimal=True, show_decimal_unit=True)
                lat_c.set_format_unit(u.deg, decimal=True, show_decimal_unit=True)
                ax.set_xlabel("ra"); ax.set_ylabel("dec")
                ax.grid(color="black", linestyle="dotted")
                wcs = _wcs
                created = True
        else:
            fig = ax.figure

        # Plot sources in world coordinates of the axes
        ax.scatter(lon, lat,
                   s=s, marker=marker, color=color, edgecolors=edgecolor,
                   alpha=alpha, zorder=zorder, label=label,
                   transform=ax.get_transform('world'), **scatter_kwargs)

        if show: plt.show()
        return fig, ax, wcs



    def apply(self, stage : str ='mask', cat: Union[pd.DataFrame, Table, None] = None, 
              columns: Sequence[str] = ("ra", "dec"), file: Optional[str] = None) -> Union[pd.DataFrame, Table]:
        """
        Apply a mask to a catalog (DataFrame/Astropy_Table) and optionally save it to disk.

        Parameters
        ----------
        stage: str
            Masking stage to use, e.g., 'mask', 'foot', 'holemap', etc.
        cat : pandas.DataFrame or astropy.table.Table
            Input catalog to which the mask will be applied.
        columns : list of str
            Columns for RA and DEC coordinates.
        file : str, optional
            Path to the output file where the result will be saved in parquet format. If None,
            result only returned.

        Returns
        -------
        pandas.DataFrame or astropy.table.Table
            The input catalog with the mask applied.
        """
        colra, coldec = columns
        # Choose stage based on its name in a pipeline
        mk = getattr(self, stage)

        idx = mk.get_values_pos(cat[colra], cat[coldec], lonlat=True)
        if file:
            cat[idx].to_parquet(file)
            print(str(len(cat[idx])),'sources within',stage,' written to:', file)
        else:
            print(str(len(cat[idx])),'sources within',stage)

        return cat[idx]



    @classmethod
    def read(cls, indir: str | os.PathLike, *, max_workers: int | None = None,
             io_block_rows: int = 200_000, per_worker_buffer_cap: int = 12_000_000,
             verbose: bool = True):
        """
        Read a SkyMaskPipe instance from disk. The input is a directoy containing a JSON file
        for metadata and a series of FITS file, one per stage. The FITS files hold the
        corresponding valid pixels with per-row Bitpack encoding. Read is perfomed in parallel
        across stages using ThreadPool workers.

        Stage geometry information is restored or inferred, and internal parameter dictionaries
        and scalars are also reinstated.

        Parameters
        ----------
        indir : str / PathLike
            Path to the input directory
        max_workers : int
            Number of workers to paralellize stage reading. Best practice is 1 worker per
            stage to be read.
        io_block_rows : int
            Number of rows to read per I/O block when streaming stage FITS files.
        per_worker_buffer_cap : int
            Maximum number of pixels to buffer per worker thread during loading.
        verbose : bool, default=True
            If True, prints status messages

        Returns
        -------
        pipe : SkyMaskPipe
            The pipeline object
        """
        indir = Path(indir)
        meta = json.loads((indir / "metadata.json").read_text(encoding="utf-8"))

        self = cls.__new__(cls)

        # ---- restore scalars
        for k, v in meta.get("scalars", {}).items():
            setattr(self, k, v)

        # ---- stages listed in metadata
        stages_meta = list(meta.get("stages", {}).items())  # [(name, {filename:...}), ...]

        # default workers: half CPUs, capped by number of stages (>=1)
        if max_workers is None:
            ncpu = os.cpu_count() or 2
            max_workers = max(1, min(len(stages_meta) or 1, max(1, ncpu // 2)))

        print_lock = threading.Lock()

        def _load_one(stage_name, filename):
            fpath = indir / filename
            if verbose:
                with print_lock:
                    print(f"[read] Loading {stage_name}  <- {filename}", flush=True)

            # >>> Exact signature from your original code <<<
            hspmap = _read_stage_fits_bitpack_fast(
                fpath,
                io_block_rows=io_block_rows,
                per_worker_buffer_cap=per_worker_buffer_cap,
                verbose=False,              # let this method handle user-facing prints
                print_lock=print_lock,      # still pass lock in case the loader uses it
                stage_name=stage_name,
            )

            if verbose:
                nside_cov = getattr(hspmap, "nside_coverage", "?")
                nside_sparse = getattr(hspmap, "nside_sparse", "?")
                n_valid = getattr(hspmap, "n_valid", "?")
                with print_lock:
                    print(
                        f"[read] Done  {stage_name}  (nside_cov={nside_cov}, "
                        f"nside_sparse={nside_sparse}, n_valid={n_valid})",
                        flush=True,
                    )
            return stage_name, hspmap

        # ---- load in parallel
        results = {}
        if stages_meta:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(_load_one, nm, info["filename"]) for nm, info in stages_meta]
                for fut in as_completed(futs):
                    nm, hspmap = fut.result()
                    results[nm] = hspmap

        # ---- attach each loaded stage (supports custom names)
        for nm, m in results.items():
            setattr(self, nm, m)

        # ---- infer geometry from first stage if missing
        first = next(iter(results.values()), None)
        if first is not None:
            def _order(nside: int) -> int:
                return int(round(math.log2(max(1, nside))))
            if not hasattr(self, "nside_cov"):
                self.nside_cov = int(first.nside_coverage)
            if not hasattr(self, "nside_out"):
                self.nside_out = int(first.nside_sparse)
            if not hasattr(self, "order_cov"):
                self.order_cov = _order(self.nside_cov)
            if not hasattr(self, "order_out"):
                self.order_out = _order(self.nside_out)

        # ---- restore _params
        self._params = dict(meta.get("params", {}))

        if verbose:
            with print_lock:
                print("[read] All stages attached.", flush=True)
        return self


    @staticmethod
    def read_single(fpath, *, io_block_rows: int = 200_000, per_worker_buffer_cap: int = 12_000_000,
        verbose: bool = True, stage_name: str | None = None, print_lock=None):
        """
        Read a single stage FITS file and return its HealSparseMap.

        Parameters
        ----------
        fpath : str or os.PathLike
            Path to the stage FITS file written by `write`.
        io_block_rows : int, optional
            Batch size for streaming I/O (passed to _read_stage_fits_bitpack_fast).
        per_worker_buffer_cap : int, optional
            Buffer cap (passed to _read_stage_fits_bitpack_fast).
        verbose : bool, optional
            If True, progress messages may be printed by the reader.
        stage_name : str or None, optional
            Name used only for logging; defaults to the file stem.
        print_lock : threading.Lock or None, optional
            Lock to serialize prints when verbose=True. A local lock is created if None.

        Returns
        -------
        HealSparseMap
            The loaded stage map
        """
        fpath = Path(fpath)
        if not fpath.is_file():
            raise FileNotFoundError(f"Stage FITS not found: {fpath}")

        if stage_name is None:
            stage_name = fpath.stem
        if print_lock is None:
            print_lock = threading.Lock()

        # Exact signature as in your codebase
        hspmap = _read_stage_fits_bitpack_fast(
            fpath,
            io_block_rows=io_block_rows,
            per_worker_buffer_cap=per_worker_buffer_cap,
            verbose=verbose,
            print_lock=print_lock,
            stage_name=stage_name)
        return hspmap


    @staticmethod
    def pix_in_zone(hsp : HealSparseMap, ra_min: float, ra_max: float, dec_min: float, dec_max: float, *,
                    dec_seg_deg: float = 1.0, pix_chunk: int = 5_000_000,
                    return_mode: str = "generator", dtype=None):
        """
        Stream valid pixels from a HealSparseMap inside an RA/Dec box, without loading the whole
        coverage into memory.

        Implementation notes:
          - Uses healpy.query_strip in RING numbering (nest=False).
          - Filters RA numerically (wrap-aware) using pix2ang with nest=False.
          - Converts surviving candidates to NEST with hp.ring2nest(nside, ipix)
            *before* calling hsp.get_values_pix(..., nest=True).
          - Works in batches to keep peak RAM flat.

        Parameters
        ----------
        hsp : healsparsemap
            Healsparse map of a stage
        ra_min, ra_max, dec_min, dec_max :  floats
            Ra-dec boundaries of the zone (in deg)
        dec_seg_deg : float
            Width of declination stripes to query the valid pixels on
        pix_chunk : int
            Batch size of pixels to process at once (ra filter+valid mask)
        return_mode : str
            If "array" return the entire pixel array. If "generator" returns an iterator over the chunks
        dtype : data type
            Data type of output pixels. Default to np.int64
        """
        if dtype is None: dtype = np.int64
        nside_sparse = hsp.nside_sparse   #int(getattr(hsp, "nside_sparse"))

        ra0_raw = float(ra_min)
        ra1_raw = float(ra_max)
        dec0 = max(-90.0, float(dec_min))
        dec1 = min( 90.0, float(dec_max))
        if dec1 < dec0:
            dec0, dec1 = dec1, dec0

        ra0 = ra0_raw % 360.0
        ra1 = ra1_raw % 360.0
        ra_wraps = ra0 > ra1  # window crosses 0°

        # Small Dec padding to avoid center/edge misses
        EPS_DEC = 1e-4
        dec0_pad = max(-90.0, dec0 - EPS_DEC)
        dec1_pad = min( 90.0, dec1 + EPS_DEC)

        seg = max(1e-3, float(dec_seg_deg))  # avoid tiny/zero bands

        def _iter_valid_pix():
            # Yield valid pixel chunks per Dec-band batch
            dec_lo = dec0_pad
            first_band = True
            tiny = 1e-12

            while dec_lo < dec1_pad - tiny:
                dec_hi = min(dec_lo + seg, dec1_pad)
                # Get colatitudes
                theta1 = np.radians(90.0 - dec_hi)
                theta2 = np.radians(90.0 - dec_lo)
                # Only the first and last bands are inclusive to avoid double-counting boundaries
                last_band = (dec_hi >= dec1_pad - tiny)
                inclusive = first_band or last_band

                # Query the pixel in a dec band. IMPORTANT -> only the RING numbering is implemented (nest=False)
                band_pix_ring = hp.query_strip(nside_sparse, theta1, theta2,
                                               inclusive=inclusive, nest=False)
                first_band = False

                if band_pix_ring.size:
                    # Process this band's pixels in manageable chunks
                    for i in range(0, band_pix_ring.size, pix_chunk):
                        sl_ring = band_pix_ring[i:i+pix_chunk]

                        # Compute RA for this chunk only (lonlat=True gives degrees) in RING scheme
                        lon, _lat = hp.pix2ang(nside_sparse, sl_ring, nest=False, lonlat=True)

                        # RA filter (wrap-aware)
                        if not ra_wraps:
                            ra_mask = (lon >= ra0) & (lon <= ra1)
                        else:
                            ra_mask = (lon >= ra0) | (lon <= ra1)
                        if not np.any(ra_mask):
                            continue

                        cand_ring = sl_ring[ra_mask]

                        # Convert *only the candidates* to NEST for the healsparse lookup
                        cand_nest = hp.ring2nest(nside_sparse, cand_ring)

                        # Validity mask from HealSparse (ask coverage only)
                        try:
                            vmask = hsp.get_values_pix(cand_nest, nest=True, valid_mask=True)
                            vmask = np.asarray(vmask, dtype=bool)
                        except TypeError:
                            vals = hsp.get_values_pix(cand_nest, nest=True)
                            if isinstance(vals, np.ma.MaskedArray):
                                vmask = ~np.asarray(vals.mask, dtype=bool)
                            else:
                                sentinel = getattr(hsp, "sentinel", None)
                                if sentinel is not None:
                                    vmask = np.asarray(vals != sentinel, dtype=bool)
                                else:
                                    vals = np.asarray(vals)
                                    vmask = np.isfinite(vals) if np.issubdtype(vals.dtype, np.number) else vals.astype(bool)

                        if np.any(vmask):
                            yield cand_nest[vmask].astype(np.uint64, copy=False)

                dec_lo = dec_hi

        if return_mode == "generator":
            # Stream results; caller processes chunk-by-chunk
            return _iter_valid_pix()

        elif return_mode == "array":
            # Stream into a compact buffer; single final conversion
            from array import array as pyarray
            buf = pyarray('Q')
            for block in _iter_valid_pix():
                if block.size:
                    buf.frombytes(np.asarray(block, dtype=np.uint64, copy=False).tobytes())
            out = np.frombuffer(buf, dtype=np.uint64)
            return out.astype(dtype, copy=False)

        else:
            raise ValueError("return_mode must be 'generator' or 'array'")


    @staticmethod
    def get_plot_order(wcs):
        """
        For a given mocpy WCS, estimate the optimal maximum order to display a moc so
        that one of its healpix pixels (of that order) falls within one display pixel on the figure.
        In general it is not worth to plot a moc at orders above this level.
        """
        # Extract WCS cdelt to get deg/px conversion factor
        cdelt = wcs.wcs.cdelt  #wcs.w.wcs.cdelt
        cdelt = np.abs((2 * np.pi / 360) * cdelt[0])
        # Minimum depth for which the resolution of a cell is contained in 1px
        depth_res = int(np.floor(np.log2(np.sqrt(np.pi / 3) / cdelt)))
        depth_res = max(depth_res, 0)
        return depth_res



    def plot_moc(self, stage: Union[HealSparseMap, str], center: Optional[SkyCoord] = None, 
                 fov: Optional[Angle] = None, clipra: Optional[tuple[float, float]] = None, 
                 clipdec: Optional[tuple[float, float]] = None, order_force: Optional[int] = None,
                 frame: str = "icrs", projection: str = "SIN", figsize: tuple[float, float] = (10.0, 5.0),
                 color: str = "green", alpha: float = 0.2, linewidth: float = 1.0, label: Optional[str] = None,
                 ax: Optional[Axes] = None, wcs: Optional[WCS] = None, show: bool = False, 
                 stream_pars: Optional[dict[str, object]] = None) -> tuple[Figure, Axes, WCS]:
        """
        Plot a MOC version of a given stage or healsparse map. Optionally clip pixels outside
        a given ra-dec box to speed up zoomed views of masks with high orders.

        Parameters
        ----------
        stage : hspmap or string
            Healsparse map-like or string corresponding to a SkyMaskPipe stage
        center : SkyCoord
            Center of plot (required on first call when ax/wcs are not provided)
        fov    : Angle
            Field of view (required on first call when ax/wcs are not provided)
        clipra : tuple[float,float]
            Clip healsparse pixels outside ra limis (in deg), before building the MOC
        clipdec : tuple[float,float]
            Clip healsparse pixels outside dec limis (in deg), before building the MOC
        frame : string
            Coordinate frame. 'icrs' | 'galactic' | ...
        projection : string
            Projection type for WCS. 'SIN', 'AIT', 'TAN', etc.
        order_force : int
            Force to plot the moc at this order
        figsize : tuple
            Figure size
        color, alpha, linewidth : matplotlib color, flot, float
            Color, transparency, border linewidth
        ax, wcs : axes type, wcs type
            Axes and WCS objects. Pass these from a previous call to layer plots
        show : bool
            Call plt.show() if True
        stream_pars : dict
            Optional dictionary with parameters passed to pix_in_zone() to control streaming

        Returns
        -------
        fig, ax, wcs : figure, axes, wcs
            The figure, the axes and the WCS objects. Useful to build layered plots
        """

        # Choose stage based on input healsparse map or the name of stage in a pipeline
        if hasattr(stage, 'valid_pixels'):
            stage = stage
        else:
            stage = getattr(self, stage)

        if stream_pars:
            if not(isinstance(stream_pars, dict)): raise Exception("stream_pars must be a valid dictionary")

        # Crop pixels outside box to speed plotting, if requested
        if (clipra is not None and clipdec is not None):
            print('Retrieving pixels inside clip box...')
            stream_pars = stream_pars or {}    # make sure we pass an empty dict al least
            pixels = self.pix_in_zone(stage, ra_min=clipra[0], ra_max=clipra[1],
                                      dec_min=clipdec[0], dec_max=clipdec[1], return_mode='array', **stream_pars)
        else:
            print('Retrieving pixels...')
            pixels = stage.valid_pixels
        print(f'Found {len(pixels)} pixels')

        # Create figure and WCS if appropiate
        created_context = False
        if ax is None or wcs is None:
            if center is None or fov is None:
                raise ValueError("When ax/wcs are not provided, you must pass center and fov.")
            fig = plt.figure(figsize=figsize)
            # Keep the WCS object to reuse later; we enter the context only for creation.
            with WCS(fig, fov=fov, center=center,
                    coordsys=frame, projection=projection,
                    rotation=Angle(0, u.deg)) as _wcs:
                ax = fig.add_subplot(1, 1, 1, projection=_wcs)
                # basic formatting only once (on first creation)
                lon = ax.coords['ra']; lat = ax.coords['dec']
                lon.set_format_unit(u.deg, decimal=True, show_decimal_unit=True)
                lat.set_format_unit(u.deg, decimal=True, show_decimal_unit=True)
                ax.set_xlabel("ra"); ax.set_ylabel("dec")
                ax.grid(color="black", linestyle="dotted")
                wcs = _wcs
                created_context = True
        else:
            fig = ax.figure

        print('Creating display moc from pixels...')
        order  = int(np.log2(stage.nside_sparse))
        order_display = self.get_plot_order(wcs)
        if order_force:
            print(f'MOC max_order is {order} --> degrading forcedly to {order_force}...')
            moc = MOC.from_healpix_cells(ipix=pixels, depth=order, max_depth=order_force)
        elif order_display < order :
            print(f'MOC max_order is {order} --> degrading to {order_display}...')
            moc = MOC.from_healpix_cells(ipix=pixels, depth=order, max_depth=order_display)
        else:
            print(f'MOC max_order is {order} --> no degrading...')
            moc = MOC.from_healpix_cells(ipix=pixels, depth=order, max_depth=order)
        # No need to actually degrade since moc is already created at the right order.
        # This has to be consistent with fill(...,optimize=False)
        # moc = moc.degrade_to_order(order_display)

        # Draw the MOC on the provided/created axes & wcs. Beware we set optimize=False because
        # the moc is already the right (possibly degraded) order
        print('Drawing plot...')
        moc.fill(ax=ax, wcs=wcs, alpha=alpha, fill=True, color=color, zorder=1, label=label, optimize=False)
        moc.border(ax=ax, wcs=wcs, alpha=max(0.6, alpha), color='k',
                   linewidth=linewidth, zorder=2)

        if show: plt.show()
        return fig, ax, wcs

    
    @staticmethod
    def frac_area_map(hsp_in: hsp.HealSparseMap, order_frac: int = 8, avg_edges: bool = False, 
                      edge_value_max: float | None = None, 
                      grow_k: int = 1, max_iters: int = 600, tol: float = 1e-4) -> hsp.HealSparseMap:
        """
        Compute the fractional area map of a healsparse map at a target HEALPix order. Optionally
        apply a harmonic averaging step to remove the artificial low-fraction ring of pixels at
        the footprint boundary.
                
        Parameters
        ----------
        hsp_in : healsparse.HealSparseMap
            Input map
        order_frac : int, default=8
            Order at which to compute the fractional coverage. Larger ``order_frac`` means finer pixels.
        avg_edges : bool, default=False
            If ``True``, post-process the fractional map to correct the systematically low values 
            that appear along the outer edge. Only a narrow **edge band**  (valid coarse pixels that 
            touch any UNSEEN neighbor) is modified. All interior pixels remain fixed.
        edge_value_max : float or None, default=None
            When ``avg_edges=True``, restrict pixel changes to edge pixels whose initial fractional 
            value is **≤ edge_value_max** (e.g., 0.15–0.30). Use this to target the undercounted pixels
            only. If ``None``, modify the entire edge band regardless of value.
        grow_k : int, default=1
            Grow the inpaint region inward by this many HEALPix **k-rings** from the detected edge 
            (0 = only the immediate edge ring).
        max_iters : int, default=600
            Maximum Gauss–Seidel iterations for the harmonic averaging on the HEALPix neighbor graph 
            (each updated pixel becomes the mean of its finite neighbors).
        tol : float, default=1e-4
            Convergence threshold on the maximum absolute change within the averaging region between 
            iterations. Iterations stop early when the change drops below this value.

        Returns
        -------
        HealSparseMap (float) with pixel values between 0 and 1
        """
        # Create fractional area map for input stage
        frac_hsp = hsp_in.fracdet_map(2**order_frac)

        # Average edges if requested
        if avg_edges:
            print(f'Fractions at edge pixels are being averaged')
            nside = int(frac_hsp.nside_sparse)
            npix  = 12 * nside * nside
        
            # Dense NEST HEALPix array
            arr_hp = frac_hsp.generate_healpix_map(nside=nside, nest=True).astype(float)
            valid  = arr_hp != hp.UNSEEN
            arr    = np.where(valid, arr_hp, np.nan)
        
            if not np.any(valid): return frac_hsp.copy()
        
            # Neighbors: shape (8, npix), with -1 where no neighbor
            pix   = np.arange(npix, dtype=np.int64)
            neigh = hp.get_all_neighbours(nside, pix, nest=True)
        
            # Find the edge ring: valid pixel with at least one UNSEEN neighbor
            has_unseen_neighbor = np.zeros(npix, dtype=bool)
            for i in range(8):
                nb = neigh[i]
                m = nb >= 0
                # Neighbor is "unseen" if not valid there
                has_unseen_neighbor[m] |= ~valid[nb[m]]
        
            edge0 = valid & has_unseen_neighbor
        
            # Optionally restrict to low-valued edge pixels
            if edge_value_max is not None:
                edge0 &= (arr <= float(edge_value_max))
        
            # Grow inward by k-rings within the valid domain
            inpaint = edge0.copy()
            for _ in range(int(grow_k)):
                touch = np.zeros(npix, dtype=bool)
                for i in range(8):
                    nb = neigh[i]
                    m  = nb >= 0
                    # any neighbor in current inpaint band?
                    touch[m] |= inpaint[nb[m]]
                inpaint |= (valid & touch)
        
            # Fixed (Dirichlet) set: all valid pixels not in the inpaint band
            fixed = valid & ~inpaint
        
            # If no unknown pixels (or no anchors), just return original
            idx_unknown = np.flatnonzero(inpaint)
            if idx_unknown.size == 0 or not np.any(fixed):  return frac_hsp.copy()
        
            # Initialize unknowns with neighbor means
            out = arr.copy()
            def neighbor_mean_for_unknowns(values, idx_unknown, neigh):
                Nu = idx_unknown.size
                sums   = np.zeros(Nu, dtype=float)
                counts = np.zeros(Nu, dtype=float)
                for i in range(8):
                    nb = neigh[i, idx_unknown]     # neighbors of each unknown pixel
                    m  = nb >= 0
                    if not np.any(m): 
                        continue
                    vals = values[nb[m]]
                    f = np.isfinite(vals)
                    if np.any(f):
                        sums[m]   += np.where(f, vals, 0.0)
                        counts[m] += f.astype(float)
                mean = np.full(Nu, np.nan, dtype=float)
                ok = counts > 0
                mean[ok] = sums[ok] / counts[ok]
                return mean
        
            init = neighbor_mean_for_unknowns(out, idx_unknown, neigh)
            out[idx_unknown] = np.where(np.isfinite(init), init, out[idx_unknown])
        
            # Gauss–Seidel harmonic updates on the inpaint band
            for _ in range(int(max_iters)):
                old = out[idx_unknown].copy()
                new = neighbor_mean_for_unknowns(out, idx_unknown, neigh)
                # keep previous value where no finite neighbors are available
                use = np.isfinite(new)
                out[idx_unknown[use]] = new[use]
                delta = np.nanmax(np.abs(out[idx_unknown] - old))
                if not np.isfinite(delta) or delta < tol:
                    break
        
            # Clip to [0,1] and convert back to HealSparse
            out = np.clip(out, 0.0, 1.0)
            hp_arr = np.where(np.isfinite(out), out, hp.UNSEEN)
            try:
                frac_hsp = hsp.HealSparseMap(healpix_map=hp_arr, nside_coverage=frac_hsp.nside_coverage, nest=True)
            except TypeError:
                frac_hsp = hsp.HealSparseMap.convert_healpix_map(hp_arr, nside_coverage=frac_hsp.nside_coverage, nest=True)

        
        print(f'Fractional area map created at order {order_frac}: {frac_hsp.n_valid} valid pixels')
        return frac_hsp

        
    
    def plot_fracmap(self, stage: Union[HealSparseMap, str], order_frac: int = 8, ax=None, wcs=None,
        center: SkyCoord | None = None, fov: Angle | None = None,
        frame: str = "icrs", projection: str = "SIN", figsize: tuple[float, float] = (10.0, 5.0),
        # image props
        vmin: float = 0.0, vmax: float = 1.0, cmap: str | None = None,
        alpha: float = 1.0, order: str | int | None = "nearest-neighbor",
        # contours
        thresholds: float | Sequence[float] | None = None,
        contour_smooth: dict | None = None, contour_kwargs: dict | None = None,
        contour_label: bool | str = False, contour_label_kwargs: dict | None = None,
        # z-order / colorbar
        zorder_img: float = 1.0, zorder_contour: float = 2.0,
        colorbar: bool = True,
        avg_edges: bool = False
    ):
        """
        Render a **fractional area map** for a pipeline stage or healsparse map and display it on WCS axes,
        optionally overlaying one or more isocontours. Works in two modes:
        
        Parameters
        ----------
        stage : healsparse.HealSparseMap or str
            The stage or map to visualize.
        order_frac : int, optional
            HEALPix order to compute the coarse fractional map. Larger orders give finer pixels. Default is 8.
        ax, wcs : matplotlib.axes._axes.Axes, mocpy.moc.WCS, optional
            When both are supplied, the function overlays the image onto the provided WCS view. If either 
            is `None`, a new figure is create at `center`+`fov`.
        center : astropy.coordinates.SkyCoord, optional
            Sky center for **Create mode**. Must be provided when `ax`/`wcs` are not.
        fov : astropy.coordinates.Angle, optional
            Field of view for **Create mode**. Must be provided when `ax`/`wcs` are not.
        frame : {'icrs', 'galactic', ...}, optional
            Coordinate system used to construct the WCS. Default 'icrs'.
        projection : {'SIN','TAN','AIT', ...}, optional
            WCS projection passed to `mocpy.WCS` (e.g., 'SIN' to match `plot_moc`). Default 'SIN'.
        figsize : tuple of float, optional
            Figure size when a new figure is created. Default (10.0, 5.0).
        vmin, vmax : float, optional
            Color stretch limits for the fractional image. Defaults 0.0–1.0.
        cmap : str or None, optional
            Matplotlib colormap name. If `None`, MPL default is used.
        alpha : float, optional
            Opacity of the fractional image layer. Default 1.0.
        order : {'nearest-neighbor','bilinear'}. Use `'nearest-neighbor'` for mask-like maps and `'bilinear'` 
            for softer display. Default 'nearest-neighbor'.
        thresholds : float or sequence of float, optional
            One or more fraction levels (e.g., `0.3`, or `[0.3, 0.5, 0.7]`) to draw as contours.
        contour_smooth : dict or None, optional
            Optional smoothing for the contours. Supported method is 
            `{'method': 'gaussian', 'sigma_pix': <float>}` where `sigma_pix` is the Gaussian
            sigma in **display pixels** (typ. 1–3). If `None`, no smoothing is applied.
        contour_kwargs : dict or None, optional
            Extra kwargs forwarded to `ax.contour` (e.g., `{'colors':'w','linewidths':1.2}`).
        contour_label : bool or str, optional
            If `True`, label each contour with a default format `f={level:.2f}`.
            If a string (e.g., `'f={level:.2f}'`), it is used as the label format.
        contour_label_kwargs : dict or None, optional
            Extra kwargs to `ax.clabel` (e.g., `{'fontsize': 9, 'inline_spacing': 10}`).
        zorder_img : float, optional
            Z-order for the fractional image (useful when layering under a MOC border). Default 1.0.
        zorder_contour : float, optional
            Z-order for contour lines. Default 2.0.
        colorbar : bool, optional
            If `True`, attach a colorbar. Default True.
        avg_edges : bool, optional
            If `True`, the fractional map produced is post-processed with a harmonic averaging that 
            fixes the the artificial low-fraction of boundary pixels. Default False.
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure used (existing or newly created).
        ax : WCSAxes
            The WCS-aware Matplotlib axes the content was drawn into.
        wcs : mocpy.moc.WCS
            The mocpy WCS object used to build the axes (handy for subsequent layers).
        im : matplotlib.image.AxesImage
            The image artist returned by `imshow` for the fractional map.
        CS : matplotlib.contour.QuadContourSet or None
            The contour set, if `thresholds` were provided; otherwise `None`.
        """
        
        # Choose stage based on input healsparse map or the name of stage in a pipeline
        if hasattr(stage, 'valid_pixels'):
            stage = stage
        else:
            stage = getattr(self, stage)

        # Create fractional area map for input stage
        frac_map = SkyMaskPipe.frac_area_map(stage, order_frac=order_frac, avg_edges=avg_edges)
        #frac_map = stage.fracdet_map(2**order_frac)
        #print(f'Fractional area map created at order {order_frac}: {frac_map.n_valid} valid pixels')
        
        # Create axes/wcs or take from the input keyworks ------------------------
        created_context = False
        if (ax is None) or (wcs is None):
            if center is None or fov is None:
                raise ValueError("When ax/wcs are not provided, you must pass `center` and `fov`.")
            fig = plt.figure(figsize=figsize)
            with WCS(fig, fov=fov, center=center, coordsys=frame,
                     projection=projection, rotation=Angle(0, u.deg)) as _wcs:
                ax = fig.add_subplot(1, 1, 1, projection=_wcs)
                # match your plot_moc axis formatting
                lon = ax.coords['ra']; lat = ax.coords['dec']
                lon.set_format_unit(u.deg, decimal=True, show_decimal_unit=True)
                lat.set_format_unit(u.deg, decimal=True, show_decimal_unit=True)
                ax.set_xlabel("ra"); ax.set_ylabel("dec")
                ax.grid(color="black", linestyle="dotted")
                wcs = _wcs
                created_context = True
        else:
            fig = ax.figure
    
        # Get the header from mocpy WCS and compute nx,ny from CRPIX1&2 -----------
        hdr = wcs.to_header()
        nx = int(round(2.0 * float(hdr['CRPIX1'])))
        ny = int(round(2.0 * float(hdr['CRPIX2'])))
        
        # Get rid of UNSEEN values and reproject -----------------------------------
        vals = frac_map.generate_healpix_map(nside=frac_map.nside_sparse, nest=True).astype(float)
        vals[vals == hp.UNSEEN] = np.nan
        arr, _ = reproject_from_healpix((vals, 'icrs'), ax.wcs, shape_out=(ny, nx), nested=True, order=order )
    
        # Draw image --------------------------------------------------------------
        im = ax.imshow(arr, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap,
                       alpha=alpha, zorder=zorder_img)
    
        # Draw contours  ----------------------------------------------------------
        CS = None
        if thresholds is not None:
            if np.isscalar(thresholds):
                levels = [float(thresholds)]
            else:
                levels = [float(t) for t in thresholds]
            levels = sorted({t for t in levels if np.isfinite(t)})
    
            ck = dict(zorder=zorder_contour)
            if contour_kwargs:
                ck.update(contour_kwargs)
    
            arr_for = np.array(arr, copy=True)
            finite = np.isfinite(arr_for)
    
            method = (contour_smooth or {}).get('method', None)
            if method in (None, 'gaussian'):
                if method == 'gaussian':
                    wgt = finite.astype(float)
                    a0  = np.where(finite, arr_for, 0.0)
                    sigma_pix = float((contour_smooth or {}).get('sigma_pix', 1.5))
                    num = gaussian_filter(a0,  sigma=sigma_pix, mode='nearest')
                    den = gaussian_filter(wgt, sigma=sigma_pix, mode='nearest')
                    arr_for = np.where(den > 0, num/den, np.nan)
    
                masked = np.ma.array(arr_for, mask=~finite)
                CS = ax.contour(masked, levels=levels, **ck)
    
                if contour_label:
                    fmt = contour_label if isinstance(contour_label, str) else 'f={level:.2f}'
                    fmt_map = {lvl: fmt.format(level=lvl) for lvl in levels}
                    lbl_kwargs = dict(inline=True, inline_spacing=10, fmt=fmt_map, fontsize=9)
                    if contour_label_kwargs: lbl_kwargs.update(contour_label_kwargs)
                    texts = ax.clabel(CS, CS.levels, **lbl_kwargs)
                    try:
                        import matplotlib.patheffects as pe
                        for t in texts:
                            t.set_path_effects([pe.withStroke(linewidth=1.5, foreground='black')])
                    except Exception:
                        pass
            else:
                raise ValueError(f"Unknown contour_smooth method: {method!r}")
    
        if colorbar:
            cb = plt.colorbar(im, ax=ax, pad=0.02)
            cb.set_label('masked fraction')
    
        # Return everything in case the user need to keep plotting
        return fig, ax, wcs, im, CS

        

    @staticmethod
    def add_moca(alwidget, stage: Union[HealSparseMap, str], color: str = "red", opacity: float = 0.25,
                 rows_per_batch: int = 2048, order_forced: Optional[int] = None) -> None:
        """
        Build a MOC from a SkyMaskPipe stage or a boolean HealSparseMap and add it
        as an overlay to an existing ipyaladin Aladin widget.

        This helper is intended to be called from :meth:`plot_moca` or directly when you
        already have an ``ipyaladin.Aladin`` instance open in a Jupyter notebook.

        Parameters
        ----------
        alwidget
            An active :class:`ipyaladin.Aladin` widget in which the MOC overlay will be drawn.
        stage : str or HealSparseMap
            Either the name of a SkyMaskPipe stage or a healsparse boolean map.
        color : str, optional
            CSS color name or hex code for both the MOC fill and border. Default is ``"red"``.
        opacity : float, optional
            Opacity of the MOC fill, between 0 (fully transparent) and 1 (opaque). Default is ``0.25``.
        rows_per_batch : int, optional
            Number of coverage rows to process per batch when building the MOC. Higher values can reduce overhead but use more memory. Default is ``2048``.
        order_forced : int, optional
            If provided, forces the generated MOC to be degraded this HEALPix order before adding it to the widget. Default is ``None``, 
            which keeps the native sparse order.

        Returns
        -------
        None
            Updates ``alwidget`` in place but does not return a value.
        """
        # Build the MOC at the chosen order
        #moc = SkyMaskPipe.moc_from_stage(stage, order_forced=order_forced)
        moc = SkyMaskPipe.moc_from_stage(stage, order_forced=order_forced, rows_per_batch=rows_per_batch)

        # Add the MOC, with good error reporting if it fails
        try:
            print('Adding MOC to widget...')
            alwidget.add_moc(moc, color=color, opacity=float(opacity),
                             fill=True, edge=True, fillColor=color)
        except Exception as e:
            # Build a concise diagnostic
            tname = type(moc_input).__name__
            preview = ""
            try:
                s = repr(moc_input)
                preview = (s[:200] + ("..." if len(s) > 200 else ""))
            except Exception:
                preview = "<unrepr-able>"
            raise RuntimeError(
                "Failed to add the MOC overlay via ipyaladin.add_moc(...).\n"
                f"- type(moc): {tname}\n"
                f"- preview: {preview}\n"
                "Hints:\n"
                "  • If it's a file path, pass it as a string or pathlib.Path to a valid MOC (FITS/JSON) file.\n"
                "  • If it's a URL, ensure it's directly fetchable and points to a MOC.\n"
                "  • If it's a mocpy object, ensure mocpy is installed and versions are compatible.\n"
            ) from e


    def plot_moca(self, stage: Union[HealSparseMap, str], color: str = "red", opacity: float = 0.25, 
                  colormap: str = "viridis", target: str = "268 -26", fov: float = 120.0, 
                  survey: str = "CDS/P/DM/I/355/gaiadr3", order_forced: Optional[int] = None):
        """
        Create an ipyaladin widget, load a HiPS survey, overlay a MOC, and set display
        customizations such as colormap and coordinate frame.

        This function is intended as a convenience wrapper around ipyaladin's `Aladin` widget.

        WARNING: this is experimental and can leave zombie processes in your browser/GPU. Close the
        browser tab and start again if you find issues.

        Parameters
        ----------
        stage : SkyMaskPipe stage or healsparse map
            Name of stage or a boolean healsparse map
        color : str, optional
            CSS color for MOC overlay fill and edge. Default is ``"red"``.
        opacity : float, optional
            Opacity for the MOC fill, between 0 (transparent) and 1 (opaque).
            Default is 0.25.
        colormap : str, optional
            Colormap to apply to the base HiPS survey (requires FITS tiles).
            Examples: ``"viridis"``, ``"inferno"``, ``"rainbow"``.
            Default is ``"viridis"``.
        target : str, optional
            Initial target position for the view. Can be ``"ra dec"`` in degrees,
            an object name resolvable by Aladin Lite, or a coordinate string.
            Default is the Galactic center, ``"268 -26"``.
        fov : float, optional
            Field of view in degrees. Must be positive. Default is 120.
        survey : str, optional
            HiPS survey identifier or base URL. Default is the Gaia DR3 density map,
            ``"CDS/P/DM/I/355/gaiadr3"``.
        order_forced : int, optional
            Force maximum order of moc passed to the widget. If None, the moc is built
            at the sparse order of the input stage. Default is None.

        Returns
        -------
        ipyaladin.Aladin
            The ipyaladin widget instance. The widget is also displayed.

        Notes
        -----
        * This helper displays the widget immediately via IPython's `display`.
          In a plain Python script, it will not render anything.

        Examples
        --------
        >>> from mocpy import MOC
        >>> moc = MOC.from_fits("example_moc.fits")
        >>> al = plot_moca(moc, color="blue", opacity=0.4, colormap="inferno")
        >>> # `al` is an ipyaladin.Aladin widget now visible in the notebook
        >>> # Use add_moca() method to build a moc from a stage and add it
        >>> # to an already existing widget
        """
        # Dependency check to detect if Aladin widget is installed
        try:
            from ipyaladin import Aladin
        except Exception as e:
            raise RuntimeError(
                "ipyaladin is required but not installed. Install with:\n"
                "  pip install ipyaladin\n"
                "Also ensure JupyterLab/Notebook has ipywidgets enabled."
            ) from e

        try:
            from IPython.display import display, Javascript
        except Exception as e:
            raise RuntimeError(
                "This function must run inside IPython (JupyterLab/Notebook)."
            ) from e

        # Create and show widget
        aladin = Aladin(target=str(target), fov=fov)
        display(aladin)

        # Load base survey, set colormap and frame
        aladin.survey = survey
        aladin.set_color_map(colormap)
        aladin.set_trait("coo_frame", "ICRSd")

        # Add the MOC, with good error reporting if it fails
        self.add_moca(aladin, stage, color=color, opacity=opacity, order_forced=order_forced)

        # Insist because sometimes these properties are not set
        aladin.set_color_map(colormap)
        aladin.set_trait("coo_frame", "ICRSd")

        return aladin


    def _resolve_stage_input(self, stage):
        """
        Normalize a `stage` (string stage name or HealSparse map) into (name, map).

        Returns
        -------
        (name, map)
            - name: canonical stage attribute name on `self` ("footmask", "propmask", ...),
                    or a descriptive label like "<map>" if the input is a raw map
                    not attached to the pipeline.
            - map : the HealSparseMap object.
        """
        name_map = {
            # add aliases if needed → canonical attribute names on `self`
            'footprint': 'foot',
            'patchmask': 'patchmask',
            'propmask': 'propmask',
            'circmask': 'circmask',
            'ellipmask': 'ellipmask',
            'starmask': 'starmask',
            'boxmask': 'boxmask',
            'zonemask': 'zonemask',
            'polymask': 'polymask',
            'mwmask': 'mwmask',
        }

        # Case 1: user passed a stage name (string)
        if isinstance(stage, str):
            key = name_map.get(stage.lower(), stage.lower())
            if not hasattr(self, key):
                raise AttributeError(f"Unknown stage '{stage}'")
            m = getattr(self, key)
            if m is None:
                raise ValueError(f"Stage '{stage}' is None")
            return key, m

        # Case 2: user passed a map object. Set a fallback label for anonymous maps not attached to self
        m = stage
        return "<map>", m


    def _empty_like_geometry(self, *, nside_cov, nside_sparse, bit_packed: bool):
        """Auxiliary method to create an empty healsparse map"""
        return hsp.HealSparseMap.make_empty(
            nside_coverage=int(nside_cov), nside_sparse=int(nside_sparse),
            dtype=np.bool_, bit_packed=bool(bit_packed) )


    def combine(self, positive, *, negative=None, order_out: Optional[int] = None,
                order_cov: Optional[int] = None, bit_packed: bool = True, verbose: bool = True, 
                output_stage: Optional[str] = None):
        """
        Combine multiple stage masks into a single mask using logical operations.

        This method supports complex boolean combinations of stage masks. The inputs in
        `positive` are grouped into OR or AND operations, while the masks in `negative`
        are subtracted at the pixel level. Coverage is handled row-by-row for efficiency,
        and input maps are automatically aligned to the requested coverage order and
        sparse order.

        Parameters
        ----------
        positive : list
            List of stage names or `HealSparseMap` objects to combine positively.
            - A string corresponds to a single stage mask.
            - A tuple or list groups multiple stages into a logical AND,
              before OR-ing with other groups.
        negative : list, optional
            List of stage names or `HealSparseMap` objects to subtract from
            the positives.
        order_out : int
            Order of the sparse (output) resolution.
        order_cov : int
            Order of the coverage resolution.
        bit_packed : bool, default=True
            If True, the resulting map is stored in bit-packed format. If
            False, it remains boolean.
        verbose : bool, default=True
            If True, prints progress messages about coverage alignment,
            stage orders, and final statistics.
        output_stage : str, optional
            Name for the ouput stage. If None, defaults to then canonical name 'mask'
            
        Returns
        -------
        mask : healsparse.HealSparseMap
            The combined mask as a `HealSparseMap` object. Stored in
            `self.mask`, with metadata attributes updated.

        Notes
        -----
        - Positives are processed row-by-row at a "work order" defined as the
          minimum of `order_out` and all positive stage orders. They are then
          expanded to the target order.
        - Negatives are converted per coverage row to the target order and
          subtracted before writing to the output map.
        - AND-groups within `positive` are intersected per row before being
          OR-ed with other groups.
        - Coverage mismatches across stages are resolved automatically using
          `change_cov_order`.

        Examples
        --------
        Combine two stages with AND, add a third stage with OR, and subtract
        a negative mask:

        >>> mask = mkp.combine(
        ...     positive=[("footmask", "propmask"), "polymask"],
        ...     negative=["starmask"],
        ...     order_out=15,
        ...     order_cov=4,
        ...     bit_packed=True,
        ...     verbose=True
        ... )
        """
        if not positive:
            raise ValueError("combine(): need at least one positive stage.")

        # Helpers   ==============================================================
        it_by_cov = globals().get("_iter_valid_by_covpix", None)

        def _iter_valid_by_cov_fallback(hspmap):
            vp = hspmap.valid_pixels
            if vp.size == 0:
                return
            nside_cov = int(hspmap.nside_coverage)
            nside_spa = int(hspmap.nside_sparse)
            ratio = nside_spa // nside_cov
            nfine = ratio * ratio
            cov = vp // nfine
            order = np.argsort(cov, kind='mergesort')
            vp = vp[order]; cov = cov[order]
            i = 0; n = vp.size
            while i < n:
                c = cov[i]
                j = i + 1
                while j < n and cov[j] == c:
                    j += 1
                yield c, vp[i:j]
                i = j

        def _iter_cov_rows(hspmap):
            return it_by_cov(hspmap) if callable(it_by_cov) else _iter_valid_by_cov_fallback(hspmap)

        def _order(nside: int) -> int:
            return int(round(math.log2(int(nside))))

        def _ensure_cov(name, m):
            # Aux function to change coverage order and display a status msg
            if verbose:
                print(f"[combine] aligning coverage for '{name}' : c{int(np.log2(m.nside_coverage))} → c{cov_ord}")
            return self.change_cov_order(m, order=cov_ord, inplace=False, verbose=False)

        class CovRowIter:
            __slots__ = ("gen", "peek", "done")
            def __init__(self, gen):
                self.gen = gen
                self.peek = None
                self.done = False
            def get_for_cov(self, cov_id):
                if self.done:
                    return None
                while True:
                    if self.peek is None:
                        try:
                            self.peek = next(self.gen)
                        except StopIteration:
                            self.done = True
                            self.peek = None
                            return None
                    if self.peek[0] < cov_id:
                        self.peek = None
                        continue
                    break
                if self.peek is not None and self.peek[0] == cov_id:
                    arr = self.peek[1]
                    try:
                        self.peek = next(self.gen)
                    except StopIteration:
                        self.done = True
                        self.peek = None
                    return arr
                return None

        def _parents_abs_for_row(m, arr_children, cov, s_ord):
            """Convert a row of a positive stage to work-order parents (absolute ids)."""
            if arr_children is None or arr_children.size == 0:
                return None
            if s_ord == work_ord:
                return np.unique(arr_children.astype(np.int64, copy=False))
            elif s_ord > work_ord:
                Delta = s_ord - work_ord
                r2 = 4**Delta
                ratio = int(m.nside_sparse // cov_ns)
                nfine_src = ratio * ratio
                base = cov * nfine_src
                parents_local = ((arr_children - base) // r2).astype(np.int64, copy=False)
                if parents_local.size == 0:
                    return None
                parents_local = np.unique(parents_local)
                parents_per_cov = 4**(work_ord - cov_ord)
                p_base = cov * parents_per_cov
                return p_base + parents_local
            else:
                # s_ord < work_ord should not happen (work_ord is min of all positives/orders)
                raise RuntimeError("Internal: work_order ended below a positive stage order.")

        def _neg_children_at_target_for_row(m, arr_pix, cov, s_ord):
            """
            Convert a row of negatives from stage order s_ord to target-order children.
            Coverage already equals cov_ns.
            [convert a negatives row to TARGET children (absolute ids)]
            """
            if arr_pix is None or arr_pix.size == 0:
                return None

            # Define geometry
            parents_per_cov_src = 4**(s_ord - cov_ord)
            base_src = cov * parents_per_cov_src

            nfine_tgt = 4**(tgt_ord - cov_ord)
            base_child = cov * nfine_tgt

            if s_ord == tgt_ord:
                # Already at target-order children (absolute)
                return arr_pix.astype(np.int64, copy=False)

            if s_ord < tgt_ord:
                # EXPAND coarser -> finer: parents at s_ord to children at tgt_ord
                r2 = 4**(tgt_ord - s_ord)
                parents_local = (arr_pix - base_src).astype(np.int64, copy=False)
                if parents_local.size == 0:
                    return None
                kids = (parents_local[:, None] * r2 + np.arange(r2, dtype=np.int64)).reshape(-1) + base_child
                return kids

            # s_ord > tgt_ord: DEGRADE finer -> coarser (map each fine child to its target child)
            r2 = 4**(s_ord - tgt_ord)
            local_child = ((arr_pix - base_src) // r2).astype(np.int64, copy=False)
            if local_child.size == 0:
                return None
            local_child = np.unique(local_child)
            return base_child + local_child

        # ======================================================================================
        # Check if user wants specific orders, otherwise get from defaults
        tgt_ord = order_out if order_out is not None else self.order_out
        cov_ord = order_cov if order_cov is not None else self.order_cov
        #tgt_ord  = int(order_out)
        #cov_ord  = int(order_cov)
        tgt_ns   = 1<<tgt_ord
        cov_ns   = 1<<cov_ord

        # Normalize positives into groups (strings -> singleton OR; tuples/lists -> AND group)
        pos_groups = []
        for it in positive:
            if isinstance(it, (tuple, list)):
                if len(it) == 0:
                    continue
                pos_groups.append([self._resolve_stage_input(nm) for nm in it])
            else:
                pos_groups.append([self._resolve_stage_input(it)])

        neg_maps = [self._resolve_stage_input(nm) for nm in (negative or [])]

        # Ensure coverage is the same for all stages
        pos_groups = [[_ensure_cov(nm, m) for nm, m in grp] for grp in pos_groups]
        neg_maps   = [_ensure_cov(nm, m) for nm, m in neg_maps]

        # These were all (stagename, stagemap) tuples. Since we already print them, just keep the stagemaps
        pos_groups = [[m for m in grp] for grp in pos_groups]
        neg_maps   = [m for m in neg_maps]


        # 1) Positives at work_order (support OR and grouped AND)   ======================================
        pos_maps_flat = [m for grp in pos_groups for m in grp]
        pos_orders = [_order(m.nside_sparse) for m in pos_maps_flat]
        work_ord = min([tgt_ord] + pos_orders)        # ≤ target order
        r2_up = 4**(tgt_ord - work_ord)               # children per work-order parent at target

        if verbose:
            print(f"[combine] target=(o{tgt_ord}, c{cov_ord}), work_order={work_ord}, r2={r2_up}")

        # Workspace at work_order to accumulate positives (parents)
        pos_work = self._empty_like_geometry(nside_cov=cov_ns, nside_sparse=1<<work_ord, bit_packed=True)

        # Build positives into pos_work
        for grp in pos_groups:
            if len(grp) == 1:
                m = grp[0]
                s_ord = _order(m.nside_sparse)
                if verbose:
                    print(f"[combine:+] stage at order {s_ord} -> work {work_ord}")
                if s_ord == work_ord:
                    for _, arr in _iter_cov_rows(m):
                        if arr.size:
                            pos_work.update_values_pix(arr, True, operation="replace")
                else:
                    Delta = s_ord - work_ord
                    r2 = 4**Delta
                    ratio = int(m.nside_sparse // cov_ns)
                    nfine_src = ratio * ratio
                    parents_per_cov = 4**(work_ord - cov_ord)
                    for cov, children in _iter_cov_rows(m):
                        if children.size == 0:
                            continue
                        base = cov * nfine_src
                        parents_local = ((children - base) // r2).astype(np.int64, copy=False)
                        if parents_local.size:
                            parents_local = np.unique(parents_local)
                            p_base = cov * parents_per_cov
                            parents = p_base + parents_local
                            pos_work.update_values_pix(parents, True, operation="replace")
            else:
                # AND-group
                grp_orders = [_order(m.nside_sparse) for m in grp]
                if verbose:
                    print(f"[combine:&] group of {len(grp)} stages: orders={grp_orders} -> work {work_ord}")
                iters = [CovRowIter(_iter_cov_rows(m)) for m in grp]
                for it in iters:  # prime
                    it.get_for_cov(-1)
                while True:
                    cov_candidates = [it.peek[0] for it in iters if it.peek is not None]
                    if not cov_candidates:
                        break
                    cov = min(cov_candidates)
                    parents_list = []
                    empty = False
                    for m, s_ord, it in zip(grp, grp_orders, iters):
                        children = it.get_for_cov(cov)
                        if children is None or children.size == 0:
                            empty = True
                            continue
                        parents_abs = _parents_abs_for_row(m, children, cov, s_ord)
                        if parents_abs is None or parents_abs.size == 0:
                            empty = True
                            continue
                        parents_list.append(parents_abs)
                    if not empty and parents_list:
                        inter = np.unique(parents_list[0])
                        for arr in parents_list[1:]:
                            inter = np.intersect1d(inter, np.unique(arr), assume_unique=True)
                            if inter.size == 0:
                                break
                        if inter.size:
                            pos_work.update_values_pix(inter, True, operation="replace")

        # 2) Negatives iterators (any order; convert per-row to target children)   ==============
        neg_states = [CovRowIter(_iter_cov_rows(m)) for m in neg_maps]
        neg_orders = [_order(m.nside_sparse) for m in neg_maps]

        # 3) Expand work parents -> target children, subtract negatives row-by-row   ======================
        out = self._empty_like_geometry(nside_cov=cov_ns, nside_sparse=tgt_ns, bit_packed=bit_packed)

        ratio = int(out.nside_sparse // cov_ns)
        nfine = ratio * ratio
        mask_row = np.empty(nfine, dtype=bool)

        parents_per_cov = 4**(work_ord - cov_ord)
        r2 = r2_up

        for cov, parents in _iter_cov_rows(pos_work):
            if parents.size == 0:
                continue
            base_child = cov * nfine
            # Parent indices in this row relative to p_base
            p_base = cov * parents_per_cov
            parents_local = (parents - p_base).astype(np.int64, copy=False)

            # Collect negatives for this row (convert each stage row to target children)
            neg_children = None
            if neg_states:
                parts = []
                for st, m, s_ord in zip(neg_states, neg_maps, neg_orders):
                    arr = st.get_for_cov(cov)
                    if arr is not None and arr.size:
                        arr_tgt = _neg_children_at_target_for_row(m, arr, cov, s_ord)
                        if arr_tgt is not None and arr_tgt.size:
                            parts.append(arr_tgt)
                if parts:
                    neg_children = np.concatenate(parts)

            if neg_children is None or neg_children.size == 0:
                # Expand blocks and write
                kids = (parents_local[:, None] * r2 + np.arange(r2, dtype=np.int64)).reshape(-1) + base_child
                out.update_values_pix(kids, True, operation="replace")
            else:
                # Mark positive children for this row
                mask_row.fill(False)
                idx = (parents_local[:, None] * r2 + np.arange(r2, dtype=np.int64)).reshape(-1)
                mask_row[idx] = True
                # Clear negative children
                off = (neg_children - base_child).astype(np.int64, copy=False)
                # Guard: only offsets within this row
                if off.size:
                    # clip to row [0, nfine)
                    sel = (off >= 0) & (off < nfine)
                    if np.any(sel):
                        mask_row[off[sel]] = False
                # Emit survivors
                loc = np.flatnonzero(mask_row)
                if loc.size:
                    out.update_values_pix(base_child + loc, True, operation="replace")

        # 4) Finalize  ==================================================================================
        res = out
        self.order_out = tgt_ord
        self.order_cov = cov_ord
        self.nside_out = tgt_ns
        self.nside_cov = cov_ns

        if verbose:
            area = float(res.get_valid_area(degrees=True))
            print(f"[combine] done: order_out={tgt_ord} (NSIDE={tgt_ns}), "
                  f"order_cov={cov_ord} (NSIDE={cov_ns}), "
                  f"valid_pix={res.n_valid:,}, area={area:.3f} deg², bit_packed={bit_packed}")

        extra_meta = {}
        return self._finalize_stage(res, default_name='mask', output_stage=output_stage, extra_meta=extra_meta)


    
    def change_sparse_order(self, stage: Union[str, HealSparseMap], order: int, *, inplace: bool = True, 
                            verbose: bool = True) -> HealSparseMap:
        """
        Change the sparse resolution for a boolean (or bit-packed boolean) stage or HealSparseMap,
        while preserving coverage and bit-packing.

        Parameters
        ----------
        stage : str or HealSparseMap
            Name of a SkymaskPipe stage or an arbitrary HealSparseMap (boolean or bit-packed)
        order : int
            Target sparse order
        inplace : bool, default=True
            If True, modifies the stage of the pipeline. If False, leaves the stage untouched and returns a new map.
            Only applies if `stage` is a string
        verbose : bool, default=True
            If True, prints status messages

        Returns
        -------
        HealSparseMap
            A HealSparseMap at the requested sparse order (modified in place or as new object.
            The encoding matches the input (boolean or bit-packed)
        """
        # Resolve input (stage name or map)
        name_label, src = self._resolve_stage_input(stage)
        attr_name = None if name_label == "<map>" else name_label

        # Set geometry & bitpacking policy
        c_ns  = int(src.nside_coverage)
        s_ns  = int(src.nside_sparse)
        c_ord = int(round(np.log2(c_ns)))
        s_ord = int(round(np.log2(s_ns)))
        t_ord = int(order)
        if t_ord < c_ord:
            raise ValueError(f"Target sparse order {t_ord} < coverage order {c_ord}.")
        tgt_ns = 1<<t_ord

        src_is_packed = bool(getattr(src, "is_bit_packed_map", False))
        if verbose:
            pack_label = "bit-packed" if src_is_packed else "bool"
            who = attr_name if attr_name else name_label

        # Build regular boolean at source geometry with the same pixels
        src_bool = hsp.HealSparseMap.make_empty(nside_coverage=c_ns, nside_sparse=s_ns, dtype=np.bool_)
        vp = src.valid_pixels
        if vp.size: src_bool.update_values_pix(vp, True)

        # Resample accordingly > do nothing, upgrade, degrade
        if t_ord == s_ord:
            out_reg = src_bool
        elif t_ord > s_ord:
            if verbose:
                print(f"[change_sparse_order] {who} requested upgrade o{s_ord} → o{t_ord}")
            out_reg = src_bool.upgrade(tgt_ns)
        else:
            if verbose:
                print(f"[change_sparse_order] {who} requested downgrade o{s_ord} → o{t_ord}")
            out_reg = src_bool.degrade(tgt_ns).astype(np.bool_)

        # Preserve original packing
        out = out_reg.as_bit_packed_map() if src_is_packed else out_reg

        # In-place update if applicable
        if attr_name and inplace: setattr(self, attr_name, out)

        if verbose:
            final_pack = "bit-packed" if getattr(out, "is_bit_packed_map", False) else "bool"
            print(f"[change_sparse_order] {who} done: o{t_ord} (NSIDE={tgt_ns}), cov=o{c_ord}, "
                  f"n_valid={out.n_valid:,}, encoding={final_pack}")

        return out


    def change_cov_order(self, stage: Union[str, HealSparseMap], order: int, *, inplace: bool = True, 
                         verbose: bool = True):
        """
        Change coverage order (nside_coverage) while preserving sparse resolution and bitpacking

        Parameters
        ----------
        stage : str or HealSparseMap
            Name of a SkymaskPipe stage or an arbitrary HealSparseMap (boolean or bit-packed)
        order : int
            Target coverage order
        inplace : bool, default=True
            If True, modifies the stage of the pipeline. If False, leaves the stage untouched and returns a new map.
            Only applies if `stage` is a string
        verbose : bool, default=True
            If True, prints status messages

        Returns
        -------
        HealSparseMap
            A HealSparseMap at the requested coverage order (modified in place or as new object.
            The encoding matches the input (boolean or bit-packed)
        """
        # Resolve input (stage name or map)
        name_label, src = self._resolve_stage_input(stage)
        attr_name = None if name_label == "<map>" else name_label

        # Set geometry & bitpacking policy
        s_ns  = int(src.nside_sparse)
        s_ord = int(round(np.log2(s_ns)))
        c_ns  = int(src.nside_coverage)
        c_ord = int(round(np.log2(c_ns)))
        t_cov_ord = int(order)
        if t_cov_ord > s_ord:
            raise ValueError(f"Coverage order {t_cov_ord} cannot exceed sparse order {s_ord}.")
        tgt_cov_ns = 1<<t_cov_ord

        src_is_packed = bool(getattr(src, "is_bit_packed_map", False))
        if verbose:
            pack_label = "bit-packed" if src_is_packed else "bool"
            who = attr_name if attr_name else name_label

        if t_cov_ord == c_ord:
            out = src
        else:
            # Rebuild map at new coverage and same sparse order
            if verbose:
                print(f"[change_cov_order] {who} requested change: c{c_ord} → c{t_cov_ord}  (sparse=o{s_ord}, keep={pack_label})")
            out_reg = hsp.HealSparseMap.make_empty(nside_coverage=tgt_cov_ns,
                                                   nside_sparse=s_ns,
                                                   dtype=np.bool_)
            vp = src.valid_pixels
            if vp.size: out_reg.update_values_pix(vp, True)
            out = out_reg.as_bit_packed_map() if src_is_packed else out_reg

        # in-place update if applicable
        if attr_name and inplace: setattr(self, attr_name, out)

        if verbose:
            final_pack = "bit-packed" if getattr(out, "is_bit_packed_map", False) else "bool"
            print(f"[change_cov_order] {who} done: c{t_cov_ord} (NSIDE={tgt_cov_ns}), sparse=o{s_ord}, "
                  f"n_valid={out.n_valid:,}, encoding={final_pack}")

        return out

    

    def build_circ_mask(self, data: Union[pd.DataFrame, str, Path] = None, order_sparse: int = 15,
            order_cov: Optional[int] = None, fmt: str = 'ascii', columns: Optional[Sequence[str]] = ['ra','dec','radius'],
            bit_packed: bool = False, n_threads: int = 4, chunk_size: int = 600_000, output_stage: Optional[str] = None):

        """
        Build a mask from input circle data and store it as a HealSparse boolean map.

        Each input row describes a circle in the sky, defined as [ra_center, dec_center, radius]
        in degrees, whera ra_center,dec_center are the center coordinates. The circles are
        pixelized at `order_sparse` to produce the set of HEALPix pixels covering it.

        Parameters
        ----------
        data : pandas.DataFrame, str, or pathlib.Path
            The circle definitions. If a DataFrame, it must contain the columns named
            in `columns`. If a path (string or Path), it is read according
            to `fmt` (delegated to astropy).
        order_sparse : int, optional
            Sparse order to pixelize the circles
        order_cov : int, optional
            Coverage order. If None, falls back to `self.order_cov`
        fmt : str, default "ascii"
            File format used when `data` is a path (e.g., "ascii", "parquet",
            "csv", or any format supported by the astropy readers)
        columns : sequence of str, optional
            Columns for the center coordinates and radius. If None, defaults
            to ["ra", "dec", "radius"]`
        bit_packed : bool, optional
            If True, return the ouput as bit-packed boolean map
        n_threads : int, default=4
            Number of threads to use in the pixelization step
        chunk_size : int, default=600000
            Number circles pixelized at once. Watch out memory if chunk_size and n_threads are both large
        output_stage : str, optional
            Name for the ouput stage. If None, defaults to then canonical name 'circmask'
            
        Returns
        -------
        HealSparseMap
            The boolean (or bit-packed) mask is stored at `self.circmask` and also
            returned to prompt
        """
        print('BUILDING CIRCLES MASK >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # Check if user wants specific orders, otherwise get from defaults
        ord_cov = order_cov if order_cov is not None else self.order_cov

        nside_sparse = 1<<order_sparse
        nside_cov = 1<<ord_cov

        # Create the empty boolean map up front
        mask = hsp.HealSparseMap.make_empty(nside_cov, nside_sparse, dtype=np.bool_)

        # Perform pixelization
        self.pixelate_circles(data, mask, fmt=fmt, order=order_sparse, columns=columns,
                              n_threads=n_threads, chunk_size=chunk_size)
        #if pix is not None and len(pix) > 0:
        #    self.circmask.update_values_pix(pix, True)

        # Force packing if desired
        if bit_packed: mask = mask.as_bit_packed_map()

        # Store calling/useful info in its own dictionary
        area_deg2 = mask.get_valid_area(degrees=True)
        npix = mask.n_valid
        print('--- Circles mask area                        :', area_deg2)
        extra_meta = { 'data': (str(data) if isinstance(data, (str, Path)) else "<mem>"), 'fmt': fmt, 
                       'columns': columns, 'bit_packed': bit_packed, 'n_threads':n_threads, 
                       'pixels': npix, 'area_deg2': area_deg2 }
        return self._finalize_stage(mask, default_name='circmask', output_stage=output_stage, extra_meta=extra_meta)


    
    def build_box_mask(self, data: Union[pd.DataFrame, str, Path] = None, order_sparse: int = 15,
            order_cov: Optional[int] = None, fmt: str = 'ascii',
            columns: Optional[Sequence[str]] = ['ra_c','dec_c','width','height'],
            bit_packed: bool = False, n_threads: int = 4, output_stage: Optional[str] = None):

        """
        Build a mask from input box data and store it as a HealSparse boolean map.

        Each input row describes a box in the sky, defined as [ra_center, dec_center,
        width, height] in degrees, whera ra_center,dec_center are the center coordinates.
        The boxes are pixelized at `order_sparse` to produce the set of HEALPix pixels covering it.

        Parameters
        ----------
        data : pandas.DataFrame, str, or pathlib.Path
            The box definitions. If a DataFrame, it must contain the columns named
            in `columns`. If a path (string or Path), it is read according
            to `fmt` (delegated to astropy).
        order_sparse : int, optional
            Sparse order to pixelize the boxes
        order_cov : int, optional
            Coverage order. If None, falls back to `self.order_cov`
        fmt : str, default "ascii"
            File format used when `data` is a path (e.g., "ascii", "parquet",
            "csv", or any format supported by the astropy readers)
        columns : sequence of str, optional
            Columns for the center coordinates and width, height. If None, defaults
            to ["ra_c", "dec_c", "width", "height"]`
        bit_packed : bool, optional
            If True, return the ouput as bit-packed boolean map
        n_threads : int, default=4
            Number of threads to use in the pixelization step
        output_stage : str, optional
            Name for the ouput stage. If None, defaults to then canonical name 'boxmask'

        Returns
        -------
        HealSparseMap
            The boolean (or bit-packed) mask is stored at `self.boxmask` and also
            returned to prompt

        """
        print('BUILDING BOXES MASK >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # Check if user wants specific orders, otherwise get from defaults
        ord_cov = order_cov if order_cov is not None else self.order_cov

        nside_sparse = 1<<order_sparse
        nside_cov = 1<<ord_cov

        # Create the empty boolean map *up front*
        mask = hsp.HealSparseMap.make_empty(nside_cov, nside_sparse, dtype=np.bool_)

        # Perform pixelization
        pix = self.pixelate_boxes(data, fmt=fmt, order=order_sparse, columns=columns, n_threads=n_threads)
        if pix is not None and len(pix) > 0:
            mask.update_values_pix(pix, True)

        # Force packing if desired
        if bit_packed: mask = mask.as_bit_packed_map()

        # Store calling/useful info in its own dictionary
        area_deg2 = mask.get_valid_area(degrees=True)
        npix = mask.n_valid
        print('--- Boxes mask area                           :', area_deg2)
        extra_meta = { 'data': (str(data) if isinstance(data, (str, Path)) else "<mem>"), 'fmt': fmt, 
                       'columns': columns, 'bit_packed': bit_packed, 'n_threads':n_threads, 
                       'pixels': npix, 'area_deg2': area_deg2 }
        return self._finalize_stage(mask, default_name='boxmask', output_stage=output_stage, extra_meta=extra_meta)

        

    def build_ellip_mask(self, data: Union[pd.DataFrame, str, Path] = None, order_sparse: int = 15,
            order_cov: Optional[int] = None, fmt: str = 'ascii',
            columns: Optional[Sequence[str]] = ['ra','dec','a','b','pa'],
            bit_packed: bool = False, output_stage: Optional[str] = None):

        """
        Build a mask from input ellipse data and store it as a HealSparse boolean map.

        Each input row describes an ellipse in the sky, defined as [ra, dec, a, b, pa]
        in degrees, where ra,dec are center coordinates; a,b are the mayor and minor half axes;
        and pa is the position angle. The ellipses are pixelized at `order_sparse` to
        produce the set of HEALPix pixels covering it.

        Parameters
        ----------
        data : pandas.DataFrame, str, or pathlib.Path
            The ellipse definitions. If a DataFrame, it must contain the columns named
            in `columns`. If a path (string or Path), it is read according
            to `fmt` (delegated to astropy).
        order_sparse : int, optional
            Sparse order to pixelize the ellipses
        order_cov : int, optional
            Coverage order. If None, falls back to `self.order_cov`
        fmt : str, default "ascii"
            File format used when `data` is a path (e.g., "ascii", "parquet",
            "csv", or any format supported by the astropy readers)
        columns : sequence of str, optional
            Columns for the ellipse data. If None, defaults to ["ra", "dec",
            "a", "b", "pa"]`
        bit_packed : bool, optional
            If True, return the ouput as bit-packed boolean map
        output_stage : str, optional
            Name for the ouput stage. If None, defaults to then canonical name 'ellipmask'
            
        Returns
        -------
        HealSparseMap
            A boolean (or bit-packed) mask is both, stored at `self.ellipmask` and returned
            to prompt
        """

        print('BUILDING ELLIPSES MASK >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # Check if user wants specific orders, otherwise get from defaults
        ord_cov = order_cov if order_cov is not None else self.order_cov

        nside_sparse = 1<<order_sparse
        nside_cov = 1<<ord_cov

        # Create the empty boolean map *up front*
        mask = hsp.HealSparseMap.make_empty(nside_cov, nside_sparse, dtype=np.bool_)

        # Perform pixelization
        pix = self.pixelate_ellipses(data, fmt=fmt, order=order_sparse, columns=columns)
        if pix is not None and len(pix) > 0:
            mask.update_values_pix(pix, True)

        # Force packing if desired
        if bit_packed: mask = mask.as_bit_packed_map()

        # Store calling/useful info in its own dictionary
        area_deg2 = mask.get_valid_area(degrees=True)
        npix = mask.n_valid
        print('--- Ellipses mask area                        :', area_deg2)
        extra_meta = { 'data': (str(data) if isinstance(data, (str, Path)) else "<mem>"), 'fmt': fmt, 
                       'columns': columns, 'bit_packed': bit_packed, 'pixels': npix, 'area_deg2': area_deg2 }
        return self._finalize_stage(mask, default_name='ellipmask', output_stage=output_stage, extra_meta=extra_meta)


    

    def build_poly_mask(self, data: Union[pd.DataFrame, str, Path] = None, order_sparse: int = 15,
            order_cov: Optional[int] = None, fmt: str = 'ascii',
            columns: Optional[Sequence[str]] = ['ra0','ra1','ra2','ra3','dec0','dec1','dec2','dec3'],
            bit_packed: bool = False, n_threads: int = 4, output_stage: Optional[str] = None):

        """
        Build a mask from input polygon data and store it as a HealSparse boolean map.

        Each input row describes a sky quadrangular polygon via 4 corner points
        (ra0, ra1, ra2, ra3, dec0, dec1, dec2, dec3) in degrees. The rectangles are
        pixelized at `order_sparse` to produce the set of HEALPix pixels covering it.

        Parameters
        ----------
        data : pandas.DataFrame, str, or pathlib.Path
            The polygon definitions. If a DataFrame, it must contain the columns named
            in `columns`. If a path (string or Path), it is read according
            to `fmt` (delegated to astropy).
        order_sparse : int, optional
            Sparse order to pixelize the polygons
        order_cov : int, optional
            Coverage order. If None, falls back to `self.order_cov`
        fmt : str, default "ascii"
            File format used when `data` is a path (e.g., "ascii", "parquet",
            "csv", or any format supported by the astropy readers)
        columns : sequence of str, optional
            Columns for the four corners. If None, defaults to ["ra0", "ra1", "ra2", "ra3"
            'dec0','dec1','dec2','dec3']`
        bit_packed : bool, optional
            If True, return the ouput as bit-packed boolean map
        n_threads : int, default=4
            Number of threads to use in the pixelization step
        output_stage : str, optional
            Name for the ouput stage. If None, defaults to then canonical name 'polymask'
            
        Returns
        -------
        HealSparseMap
            The boolean (or bit-packed) mask is stored at `self.polymask` and also
            returned to prompt
        """

        print('BUILDING POLYGON MASK >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # Check if user wants specific orders, otherwise get from defaults
        ord_cov = order_cov if order_cov is not None else self.order_cov

        nside_sparse = 1<<order_sparse
        nside_cov = 1<<ord_cov

        # Create the empty boolean map *up front*
        mask = hsp.HealSparseMap.make_empty(nside_cov, nside_sparse, dtype=np.bool_)

        # Perform pixelization
        pix = self.pixelate_polys(data, fmt=fmt, order=order_sparse, columns=columns, n_threads=n_threads)
        if pix is not None and len(pix) > 0:
            mask.update_values_pix(pix, True)

        # Force packing if desired
        if bit_packed: mask = mask.as_bit_packed_map()

        # Store calling/useful info in its own dictionary
        area_deg2 = mask.get_valid_area(degrees=True)
        npix = mask.n_valid
        print('--- Polygon mask area                           :', area_deg2)
        extra_meta = { 'data': (str(data) if isinstance(data, (str, Path)) else "<mem>"), 'fmt': fmt, 
                       'columns': columns, 'bit_packed': bit_packed, 'n_threads':n_threads, 
                       'pixels': npix, 'area_deg2': area_deg2 }
        return self._finalize_stage(mask, default_name='polymask', output_stage=output_stage, extra_meta=extra_meta)

    

    def build_zone_mask(self, data: Union[pd.DataFrame, str, Path] = None,
                        order_sparse: int = 15, order_cov: Optional[int] = None,
                        fmt: str = 'ascii',
                        columns: Optional[Sequence[str]] = ['ra1','dec1','ra2','dec2'],
                        bit_packed: bool = False, output_stage: Optional[str] = None):
        """
        Build a boolean HealSparse mask from rectangular sky zones. A zone is defined
        by ra boundaries (major circles) and dec limits (minor circles).

        Each input row describes a sky-aligned rectangle via two corner points
        (ra1, dec1) and (ra2, dec2) in degrees. The rectangles are pixelized at
        `order_sparse` to produce the set of HEALPix pixels covering it.

        Parameters
        ----------
        data : pandas.DataFrame | str | pathlib.Path | None
            The zone definitions. If a DataFrame, it must contain the columns named
            in `columns`. If a path (string or Path), it is read according
            to `fmt` (delegated to astropy).
        order_sparse : int, optional
            Sparse order to pixelize the zones
        order_cov : int, optional
            Coverage order. If None, falls back to `self.order_cov`
        fmt : str, default "ascii"
            File format used when `data` is a path (e.g., "ascii", "parquet",
            "csv", or any format supported by the astropy readers)
        columns : sequence of str, optional
            Columns for the two corners. If None, defaults to ("ra1", "dec1", "ra2", "dec2")
        bit_packed : bool, optional
            If True, return the ouput as bit-packed boolean map
        output_stage : str, optional
            Name for the ouput stage. If None, defaults to then canonical name 'zonemask'
            
        Returns
        -------
        HealSparseMap
            The boolean (or bit-packed) mask is stored at `self.circmask` and also
            returned to prompt

        See Also
        --------
        pixelate_zones : Converts rectangular zones to pixels at a given order
        """
        print('BUILDING ZONES MASK >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # Check if user wants specific orders, otherwise get from defaults
        ord_cov = order_cov if order_cov is not None else self.order_cov

        nside_sparse = 1<<order_sparse
        nside_cov = 1<<ord_cov

        # Create the empty boolean map *up front*
        mask = hsp.HealSparseMap.make_empty(nside_cov, nside_sparse, dtype=np.bool_)

        # Perform pixelization
        pix = self.pixelate_zones(data, fmt=fmt, order=order_sparse, columns=columns)
        if pix is not None and len(pix) > 0:
            mask.update_values_pix(pix, True)

        # Force packing if desired
        if bit_packed: mask = mask.as_bit_packed_map()

        # Store calling/useful info in its own dictionary
        area_deg2 = mask.get_valid_area(degrees=True)
        npix = mask.n_valid
        print('--- Zones mask area                           :', area_deg2)
        extra_meta = { 'data': (str(data) if isinstance(data, (str, Path)) else "<mem>"), 'fmt': fmt, 
                       'columns': columns, 'bit_packed': bit_packed, 'pixels': npix, 'area_deg2': area_deg2 }
        return self._finalize_stage(mask, default_name='zonemask', output_stage=output_stage, extra_meta=extra_meta)


        
    def build_milkyway_mask(self, order_sparse: int = 8, order_cov: Optional[int] = None,
                            b0_deg: float = 15., bulge_a_deg: float = 25., bulge_b_deg: float = 20.,
                            bit_packed: bool = False, output_stage: Optional[str] = None):
        """
        Build a boolean HealSparse mask for the plane and bulge of the Milky Way.

        Parameters
        ----------
        order_sparse : int, optional
            Sparse order to pixelize the galaxy mask
        order_cov : int, optional
            Coverage order. If None, falls back to `self.order_cov`
        b0_deg : float
            Half-thickness of the Galactic plane band (|b| <= b0_deg)
        bulge_a_deg : float
            Semi-axis along Galactic longitude for the bulge ellipse (degrees)
        bulge_b_deg : float
            Semi-axis along Galactic latitude for the bulge ellipse (degrees)
        bit_packed : bool, optional
            If True, return the ouput as bit-packed boolean map
        output_stage : str, optional
            Name for the ouput stage. If None, defaults to then canonical name 'mwmask'
            
        Returns
        -------
        HealSparseMap
            The boolean (or bit-packed) mask is stored at `self.mwmask` and also
            returned to prompt
        """
        print('BUILDING MILKY WAY MASK >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # Check if user wants specific orders, otherwise get from defaults
        ord_cov = order_cov if order_cov is not None else self.order_cov

        nside_sparse = 1<<order_sparse
        nside_cov = 1<<ord_cov

        # Create the empty boolean map up front
        mask = hsp.HealSparseMap.make_empty(nside_cov, nside_sparse, dtype=np.bool_)

        # Create MOC of the MW plane and bulge
        moc_pb = SkyMaskPipe.gal_plane_bulge_moc(max_depth=order_sparse, b0_deg=b0_deg,
                                                 bulge_a_deg=bulge_a_deg, bulge_b_deg=bulge_b_deg)

        # Get pixels and update mwmask. No beed for facy streaming since its unnlikey to require high orders
        pixels = moc_pb.flatten().astype(np.int64)
        mask.update_values_pix(pixels, True)

        # Force packing if desired
        if bit_packed: mask = mask.as_bit_packed_map()

        # Store calling/useful info in its own dictionary
        area_deg2 = mask.get_valid_area(degrees=True)
        npix = mask.n_valid
        print('--- Milky Way mask area                          :', area_deg2)
        extra_meta = { 'b0_deg': b0_deg, 'bulge_a_deg': bulge_a_deg, 'buge_b_deg': bulge_b_deg, 
                       'pixels': npix, 'area_deg2': area_deg2 }        
        return self._finalize_stage(mask, default_name='mwmask', output_stage=output_stage, extra_meta=extra_meta)

        

    @staticmethod
    def moc_from_stage(stage, order_forced: int | None = None,
                       rows_per_batch: int = 2048,    # how many coverage rows per micro-MOC
                       progress: bool = True, pbar_kwargs: dict | None = None) -> MOC:
        """
        Build a MOC from a stage's healSparse boolean map without materializing all valid
        pixels at once, by streaming sparse pixels inside coverage pixels.

        This routine consumes coverage pixels one-by-one via ``_iter_valid_by_covpix(hsp)``,
        immediately **coarsens** each row’s sparse pixels to the target order, performs
        an in-row ``unique()``, accumulates a small batch of rows, builds a micro-MOC,
        and **union-reduces** partial MOCs as it goes.

        Parameters
        ----------
        stage : str or healSparse map
            Either the name of a SkyMaskPipe stage attribute (e.g., ``"starmask"``),
            or a healSparse boolean map object itself (regular or bit-packed).
        order_forced : int, optional
            Target MOC order (depth). By default the map’s ``order_sparse`` is used.
        rows_per_batch : int, default 2048
            Number of non-empty coverage pixels to accumulate before building each
            micro-MOC.
        progress : bool, default True
            If True, display a tqdm progress bar over coverage rows. If tqdm is not
            available, execution proceeds silently.
        pbar_kwargs : dict, optional
            Extra keyword arguments forwarded to ``tqdm.auto.tqdm`` (e.g.,
            ``{"leave": True}``, ``{"desc": "Building MOC"}``).

        Returns
        -------
        moc : mocpy.MOC
            The union of all micro-MOCs at ``order_forced`` (or ``order_sparse`` if
            not forced).


        Memory behavior
        ---------------
        - Never calls ``hsp.valid_pixels`` on the whole map.
        - Per row: coarsen **immediately** from ``order_sparse`` to target order
          using parent mapping (NESTED scheme), then sort/unique within the row.
        - Builds micro-MOCs every ``rows_per_batch`` rows and union-reduces partials
          promptly to bound the number of large objects alive at once.

        Assumptions
        -----------
        - Input healSparse map is in **NESTED** ordering (parent mapping via integer
          division is used).
        - ``_iter_valid_by_covpix(hsp)`` yields ``(covpix, arr)`` where ``arr`` is a
          1-D array of sparse-order pixel indices (dtype convertible to int64) for
          that coverage pixel. It may yield rows in any order.
        """
        from math import log2

        # Resolve stage -> healsparse map
        if hasattr(stage, 'nside_sparse'): hsp = stage

        order_sparse = int(round(log2(hsp.nside_sparse)))
        order_target = order_forced if order_forced is not None else order_sparse
        if order_target > order_sparse:
            raise ValueError(f"order_target ({order_target}) > order_sparse ({order_sparse})")

        # child → parent factor in NESTED scheme
        delta = order_sparse - order_target
        parent_factor = 4 ** delta if delta > 0 else 1

        # Prepare tqdm (auto total if possible)
        pbar = None
        if progress:
            total_covpix = None
            try:
                cov = getattr(hsp, "coverage_mask", None)
                if cov is not None:
                    cov = np.asarray(cov)
                    if cov.dtype == np.bool_ or np.issubdtype(cov.dtype, np.bool_):
                        total_covpix = int(np.count_nonzero(cov))
                    else:
                        # if it's an index array / anything else, use its length
                        total_covpix = int(cov.size)
                if total_covpix is None:
                    # fallback: count iterator (usually only a few thousand rows)
                    total_covpix = sum(1 for _ in _iter_covpix(hsp))
            except Exception:
                total_covpix = None  # show indeterminate bar if detection fails

            try:
                from tqdm.auto import tqdm
                kb = dict(total=total_covpix, unit="row", dynamic_ncols=True, leave=False)
                if pbar_kwargs:
                    kb.update(pbar_kwargs)
                pbar = tqdm(desc=f"Streaming pixels to build MOC at order {order_target}", **kb)
            except Exception:
                pbar = None  # tqdm not available; continue silently

        # Streaming accumulators
        micro_ipix = []   # list of np.ndarray[int64] at target order
        partials = []     # list of tiny MOCs to union later
        row_counter = 0
        micros_built = 0

        # Main streaming loop over coverage pixels ------------------
        for _covpix, arr in _iter_valid_by_covpix(hsp):
            if pbar is not None: pbar.update(1)

            if arr is None or arr.size == 0: continue

            # Ensure int64 without a copy when possible
            arr = np.asarray(arr, dtype=np.int64, order='C')

            # Coarsen immediately (order_sparse → order_target)
            parents = (arr // parent_factor) if delta > 0 else arr

            # Unique within the row to keep the batch small
            if parents.size > 1 and (parents[1:] < parents[:-1]).any(): parents.sort()
            parents = np.unique(parents)

            # Accumulate
            if parents.size:
                micro_ipix.append(parents)
                row_counter += 1

            # Build a micro-MOC every rows_per_batch rows
            if row_counter >= rows_per_batch:
                ipix = np.unique(np.concatenate(micro_ipix)) if len(micro_ipix) > 1 else micro_ipix[0]
                micro_ipix.clear()
                row_counter = 0

                # All at the same order_target now
                m = MOC.from_healpix_cells(ipix=ipix, depth=order_target, max_depth=order_target)

                partials.append(m)
                micros_built += 1
                if pbar is not None:
                    pbar.set_postfix_str(f"micros={micros_built} partials={len(partials)}")

                # Union-reduce to avoid long chains (kept identical to your version)
                while len(partials) >= 2:
                    a = partials.pop()
                    b = partials.pop()
                    partials.append(a.union(b))

        # Flush any remainder
        if micro_ipix:
            ipix = np.unique(np.concatenate(micro_ipix)) if len(micro_ipix) > 1 else micro_ipix[0]
            micro_ipix.clear()
            m = MOC.from_healpix_cells(ipix=ipix, depth=order_target, max_depth=order_target)
            partials.append(m)
            micros_built += 1
            if pbar is not None:
                pbar.set_postfix_str(f"micros={micros_built} partials={len(partials)}")

        # Close the bar before the potentially heavy final reduction
        if pbar is not None:
            pbar.close()
            pbar = None

        # Final reduction / empty case
        if not partials:
            return MOC.from_healpix_cells(ipix=np.array([], dtype=np.int64),
                   depth=np.array([], dtype=np.int16), max_depth=order_target)

        moc = partials[0]
        for p in partials[1:]:
            moc = moc.union(p)
        return moc




    @staticmethod
    def _otsu_threshold(gray_0to255: np.ndarray) -> int:
        # Auxliary for image_to_healsparse()
        """
        Return Otsu threshold (0..255) for a grayscale uint8 image. 0tsu method assumes the image
        has two intensity classes (background vs foreground) and chooses the threshold that
        best separates them by minimizing within-class variance.
        """
        hist = np.bincount(gray_0to255.ravel(), minlength=256).astype(np.float64)
        p = hist / hist.sum()
        omega = np.cumsum(p)
        mu = np.cumsum(p * np.arange(256))
        mu_t = mu[-1]
        denom = omega * (1 - omega)
        denom[denom == 0] = np.nan
        sigma_b2 = (mu_t * omega - mu) ** 2 / denom
        k = int(np.nanargmax(sigma_b2))
        return k


    @staticmethod
    def _load_binary_mask(image_path: str, invert: bool = False, blur_radius: float = 0.0,
                          threshold: int | None = None, expand_px: int = 0) -> np.ndarray:
        # Auxliary for image_to_healsparse()
        """
        Load image -> grayscale -> (optional) blur -> threshold -> (optional) dilate.
        Returns a boolean array mask[y, x] with True = "ink"/foreground.
        """
        from PIL import Image, ImageOps, ImageFilter

        im = Image.open(image_path).convert("L")          # grayscale
        im = ImageOps.autocontrast(im)                    # normalize a bit
        if blur_radius and blur_radius > 0:
            im = im.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))

        arr = np.asarray(im, dtype=np.uint8)
        if threshold is None:
            thr = SkyMaskPipe._otsu_threshold(arr)
        else:
            thr = int(threshold)

        if invert:
            mask = arr < thr
        else:
            mask = arr >= thr

        if expand_px and expand_px > 0:
            # Simple morphological dilation using PIL’s MaxFilter
            im_bin = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
            im_dil = im_bin.filter(ImageFilter.MaxFilter(size=2*int(expand_px)+1))
            mask = (np.asarray(im_dil) > 0)

        return mask


    @staticmethod
    def _rotation(x, y, angle_deg: float):
        # Auxliary for image_to_healsparse()
        """Rotate 2D coordinates (x, y) by angle_deg (counterclockwise) in radians."""
        if angle_deg == 0.0:
            return x, y
        a = np.deg2rad(angle_deg)
        ca, sa = np.cos(a), np.sin(a)
        xr = ca * x - sa * y
        yr = sa * x + ca * y
        return xr, yr


    @staticmethod
    def image_to_healsparse(image_path: str, *,
        order_sparse: int = 9, order_cov: int = 4,
        ra0_deg: float = 0.0, dec0_deg: float = 45.0,
        width_deg: float = 30.0, height_deg: float | None = None,
        rotation_deg: float = 0.0, invert: bool = False, blur_radius: float = 0.0,
        threshold: int | None = None,    # 0..255; None = Otsu
        expand_px: int = 0,              # dilate mask in pixels
        samples_per_pix: int = 1,        # 1 or 4 (for thicker/anti-aliased edges)
        edge_eps: float = 0.0,           # keep if inside OR within eps in plane units
        chunk_size: int = 100_000, return_bit_packed: bool = False) -> hsp.HealSparseMap:
        """
        Rasterize an arbitrary image silhouette into a boolean HealSparseMap.

        The image is mapped into a gnomonic (tangent-plane) patch centered on (ra0, dec0).
        The on-sky size is ~width_deg × height_deg. Rotation is in the tangent plane.

        Parameters
        ----------
        image_path : str
            Path to the figure to draw (any format PIL can read).
            Foreground (ink) will become True pixels in the map after thresholding.
        order_sparse : int
            HEALPix order (NESTED) for the sparse map (NSIDE=2**order_sparse).
        order_cov : int
            Coverage order; must satisfy order_sparse >= order_cov.
        ra0_deg, dec0_deg : float
            Center of the drawing in ICRS degrees.
        width_deg, height_deg : float | None
            Angular size of the drawing. If height_deg is None, it’s set by image aspect ratio.
        rotation_deg : float
            Rotation of the image (counterclockwise) in the tangent plane.
        invert : bool
            If True, invert the thresholded mask (useful if the background is dark).
        blur_radius : float
            Gaussian blur (px) before thresholding (helps noisy edges).
        threshold : int | None
            Manual threshold (0..255). If None, use Otsu.
        expand_px : int
            Morphological dilation in pixels after thresholding (thickens lines).
        samples_per_pix : int
            1 or 4. If 4, each HEALPix pixel center is sampled at 4 subpoints.
        edge_eps : float
            Extra acceptance band in tangent-plane units; 0.0 is typical.
        chunk_size : int
            How many sparse pixels to process per batch.
        return_bit_packed : bool
            If True, return a bit-packed boolean HealSparseMap.

        Returns
        -------
        A healparse map with your favorite image printed!
        """
        if order_sparse < order_cov:
            raise ValueError("order_sparse must be >= order_cov.")

        # Load silhouette mask
        mask = SkyMaskPipe._load_binary_mask(image_path, invert=invert,
               blur_radius=blur_radius, threshold=threshold, expand_px=expand_px)
        H, W = mask.shape

        # Determine on-sky height from aspect ratio if not given
        if height_deg is None: height_deg = width_deg * (H / W)

        # Gnomonic geometry
        nside_sparse = 2 ** order_sparse
        nside_cov = 2 ** order_cov
        dlevel = order_sparse - order_cov
        child_factor = 4 ** dlevel

        ra0 = np.deg2rad(ra0_deg % 360.0)
        dec0 = np.deg2rad(dec0_deg)
        sdec0, cdec0 = np.sin(dec0), np.cos(dec0)

        # Scale: map sky box width/height to plane via tan
        tx = np.tan(np.deg2rad(width_deg) / 2.0)
        ty = np.tan(np.deg2rad(height_deg) / 2.0)

        # A safe circular cap that contains the whole patch
        half_diag = 0.5 * np.deg2rad(np.hypot(width_deg, height_deg))
        cap_radius = min(np.deg2rad(89.0), half_diag * 1.05)

        # Coverage candidates: centers within cap (with a small pad for coverage pixel size)
        cov_area_deg2 = hp.nside2pixarea(nside_cov, degrees=True)
        cov_radius_pad = np.deg2rad(np.sqrt(cov_area_deg2 / np.pi))

        cov_npix = hp.nside2npix(nside_cov)
        cov_pix = np.arange(cov_npix, dtype=np.int64)
        theta_cov, phi_cov = hp.pix2ang(nside_cov, cov_pix, nest=True)
        ra_cov = phi_cov
        dec_cov = (np.pi/2.0) - theta_cov
        dalpha_cov = (ra_cov - ra0 + np.pi) % (2*np.pi) - np.pi
        cosg_cov = sdec0*np.sin(dec_cov) + cdec0*np.cos(dec_cov)*np.cos(dalpha_cov)
        cosg_cov = np.clip(cosg_cov, -1.0, 1.0)
        gamma_cov = np.arccos(cosg_cov)
        cov_candidates = cov_pix[gamma_cov <= (cap_radius + cov_radius_pad)]

        # Prepare sub-sampling offsets (in plane units) for anti-alias/thick edges if requested
        if samples_per_pix == 4:
            # Offsets are small fractions of plane scale
            sub = np.array([[-0.25, -0.25], [+0.25, -0.25], [-0.25, +0.25], [+0.25, +0.25]])
        else:
            sub = np.array([[0.0, 0.0]])

        valid_parts = []
        for p in cov_candidates:
            start = p * child_factor
            stop = (p + 1) * child_factor

            for base in range(start, stop, chunk_size):
                end = min(base + chunk_size, stop)
                child_idx = np.arange(base, end, dtype=np.int64)

                theta, phi = hp.pix2ang(nside_sparse, child_idx, nest=True)
                ra = phi
                dec = (np.pi/2.0) - theta

                # Pre-cut by the cap
                dalpha = (ra - ra0 + np.pi) % (2*np.pi) - np.pi
                cosg = sdec0*np.sin(dec) + cdec0*np.cos(dec)*np.cos(dalpha)
                cosg = np.clip(cosg, -1.0, 1.0)
                gamma = np.arccos(cosg)
                m_cap = gamma <= cap_radius
                if not np.any(m_cap):
                    continue

                ra = ra[m_cap]
                dec = dec[m_cap]
                dalpha = dalpha[m_cap]
                idx_cap = child_idx[m_cap]

                # Gnomonic projection
                denom = sdec0*np.sin(dec) + cdec0*np.cos(dec)*np.cos(dalpha)
                good = denom > 1e-12
                if not np.any(good):
                    continue

                ra = ra[good]; dec = dec[good]; dalpha = dalpha[good]; idx_cap = idx_cap[good]
                denom = denom[good]

                x = (np.cos(dec) * np.sin(dalpha)) / denom
                y = (cdec0*np.sin(dec) - sdec0*np.cos(dec)*np.cos(dalpha)) / denom

                # Optional rotation in plane
                x, y = SkyMaskPipe._rotation(x, y, rotation_deg)

                # Normalize to image box
                xn = x / tx
                yn = y / ty

                # Subsample (vectorized): accept if ANY subpoint falls inside the foreground
                accept = np.zeros(xn.shape, dtype=bool)
                for dx, dy in sub:
                    xn_s = xn + (dx / (W if W>0 else 1))
                    yn_s = yn + (dy / (H if H>0 else 1))

                    # Map to pixel coords (u, v); note v grows downward
                    u = (xn_s * 0.5 + 0.5) * W
                    v = (-(yn_s) * 0.5 + 0.5) * H

                    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
                    if not np.any(inside):
                        continue

                    ui = u[inside].astype(np.int64)
                    vi = v[inside].astype(np.int64)

                    hit = np.zeros_like(inside)
                    hit_idx = mask[vi, ui]
                    hit[inside] = hit_idx

                    if edge_eps > 0.0:
                        # keep near-edges too: points slightly outside but within eps of box
                        near = (~inside) & (np.abs(xn_s) <= 1.0 + edge_eps) & (np.abs(yn_s) <= 1.0 + edge_eps)
                        hit |= near

                    accept |= hit

                if np.any(accept): valid_parts.append(idx_cap[accept])

        valid_pixels = (np.unique(np.concatenate(valid_parts))
                        if valid_parts else np.array([], dtype=np.int64))

        hmap = hsp.HealSparseMap.make_empty(nside_coverage=2**order_cov,
                                            nside_sparse=2**order_sparse, dtype=np.bool_)

        if valid_pixels.size: hmap[valid_pixels] = True

        print(f'Map generated from image -> valid pixels = {hmap.n_valid}')
        return hmap.as_bit_packed_map() if return_bit_packed else hmap



    @staticmethod
    def _compute_bitshift(nside_from, nside_to):
        d = int(np.log2(nside_to) - np.log2(nside_from))
        if d < 0 or (1 << d) * nside_from != nside_to:
            raise ValueError("nside_randoms must be a power-of-two multiple of nside_sparse.")
        return 2 * d  # NESTED: +2 bits per order



    def makerans(self, stage: Union[str, "HealSparseMap"] = "mask", nr: int = 50_000, 
                 file: Optional[os.PathLike[str] | str] = None, nside_randoms: int = 2**23, 
                 rng: Optional[RandomState] = None, **kwargs) -> pd.DataFrame:
        """
        Generate uniform randoms over a HealSparse boolean (or bit-packed) map without materializing
        the global valid-pixels list.

        Parameters
        ----------
        stage : str
            Mask stage name in this SkyMaskPipe (e.g., 'mask', 'foot', 'propmask', ...)
        nr : int
            Number of random points to generate.
        file : str or None
            If provided, path to a .parquet file to write (index=False).
        nside_randoms : int or None
            Target NESTED nside for placing randoms (must be a power-of-two multiple of map nside). Defaults to 2**23.
        rng : np.random.RandomState or None
            RNG to use. If None, uses np.random.RandomState().
        **kwargs :
            Extra kwargs passed to pandas.DataFrame.to_parquet().

        Returns
        -------
        pandas.DataFrame
            Columns: 'ra', 'dec' (degrees).
        """
        if rng is None: rng = np.random.RandomState()

        # Resolve stage -> healsparse map
        mk = stage if hasattr(stage, 'nside_sparse') else getattr(self, stage)
        nside_sparse = mk.nside_sparse

        # Choose default fine grid (power-of-two multiple of nside_sparse)
        #if nside_randoms is None:
        #    #nside_randoms = min(2**23, nside_sparse * (2**3))
        bit_shift = self._compute_bitshift(nside_sparse, nside_randoms)

        # ---------- Pass 1: counts per iterator chunk (streaming) ----------
        counts = [pix.size for pix in mk.iter_valid_pixels_by_covpix()]
        if not counts or nr <= 0:
            df = pd.DataFrame({'ra': np.empty(0, dtype=float), 'dec': np.empty(0, dtype=float)})
            if file:
                df.to_parquet(file, index=False, **kwargs)
                print("0 randoms written (empty mask).")
            return df

        counts = np.asarray(counts, dtype=np.int64)
        total = counts.sum()
        if total == 0:
            df = pd.DataFrame({'ra': np.empty(0, dtype=float), 'dec': np.empty(0, dtype=float)})
            if file:
                df.to_parquet(file, index=False, **kwargs)
                print("0 randoms written (empty mask).")
            return df

        alloc = rng.multinomial(nr, counts / total)

        # ---------- Pass 2: sample within each iterator chunk ----------
        ras, decs = [], []
        for i, pix in enumerate(mk.iter_valid_pixels_by_covpix()):
            take = int(alloc[i])
            if take <= 0 or pix.size == 0:
                continue

            parents = np.asarray(pix, dtype=np.int64)  # absolute NESTED pixel ids @ nside_sparse
            choice = rng.randint(0, parents.size, size=take)
            parents_sel = parents[choice]

            # Refine to nside_randoms (append random sub-bits)
            subbits = rng.randint(0, 1 << bit_shift, size=take, dtype=np.int64)
            randpix = (parents_sel << bit_shift) + subbits

            ra, dec = hp.pix2ang(nside_randoms, randpix, nest=True, lonlat=True)
            ras.append(ra); decs.append(dec)

        rra  = np.concatenate(ras) if ras else np.empty(0, dtype=float)
        rdec = np.concatenate(decs) if decs else np.empty(0, dtype=float)

        df = pd.DataFrame({'ra': rra.astype(float, copy=False), 'dec': rdec.astype(float, copy=False)})

        if file:
            df.to_parquet(file, index=False, **kwargs)
            print(f"{len(df)} randoms written to: {file}")

        return df

