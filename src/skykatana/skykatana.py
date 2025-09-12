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
from matplotlib.axes import Axes


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
    A class to work with healsparse sky masks in a pipeline way

    Attributes
    ----------
    order_cov : int
        Coverage order of (all) healsparse maps.
    order_foot : int
        Order for the footprint map
    order_patch : int
        Order for the patch map
    order_prop : int
        Order for the property map
    order_star : int
        Order for the bright star mask (queried from online catalog)
    """

    # Class scalar attributes to be saved in JSON file
    _SCALAR_ATTRS = ["order_cov","order_foot","order_patch","order_prop","order_star",
                     "order_circ", "order_box","order_ellip","order_poly","order_zone","order_out"]
    _BITPACK_FITS_VERSION = 1   # File format version in case we update


    def __init__(self, **kwargs):

        # Default values for orders, when not provided
        defaults = {
            'order_cov':       4,
            'order_foot':      13,
            'order_patch':     13,
            'order_prop':      15,
            'order_star':      15,
            'order_circ':      15,
            'order_box':       15,
            'order_ellip':     15,
            'order_poly':      15,
            'order_zone':      15,
            'order_out':       15
        }

        for (prop, val) in defaults.items():
            setattr(self, prop, kwargs.get(prop, val))

        self.nside_out       = 1<<self.order_out
        self.nside_cov       = 1<<self.order_cov
        self.footmask        = None
        self.propmap         = None
        self.starmask        = None
        self.circmask        = None
        self.boxmask         = None
        self.ellipmask       = None
        self.polymask        = None
        self.zonemask        = None
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
            "zonemask", "mask",
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


    # Generic helper to stash params for any stage ----------
    def _store_params(self, stage: str, **params: Any) -> None:
        def _norm(v):
            if isinstance(v, (str, Path)):       # normalize paths & strings
                return str(v)
            if isinstance(v, list):              # lists → tuples for immutability-ish logs
                return tuple(v)
            return v
        packed = {k: _norm(v) for k, v in params.items()}
        self._params.setdefault(stage, {}).update(packed)



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
    def readQApatches(qafile):
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
    def parse_condition(condition_str):
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
    def filter_and_pixelate_patches(file, qatable, filt=None, order=13):
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
        order : int
            Pixelization order
        filt : str
            Contition(s) to apply to patches, e.g filt='(ra>20) and (dec<10)'. If None,
            all patches will be considered

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
    def pixelate_circles(data, fmt='ascii', columns=['ra', 'dec', 'radius'], order=15,
                         delta_depth=2, n_threads=4):
        """
        Read circular regions around bright stars, pixelize them and return the (unique) pixels inside.
        Coordinates and distances should be in degrees.

        Parameters
        ----------
        data : pd.Dataframe or str or Path
            Pandas dataFrame or path to file
        fmt : str
            Format of file, e.g. 'ascii', 'parquet', or any accepted by astropy.table
        columns : list of str
            Colums for ra, dec, radius of circles
        order : int
            Pixelization order
        delta_depth : int
            Delta to higher orders to improve pixelization
        n_threads : int
            Number of threads. Set to None to use all available threads

        Returns
        -------
        ndarray
            Array of pixels
        """
        colra, coldec, colrad = columns
        if isinstance(data, pd.DataFrame):
            print('--- Pixelating circles from DataFrame')
            table = data
        elif isinstance(data, (str, PosixPath)):
            print('--- Pixelating circles from', data)
            table = Table.read(data, format=fmt)
        print('    Order ::',order)

        mocs = MOC.from_cones(
            lon=Longitude(table[colra], unit='deg'), lat=Latitude(table[coldec], unit='deg'), radius=Angle(table[colrad], unit='deg'),
            max_depth=order, delta_depth=delta_depth, n_threads=n_threads)

        hp_index = np.concatenate([moc.flatten() for moc in mocs])
        print('    done')
        return np.unique(hp_index).astype(np.int64)


    @staticmethod
    def pixelate_ellipses(data, fmt='ascii', columns=['ra','dec','a','b','pa'], order=15, delta_depth=2):
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
    def pixelate_boxes(data, fmt='ascii', columns=['ra_c','dec_c','width','height'], order=15, n_threads=4):
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
    def pixelate_zones(data, fmt='ascii', columns=['ra1','dec1','ra2','dec2'], order=15):
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
    def pixelate_polys(data, fmt='ascii', columns=['ra0','ra1','ra2','ra3','dec0','dec1','dec2','dec3'],
                       n_threads=4, order=15):
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




    def build_star_mask_online(self, starq, order_star: Optional[int] = None, order_cov: Optional[int] = None,
                               columns: Optional[Sequence[str]] = ['ra','dec','radius'],
                               bit_packed: Optional[bool] = None, n_threads=4):
        """
        Build a bright-star mask on the fly by querying a remote catalog.

        This method constructs a `HealSparseMap` mask of bright stars by:
        (1) defining a search region from a user-supplied `search_stage`,
        (2) querying the Gaia catalog in chunks of sky defined by a MOC,
        (3) applying a custom radius function to compute star exclusion radii,
        (4) pixelizing the stars into HEALPix pixels at the requested order,
        and (5) streaming results directly into a sparse mask. The final mask
        is stored in the `starmask` attribute.

        Parameters
        ----------
        starq : dict
            Dictionary of parameters controlling the star mask construction.
            Required keys:
              - ``search_stage`` : `HealSparseMap` defining the search region.
              - ``cat`` : catalog identifier to open with `lsdb.open_catalog`.
              - ``columns`` : list of columns to load from the Gaia catalog.
              - ``gaia_gmag_lims`` : tuple ``(gmin, gmax)`` magnitude limits.
              - ``gaia_b_lims`` : tuple ``(b_south, b_north)`` Galactic latitude cuts.
              - ``radfunction`` : callable that takes a DataFrame and assigns radii to stars.
            Optional keys:
              - ``max_area_single`` : maximum deg² before splitting (default 500).
              - ``target_chunk_area`` : target deg² per MOC chunk (default 300).
              - ``coarse_order_bfs`` : order for initial chunk splitting (default 5).
        order_star : int, optional
            Sparse order for pixelizing stars. Defaults to `self.order_star`.
        order_cov : int, optional
            Coverage order. Defaults to `self.order_cov`.
        columns : sequence of str, optional
            Column names expected in the star DataFrame. Defaults to
            ``['ra', 'dec', 'radius']``.
        bit_packed : bool, optional
            If True, convert the final mask to bit-packed format.
        n_threads : int, default=4
            Number of threads used during circle pixelization.

        Returns
        -------
        starmask : healsparse.HealSparseMap
            The star mask as a `HealSparseMap`, also stored as `self.starmask`
        """

        print('BUILDING STAR MASK >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        if not(isinstance(starq, dict)): raise Exception("starq must be a valid dictionary")

        order_sparse = order_star if order_star is not None else self.order_star
        ord_cov = order_cov if order_cov is not None else self.order_cov

        nside_sparse = 1<<order_sparse
        nside_cov = 1<<ord_cov

        # Create the empty boolean map up front
        self.starmask = hsp.HealSparseMap.make_empty(nside_cov, nside_sparse, dtype=np.bool_, bit_packed=True)

        MAX_DEPTH = 8     # The moc of the search_stage will be degraded to this order.
                          # 8 means 822" pixels, so is fine for including stars with up to ~411" radii
        search_stage = starq['search_stage']    # This is the area for searching Gaia stars
        if not isinstance(search_stage, hsp.HealSparseMap):
            raise TypeError('search_stage must be a valid HealSparseMap')

        # Build search MOC
        order_search_stg = int(np.log2(search_stage.nside_sparse))
        moc = MOC.from_healpix_cells(ipix=search_stage.valid_pixels, depth=order_search_stg, max_depth=MAX_DEPTH)

        # Open Gaia (already filtered by columns/mag/lat outside MW plane)
        gaia = lsdb.open_catalog(
            starq['cat'],
            columns=starq['columns'],
            search_filter=lsdb.BoxSearch(ra=[0.,360.], dec=[-89.99999, 10.]),  # hardcode dec<10 for LSST
            filters=[["phot_g_mean_mag", ">", starq['gaia_gmag_lims'][0]],
                     ["phot_g_mean_mag", "<", starq['gaia_gmag_lims'][1]]]
        )
        bsouth, bnorth = starq['gaia_b_lims']
        gaiat = gaia.query(f"(b < {bsouth}) or (b > {bnorth})")

        # Split moc parameters
        max_area_sing     = starq.get('max_area_single',   500.0)
        target_chunk_area = starq.get('target_chunk_area', 300.0)
        coarse_order_bfs  = starq.get('coarse_order_bfs',  5)

        # Get chunks (or a single piece)
        moc_area = getarea_moc(moc)
        if moc_area < max_area_sing:
            print(f'Area of search_stage is {moc_area:.2f} deg2 -> no splitting')
            chunk_mocs = [moc.add_neighbours()]  # 1px border at max depth
        else:
            print(f'Area of search_stage is {moc_area:.2f} deg2 -> splitting...')
            chunk_mocs = split_moc_into_chunks(moc, target_deg2=target_chunk_area, coarse_order=coarse_order_bfs)
            # refine to original depth + add border
            chunk_mocs = [c.intersection(moc).add_neighbours() for c in chunk_mocs]
            chunk_areas = [getarea_moc(mi) for mi in chunk_mocs]
            print(f"Got {len(chunk_mocs)} chunks of areas [{' '.join(f'{a:6.2f}' for a in chunk_areas)}] deg²")


        # Star radii function
        radfunction = starq['radfunction']
        if not callable(radfunction):
            raise TypeError("radfunction must be a callable that accepts a DataFrame with optional kwargs")

        # ---- STREAMING PIPELINE ----
        for i, mi in enumerate(chunk_mocs, 1):
            print(f'Chunk {i}/{len(chunk_mocs)}: querying URL catalog ...')
            # Get stars in this chunk only
            s = gaiat.moc_search(moc=mi).compute()

            # Add radii in-place
            radfunction(s)

            # Pixelate *this chunk only*
            pix = self.pixelate_circles(s, order=order_sparse, columns=columns, n_threads=n_threads)
            if pix is None or len(pix) == 0:
                continue

            # Optional (small) per-chunk de-dup to reduce work:
            # cheaper than global unique, and bounded in size
            pix = np.asarray(pix, dtype=np.int64)
            pix = np.unique(pix)

            # Stream directly into the sparse map (idempotent)
            self.starmask.update_values_pix(pix, True)

            # Free memory promptly
            del s, pix
            gc.collect()

        # Force packing if desired
        if bit_packed: self.starmask = self.starmask.as_bit_packed_map()

        # Store calling/useful info in its own dictionary
        area_deg2 = self.starmask.get_valid_area(degrees=True)
        npix = self.starmask.n_valid
        starq['search_stage'] = '<dummy>'  # for now just set a dummy string for the search_stage to avoid
                                           # storing the actual map. We should fix this later
        self._store_params('starmask',
            starq=starq,
            order_star=order_sparse, order_cov=ord_cov, columns=columns, bit_packed=bit_packed,
            n_threads=n_threads, pixels=npix, area_deg2=area_deg2)

        print('--- Star mask area                         :', area_deg2)
        return self.starmask



    @staticmethod
    def reproject_nside_coverage(hspmap, newcov):
        """
        Change the nside_coverage of a healsparse map. Useful to bring boolean maps to a common coverage,
        allowing logic combinations between them

        Parameters
        ----------
        hsmap
            Healsparse boolean map

        Returns
        -------
        hsmap
            Healsparse boolean map
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


    def build_prop_mask(self, prop_maps, thresholds, comparisons,
                        order_prop=None, order_cov=None, bit_packed=None):
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

        Returns
        -------
        mask_map : HealSparseMap
            Healsparse boolean map whose pixels meet all criteria
        """

        print('BUILDING PROPERTY MAP >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # Check if user wants specific orders, otherwise get from defaults
        order_sparse = order_prop if order_prop is not None else self.order_prop
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
        self.propmask = hsp.HealSparseMap.make_empty(prop_maps[0].nside_coverage, prop_maps[0].nside_sparse, dtype=np.bool_)
        self.propmask[pixels] = True
        #self.order_prop = prop_maps[0].nside_sparse

        # Change to desired coverage order. This honors the input keyword parameter, which itself defaults
        # to the pipeline value (order_cov) when not specified
        if nside_cov != prop_maps[0].nside_coverage:
            print(f'--- Propertymap coverage order changed to {ord_cov}')
            self.propmask = self.change_cov_order(self.propmask, ord_cov, inplace=True, verbose=False)
            #self.propmask = self.reproject_nside_coverage(self.propmask, self.nside_cov)

        # Change to desired sparse order.This honors the input keyword parameter, which itself defaults
        # to the pipeline value (order_prop) when not specified
        if nside_sparse != prop_maps[0].nside_sparse:
            print(f'--- Propertymap sparse order changed to {order_sparse}')
            self.propmask = self.change_sparse_order(self.propmask, order_sparse, inplace=True, verbose=False)

        # Force packing if desired
        if bit_packed: self.propmask = self.propmask.as_bit_packed_map()

        # Store calling/useful info in its own dictionary
        area_deg2 = self.propmask.get_valid_area(degrees=True)
        npix = self.propmask.n_valid
        self._store_params('propmask',
            prop_maps=pmstring, thresholds=thresholds, comparisons=comparisons, order_prop=order_sparse,
            order_cov=ord_cov, bit_packed=bit_packed, pixels=npix, area_deg2=area_deg2)

        print('--- Propertymap mask area                       :', area_deg2)
        return self.propmask



    def build_patch_mask(self, patchfile=None, qafile=None, order_patch=None, order_cov=None,
                         filt="(gmag_psf_depth>26.2) and (rmag_psf_depth>25.9) and (imag_psf_depth>25.7)",
                         bit_packed=None):
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
        order_patch : int
            Pixelization order
        filt : string
            Contition(s) to apply to patches. If None, all patches will be considered
        bit_packed : bool, optional
            If True, return the ouput as bit-packed boolean map

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
        order_sparse = order_patch if order_patch is not None else self.order_patch
        ord_cov = order_cov if order_cov is not None else self.order_cov

        nside_sparse = 1<<order_sparse
        nside_cov = 1<<ord_cov


        # Read patch qa list
        qatable = self.readQApatches(qafile)

        # Create the empty boolean map *up front*
        self.patchmask = hsp.HealSparseMap.make_empty(nside_cov, nside_sparse, dtype=np.bool_)

        for p in patchfile:
            fpx = self.filter_and_pixelate_patches(p, qatable, order=order_sparse, filt=filt)
            self.patchmask[fpx] = True

        # Force packing if desired
        if bit_packed: self.patchmask = self.patchmask.as_bit_packed_map()

        # Store calling/useful info in its own dictionary
        area_deg2 = self.patchmask.get_valid_area(degrees=True)
        npix = self.patchmask.n_valid
        self._store_params('patchmask',
            patchfile=patchfile, qafile=qafile, order_zone=order_sparse, order_cov=ord_cov,
            filt=filt, bit_packed=bit_packed, pixels=npix, area_deg2=area_deg2)

        print('--- Patch mask area                           :', area_deg2)
        return self.patchmask




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
        Normalize inputs and return an array of pixel indices (nest scheme) for all sources.
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


    def build_foot_mask(self, sources, order_foot: Optional[int] = None, order_cov: Optional[int] = None,
                        columns: Iterable[str] = ("ra", "dec"), *, remove_isopixels: bool = False,
                        erode_borders: bool = False, mapping: bool = False, bit_packed: bool = True):
        """
        Create a footprint mask of a source catalog (from any astropy-supported table or HATS catalog),
        pixelated at a given order. Optionally remove isolated empty pixels and erode borders
        around empty zones. For details see remove_isopixels() and erode_borders().

        Parameters
        ----------
        sources : str | astropy.table.Table | pandas.DataFrame | HATS/lsdb Catalog
            Input catalog or path (any format supported by astropy, or a HATS catalog).
        order_foot : int, optional
            Pixelization order. If None, uses self.order_foot.
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

        Returns
        -------
        healsparse.HealSparseMap
            Boolean footprint map
        """
        columns = tuple(columns)
        # Check if user wants specific orders, otherwise get from defaults
        order_sparse = order_foot if order_foot is not None else self.order_foot
        ord_cov = order_cov if order_cov is not None else self.order_cov

        nside_sparse = 1 << order_sparse
        nside_cov = 1 << ord_cov

        print("BUILDING FOOT MASK >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("    Order ::", order_sparse)

        # Prepare empty output footprint map
        self.footmask = hsp.HealSparseMap.make_empty(nside_cov, nside_sparse, dtype=np.bool_)

        # Compute pixels exactly once (except for HATS+mapping which is distributed)
        pixels = self._pixels_from_sources(sources, nside_sparse, columns, mapping=mapping,
                                           order_foot=order_sparse, order_cov=ord_cov)

        # Update map values for pixels that have objects
        if pixels.size: self.footmask.update_values_pix(pixels, True, operation="or")

        # Optional post-processing
        if remove_isopixels: self.footmask = self.remove_isopixels(self.footmask)
        if erode_borders: self.footmask = self.erode_borders(self.footmask)

        # Force packing if desired
        if bit_packed: self.footmask = self.footmask.as_bit_packed_map()

        # Store calling/useful info in its own dictionary
        area_deg2 = self.footmask.get_valid_area(degrees=True)
        npix = self.footmask.n_valid
        if hasattr(sources, "hc_structure") and hasattr(sources.hc_structure, "catalog_path"):
            sources_string = sources.hc_structure.catalog_path  # extract lsdb path
        elif isinstance(sources, (str, Path)):
            sources_string = str(sources)                        # get simply the string
        else:
            sources_string = "<mem>"                             # fallback when input from mem
        self._store_params('footmask',
            sources=sources_string, order_foot=order_sparse, order_cov=ord_cov,
            columns=columns, remove_isopixels=remove_isopixels, erode_borders=erode_borders,
            bit_packed=bit_packed, pixels=npix, area_deg2=area_deg2)

        print('--- Foot mask area                         :', area_deg2)
        return self.footmask


    def BAK_build_footprint_mask(self, sources, order_foot=None, columns=['ra','dec'],
                                remove_isopixels=False, erode_borders=False, mapping=False):
        """
        Create a footprint map of a source catalog (from any astropy-supported table or HATS),
        pixelated at a given order. Optionally remove isolated empty pixels and erode borders
        around empty zones. For details see remove_isopixels() and erode_borders()

        Parameters
        ----------
        sources : str, astropy.table.Table, pandas.DataFrame, or HATS catalog
            Input catalog or path to catalog (any format supported by astropy or lsdb.read_hats for HATS)
        order_foot : int
            Pixelization order
        columns : list of str
            Columns for ra, dec
        remove_isopixels : bool
            Remove isolated (empty) pixels surrounded by 8 non-empty pixels
        erode_borders : bool
            Detect and remove border pixels around holes
        mapping : bool
            If True, distribute pixelization across HATS partitions. A Dask cluster must be running

        Returns
        -------
        hsp_map
            Healsparse boolean map
        """

        def footpartition(df, pixel, order_foot=15, order_cov=6, columns=['ra','dec']):
            # Returns a df with the list of pixels of the input df
            nside_foot   = 1<<order_foot
            nside_cov    = 1<<order_cov
            foot = hsp.HealSparseMap.make_empty(nside_cov, nside_foot, dtype=np.bool_)
            pixels = hp.ang2pix(nside_foot, df[columns[0]], df[columns[1]], nest=True, lonlat=True)
            foot.update_values_pix(pixels, True, operation='or')   #np.full_like(pixels, True, dtype=np.bool_)
            outdf = pd.DataFrame(foot.valid_pixels, columns=['pxs'])
            return outdf

        metafootpartition = pd.DataFrame([{"pxs": 0}])


        if order_foot: self.order_foot = order_foot
        colra, coldec = columns
        print('BUILDING FOOTPRINT MAP >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

        # Create empty healsparse foot
        nside_foot = 1<<self.order_foot
        self.foot = hsp.HealSparseMap.make_empty(self.nside_cov, nside_foot, dtype=np.bool_)

        # Read sources and pixelize based on input type
        srcs = None
        if str(sources.__class__) == "<class 'lsdb.catalog.catalog.Catalog'>" :
            print('--- Pixelating HATS catalog')
            import lsdb
            srcs = sources
            if mapping:
                print(f"    Partitions for mapping: {srcs.npartitions:<7}")
                pixdf = srcs.map_partitions(footpartition, include_pixel=True, meta=metafootpartition,
                                            order_foot=self.order_foot, order_cov=self.order_cov, columns=columns).compute()
                pixels = np.array(pixdf['pxs'].values)
            else:
                srcs = srcs[columns].compute()
                pixels = hp.ang2pix(nside_foot, srcs[colra], srcs[coldec], nest=True, lonlat=True) # get pixel nr for each object
        elif isinstance(sources, Table):
            print('--- Pixelating sources')
            srcs = sources
            pixels = hp.ang2pix(nside_foot, srcs[colra], srcs[coldec], nest=True, lonlat=True) # get pixel nr for each object
        elif isinstance(sources, pd.DataFrame):
            print('--- Pixelating sources')
            srcs = Table.from_pandas(sources)
            pixels = hp.ang2pix(nside_foot, srcs[colra], srcs[coldec], nest=True, lonlat=True) # get pixel nr for each object
        elif isinstance(sources, str):
            print('--- Pixelating sources from:', sources)
            srcs = Table.read(sources)
            pixels = hp.ang2pix(nside_foot, srcs[colra], srcs[coldec], nest=True, lonlat=True) # get pixel nr for each object
        else:
            raise ValueError("sources must be a str, astropy.table.Table, or pandas.DataFrame")

        print('    Order ::',self.order_foot)

        # Update map values for pixels that have objects
        self.foot.update_values_pix(pixels, True, operation='or')

        # Remove isolated empty pixels and borders around holes, if requested
        if remove_isopixels:
            self.foot = self.remove_isopixels(self.foot)

        if erode_borders:
            self.foot = self.erode_borders(self.foot)

        print('--- Footprint map area                    :', self.foot.get_valid_area(degrees=True))



    @staticmethod
    def intersect_boolmask(mask1, mask2, bit_packed: "Optional[bool]" = None):
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
    def subtract_boolmask(mask1, mask2, bit_packed: "Optional[bool]" = None):
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
        xx, yy = hsp.make_uniform_randoms_fast(mk, nr)

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



    def makerans(self, stage='mask', nr=50_000, file=None, **kwargs):
        """
        Generate random points over a given mask and optionally save it to disk

        Parameters
        ----------
        stage : str
            Masking stage to use, e.g. 'mask', 'foot', 'holemap', etc.
        nr : int
            Number of randoms
        file : str, optional
            Output file (any format supported by astropy)
        kwargs : kwargs
            Extra arguments passed to astropy's write(), e.g. format="parquet"

        Returns
        -------
        dataframe/astropy_table
            Input catalog with mask applied
        """
        # Choose stage based on its name in a pipeline
        mk = getattr(self, stage)

        rra, rdec = hsp.make_uniform_randoms_fast(mk, nr)
        tt = Table([rra, rdec],names=['ra','dec'])
        if file:
            tt.write(file, overwrite=True, **kwargs)
            print(str(nr),'randoms written to:', file)

        return tt


    def apply(self, stage='mask', cat=None, columns=['ra','dec'], file=None):
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



    # =========================
    # ===== Helper Utils  =====
    # =========================
    @staticmethod
    def _is_bool_dtype(dt):
        import numpy as np
        try:
            return np.issubdtype(dt, np.bool_)
        except Exception:
            return False

    def _assert_bool_map(self, m, name="map"):
        if not self._is_bool_dtype(m.dtype):
            raise TypeError(f"{name}: expected boolean HealSparseMap; got dtype={m.dtype!r}")

    def _is_bit_packed(self, m) -> bool:
        return bool(getattr(m, "is_bit_packed_map", False))

    def _empty_like_geometry(self, *, nside_cov, nside_sparse, bit_packed: bool):
        #import numpy as np, healsparse as hsp
        return hsp.HealSparseMap.make_empty(
            nside_coverage=int(nside_cov),
            nside_sparse=int(nside_sparse),
            dtype=np.bool_,
            bit_packed=bool(bit_packed),
        )

    def _to_bit_packed(self, m):
        """Return a bit-packed boolean map with the same geometry as m."""
        self._assert_bool_map(m, "to_bit_packed")
        return m if self._is_bit_packed(m) else m.as_bit_packed_map()

    def _to_unpacked(self, m):
        """Return an unpacked (byte-per-pixel) boolean map with the same geometry as m."""
        self._assert_bool_map(m, "to_unpacked")
        if not self._is_bit_packed(m):
            return m
        out = self._empty_like_geometry(
            nside_cov=m.nside_coverage, nside_sparse=m.nside_sparse, bit_packed=False
        )
        vp = m.valid_pixels
        if vp.size:
            out.update_values_pix(vp, True, operation="replace")
        return out

    # ===========================================
    # ===== Sparse (order) upgrade / degrade =====
    # ===========================================
    def upgrade_sparse_order(self, stage, new_order, *, max_children_batch: int = 20_000_000):
        """
        Increase nside_sparse to 2**new_order (NESTED). Output keeps the SAME packing as input.
        True parents expand to ALL True children.
        """
        import numpy as np
        self._assert_bool_map(stage, "upgrade_sparse_order")
        src_packed = self._is_bit_packed(stage)

        old_nside = int(stage.nside_sparse)
        new_nside = int(1<<new_order)
        if new_nside == old_nside:
            return stage
        if new_nside < old_nside:
            raise ValueError("upgrade_sparse_order: target must be finer than current.")

        # children per parent (NESTED): 4**Δorder
        delta_order = int(new_order) - int(np.log2(old_nside))
        if delta_order <= 0:
            raise ValueError("upgrade_sparse_order: non-positive Δorder computed.")
        r2 = 4**delta_order

        out = self._empty_like_geometry(
            nside_cov=stage.nside_coverage, nside_sparse=new_nside, bit_packed=src_packed
        )
        vp = stage.valid_pixels
        if vp.size == 0:
            return out

        parents_per_batch = max(1, max_children_batch // r2)
        off = np.arange(r2, dtype=np.int64)

        for i0 in range(0, vp.size, parents_per_batch):
            p = vp[i0:i0 + parents_per_batch].astype(np.int64)
            base = p * r2
            children = (base[:, None] + off).reshape(-1)
            out.update_values_pix(children, True, operation="replace")

        return out

    def degrade_sparse_order(self, stage, new_order, *, max_parent_batch: int = 25_000_000):
        """
        Decrease nside_sparse to 2**new_order (NESTED). Output keeps the SAME packing as input.
        Parent=True if ANY child=True (OR semantics).
        """
        import numpy as np
        self._assert_bool_map(stage, "degrade_sparse_order")
        src_packed = self._is_bit_packed(stage)

        old_nside = int(stage.nside_sparse)
        new_nside = int(1<<new_order)
        if new_nside == old_nside:
            return stage
        if new_nside > old_nside:
            raise ValueError("degrade_sparse_order: target must be coarser than current.")

        delta_order = int(np.log2(old_nside)) - int(new_order)
        if delta_order <= 0:
            raise ValueError("degrade_sparse_order: non-positive Δorder computed.")
        r2 = 4**delta_order  # children per parent to collapse

        out = self._empty_like_geometry(
            nside_cov=stage.nside_coverage, nside_sparse=new_nside, bit_packed=src_packed
        )
        vp = stage.valid_pixels
        if vp.size == 0:
            return out

        for i0 in range(0, vp.size, max_parent_batch):
            parents = (vp[i0:i0 + max_parent_batch] // r2).astype(np.int64)
            # Dedup within batch to reduce writes
            if parents.size:
                parents = np.unique(parents)
                out.update_values_pix(parents, True, operation="replace")

        return out

    # ==================================
    # ===== Coverage NSIDE changer  =====
    # ==================================
    def change_nside_coverage(self, stage, new_order_cov):
        """
        Change nside_coverage to 2**new_order_cov. Output keeps the SAME packing as input.
        """
        self._assert_bool_map(stage, "change_nside_coverage")
        src_packed = self._is_bit_packed(stage)

        current_cov = int(stage.nside_coverage)
        target_cov = int(1<<new_order_cov)
        if target_cov == current_cov:
            return stage

        out = self._empty_like_geometry(
            nside_cov=target_cov, nside_sparse=stage.nside_sparse, bit_packed=src_packed
        )
        vp = stage.valid_pixels
        if vp.size:
            out.update_values_pix(vp, True, operation="replace")
        return out

    # ==================================
    # ===== One-shot stage regridding ===
    # ==================================
    def regrid_stage(self, stage, *, order_out: int, order_cov: int):
        """
        Return `stage` at (order_out, order_cov), preserving the stage's packing.
        """
        self._assert_bool_map(stage, "regrid_stage")

        # 1) sparse order first
        current_order = int(__import__("numpy").log2(stage.nside_sparse))
        if current_order < order_out:
            stage = self.upgrade_sparse_order(stage, order_out)
        elif current_order > order_out:
            stage = self.degrade_sparse_order(stage, order_out)

        # 2) coverage order
        stage = self.change_nside_coverage(stage, order_cov)
        return stage



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
    def pix_in_zone(hsp, ra_min: float, ra_max: float, dec_min: float, dec_max: float, *,
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
        stage : hspmap
            Healsparse map of a stage
        ra_min, ra_max, dec_min, dec_flat :  floats
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



    def plot_moc(self, stage, center=None, fov=None, clipra=None, clipdec=None, order_force=None,
                 frame='icrs', projection='SIN', figsize=(10, 5),
                 color='green', alpha=0.2, linewidth=1.0, label=None,
                 ax=None, wcs=None, show=False, stream_pars=None):
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
            print('Looking pixels inside clip box...')
            stream_pars = stream_pars or {}    # make sure we pass an empty dict al least
            pixels = self.pix_in_zone(stage, ra_min=clipra[0], ra_max=clipra[1],
                                      dec_min=clipdec[0], dec_max=clipdec[1], return_mode='array', **stream_pars)
        else:
            print('Looking pixels...')
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



    def _resolve_stage_input(self, stage):
        """
        Normalize a `stage` (string stage name or HealSparse map) into (name, map).

        Returns
        -------
        (name, map)
            - name: canonical stage attribute name on `self` ("foot", "propmap", ...),
                    or a descriptive label like "<map>" if the input is a raw map
                    not attached to the pipeline.
            - map : the HealSparseMap object.
        """
        name_map = {
            # add aliases if needed → canonical attribute names on `self`
            'footprint': 'foot',
            'patchmask': 'patchmask',
            'propmap': 'propmap',
            'circmask': 'circmask',
            'ellipmask': 'ellipmask',
            'starmask': 'starmask',
            'boxmask': 'boxmask',
            'zonemask': 'zonemask',
            'polymask': 'polymask',
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



    def combine(self, *, positive, negative=None, order_out: Optional[int] = None,
                order_cov: Optional[int] = None, bit_packed: bool = True, verbose: bool = True):
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
        self.mask = res
        self.order_out = tgt_ord
        self.order_cov = cov_ord
        self.nside_out = tgt_ns
        self.nside_cov = cov_ns

        if verbose:
            area = float(res.get_valid_area(degrees=True))
            print(f"[combine] done: order_out={tgt_ord} (NSIDE={tgt_ns}), "
                  f"order_cov={cov_ord} (NSIDE={cov_ns}), "
                  f"valid_pix={res.n_valid:,}, area={area:.3f} deg², bit_packed={bit_packed}")

        return res




    def change_sparse_order(self, stage, order: int, *, inplace: bool = True, verbose: bool = True):
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



    def change_cov_order(self, stage, order: int, *, inplace: bool = True, verbose: bool = True):
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



    def build_circ_mask(self, data: Union[pd.DataFrame, str, Path] = None, order_circ: Optional[int] = None,
            order_cov: Optional[int] = None, fmt: str = 'ascii',
            columns: Optional[Sequence[str]] = ['ra','dec','radius'],
            bit_packed: Optional[bool] = None, n_threads: int = 4):

        """
        Build a mask from input circle data and store it as a HealSparse boolean map.

        Each input row describes a circle in the sky, defined as [ra_center, dec_center, radius]
        in degrees, whera ra_center,dec_center are the center coordinates. The circles are
        pixelized at `order_circ` to produce the set of HEALPix pixels covering it.

        Parameters
        ----------
        data : pandas.DataFrame, str, or pathlib.Path
            The circle definitions. If a DataFrame, it must contain the columns named
            in `columns`. If a path (string or Path), it is read according
            to `fmt` (delegated to astropy).
        order_circ : int, optional
            Sparse order to pixelize the circles. If None, falls back to `self.order_circ`
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

        Returns
        -------
        HealSparseMap
            A boolean (or bit-packed) mask is both, stored at `self.circmask` and returned
            to prompt
        """
        print('BUILDING CIRCLES MASK >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # Check if user wants specific orders, otherwise get from defaults
        order_sparse = order_circ if order_circ is not None else self.order_circ
        ord_cov = order_cov if order_cov is not None else self.order_cov

        nside_sparse = 1<<order_sparse
        nside_cov = 1<<ord_cov

        # Create the empty boolean map *up front*
        self.circmask = hsp.HealSparseMap.make_empty(nside_cov, nside_sparse, dtype=np.bool_)

        # Perform pixelization
        pix = self.pixelate_circles(data, fmt=fmt, order=order_sparse, columns=columns, n_threads=n_threads)
        if pix is not None and len(pix) > 0:
            self.circmask.update_values_pix(pix, True)

        # Force packing if desired
        if bit_packed: self.circmask = self.circmask.as_bit_packed_map()

        # Store calling/useful info in its own dictionary
        area_deg2 = self.circmask.get_valid_area(degrees=True)
        npix = self.circmask.n_valid
        self._store_params('circmask',
            data=(str(data) if isinstance(data, (str, Path)) else "<mem>"),
            order_circ=order_sparse, order_cov=ord_cov,
            fmt=fmt, columns=columns, bit_packed=bit_packed, n_threads=n_threads,
            pixels=npix, area_deg2=area_deg2)

        print('--- Circles mask area                        :', area_deg2)
        return self.circmask


    def build_box_mask(self, data: Union[pd.DataFrame, str, Path] = None, order_box: Optional[int] = None,
            order_cov: Optional[int] = None, fmt: str = 'ascii',
            columns: Optional[Sequence[str]] = ['ra_c','dec_c','width','height'],
            bit_packed: Optional[bool] = None, n_threads: int = 4):

        """
        Build a mask from input box data and store it as a HealSparse boolean map.

        Each input row describes a box in the sky, defined as [ra_center, dec_center,
        width, height] in degrees, whera ra_center,dec_center are the center coordinates.
        The boxes are pixelized at `order_box` to produce the set of HEALPix pixels covering it.

        Parameters
        ----------
        data : pandas.DataFrame, str, or pathlib.Path
            The box definitions. If a DataFrame, it must contain the columns named
            in `columns`. If a path (string or Path), it is read according
            to `fmt` (delegated to astropy).
        order_poly : int, optional
            Sparse order to pixelize the boxes. If None, falls back to `self.order_box`
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

        Returns
        -------
        HealSparseMap
            A boolean (or bit-packed) mask is both, stored at `self.boxmask` and returned
            to prompt
        """
        print('BUILDING BOXES MASK >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # Check if user wants specific orders, otherwise get from defaults
        order_sparse = order_box if order_box is not None else self.order_box
        ord_cov = order_cov if order_cov is not None else self.order_cov

        nside_sparse = 1<<order_sparse
        nside_cov = 1<<ord_cov

        # Create the empty boolean map *up front*
        self.boxmask = hsp.HealSparseMap.make_empty(nside_cov, nside_sparse, dtype=np.bool_)

        # Perform pixelization
        pix = self.pixelate_boxes(data, fmt=fmt, order=order_sparse, columns=columns, n_threads=n_threads)
        if pix is not None and len(pix) > 0:
            self.boxmask.update_values_pix(pix, True)

        # Force packing if desired
        if bit_packed: self.boxmask = self.boxmask.as_bit_packed_map()

        # Store calling/useful info in its own dictionary
        area_deg2 = self.boxmask.get_valid_area(degrees=True)
        npix = self.boxmask.n_valid
        self._store_params('boxmask',
            data=(str(data) if isinstance(data, (str, Path)) else "<mem>"),
            order_box=order_sparse, order_cov=ord_cov,
            fmt=fmt, columns=columns, bit_packed=bit_packed, n_threads=n_threads,
            pixels=npix, area_deg2=area_deg2)

        print('--- Boxes mask area                           :', area_deg2)
        return self.boxmask



    def build_ellip_mask(self, data: Union[pd.DataFrame, str, Path] = None, order_ellip: Optional[int] = None,
            order_cov: Optional[int] = None, fmt: str = 'ascii',
            columns: Optional[Sequence[str]] = ['ra','dec','a','b','pa'],
            bit_packed: Optional[bool] = None):

        """
        Build a mask from input ellipse data and store it as a HealSparse boolean map.

        Each input row describes an ellipse in the sky, defined as [ra, dec, a, b, pa]
        in degrees, where ra,dec are center coordinates; a,b are the mayor and minor half axes;
        and pa is the position angle. The rectangles are pixelized at `order_ellip` to
        produce the set of HEALPix pixels covering it.

        Parameters
        ----------
        data : pandas.DataFrame, str, or pathlib.Path
            The ellipse definitions. If a DataFrame, it must contain the columns named
            in `columns`. If a path (string or Path), it is read according
            to `fmt` (delegated to astropy).
        order_ellip : int, optional
            Sparse order to pixelize the ellipses. If None, falls back to `self.order_ellip`
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

        Returns
        -------
        HealSparseMap
            A boolean (or bit-packed) mask is both, stored at `self.ellipmask` and returned
            to prompt
        """

        print('BUILDING ELLIPSES MASK >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # Check if user wants specific orders, otherwise get from defaults
        order_sparse = order_ellip if order_ellip is not None else self.order_ellip
        ord_cov = order_cov if order_cov is not None else self.order_cov

        nside_sparse = 1<<order_sparse
        nside_cov = 1<<ord_cov

        # Create the empty boolean map *up front*
        self.ellipmask = hsp.HealSparseMap.make_empty(nside_cov, nside_sparse, dtype=np.bool_)

        # Perform pixelization
        pix = self.pixelate_ellipses(data, fmt=fmt, order=order_sparse, columns=columns)
        if pix is not None and len(pix) > 0:
            self.ellipmask.update_values_pix(pix, True)

        # Force packing if desired
        if bit_packed: self.ellipmask = self.ellipmask.as_bit_packed_map()

        # Store calling/useful info in its own dictionary
        area_deg2 = self.ellipmask.get_valid_area(degrees=True)
        npix = self.ellipmask.n_valid
        self._store_params('ellipmask',
            data=(str(data) if isinstance(data, (str, Path)) else "<mem>"),
            order_ellip=order_sparse, order_cov=ord_cov,
            fmt=fmt, columns=columns, bit_packed=bit_packed,
            pixels=npix, area_deg2=area_deg2)

        print('--- Ellipses mask area                        :', area_deg2)
        return self.ellipmask


    def build_poly_mask(self, data: Union[pd.DataFrame, str, Path] = None, order_poly: Optional[int] = None,
            order_cov: Optional[int] = None, fmt: str = 'ascii',
            columns: Optional[Sequence[str]] = ['ra0','ra1','ra2','ra3','dec0','dec1','dec2','dec3'],
            bit_packed: Optional[bool] = None, n_threads: int = 4):

        """
        Build a mask from input polygon data and store it as a HealSparse boolean map.

        Each input row describes a sky quadrangular polygon via 4 corner points
        (ra0, ra1, ra2, ra3, dec0, dec1, dec2, dec3) in degrees. The rectangles are
        pixelized at `order_poly` to produce the set of HEALPix pixels covering it.

        Parameters
        ----------
        data : pandas.DataFrame, str, or pathlib.Path
            The polygon definitions. If a DataFrame, it must contain the columns named
            in `columns`. If a path (string or Path), it is read according
            to `fmt` (delegated to astropy).
        order_poly : int, optional
            Sparse order to pixelize the polygons. If None, falls back to `self.order_poly`
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

        Returns
        -------
        HealSparseMap
            A boolean (or bit-packed) mask is both, stored at `self.polymask` and returned
            to prompt
        """

        print('BUILDING POLYGON MASK >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # Check if user wants specific orders, otherwise get from defaults
        order_sparse = order_poly if order_poly is not None else self.order_poly
        ord_cov = order_cov if order_cov is not None else self.order_cov

        nside_sparse = 1<<order_sparse
        nside_cov = 1<<ord_cov

        # Create the empty boolean map *up front*
        self.polymask = hsp.HealSparseMap.make_empty(nside_cov, nside_sparse, dtype=np.bool_)

        # Perform pixelization
        pix = self.pixelate_polys(data, fmt=fmt, order=order_sparse, columns=columns, n_threads=n_threads)
        if pix is not None and len(pix) > 0:
            self.polymask.update_values_pix(pix, True)

        # Force packing if desired
        if bit_packed: self.polymask = self.polymask.as_bit_packed_map()

        # Store calling/useful info in its own dictionary
        area_deg2 = self.polymask.get_valid_area(degrees=True)
        npix = self.polymask.n_valid
        self._store_params('polymask',
            data=(str(data) if isinstance(data, (str, Path)) else "<mem>"),
            order_poly=order_sparse, order_cov=ord_cov,
            fmt=fmt, columns=columns, bit_packed=bit_packed, n_threads=n_threads,
            pixels=npix, area_deg2=area_deg2)

        print('--- Polygon mask area                           :', area_deg2)
        return self.polymask



    def build_zone_mask(self, data: Union[pd.DataFrame, str, Path] = None,
                        order_zone: Optional[int] = None, order_cov: Optional[int] = None,
                        fmt: str = 'ascii',
                        columns: Optional[Sequence[str]] = ['ra1','dec1','ra2','dec2'],
                        bit_packed: Optional[bool] = None):
        """
        Build a boolean HealSparse mask from rectangular sky zones. A zone is defined
        by ra boundaries (major circles) and dec limits (minor circles).

        Each input row describes a sky-aligned rectangle via two corner points
        (ra1, dec1) and (ra2, dec2) in degrees. The rectangles are pixelized at
        `order_zone` to produce the set of HEALPix pixels covering it.

        Parameters
        ----------
        data : pandas.DataFrame | str | pathlib.Path | None
            The zone definitions. If a DataFrame, it must contain the columns named
            in `columns`. If a path (string or Path), it is read according
            to `fmt` (delegated to astropy).
        order_zone : int, optional
            Sparse order to pixelize the zones. If None, falls back to `self.order_zone`
        order_cov : int, optional
            Coverage order. If None, falls back to `self.order_cov`
        fmt : str, default "ascii"
            File format used when `data` is a path (e.g., "ascii", "parquet",
            "csv", or any format supported by the astropy readers)
        columns : sequence of str, optional
            Columns for the two corners. If None, defaults to ("ra1", "dec1", "ra2", "dec2")
        bit_packed : bool, optional
            If True, return the ouput as bit-packed boolean map

        Returns
        -------
        HealSparseMap
            A boolean (or bit-packed) mask is both, stored at `self.zonemask` and returned
            to prompt

        Notes
        -----
        - All angles must be in degrees
        - This method stores useful metadata (e.g., orders, format, input source,
          pixel count, area) via `self._store_params("zonemask")`

        See Also
        --------
        pixelate_zones : Converts rectangular zones to pixels at a given order
        """
        print('BUILDING ZONES MASK >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # Check if user wants specific orders, otherwise get from defaults
        order_sparse = order_zone if order_zone is not None else self.order_zone
        ord_cov = order_cov if order_cov is not None else self.order_cov

        nside_sparse = 1<<order_sparse
        nside_cov = 1<<ord_cov

        # Create the empty boolean map *up front*
        self.zonemask = hsp.HealSparseMap.make_empty(nside_cov, nside_sparse, dtype=np.bool_)

        # Perform pixelization
        pix = self.pixelate_zones(data, fmt=fmt, order=order_sparse, columns=columns)
        if pix is not None and len(pix) > 0:
            self.zonemask.update_values_pix(pix, True)

        # Force packing if desired
        if bit_packed: self.zonemask = self.zonemask.as_bit_packed_map()

        # Store calling/useful info in its own dictionary
        area_deg2 = self.zonemask.get_valid_area(degrees=True)
        npix = self.zonemask.n_valid
        self._store_params('zonemask',
            data=(str(data) if isinstance(data, (str, Path)) else "<mem>"),
            order_zone=order_sparse, order_cov=ord_cov,
            fmt=fmt, columns=columns, bit_packed=bit_packed,
            pixels=npix, area_deg2=area_deg2)

        print('--- Zones mask area                           :', area_deg2)
        return self.zonemask

