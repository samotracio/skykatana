---
title: Pixelization with GPU
icon: material/gamepad-circle-outline
---
Skykatana includes an <span class='emph1'>optional GPU backend</span> for pixelizing circles (spherical cones) on the sky with *(RA, Dec, radius)* into HEALPix pixels. This is most useful when you need to pixelize very large numbers of circles (e.g., bright-star masks), where the default CPU approach becomes slow or memory-heavy.

The GPU backend is designed to be a **drop-in alternative** to the default CPU pixelization route and is intended to produce the **same set of pixels** as the CPU reference implementation based on `mocpy` pixelization.

<hr class='sep1'>

### <span class='emph2'>Requirements and recommended installation</span>

- An **NVIDIA GPU** with a working NVIDIA driver.
- A Python environment with GPU dependencies (`numba` + `cupy`).
- A CUDA runtime/toolchain compatible with your NVIDIA driver.

**Important rule:** the CUDA runtime/toolchain in your environment should not be newer than what your driver supports. If it is, you may see errors such as “Unsupported PTX version”.

The recommended way is to use the provided conda environment file:

```bash
conda env create -f environment-gpu.yml
conda activate skykatana-gpu
pip install -e .
```

### <span class='emph2'>The `pixelizer` argument</span>

Several Skykatana “build” functions that pixelize circles support selecting the backend via:

- `pixelizer="cpu"` *(default)*
- `pixelizer="gpu"` *(requires GPU dependencies)*

The main entry points are:

- `build_circ_mask(..., pixelizer=...)`
- `build_star_mask_online(..., pixelizer=...)`

The output is always a stage that is finalized/stored exactly like the CPU route. Depending on the GPU/CPU models and mask size, you can expect a <span class='emph1'>~x4 to x7 increase in performance</span> compared to the CPU case.

---

### <span class='emph2'>GPU keyword arguments</span>

When `pixelizer="gpu"`, you can pass a `gpu_kwargs` dictionary to tune performance. The most important keys are:

- `stream_batch` *(int)*  
  Number of circles processed per streamed batch on the GPU.  
  Typical values: `10_000`–`1_000_000` (default `20_000`).

- `threads_per_block` *(int)*  
  CUDA threads per block used for GPU kernels.  
  Typical values: `128` (default), sometimes `256` depending on GPU.

- `group_by_covpix` *(bool)*  
  If `True`, circles are grouped by coverage pixel to improve locality and reduce overhead.  
  Recommended: `True`.

- `keep_on_gpu` *(bool)*  
  If `True`, internal buffers are kept on the GPU across streamed batches to reduce transfers.  
  Recommended: `True` when GPU memory allows.

- `timing_split` *(bool)*  
  If `True`, prints a timing breakdown (useful for profiling and debugging).  
  Recommended: `False` by default.

- `benchmark_mode` *(bool)*  
  If `True`, enables additional kernel timing counters used for profiling.  
  Recommended: `False` unless benchmarking.

Advanced keys (most users should not need these):

- `covpix_list` *(array-like of int)*: explicit list of coverage pixels used to pre-allocate rows.
- `bitpack_accumulator`: reuse an accumulator across calls/chunks for advanced workflows.

---

### Example: `build_circ_mask` with GPU

```python
mask = pipe.build_circ_mask(
    data=df_circles,
    order_sparse=15,
    pixelizer="gpu",
    gpu_kwargs={
        "stream_batch": 20000,
        "threads_per_block": 128,
    },
)
```

CPU (default):

```python
mask = pipe.build_circ_mask(
    data=df_circles,
    order_sparse=15,
    pixelizer="cpu",
)
```

---

### Example: `build_star_mask_online` with GPU

```python
starmask = pipe.build_star_mask_online(
    starq=starq,
    order_sparse=15,
    chunk_size=600000,
    pixelizer="gpu",
    gpu_kwargs={"stream_batch": 600000},
)
```

CPU:

```python
starmask = pipe.build_star_mask_online(
    starq=starq,
    order_sparse=15,
    chunk_size=600000,
    pixelizer="cpu",
    n_threads=4,
)
```

---

### <span class='emph2'>How it works (in-depth)</span>

#### <span class='emph2'>1) What “pixelizing circles” means in Skykatana</span>

Each input row represents a sky circle with `center: (ra_deg, dec_deg)` and `radius: radius_deg`. Pixelization produces the set of HEALPix NESTED pixels at a target sparse order (e.g., `order_sparse=15`) that **cover** the circle. Those pixels are then written into a boolean sky mask fully compatible with a **HealSparseMap**, and often in **bit-packed** form for memory efficiency.

#### <span class='emph2'>2) CPU route</span>

The CPU route typically looks like:

1. For each circle (or chunk of circles), compute the pixel set (via `mocpy` / `cdshealpix` semantics).
2. Merge pixels into the output `HealSparseMap` (often by updating many pixel indices).

#### <span class='emph2'>3) GPU route</span>

The GPU route keeps the same *mathematical definition* of which pixels belong to a circle, but changes the *dataflow*. Instead of generating large pixel lists on CPU, it computes coverage on GPU and **paints directly into a bit-packed mask buffer**. Concretely, the GPU backend:

1. **Streams circles in batches** (controlled by `stream_batch`) to keep memory bounded.
2. For each circle, determines the set of target HEALPix pixels (NESTED) it covers using a traversal strategy consistent with `mocpy` “small_cones”.
3. Instead of emitting pixel indices, it performs **bitwise OR painting** into a packed representation:
    - the output mask is represented as `uint64` words,
    - each output pixel maps to exactly one bit inside a `uint64` word,
    - painting is `word |= (1 << bit)` (plus range-fills for large contiguous runs when possible).

This avoids allocating or returning giant arrays of pixel IDs.

#### <span class='emph2'>4) Bit-packed mask layout</span>

Skykatana’s packed masks can be thought of as a **coverage layer** (order `order_cov`) that tells which coarse sky tiles exist; and within each coverage pixel, a fixed-size bit array that represents all fine pixels at `order_sparse`. For a target `order_sparse` and a chosen `order_cov`, each fine pixel belongs to exactly one coverage pixel, and its **offset** within that coverage pixel is a small integer. That offset maps to:

- `word_index = offset // 64`
- `bit_index = offset % 64`

The GPU painter uses this mapping to update bits directly. This layout enables very compact memory usage, efficient OR-merges and a clean path to build a `HealSparse` compliant map at the end.

#### <span class='emph2'>5) Accumulator model</span>

To support “streaming” and repeated calls (for chunks of sky or chunks of stars), the GPU pixelizer uses an accumulator pattern:

- A **GPU accumulator** holds the device-side buffers needed for painting efficiently.
- A **CPU-side packed-block dictionary** accumulates the final result:
    - keys: `covpix` (coverage pixel indices at `order_cov`)
    - values: `uint64[words_per_cov]` arrays (the packed bits)

Each streamed batch paints into GPU buffers, then the painted packed blocks are copied back and **OR-merged** into the CPU dictionary. OR-merging is associative and order-independent, so this is safe and stable.

At the end of the build function, Skykatana finalizes this packed dictionary into a real `HealSparseMap` object in memory.

#### <span class='emph2'>6) How Skykatana chooses which coverage pixels exist</span>

In GPU mode, the painter needs to know which coverage pixels (at `order_cov`) it is allowed to write into. This controls allocation and prevents accidental growth. Skykatana typically builds a **coverage pixel superset** for each chunk of work:

- in `build_star_mask_online`, it derives it from the current chunk MOC (plus border) and degrades it to `order_cov`,
- in `build_circ_mask`, it derives it from the input circles.

This superset is passed to the GPU backend as `covpix_list` and it should include every coverage pixel that any circle might touch. If it is too small, pixels could be dropped (and the code should warn/raise).



