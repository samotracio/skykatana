---
title: IO format
icon: material/storage-tank
---
# IO format

<span class="sc">SkyKatana</span> persists a `SkyMaskPipe` objecs to a directory containing one JSON metadata file plus <span class='emph1'>one FITS table per stage</span>, using a compact <span class='emph1'>bit-packed</span> boolean encoding designed for high-throughput streaming I/O, minimal impact on RAM usage, and compact file size.

## Example Directory layout

```
<outdir>/
  ├─ metadata.json
  ├─ footmask.fits
  ├─ propmask.fits
  ├─ starmask.fits
  └─ … (one <stage>.fits per discovered stage)
```

Stages are discovered dynamically from the in-memory object (attributes ending in `mask` that are valid HealSparse maps).

## `metadata.json`
The JSON captures:

- `format`: `"skymaskpipe-bitpack-fits-stream"`
- `version`: internal on-disk format version
- `class`: class name (`SkyMaskPipe`)
- `stages`: mapping from stage name → `{ "filename": "<stage>.fits" }`
- `scalars`: selected scalar attributes saved from the pipeline
- `params`: JSON-safe copy of per-stage parameter dictionaries

This structure is created during <span class="met">write()</span> and later restored by <span class="met">read()</span>.

## Stage FITS format (bit-packed)
Each stage is a single-table FITS file with **one row per coverage pixel** that has at least one “child” sparse pixel set to True.

### Table schema

- `COVPIX` (type `K`): coverage pixel index (NESTED)
- `ENC` (type `B`): row occupancy flag (always 1 for non-empty rows)
- `PACKED` (type `PB()`): a **byte array** holding the little-endian bit-packed bitmap of set child pixels within that coverage row

The table header stores geometry and encoding:

- `NSIDE_COV`, `NSIDE_SPA` (coverage & sparse resolutions)
- `DTYPE='bool'`
- `ENCOD='BITPACK'`
- `NFINE` (= `(nside_sparse/nside_coverage)^2`, children per coverage pixel)
- `BITORD='L'` (little-endian bit order within each byte)

### Why bit-packing?

Instead of storing a dense boolean array per coverage pixel (wasteful when occupancy is sparse), <span class="sc">SkyKatana</span>  writes only the **minimal number of bytes** needed to represent the highest child offset, with bits set for active children. Packing reduces I/O volume and memory pressure while preserving exact geometry.

## Algorithms

### Writing (<span class="met">write()</span> → `_write_stage_fits_bitpack`)

1. **Discover stages** in the object (e.g., `footmask`, `propmask`, …).  
2. For each stage, iterate **valid sparse pixels grouped by coverage pixel**.  
3. For each row:
   - Compute child offsets `off = fine_pix - (covpix * NFINE)` (sorted).
   - **Pack offsets into bytes** without allocating a full row bitmap:

     ```
     byte_idx = off // 8
     bit_pos  = off % 8
     contrib  = (1 << bit_pos)
     add.at(out_u16, byte_idx, contrib)  # acts like bitwise OR; no duplicates
     packed = out_u16.astype(uint8)
     ```

4. **Stream in batches** to FITS (`rows_per_batch`), appending `COVPIX`, `ENC=1`, and `PACKED` arrays; flush at batch boundaries.

5. After all stages, write `metadata.json` and atomically move the temp directory into place.


### Reading (<span class="met">read()</span> → `_read_stage_fits_bitpack_fast`)

1. Load `metadata.json`, restore scalar attributes, enumerate stages, and choose a **thread pool** size.
2. For each stage (in parallel workers):
   - Open FITS, **validate encoding**, and **recreate an empty HealSparse map** with `bit_packed=True`.
   - **Block streaming**: read rows in chunks (`io_block_rows`). Before expanding, estimate how many pixels the block will produce using **byte-level popcount**. A population count (popcount) operation returns the number of bits set to 1 in a binary word while a byte-level popcount applies that to each 8-bit byte.
   - **Expand `PACKED` → child pixel IDs**:
     - For row `i`, compute `base = covpix[i] * NFINE`.
     - Iterate bytes; for each set bit, append `base + bit_index` to the buffer.
     - After a buffer fill or end of block, **update** the HealSparse map.

3. Attach each loaded map back onto the newly constructed `SkyMaskPipe` instance under its recorded stage name.

!!! Notes

    * The read path is **idempotent** w.r.t. geometry, i.e. reading the same on-disk directory multiple times produces exactly the same in-memory result, without side-effects that accumulate or alter data.
    * The writer enforces `nside_sparse % nside_coverage == 0` to ensure `NFINE` is an integer.

## Performance characteristics

- **Streaming** on both write and read minimizes peak memory
- **Bit-packing** reduces on-disk footprint and I/O.
- **Parallel stage loading** hides latency across multiple FITS files.

## Forward/Backward compatibility

A simple on-disk **version** is stored in `metadata.json` (`_BITPACK_FITS_VERSION`), enabling improvements of the layout while keeping older readers able to validate or branch on version.

