---
title: Healsparse map basics
icon: material/map-marker
---
# Healsparse map basics

[Healsparse](https://github.com/LSSTDESC/healsparse) provides a sparse, memory‑efficient way to represent high‑resolution HEALPix maps. This section summarizes the key concepts you need when working with stages in <span class="sc">SkyKatana</span>. For more detailed infomation follow healsparse documentation.

---

## Core concept: HealSparse map

A **HealSparseMap** is a two‑layer representation of a sky map:

| Term | Meaning |
|------|--------|
| **`:::py3 nside_sparse`** | The *fine* HEALPix resolution where actual data are stored. |
| **`:::py3 nside_coverage`** | A coarser HEALPix map that marks which “coverage pixels” contain any valid data (the low‑resolution footprint). |
| **Valid / sparse pixels** | Only fine pixels inside active coverage pixels are stored; all other fine pixels are treated as missing. |
| **Sentinel** | A special invalid value (like HEALPix’s `:::py3 UNSEEN`) indicating pixels with no data. |

This design lets the map behave like a full‑sky HEALPix map while keeping memory proportional only to the number of populated regions.

---

## Boolean HealSparse maps

A **boolean HealSparse map** is a map whose values are just `True` or `False`. Typical uses include masks and coverage footprints. Characteristics:

  * Sentinel is always `False`.
  * Logical operations (`&`, `|`, `^`, `~`) are implemented efficiently.
  * You can query or iterate over valid pixels just like with numeric maps.

---

## Bit‑packed boolean maps

For extremely high‑resolution boolean maps, HealSparse supports a **bit‑packed** representation:

* *Storage*: packs eight booleans into a single byte, cutting memory and disk usage by roughly a factor of eight compared to a normal boolean array.
* *Performance*: while pixel lookups remain fast (bit tests within a byte),  iterating over *all* valid pixels requires scanning bits, so it’s slightly slower than a plain boolean array.
* *Constraints*: sentinel must be `:::py3 False`; the packing only applies to boolean maps.
  
---

## Understanding coverage vs. sparse nside

### Coverage layer
* Think of this as a quick-look index of where the map has data.
* At query time, the library first checks the coverage pixel to decide whether it even needs to look at finer pixels—this is why large empty sky regions cost almost no memory.
* Choosing a lower coverage NSIDE (coarser tiles) means fewer coverage pixels and a smaller index table, but each coverage pixel covers a bigger sky area.

### Sparse layer
* Holds the actual science values at `:::py3 NSIDE_SPARSE` resolution.
* Can be arbitrarily higher than the coverage NSIDE—often many powers of two higher.
* Only pixels inside active coverage cells are allocated.
* The sparse NSIDE must be an integer power-of-two multiple of the coverage NSIDE so that each coverage pixel can be subdivided cleanly.

### Choosing values
* `:::py3 NSIDE_COVERAGE` controls how finely the “index” can localize data. Smaller NSIDE (coarser) → smaller index, but queries may read more empty fine pixels per coarse cell. Typical values for masks are 16, 32, 64 or 128.
* `:::py3 NSIDE_SPARSE` sets the science resolution: pick this according to the angular detail you need (e.g., NSIDE 4096 for ~0.86 arcmin pixels).The table below shows for an order n, the nside (2<sup>n</sup>), number of pixels N<sub>pix</sub> (12Nside<sup>2</sup>) and pixel size defined by HEALpix.

| n  | Nside | N<sub>pix</sub> | θ<sub>pix</sub> |
|----|------------|-------------------|---------------------|
| 0  | 1     | 12            | 58.6°  |
| 1  | 2     | 48            | 29.3°  |
| 2  | 4     | 192           | 14.7°  |
| 3  | 8     | 768           | 7.33°  |
| 4  | 16    | 3072          | 3.66°  |
| 5  | 32    | 12288         | 1.83°  |
| 6  | 64    | 49152         | 55.0′  |
| 7  | 128   | 196608        | 27.5′  |
| 8  | 256   | 786432        | 13.7′  |
| 9  | 512   | 3,145,728     | 6.87′  |
| 10 | 1024  | 12,582,912    | 3.44′  |
| 11 | 2048  | 50,331,648    | 1.72′  |
| 12 | 4096  | 201,326,592   | 51.5″  |
| 13 | 8192  | 805,306,368   | 25.8″  |
| 14 | 2<sup>14</sup>  | 3.22 × 10<sup>9</sup>   | 12.9″  |
| 15 | 2<sup>15</sup>  | 1.29 × 10<sup>10</sup>  | 6.44″  |
| 16 | 2<sup>16</sup>  | 5.15 × 10<sup>10</sup>  | 3.22″  |
| 17 | 2<sup>17</sup>  | 2.06 × 10<sup>11</sup>  | 1.61″  |

## Typical healsparse workflow

1. **Create** an empty map:

   ```python
   import healsparse as hsp
   m = hsp.HealSparseMap.make_empty(nside_coverage=32,
                                    nside_sparse=4096,
                                    dtype=bool,
                                    bit_packed=True)  # optional
   ```

2. **Update** pixels with valid data (for booleans, set `True`).
3. **Query** for values, valid pixel lists, or coverage information.

