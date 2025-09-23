---
title: HATS & LSDB basics
icon: material/database
---
# HATS & LSDB basics

[HATS](https://github.com/astronomy-commons/hats) (Hierarchical Adaptive Tiling Scheme) and [LSDB](https://github.com/astronomy-commons/lsdb) (Large Survey DataBase) are complementary frameworks for storing and querying **very large astronomical catalogs**—hundreds of millions or even billions of sources—so that spatial operations remain fast and memory‑efficient. Both are built around the HEALPix pixelization of the sky but add crucial layers of **hierarchical, adaptive partitioning** and rich metadata. This section cover a few basic concepts relative to <span class='sc'>SkyKatana</span>. For more detailed infomation please follow their respective documentation.

---

## HATS: Hierarchical Adaptive Tiling Scheme

### Adaptive tiling
* **Goal:** balance file sizes and source counts across the sky.  
* **Mechanism:** the sky is first divided into coarse HEALPix pixels. Any pixel that contains more than a target number of sources is recursively subdivided to higher HEALPix orders until the density per tile falls below the target.  
* **Result:** sparse regions remain at low order (few large files), while crowded regions—e.g. the Galactic plane—are stored in many small, high‑order tiles.  
* **Benefit:** every Parquet partition ends up with roughly uniform row counts, which is essential for parallel processing and predictable I/O.

### Directory layout and metadata
HATS defines a strict directory tree such as:

```
/catalog_name/
  partition_info.csv
  properties/
  dataset/
    Norder=1/
      Dir=0/
        Npix=0.parquet
        Npix=1.parquet
    Norder=J/
      Dir=…/
        Npix=…parquet
```

* **Norder** encodes the HEALPix order (log₂ of NSIDE).  
* **Npix** is the HEALPix pixel index.  
* `partition_info.csv` tracks every partition’s order, pixel index, number of sources, bounding geometry, and optional statistics.  
* The `properties/` directory stores global metadata: column definitions, units, coordinate system, and provenance.

This scheme allows clients to discover and stream just the subset of Parquet files that overlap a requested region, avoiding any full‑catalog scan.

### Margins
Cross‑matching or neighbor searches near partition edges require access to objects just outside the nominal pixel. HATS supports **margin buffers**, storing sources that lie slightly outside each tile’s boundary.  A query on one tile therefore sees all neighbors within a configurable angular distance without loading adjacent partitions.

---

## LSDB: Large Survey DataBase

LSDB builds directly on the HATS layout and provides the computational layer for analysis, querying  and cross‑matching.

### Spatial query engine
* Uses the **hierarchical index** from HATS to find which partitions overlap a cone, polygon, or arbitrary region.  
* Reads only those Parquet files and only the required columns, leveraging the columnar nature of Parquet to minimize I/O.

### Parallel and distributed execution
* Integrates with **Dask** for task scheduling. Each partition is an independent computation unit, so filters, aggregations, and spatial joins run across a cluster or on multi‑core machines with near‑linear scaling.  
* Adaptive partitioning ensures each worker receives roughly equal amounts of data, avoiding “hot spots” that would otherwise slow down parallel jobs.

### Cross‑matching and neighbor searches
* LSDB performs one‑to‑many or many‑to‑many spatial joins by combining HATS’ margin buffers with efficient cone‑search algorithms.  
* Because the partitioning respects HEALPix hierarchy, LSDB can pre‑filter candidate matches by HEALPix pixel before computing exact great‑circle separations, dramatically reducing the number of expensive distance calculations.

---

## Key principles of the spatial organization

| Concept | Purpose | Insight |
|--------|--------|--------|
| **Hierarchical HEALPix indexing** | Provides a mathematically exact, equal‑area tessellation of the sphere that can be refined to any order. | Enables region queries and neighbor searches to be reduced to pixel lookups before fine‑grained calculations. |
| **Adaptive partitioning** | Adjusts HEALPix order per region to keep file sizes and source counts roughly uniform. | Prevents dense regions (e.g. Galactic plane) from dominating runtime and storage while avoiding a huge number of empty tiles in sparse areas. |
| **Margin caching** | Stores sources beyond tile boundaries. | Guarantees completeness of cross‑matches and density estimates near edges without forcing the query to read neighboring partitions. |
| **Columnar Parquet storage** | Data are stored column‑wise inside each partition. | Allows queries to read only the columns actually needed (e.g., `ra`, `dec`), lowering I/O and memory use for common spatial selections. |
| **Metadata‑driven discovery** | Partition tables and catalog properties describe the geometry and schema. | Applications can programmatically discover which files to read and how to interpret them without external bookkeeping. |


