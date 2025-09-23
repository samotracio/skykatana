---
title: Introduction
icon: material/page-first
---
# SkyKatana : masks in a pipeline way

<span class="sc">SkyKatana</span> is a package to construct and maniputate boolean spatial masks on the celestial sphere covering tens of thousands of square degrees. Starting from (i) a list of *geometric shapes*, (ii) an input *catalog of sources*, or (iii) any *healsparse map*, Skykatana creates pixelized masks in [healsparse](https://github.com/LSSTDESC/healsparse) format with the purpose of excluding areas due to various effects such as bright stars, extended sources, regions of high extinction, bad seeing, etc. 

:material-checkbox-multiple-blank-circle-outline: &nbsp; **We call these masks <span class='emph1'>stages</span>, which are stored in a pipeline object along with metadata.**

:material-checkbox-multiple-blank-circle-outline: &nbsp; **Stages can be logically combined into a "final" mask by performing unions, subtractions and intersections.**

:material-checkbox-multiple-blank-circle-outline: &nbsp; **For each stage you can quickly generate random points, visualize its MOC (Multi-Order Coverage map), overlay catalogs or geometrical shapes, and apply the mask to an arbitrary catalog to restrict to sources located within the mask.**

For creating bright star masks, <span class="sc">SkyKatana</span> has special methods to query and pixelate large online catalogs in HATS/LSDB format (e.g. Gaia), running in systems with limited memory resources. For example, in the Rubin Science Platform (RSP) it can create a completely customized star mask over 5000 deg<sup>2</sup> in just a few minutes. 

Although mainly designed to work with the upcoming half-sky dataset of the Vera Rubin Observatory, it is flexible to accomodate the particular details of other surveys such as the the Nancy Grace Roman Telescope, the Subaru HSC-SSP, and most wide-field surveys.  

