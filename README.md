Skykatana
==================================

**Create pixelized spatial masks sky for large astronomical surveys in a pipeline way**

**Check /code/example_hsc.ipynb for real usage**

This pacakge allows to create and maniputate spatial masks on the celestial sphere
by combinings healsparse pixel maps due to various effects such as bright stars,
regions with low depth, bad seeing, extended sources. We call these partial maps 
"stages", which are combined into a final mask. For each stage you can generate
randomm points, do plots overlaying bright stars and boxes, and apply the map
to cut out sources liying outside.


Main Class
-------------
* ``SkyMaskPipe``
    Main class for assembling and handling pixelized mask

Main Methods
-------------
* ``build_footprint_mask, build_patch_mask, build_holes_mask, etc``
    Generate maps for each stage
* ``combine_mask``
    Merges the maps create above to general a final mask
* ``plot``
    Visualize a mask stage by plotting randoms. Options to zoom, oveplot stars, etc.
* ``plot2compare``
    Compare input sources on the left and a mask stage on the right
* ``makerans``
    Generate randoms over a mask stage
* ``apply``
    Cut out sources outside of a given mask stage

Dependencies
------------
1. numpy, astropy, matplotlib, tqdm
2. healpy, mocpy, lsdb
3. healsparse

Except for healsparse, most dependencies will be covered by installing hipscat-import and lsbd


To Do
-----
- [ ] Something 1
- [ ] Something 2


Credits
-------
