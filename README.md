Skykatana
==================================

**Create pixelized spatial masks in the sky for large astronomical surveys**

For an input catalog of HSC sources, this creates a pixelized mask or map in healsparse format, excluding areas\
due to various reasons such as bright stars, patches with low depth, etc. It also allows to combine this \
mask with arbitrary regions defined by the user.

Check example_hsc.ipynb for real usage!!!


Main Class
-------------
* ``SkyMaskPipe``
    Main class for assembling and handling pixelized mask

Main Methods
-------------
* ``build_footprint_mask, build_patch_mask, build_holes_mask, etc``
    Generate
* ``combine_mask``
    Merges the maps create above to general a final mask
* ``plot``
    Quickly visualize a mask
* ``makerans``
    Generate randoms over a mask
* ``apply``
    Cut out sources outside of a given mask

Example of use
--------------
    # Raw example
    from skykatana import SkyMaskPipe
    skp = SkyMaskPipe()
    mkp.build_patch_mask(patchfile=PATCH_FILE, qafile=QA_FILE)
    mkp.build_holes_mask(star_regs=STARS_REGIONS, box_regs=BOX_STARS_REGIONS)
    mkp.build_extended_mask(ellip_regs=ELLIP_REGIONS, fmt='parquet')
    mkp.combine_mask()


Dependencies
------------
1. numpy, astropy, matplotlib, tqdm
2. healpy, mocpy, lsdb
3. healsparse

Except for healsparse, most dependencies will be install by installing hipscat-import and lsbd


To Do
-----
- [ ] Something

Credits
-------
