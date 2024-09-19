# <img src="./docs/croppedSCR.png">
---
## **Cutting pixelized sky masks in a pipeline way**
---
**Skykatana** is a pacakge to create and maniputate spatial masks on the
celestial sphere, by combinings healsparse pixel maps accounting for various effects
such as cutting out regions around bright stars, low depth, bad seeing, extended sources, among others.
We call these partial maps **stages**, which are then combined into a final mask.

For each stage you can generate random points, quickly visualize masks, do plots overlaying
bright stars, and apply the mask to an arbitrary catalog to select sources located inside.

Although mainly designed to work with the HSC-SSP survey, it is flexible to accomodate
other surveys such as the upcoming half-sky dataset of the Vera Rubin Observatory.

---
* **Quick intro notebook: /code/quick_example_hsc.ipynb**
* **In-depth notebook: /code/indepth_usage.ipynb**
---

Main Class
-------------
* ``SkyMaskPipe()``
    Main class for assembling and handling pixelized mask

Main Methods
-------------
* ``build_footprint_mask(), build_patch_mask(), build_holes_mask(), etc``
    --> Generate maps for each stage
* ``combine_mask()``
    --> Merge the maps created above to generate a final mask
* ``plot()``
    --> Visualize a mask stage by plotting randoms. Options to zoom, oveplot stars, etc.
* ``plot2compare()``
    --> Compare input sources on the left and a mask stage on the right
* ``makerans()``
    --> Generate randoms over a mask stage
* ``apply()``
    --> Cut out sources outside of a given mask stage

Dependencies
------------
1. numpy, astropy, matplotlib, tqdm
2. healsparse, healpy, mocpy, lsdb

Except for healsparse, most dependencies will be covered by installing hipscat-import and lsbd

Credits
-------
* [Emilio Donoso](mailto:emiliodon@gmail.com), [Mariano Dominguez](mailto:mariano.dominguez@unc.edu.ar),
[Claudio Lopez](mailto:yoclaudioantonio1@gmail.com)
