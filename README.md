# <img src="./docs/croppedSCR.png">
---
## **Cutting pixelized sky masks in a pipeline way**
---
**Skykatana** is a pacakge to create and maniputate boolean spatial masks on the
celestial sphere, by combining [healsparse](https://github.com/LSSTDESC/healsparse) pixel maps
accounting for various effects such as cutting out regions around bright stars, low depth, bad
seeing, extended sources, among others. We call these partial maps **stages**, which are then
combined into a final mask.

For each stage you can generate random points, quickly visualize masks, do plots overlaying
bright stars, and apply the mask to an arbitrary catalog to select sources located inside.

Although mainly designed to work with the [HSC-SSP survey](https://hsc-release.mtk.nao.ac.jp/doc/),
it is flexible to accomodate other surveys such as the upcoming half-sky dataset of the
[Vera Rubin Observatory](https://rubinobservatory.org/).

Main Class
-------------
* ``SkyMaskPipe()``
    Main class for assembling and handling pixelized masks

Main Methods
-------------
* ``build_footprint_mask(), build_patch_mask(), build_holes_mask(), buld_propmap_mask(), etc``
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
* [lsdb](https://github.com/astronomy-commons/lsdb), [healsparse](https://github.com/LSSTDESC/healsparse),
[tqdm](https://github.com/tqdm/tqdm), [healpy](https://github.com/healpy/healpy)

Install
-------
There are two ways to get skykatana:
1. `pip install skykatana`
2. Clone the repo, switch to the pacakge directory and do `pip install .` . This has the advantage that you will
get the latest version and example notebooks.

Example Dataset
---------------
There a small dataset of ~8 million HSC sources to start using the package. Get it [here](https://drive.google.com/file/d/1Fft9E9uD1eXs-8Dxb8bp5ou1bEtCTkgr/view?usp=sharing) 
(170 MB) and decompress it. Then, adjust the folder location in the provided notebooks and just run them. 

Documentation
-------------
* A quick introductory notebook is availables [here](https://github.com/samotracio/skykatana/blob/main/notebooks/quick_example_hsc.ipynb)
* An indepth tutorial notebook can be found [here](https://github.com/samotracio/skykatana/blob/main/notebooks/indepth_usage.ipynb)
* The full documentation is available [here](https://skykatana.readthedocs.io/en/latest/)

Credits
-------
* Main author: [Emilio Donoso](mailto:emiliodon@gmail.com)
* Contributors: [Mariano Dominguez](mailto:mariano.dominguez@unc.edu.ar),
[Claudio Lopez](mailto:yoclaudioantonio1@gmail.com), [Konstantin Malanchev](mailto:hombit@gmail.com)

Acknowledgements
----------------
This software was partially developed with the generous support of the [LINCC Frameworks Incubator Program](https://lsstdiscoveryalliance.org/programs/lincc-frameworks/incubators/) using LINCC resources. The [healsparse](https://github.com/LSSTDESC/healsparse) code was written by Eli Rykoff and Javier Sanchez

