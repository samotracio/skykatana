---
title: Auxiliary routines
icon: material/robot-happy-outline
---
# Auxiliary routines

## Synthetic star data

<span class='sc'>SkyKatana</span> includes a module (aux_mockdata.py) with simple routines to generate mock datasets suitable for pixelization. These can be useful for testing or developing new algorithms. These routines include:

* `make_random_sphere()` > generate random (ra, dec) points uniformly over the shpere
* `generate_powerlaw_radii()` > generate power-law radii between a minimum and a maximum radius
* `make_random_stars()` > use functions above to generate a (ra,dec,radius) mock datasets

Each function has a detailed docstring available, but you will most likely use it as: 

```python
from skykatana.aux_mockdata import make_random_stars
starcat = make_random_stars(100000, ralim=[0,45], declim=[10,30], 
                            min_radius=20., max_radius=200., power_law_index=-2.2)
```

This code will generate a 3-column pandas dataframe containing 100k synthetic stars with random positions in an area bounded by the given ra/dec limits, and radii between 20 and 200 arcsec, following the given power law.

