---
title: API
icon: simple/autoprefixer
---

# API Documentation

## <span class='mgroup'>:material-label-multiple-outline: Mask building & ops</span>
<hr class='sep2'>
::: skykatana.SkyMaskPipe
    options:
      show_docstring_attributes: false
      show_root_toc_entry: false
      show_root_heading: false
      show_docstring_description: true
      members:
        - build_circ_mask
        - build_box_mask
        - build_zone_mask
        - build_ellip_mask
        - build_poly_mask
        - build_prop_mask
        - build_patch_mask
        - build_milkyway_mask
        - build_foot_mask
        - build_star_mask_online
        - apply
        - combine
        - makerans

## <span class='mgroup'>:material-label-multiple-outline: Visualization</span>
<hr class='sep2'>
::: skykatana.SkyMaskPipe
    options:
      show_docstring_attributes: false
      show_root_toc_entry: false
      show_root_heading: false
      show_docstring_description: true
      members:
        - plot
        - plot_srcs
        - plot_moc
        - plot_moca
        - plot_fracmap
        - add_moca
        - moc_from_stage

## <span class='mgroup'>:material-label-multiple-outline: Pixelization</span>
<hr class='sep2'>
::: skykatana.SkyMaskPipe
    options:
      show_docstring_attributes: false
      show_root_toc_entry: false
      show_root_heading: false
      show_docstring_description: true
      members:
        - pixelate_circles
        - pixelate_ellipses
        - pixelate_boxes
        - pixelate_zones
        - pixelate_polys
        - filter_and_pixelate_patches
        
## <span class='mgroup'>:material-label-multiple-outline: Auxiliary</span>
<hr class='sep2'>
::: skykatana.SkyMaskPipe
    options:
      show_docstring_attributes: false
      show_root_toc_entry: false
      show_root_heading: false
      show_docstring_description: true
      members:
        - change_sparse_order
        - change_cov_order
        - intersect_boolmask
        - subtract_boolmask
        - pix_in_zone
        - get_plot_order
        - image_to_healsparse
        - remove_isopixels
        - erode_borders
        - gal_plane_bulge_moc
        - frac_area_map
        - stage_meta

## <span class='mgroup'>:material-label-multiple-outline: IO Methods</span>
<hr class='sep2'>
::: skykatana.SkyMaskPipe
    options:
      show_docstring_attributes: false
      show_root_toc_entry: false
      show_root_heading: false
      show_docstring_description: true
      members:
        - read
        - read_single
        - write

        

