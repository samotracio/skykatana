import numpy as np
import healsparse as hsp
import healpy as hp
import matplotlib.pyplot as plt
import lsdb
from astropy.table import Table, join
from astropy.coordinates import Angle, Latitude, Longitude, SkyCoord
from mocpy import MOC
from tqdm import tqdm
import re
import pickle



class SkyMaskPipe:
    """

    A class to work with healsparse sky masks in a pipeline way

    Attributes
    ----------
    order_cov : int
        Coverage order of (all) healsparse maps.
    order_foot : int
        Order for the footprint map
    order_patch : int
        Order for the patch map
    order_prop : int
        Order for the property map
    order_holes : int
        Order for the holes map due to bright stars/boxes
    order_extended : int
        Order for the extended sources map
    order_user : int
        Order for user defined map
    """

    def __init__(self, **kwargs):
        defaults = {
            'order_cov':    4,
            'order_foot':   13,
            'order_patch':  13,
            'order_prop':   15,
            'order_holes':  15,
            'order_user':   15,
            'order_extended': 15,
            'order_out':    15
        }

        for (prop, val) in defaults.items():
            setattr(self, prop, kwargs.get(prop, val))

        self.nside_out   = 2**self.order_out
        self.nside_cov   = 2**self.order_cov
        self.foot        = None
        self.patchmap    = None
        self.propmap     = None
        self.holemap     = None
        self.usermap     = None
        self.extendedmap = None
        self.mask        = None
        self.hatscat     = None
        self.qafile      = None
        self.patchfile   = None
        self.star_regs   = None
        self.box_regs    = None
        self.circ_uregs  = None
        self.poly_uregs  = None
        self.ellip_regs  = None


    def __str__(self):
        # ================================================================
        # Create a summary of a SkyMaskPipe object that works with print()
        #================================================================
        import healpy as hp

        def summarize_stage(name, hsp_map):
            if hsp_map is None:
                return f"{name:<15}: Not defined"
            else:
                nside_cov = hsp_map.nside_coverage
                nside_sparse = hsp_map.nside_sparse
                npix = hsp_map.n_valid
                area = hsp_map.get_valid_area(degrees=True)
                pix_area_deg2 = hp.nside2pixarea(nside_sparse, degrees=True)
                pix_size_arcsec = (pix_area_deg2 ** 0.5) * 3600  # deg2arcsec
                return (f"{name:<15}: nside_cov={nside_cov:<5}  "
                        f"nside_sparse={nside_sparse:<5}  "
                        f"valid_pix={npix:<7}  area={area:6.2f} deg²  "
                        f"pix_size={pix_size_arcsec:6.1f}\"")

        lines = [
            summarize_stage("foot", self.foot),
            summarize_stage("patchmap", self.patchmap),
            summarize_stage("propmap", self.propmap),
            summarize_stage("holemap", self.holemap),
            summarize_stage("extendedmap", self.extendedmap),
            summarize_stage("usermap", self.usermap),
            summarize_stage("mask", self.mask)
        ]
        return "\n".join(lines)

    # #########################
    # Uncomment this if you want to display the summary just by typing the name of the pipeline object
    #__repr__ = __str__
    ###########################


    @staticmethod
    def readQApatches(qafile):
        """
        Read contents of HSC QA patch list.

        See https://hsc-release.mtk.nao.ac.jp/schema/#pdr3.pdr3_wide.patch_qa

        Parameters
        ----------
        qafile : string
            File path

        Returns
        -------
        astropy table
            Table with QA patches
        """
        pqa = Table.read(qafile)
        return pqa



    @staticmethod
    def parse_condition(condition_str):
        """
        Parses a condition string, converting column names into a format usable for direct evaluation over an astropy table.
        For example, converts "(ra>20) and (dec<30)" into "(table['ra']>20) & (table['dec']<30)".

        Parameters
        ----------
        condition_str : str
            Condition string, e.g., "(gmag>20.4) and (imag>19.8)"

        Returns
        -------
        str
            Condition in new format
        """
        # Regex to match alphanumeric + underscores after a '(' and before a comparison operator (<, >, =, !=)
        pattern = r'\(([\w_]+)(?=[<>!=])'

        # Substitute the matched column names with table['column_name']
        condition_str = re.sub(pattern, r"(table['\1']", condition_str)

        # Use regex to replace 'and' and 'or' as standalone words, not as substrings within column names
        condition_str = re.sub(r'\band\b', '&', condition_str)
        condition_str = re.sub(r'\bor\b', '|', condition_str)

        return condition_str


    @staticmethod
    def filter_and_pixelate_patches(file, qatable, filt=None, order=13):
        """
        Reads a file with HSC patches, matches it against the QA table containing quality measures
        of each patch, filter those that meet some depth/seeing/etc critera, and return their
        pixels at a given order. Patches are pixelized as quadrangle polygons.

        Modify the patch aceptance criteria to suit your purposes, e.g. accepting only those above
        a minimum depth threshold in a given band, etc.

        Parameters
        ----------
        file : str
            HSC patch file (defult parquet, but anything astropy can read)
        qatable : astropy table
            Table of patches with QA measurements
        order : int
            Pixelization order
        filt : str
            Contition(s) to apply to patches, e.g filt='(ra>20) and (dec<10)'. If None,
            all patches will be considered

        Returns
        -------
        ndarray
            Array of pixels inside patches fulfilling the patch selection criteria
        """

        # Read patch file
        print('--- Processing', str(file))
        print('    Order ::',order)
        table = Table.read(file)
        # Construct skymap_id to merge the patch table and the QA table
        table['skymap_id'] = table['Tract']*10000 + table['Patch1']*100 + table['Patch2']
        table = join(table, qatable, keys='skymap_id')
        print('    Patches with QA                       :', len(table))

        if filt:
            # Apply uniformity criteria to patches based on minimum depth. This gets rids of border zones.
            filt_goodstr = __class__.parse_condition(filt)
            idx = eval(filt_goodstr)
            print('    Patches with QA fulfilling conditions :', len(table[idx]))
        else:
            idx = np.full(len(table), fill_value=True, dtype=np.bool_)
            print('    Patches adopted (no filters applied) :', len(table[idx]))

        # Change to float64 because moc complains otherise
        table['ra0']=table['ra0'].astype(np.float64)
        table['ra1']=table['ra1'].astype(np.float64)
        table['ra2']=table['ra2'].astype(np.float64)
        table['ra3']=table['ra3'].astype(np.float64)
        table['dec0']=table['dec0'].astype(np.float64)
        table['dec1']=table['dec1'].astype(np.float64)
        table['dec2']=table['dec2'].astype(np.float64)
        table['dec3']=table['dec3'].astype(np.float64)

        # Create list of skycoords for each patch
        sks = [ SkyCoord([p['ra0'], p['ra1'], p['ra2'], p['ra3']], [p['dec0'], p['dec1'], p['dec2'], p['dec3']], unit='deg') for p in table[idx]]
        # Generate moc from the polygon of the four vertices of each patch.
        moc_ptchs = MOC.from_polygons(sks, max_depth=order)
        # Return flat and unique pixels at max_depth
        hp_index = np.concatenate([p.flatten() for p in moc_ptchs])
        hp_index = np.unique(hp_index).astype(np.int64)
        print('    Surviving patch pixels                :', hp_index.shape[0])

        return hp_index


    @staticmethod
    def remove_isopixels(hsmap):
        """
        Remove empty isolated pixels (i.e. one False pixel surrounded by 8 True pixels),
        by settiing it them True. This can help for example when pixelating sources
        with just the right order so that a few artificial empty pixels appear

        Parameters
        ----------
        hsmap
            Healsparse boolean map

        Returns
        -------
        hsmap
            Healsparse boolean map
        """
        print('    ...removing isolated pixels...')
        from collections import Counter
        active_pixels = hsmap.valid_pixels
        neighbors_deact = []
        for pix in tqdm(active_pixels):
            neighbors = hp.get_all_neighbours(hsmap.nside_sparse, pix, nest=True)
            neighbors_deact.append(neighbors[~hsmap[neighbors]])

        counts = Counter(np.hstack(neighbors_deact))
        val8times = [key for key, count in counts.items() if count == 8]
        val8times = np.hstack(val8times)
        hsmap.update_values_pix(val8times, np.full_like(val8times, True, dtype=np.bool_), operation='replace')
        return hsmap


    @staticmethod
    def erode_borders(hsmap):
        """
        Remove the borders of holes in the mask, i.e. detect the pixels that delineate
        zones set to False (completely surrounded by pixels set to True) as well as
        the external border of regions, and set those border pixels off. This can
        help to remove jagged boundaries around empty regions, when pixelated
        at relatively coarse resolutions.

        Parameters
        ----------
        hsmap
            Healsparse boolean map

        Returns
        -------
        hsmap
            Healsparse boolean map
        """
        print('    ...eroding borders...')
        nborders = 8
        active_pixels = hsmap.valid_pixels
        active_pixel_set = set(active_pixels)
        filtered_pixels = []
        for pix in tqdm(active_pixels):
            neighbors = hp.get_all_neighbours(hsmap.nside_sparse, pix, nest=True)
            active_neighbors_count = 0
            for neighbor in neighbors:
                if neighbor in active_pixel_set:
                    active_neighbors_count += 1
            if active_neighbors_count < nborders:
                filtered_pixels.append(pix)

        filtered_pixels = np.array(filtered_pixels)
        hsmap.update_values_pix(filtered_pixels, np.full_like(filtered_pixels, False, dtype=np.bool_), operation='replace')
        return hsmap


    @staticmethod
    def pixelate_circles(file, fmt='ascii', columns=['ra', 'dec', 'radius'], order=15):
        """
        Read circular regions around bright stars, pixelize them and return the (unique) pixels inside.
        Coordinates and distances should be in degrees.

        Parameters
        ----------
        file : str
            Path to file
        fmt : str
            Format of file, e.g. 'ascii', 'parquet', or any accepted by astropy.table
        columns : list of str
            Colums for ra, dec, radius of circles
        order : int
            Pixelization order

        Returns
        -------
        ndarray
            Array of pixels
        """
        colra, coldec, colrad = columns

        MOCPY_THREADS = None     # number of threads to use with mocpy (none=all)
        MOCPY_DEPTH_DELTA = 3    # precision of healpix filtering when applying from_cones()
        print('--- Pixelating circles from', file)
        print('    Order ::',order)
        table = Table.read(file, format=fmt)
        mocs = MOC.from_cones(
            lon=Longitude(table[colra], unit='deg'), lat=Latitude(table[coldec], unit='deg'), radius=Angle(table[colrad], unit='deg'),
            max_depth=order, delta_depth=MOCPY_DEPTH_DELTA, n_threads=MOCPY_THREADS)

        hp_index = np.concatenate([moc.flatten() for moc in mocs])
        print('    done')
        return np.unique(hp_index).astype(np.int64)


    @staticmethod
    def pixelate_ellipses(file, fmt='ascii', columns=['ra','dec','a','b','pa'], order=15):
        colra, coldec, cola, colb, colpa = columns
        """
        Read elliptical regions around extended sources, pixelize them and return the (unique)
        pixels inside. Coordinates and distances should be in degrees.

        Parameters
        ----------
        file : str
            Path to file
        fmt : str
            Format of file, e.g. 'ascii', 'parquet', or any accepted by astropy.table
        columns : list of str
            Columns for ra, dec, a, b and pa (position angle) of ellipses
        order : int
            Pixelization order

        Returns
        -------
        ndarray
            Array of pixels
        """
        MOCPY_DEPTH_DELTA = 3    # precision of healpix filtering when applying from_cones()
        print('--- Pixelating ellipses from', file)
        print('    Order ::',order)
        table = Table.read(file, format=fmt)

        mocs = []
        for ra, dec, a_axis, b_axis, pa_angle in zip(table[colra], table[coldec], table[cola], table[colb], table[colpa]):
            moc = MOC.from_elliptical_cone(
            lon=Longitude(ra, unit='deg'), lat=Latitude(dec, unit='deg'), a=Angle(a_axis, unit='deg'), b=Angle(b_axis, unit='deg'),
            pa=Angle(pa_angle, unit='deg'), max_depth=order, delta_depth=MOCPY_DEPTH_DELTA )
            pixels = moc.flatten().astype(np.int64)
            mocs.append(pixels)

        hp_index = np.hstack(mocs)
        print('    done')
        return np.unique(hp_index)


    @staticmethod
    def pixelate_boxes(file, fmt='ascii', columns=['ra_c','dec_c','width','height'], order=15):
        """
        Read box regions around bright stars, pixelize them and return the (unique) pixels inside.
        Coordinates and distances should be in degrees.

        Parameters
        ----------
        file : str
            Path to file
        fmt : str
            Format of file, e.g. 'ascii', 'parquet', or any accepted by astropy.table
        columns : list of str
            Columns for ra_center, dec_center, width and height of boxes
        order : int
            Pixelization order

        Returns
        -------
        ndarray
            Array of pixels
        """
        colra, coldec, colw, colh = columns

        MOCPY_THREADS = None     # number of threads to use with mocpy (none=all)
        print('--- Pixelating boxes from', file)
        print('    Order ::',order)
        table = Table.read(file, format=fmt)
        ra_center = Longitude(table[colra], unit='deg')
        dec_center = Latitude(table[coldec], unit='deg')
        width = Angle(table[colw], unit='deg')
        height = Angle(table[colh], unit='deg')

        width_larger = width > height
        a = np.where(width_larger, 0.5 * width, 0.5 * height)
        b = np.where(width_larger, 0.5 * height, 0.5 * width)
        angle = np.where(width_larger, Angle(90, 'deg'), 0)

        # Boxes seem to strech at high declination. For now, multiply by cos(dec) #######
        a = a*np.cos(table[coldec].value*np.pi/180.)

        mocs = MOC.from_boxes(
            lon=ra_center, lat=dec_center, a=a, b=b, angle=angle,
            max_depth=order, n_threads=MOCPY_THREADS)

        hp_index = np.concatenate([moc.flatten() for moc in mocs])
        print('    done')
        return np.unique(hp_index).astype(np.int64)



    @staticmethod
    def pixelate_polys(file, fmt='ascii', columns=['ra0','ra1','ra2','ra3','dec0','dec1','dec2','dec3'], order=15):
        """
        Read quadrangular polygons, pixelize them and return the (unique) pixels inside.
        Input data must have 8 columns for the coordinates of the 4 vertices.
        Coordinates and distances should be in degrees.

        Parameters
        ----------
        file : str
            Path to file
        fmt : str
            Format of file, e.g. 'ascii', 'parquet', or any accepted by astropy.table
        columns : list of str
            Columns for ra, dec for each of the four vertexs
        order : int
            Pixelization order

        Returns
        -------
        ndarray
            Array of pixels
        """
        cr0, cr1, cr2, cr3, cd0, cd1, cd2, cd3 = columns

        # Read poly file
        print('--- Pixelating polygons from', file)
        print('    Order ::',order)
        table = Table.read(file, format=fmt)

        # Change to float64 because otherwise moc complains
        table[cr0]=table[cr0].astype(np.float64)
        table[cr1]=table[cr1].astype(np.float64)
        table[cr2]=table[cr2].astype(np.float64)
        table[cr3]=table[cr3].astype(np.float64)
        table[cd0]=table[cd0].astype(np.float64)
        table[cd1]=table[cd1].astype(np.float64)
        table[cd2]=table[cd2].astype(np.float64)
        table[cd3]=table[cd3].astype(np.float64)

        # Create list of skycoords for each poly
        sks = [ SkyCoord([p[cr0], p[cr1], p[cr2], p[cr3]], [p[cd0], p[cd1], p[cd2], p[cd3]], unit='deg') for p in table]
        # Generate moc from the polygon of the four vertices of each patch.
        moc_plys = MOC.from_polygons(sks, max_depth=order)
        # Return flat and unique pixels at max_depth
        hp_index = np.concatenate([p.flatten() for p in moc_plys])
        hp_index = np.unique(hp_index).astype(np.int64)

        return hp_index



    def build_holes_mask(self, star_regs=None, box_regs=None, fmt='parquet', order_holes=None,
                         columns_circ=['ra','dec','radius'],
                         columns_box=['ra_c','dec_c','width','height']):
        """
        Create a holes map corresponding to (circular and/or rectangular regions) around bright stars.
        Note by default this map's value is set to True, as combine_mask() use it as negative while
        combining with other maps

        Parameters
        ----------
        star_regs : str
            Path to file (circular astropy-regions)
        box_regs : bool
            Path to file (box astropy-regions)
        fmt : str
            Format of file, e.g. 'ascii', 'parquet', or any accepted by astropy.table
        order_holes : int
            Pixelization order
        columns_circ : list of str
            Colums for ra, dec, radius of circles
        columns_box : list of str
            Colums for ra_center, dec_center, width, height of boxes

        Returns
        -------
        hsp_map
            Healsparse boolean map
        """
        print('BUILDING BRIGHT STAR HOLES MAP >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

        if order_holes: self.order_holes=order_holes

        if star_regs:
            self.star_regs=star_regs
            self.star_regs_fmt = fmt
            self.star_regs_columns = columns_circ
            star_pixels = self.pixelate_circles(self.star_regs, fmt=fmt, order=self.order_holes, columns=columns_circ)

        if box_regs:
            self.box_regs=box_regs
            self.box_regs_fmt = fmt
            self.box_regs_columns = columns_box
            box_pixels = self.pixelate_boxes(self.box_regs, fmt=fmt, order=self.order_holes, columns=columns_box)

        # Empty boolean map
        nside_holes = 2**self.order_holes
        self.holemap = hsp.HealSparseMap.make_empty(self.nside_cov, nside_holes, dtype=np.bool_)
        # Set the holes to True to invert them later
        if star_regs: self.holemap[star_pixels] = True
        if box_regs: self.holemap[box_pixels] = True

        print('--- Holes map area                        :', self.holemap.get_valid_area(degrees=True))



    def build_user_mask(self, circ_uregs=None, poly_uregs=None, fmt='ascii', order_user=None,
                        columns_ucirc=['ra','dec','radius'],
                        columns_upoly=['ra0','ra1','ra2','ra3','dec0','dec1','dec2','dec3'],
                        usermap_type = 'positive'):
        """
        Builds a user defined mask from a list of circular and/or quadrangular polygons.

        Parameters
        ----------
        circ_uregs : str
            Path to file (circular astropy-regions)
        poly_uregs : str
            Path to file (polygon astropy-regions)
        fmt : str
            Format of file, e.g. 'ascii', 'parquet', or any accepted by astropy.table
        order_user : int
            Pixelization order
        columns_ucirc : list of str
            Columns for ra, dec, radius
        columns_upoly : list of str
            Columns for ra, dec for each of the four vertexs
        usermap_type : str
            If 'positive'('negative'), combine_mask() will perform an intersection(subtraction)

        Returns
        -------
        hsp_map
            Healsparse boolean map
        """

        if circ_uregs: self.circ_uregs=circ_uregs
        if poly_uregs: self.poly_uregs=poly_uregs
        if order_user: self.order_user=order_user
        print('BUILDING USER DEFINED MAP >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

        # Empty boolean map
        nside_user = 2**self.order_user
        self.usermap = hsp.HealSparseMap.make_empty(self.nside_cov, nside_user, dtype=np.bool_)

        if self.circ_uregs:
            circle_pixels = self.pixelate_circles(self.circ_uregs, fmt=fmt, order=self.order_user, columns=columns_ucirc)
            self.usermap[circle_pixels] = True

        if self.poly_uregs:
            poly_pixels = self.pixelate_polys(self.poly_uregs, fmt=fmt, order=self.order_user, columns=columns_upoly)
            self.usermap[poly_pixels] = True

        if usermap_type in ['positive','negative']:
            self.usermap_type = usermap_type
        else:
            raise Exception('usermap_type should be "postive" or "negative"')

        print('--- User map area                         :', self.usermap.get_valid_area(degrees=True))



    def build_extended_mask(self, ellip_regs=None, fmt='ascii', order_extended=None,
                            columns_ellip=['ra','dec','a','b','pa']):
        """
        Create a holes map corresponding to (elliptical) extended sources.
        Note by default this map's value is set to True, as combine_mask() use it as negative while
        combining with other maps

        Parameters
        ----------
        ellip_regs : str
            Path to file (elliptical astropy-regions)
        fmt : str
            Format of file, e.g. 'ascii', 'parquet', or any accepted by astropy.table
        order_extended : int
            Pixelization order
        columns_ellip : list of str
            Columns for ra, dec, a, b and p.a. for each ellipse

        Returns
        -------
        hsp_map
            Healsparse boolean map
        """

        if ellip_regs: self.ellip_regs=ellip_regs
        if order_extended: self.order_extended=order_extended
        print('BUILDING EXTENDED MAP >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

        # Empty boolean map
        nside_extended = 2**self.order_extended
        self.extendedmap = hsp.HealSparseMap.make_empty(self.nside_cov, nside_extended, dtype=np.bool_)

        if self.ellip_regs:
            ellip_pixels = self.pixelate_ellipses(self.ellip_regs, fmt=fmt, order=self.order_user, columns=columns_ellip)
            self.extendedmap[ellip_pixels] = True

        print('--- Extended map area                     :', self.extendedmap.get_valid_area(degrees=True))


    @staticmethod
    def reproject_nside_coverage(hspmap, newcov):
        """
        Change the nside_coverage of a healsparse map. Useful to bring boolean maps to a common coverage,
        allowing logic combinations between them

        Parameters
        ----------
        hsmap
            Healsparse boolean map

        Returns
        -------
        hsmap
            Healsparse boolean map
        """
        oldcov = hspmap.nside_coverage
        # Get all sparse pixels with valid data
        ipix = hspmap.valid_pixels
        values = hspmap.get_values_pix(ipix)

        # Create new map with same nside_sparse, but new coverage resolution
        new_map = hsp.HealSparseMap.make_empty(
            nside_coverage=newcov,
            nside_sparse=hspmap.nside_sparse,
            dtype=hspmap.dtype
        )

        # Insert data into the new map
        new_map.update_values_pix(ipix, values)

        print(f'    Warning: nside_coverage changed from {oldcov} to {newcov}')
        return new_map


    def build_propmap_mask(self, prop_maps, thresholds, comparisons):
        """
        Build a HealSparse boolean mask based on multiple property map thresholds

        Parameters
        ----------
        prop_maps : list of HealSparseMap or list of str
            One or more HealSparse maps to threshold, either as objects or file paths
        thresholds : float or list of float
            Threshold value(s) for each property map
        comparisons : str or list of {'gt', 'lt', 'ge', 'le'}
            Comparison operator(s) for each threshold

        Returns
        -------
        mask_map : HealSparseMap
            Healsparse boolean map
        """

        import os
        from pathlib import Path

        print('BUILDING PROPERTY MAP >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

        # Normalize inputs to lists
        if not isinstance(prop_maps, (list, tuple)):
            prop_maps = [prop_maps]
        if not isinstance(thresholds, (list, tuple)):
            thresholds = [thresholds]
        if not isinstance(comparisons, (list, tuple)):
            comparisons = [comparisons]

        if not (len(prop_maps) == len(thresholds) == len(comparisons)):
            raise ValueError("prop_maps, thresholds, and comparisons must be of the same length")

        # Convert file paths to HealSparseMap objects
        resolved_maps = []
        for pm in prop_maps:
            if isinstance(pm, (str, os.PathLike)):               # if isinstance(pm, str):
                print('--- Processing', str(pm))
                resolved_maps.append(hsp.HealSparseMap.read(str(pm)))    #resolved_maps.append(hsp.HealSparseMap.read(pm))
            elif isinstance(pm, hsp.HealSparseMap):
                resolved_maps.append(pm)
            else:
                raise TypeError(f"Each prop_map must be a str (path) or a HealSparseMap instance, not {type(pm)}")

        prop_maps = resolved_maps

        # Verify that all maps have the same resolution
        cov_res_set = {pm.nside_coverage for pm in prop_maps}
        sparse_res_set = {pm.nside_sparse for pm in prop_maps}

        if len(cov_res_set) != 1 or len(sparse_res_set) != 1:
            raise ValueError("All input property maps must have the same nside_coverage and nside_sparse")


        ops = {
            'gt': lambda v, t: v > t,
            'ge': lambda v, t: v >= t,
            'lt': lambda v, t: v < t,
            'le': lambda v, t: v <= t,
        }

        # Start with the pixel set from the first map
        pixels = prop_maps[0].valid_pixels.copy()

        for i, (prop_map, threshold, comparison) in enumerate(zip(prop_maps, thresholds, comparisons)):
            if comparison not in ops:
                raise ValueError(f"Invalid comparison: {comparison}")

            this_pixels = prop_map.valid_pixels
            this_values = prop_map.get_values_pix(this_pixels)

            # Apply comparison and filter
            selected = this_pixels[ops[comparison](this_values, threshold)]

            # Intersect with current valid pixel set
            pixels = np.intersect1d(pixels, selected, assume_unique=True)

            if len(pixels) == 0:
                raise ValueError(f"0 pixels remaining after condition {i} ({comparison} {threshold})")

        # Build final mask prop_maps[0].nside_coverage
        #self.propmap = hsp.HealSparseMap.make_empty(self.nside_cov, prop_maps[0].nside_sparse, dtype=np.bool_)
        self.propmap = hsp.HealSparseMap.make_empty(prop_maps[0].nside_coverage, prop_maps[0].nside_sparse, dtype=np.bool_)
        self.propmap[pixels] = True
        self.order_prop = prop_maps[0].nside_sparse

        # Change nside_coverage if needed
        if self.nside_cov != prop_maps[0].nside_coverage:
            self.propmap = self.reproject_nside_coverage(self.propmap, self.nside_cov)

        print('--- Property map area                     :', self.propmap.get_valid_area(degrees=True))



    def build_patch_mask(self, patchfile=None, qafile=None, order_patch=None,
                         filt="(gmag_psf_depth>26.2) and (rmag_psf_depth>25.9) and (imag_psf_depth>25.7)"):
        """
        For a series of HSC patches, matches them against the QA table containing
        quality measurements, filter those that meet some depth/seeing/etc critera,
        and returns a pixelated map of all accepted patches.

        Parameters
        ----------
        patchfile : list of str
            HSC patch files (e.g. for hectomap, sping, autumn, aegis)
        qafile : str
            File with the table of patches with QA measurements
        order_patch : int
            Pixelization order
        filt : string
            Contition(s) to apply to patches. If None, all patches will be considered

        Returns
        -------
        hsp_map
            Healsparse boolean map

        The filt keyword
        ----------------
        The filt keyword can be a string to filter which patches will be pixelized later. For example:
        filt='(imag_psf_depth>26) and (rmag_psf_depth>26.1)'. If filt=None, no filtering will be applied
        and all patches will be used.
        """

        if patchfile: self.patchfile=patchfile
        if qafile: self.qafile=qafile
        if order_patch: self.order_patch=order_patch

        print('BUILDING PATCH MAP >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # Read patch qa list
        qatable = self.readQApatches(self.qafile)

        # Create empty map
        nside_patch  = 2**self.order_patch
        self.patchmap = hsp.HealSparseMap.make_empty(self.nside_cov, nside_patch, dtype=np.bool_)

        for p in self.patchfile:
            fpx = self.filter_and_pixelate_patches(p, qatable, order=self.order_patch, filt=filt)
            self.patchmap[fpx] = True

        print('--- Patch map area                        :', self.patchmap.get_valid_area(degrees=True))



    def build_footprint_mask(self, hatscat=None, order_foot=None, columns=['ra','dec'],
                             remove_isopixels=False, erode_borders=False):
        """
        Create a footprint map of a source catalog, pixelated at a given order. Optionally remove isolated
        empty pixels and erode borders around empty zones. For details see remove_isopixels() and erode_borders()

        Parameters
        ----------
        hatscat : str
            Path to catalog (HATS directory)
        order_foot : int
            Pixelization order
        remove_isopixels : bool
            Remove isolated (empty) pixels surrounded by 8 non-empty pixels
        erode_borders : bool
            Detect and remove border pixels around holes
        columns : list of str
            Columns for ra, dec

        Returns
        -------
        hsp_map
            Healsparse boolean map
        """

        if hatscat: self.hatscat=hatscat
        if columns: self.hatscat_columns=columns
        if order_foot: self.order_foot=order_foot
        colra, coldec = columns
        print('BUILDING FOOTPRINT MAP >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

        # Create empty healsparse empty
        nside_foot   = 2**self.order_foot
        self.foot = hsp.HealSparseMap.make_empty(self.nside_cov, nside_foot, dtype=np.bool_)

        # Read (ra,dec) from a HATS catalog
        print('--- Pixelating HATS catalog from:', self.hatscat)
        print('    Order ::',self.order_foot)
        srcs = lsdb.read_hats(self.hatscat, columns=columns).compute()

        # Get pixel number for each object
        pixels = hp.ang2pix(nside_foot, srcs[colra], srcs[coldec], nest=True, lonlat=True)

        # Update map values for pixels that have objects
        self.foot.update_values_pix(pixels, np.full_like(pixels, True, dtype=np.bool_), operation='or')

        # Remove isolated empty pixels and borders aound holes, if requested
        if remove_isopixels:
            self.foot = self.remove_isopixels(self.foot)

        if erode_borders:
            self.foot = self.erode_borders(self.foot)

        print('--- Footprint map area                    :', self.foot.get_valid_area(degrees=True))


    def combine_mask(self, apply_patchmap=True, apply_propmap=False, apply_holemap=True,
                     apply_extendedmap=True, apply_usermap=False):
        """
        Combine a footprint map with **4 (optional) masks**:

        1) A patch map containing valid patches

        2) A property map containing valid pixels meeting a given threshold of a certain property

        3) A holes map due to bright stars/boxes

        4) A holes map due to extended sources

        5) A user defined map of arbitrary regions

        Parameters
        ----------
        apply_patchmap : bool
            Apply patch map of accepted patches
        apply_propmap : bool
            Apply property map of accepted pixels
        apply_holemap : bool
            Apply holes map due to bright stars and boxes
        apply_extendedmap : bool
            Apply holes map due to extended sources
        apply_usermap : bool
            Apply map of user defined regions

        Returns
        -------
        hsp_map
            Healsparse boolean map
        """

        print('COMBINING MAPS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

        # Upgrade maps if needed up to the desired resolution
        otmp = int(np.log2(self.foot.nside_sparse))
        if otmp < self.order_out:
            print('--- footprint order upgraded to:', self.order_out)
            self.foot = self.foot.upgrade(self.nside_out)
            self.order_foot = self.order_out

        if apply_patchmap and self.patchmap:
            otmp = int(np.log2(self.patchmap.nside_sparse))
            if otmp < self.order_out:
                print('--- patchmap order upgraded to: ', self.order_out)
                self.patchmap = self.patchmap.upgrade(self.nside_out)
                self.order_patch = self.order_out

        if apply_holemap and self.holemap:
            otmp = int(np.log2(self.holemap.nside_sparse))
            if otmp < self.order_out:
                print('--- holemap order upgraded to: ', self.order_out)
                self.holemap = self.holemap.upgrade(self.nside_out)
                self.order_holes = self.order_out

        if apply_propmap and self.propmap:
            otmp = int(np.log2(self.propmap.nside_sparse))
            if otmp < self.order_out:
                print('--- propmap order upgraded to: ', self.order_out)
                self.propmap = self.propmap.upgrade(self.nside_out)
                self.order_prop = self.order_out

        if apply_usermap and self.usermap:
            otmp = int(np.log2(self.usermap.nside_sparse))
            if otmp < self.order_out:
                print('--- usermap order upgraded to: ', self.order_out)
                self.usermap = self.usermap.upgrade(self.nside_out)
                self.order_user = self.order_out

        if apply_extendedmap and self.extendedmap:
            otmp = int(np.log2(self.extendedmap.nside_sparse))
            if otmp < self.order_out:
                print('--- extendedmap order upgraded to: ', self.order_out)
                self.extendedmap = self.extendedmap.upgrade(self.nside_out)
                self.order_extended = self.order_out

        # Create empty map to contain the final mask and perform combination
        self.mask = hsp.HealSparseMap.make_empty(self.nside_cov, self.nside_out, dtype=np.bool_)

        # Start from footprint map
        self.mask |= self.foot
        # Should we consider an anternative flow with no footmap, and starting from
        # the usermap?

        if apply_patchmap:
            if self.patchmap:
                self.mask = self.intersect_boolmask(self.patchmap, self.mask)
            else:
                raise Exception('patchmap not defined')

        if apply_propmap:
            if self.propmap:
                self.mask = self.intersect_boolmask(self.propmap, self.mask)
            else:
                raise Exception('propmap not defined')

        if apply_holemap:
            if self.holemap:
                self.mask = self.mask & (~self.holemap)
            else:
                raise Exception('holemap not defined')

        if apply_extendedmap:
            if self.extendedmap:
                self.mask = self.mask & (~self.extendedmap)
            else:
                raise Exception('extendedmap not defined')

        if apply_usermap:
            if self.usermap:
                if self.usermap_type == 'positive':
                    self.mask = self.intersect_boolmask(self.usermap, self.mask)
                elif self.usermap_type == 'negative':
                    self.mask = self.mask & (~self.usermap)
            else:
                raise Exception('usermap not defined')

        #if apply_usermap:
        #    if self.usermap:
        #        self.mask = self.intersect_boolmask(self.usermap, self.mask)
        #    else:
        #        raise Exception('usermap not defined')

        print('--- Combined map area                     :', self.mask.get_valid_area(degrees=True))



    @staticmethod
    def intersect_boolmask(mask1, mask2):
        """
        Intersect two arbitrary boolean masks in healsparse format.

        Parameters
        ----------
        mask1 : hsp_map
            Healsparse boolean map 1
        mask2 : hsp_map
            Healsparse boolean map 2

        Returns
        -------
        hsp_map
            Healsparse boolean map
        """

        if mask1.nside_sparse != mask2.nside_sparse:
            raise Exception('Maps have different nside_sparse')

        tmp = mask1 & mask2
        msk = mask2 & tmp
        return msk


    def plot(self, stage='mask', nr=50_000, s=0.5, figsize=[12,6], xwin=None, ywin=None,
             plot_stars=False, plot_boxes=False, use_srcs=False, ax=None, **kwargs):
        """
        Visualize a mask by means of its randoms points

        Parameters
        ----------
        stage : string
            Masking stage to use, e.g. 'mask', 'foot', 'holemap', etc.
        nr : integer
            Number of randoms
        s : float
            Point size
        figsize : list of floats
            Figure size
        xwin : list of floats
            Plot limits in ra, e.g. xwin=[226.5,227.5]
        ywin : list of floats
            Plot limits in dec, e.g. ywin=[10.,11.]
        plot_stars : bool
            Overlay circles due to bright stars
        plot_boxes : bool
            Overlay boxes due to bright stars
        use_srcs : bool
            Plot input sources used to build the footprint map, instead of random points. Note this means that no mask of any kind is actually plotted
        ax : axes
            If given, plot will be added to the axes object provided
        kwargs : [key=val]
            Adittional keyword arguments passed to mataplolib.scatter()
        """
        if stage == 'foot':        mk = self.foot
        if stage == 'patchmap':    mk = self.patchmap
        if stage == 'propmap':     mk = self.propmap
        if stage == 'holemap':     mk = self.holemap
        if stage == 'extendedmap': mk = self.extendedmap
        if stage == 'usermap':     mk = self.usermap
        if stage == 'mask':        mk = self.mask

        if use_srcs:
            # Use catalog for scatter plot
            srcs = lsdb.read_hats(self.hatscat, columns=self.hatscat_columns).compute()
            xx, yy = srcs[self.hatscat_columns[0]], srcs[self.hatscat_columns[1]]
            stage = 'sources'
        else:
             # Use randoms for scatter plot
             xx, yy = hsp.make_uniform_randoms_fast(mk, nr)

        # Do plot ------------------------------------------
        if not(ax): fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(xx, yy, s=s, **kwargs)
        ax.set_title(stage)
        if xwin: ax.set_xlim(xwin)
        if ywin: ax.set_ylim(ywin)
        xwin=ax.get_xlim()  ;  ywin=ax.get_ylim()

        if plot_stars:
            stars = Table.read(self.star_regs, format=self.star_regs_fmt)
            colra, coldec, colrad = self.star_regs_columns
            idx = (stars[colra]>xwin[0]) & (stars[colra]<xwin[1]) & (stars[coldec]>ywin[0]) & (stars[coldec]<ywin[1])
            ts = stars[idx]
            for i in range(len(ts)):
                star_ra, star_dec, star_rad = ts[colra][i], ts[coldec][i], ts[colrad][i]
                circ = plt.Circle((star_ra, star_dec), star_rad, color='r', fill=False, linewidth=0.4)
                ax.add_artist(circ)
                #print(i, (ts[colra][i], ts[coldec][i]), ts[colrad][i])

        if plot_boxes:
            # read boxes to overplot boxes
            boxes = Table.read(self.box_regs, format=self.box_regs_fmt)
            ra_c, dec_c, width, height = self.box_regs_columns
            boxes['corner_ra']=boxes[ra_c]-0.5*boxes[width]   # assume no box crosses 360 boundary
            boxes['corner_dec']=boxes[dec_c]-0.5*boxes[height]

            idxb = (boxes[ra_c]>xwin[0]) & (boxes[ra_c]<xwin[1]) & (boxes[dec_c]>ywin[0]) & (boxes[dec_c]<ywin[1])
            tsb = boxes[idxb]
            for i in range(len(tsb)):
                box_ra, box_dec, box_sx, box_sy = tsb['corner_ra'][i], tsb['corner_dec'][i], tsb[width][i], tsb[height][i]
                rec = plt.Rectangle((box_ra, box_dec), box_sx, box_sy, color='r', fill=False, linewidth=0.4)
                ax.add_artist(rec)
                #print(i, box_ra, box_dec, box_sx, box_sy)

        plt.tight_layout()
        #if not(ax): plt.show()



    def plot2compare(self, stage='mask', nr=50_000, s=0.5, figsize=[12,6], xwin=None, ywin=None,
                     plot_stars=False, plot_boxes=False, **kwargs):
        """
        Compare input sources and random points generate over a mask

        Parameters
        ----------
        stage : string
            Masking stage to use, e.g. 'mask', 'foot', 'holemap', etc.
        nr : int
            Number of randoms
        s : float
            Point size
        figsize : list of floats
            Figure size
        xwin : list of floats
            plot limits in ra, e.g. xwin=[226.5,227.5]
        ywin : list of floats
            plot limits in dec, e.g. ywin=[10.,11.]
        plot_stars : bool
            Overlay circles due to bright stars
        plot_boxes : bool
            Overlay boxes due to bright stars
        kwargs : [key=val]
            Adittional keyword arguments passed to mataplolib.scatter()
        """
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=figsize)

        self.plot(stage=stage, nr=nr, s=s, figsize=[figsize[0]*0.5, figsize[1]], xwin=xwin, ywin=ywin,
                  plot_stars=plot_stars, plot_boxes=plot_boxes, use_srcs=True, ax=ax1 ,**kwargs)

        self.plot(stage=stage, nr=nr, s=s, figsize=[figsize[0]*0.5, figsize[1]], xwin=xwin, ywin=ywin,
                  plot_stars=plot_stars, plot_boxes=plot_boxes, use_srcs=False, ax=ax2, **kwargs)




    def makerans(self, stage='mask', nr=50_000, file=None, **kwargs):
        """
        Generate random points over a given mask and optionally save it to disk

        Parameters
        ----------
        stage : str
            Masking stage to use, e.g. 'mask', 'foot', 'holemap', etc.
        nr : int
            Number of randoms
        file : str, optional
            Output file (fits table)  XXXX change to parquet!

        Returns
        -------
        dataframe/astropy_table
            Input catalog with mask applied
        """
        if stage == 'foot':        mk = self.foot
        if stage == 'patchmap':    mk = self.patchmap
        if stage == 'propmap':     mk = self.propmap
        if stage == 'holemap':     mk = self.holemap
        if stage == 'extendedmap': mk = self.extendedmap
        if stage == 'mask':        mk = self.mask
        if stage == 'usermap':     mk = self.usermap
        rra, rdec = hsp.make_uniform_randoms_fast(mk, nr)
        tt = Table([rra, rdec],names=['ra','dec'], **kwargs)
        if file:
            tt.write(file, overwrite=True)
            print(str(nr),'randoms written to:', file)

        return tt


    def apply(self, stage='mask', cat=None, columns=['ra','dec'], file=None):
        """
        Apply a mask to a catalog (DataFrame or Astropy Table) and optionally save it to disk.

        Parameters
        ----------
        stage: str
            Masking stage to use, e.g., 'mask', 'foot', 'holemap', etc.
        cat : pandas.DataFrame or astropy.table.Table
            Input catalog to which the mask will be applied.
        columns : list of str
            Columns for RA and DEC coordinates.
        file : str, optional
            Path to the output file where the result will be saved in parquet format. If not provided, the result is not saved.

        Returns
        -------
        pandas.DataFrame or astropy.table.Table
            The input catalog with the mask applied.
        """
        colra, coldec = columns
        if stage == 'foot':        mk = self.foot
        if stage == 'patchmap':    mk = self.patchmap
        if stage == 'propmap':     mk = self.propmap
        if stage == 'holemap':     mk = self.holemap
        if stage == 'extendedmap': mk = self.extendedmap
        if stage == 'mask':        mk = self.mask
        if stage == 'usermap':     mk = self.usermap
        idx = mk.get_values_pos(cat[colra], cat[coldec], lonlat=True)
        if file:
            cat[idx].to_parquet(file)
            print(str(len(cat[idx])),'sources within',stage,' written to:', file)
        else:
            print(str(len(cat[idx])),'sources within',stage)

        return cat[idx]


    def write(self, filename):
        """
        Save a SkyMaskPipe object to disk (in a pickled, compressed, optimized format)

        Parameters
        ----------
        filename : str
            Path to the output file

        """
        import gzip, pickle, pickletools
        with gzip.open(filename, "wb") as f:
            pickled = pickle.dumps(self)
            optimized_pickle = pickletools.optimize(pickled)
            f.write(optimized_pickle)
            print(filename, 'saved to disk')


    @staticmethod
    def read(filename):
        """
        Read a SkyMaskPipe object from disk

        Parameters
        ----------
        file : str
            Path to file

        Returns
        -------
        SkyMaskPipe
            The SkyMaskPipe instance
        """
        import gzip, pickle, pickletools
        with gzip.open(filename, 'rb') as f:
            p = pickle.Unpickler(f)
            mk = p.load()
        return mk


