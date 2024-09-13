import numpy as np
import healsparse as hsp
import healpy as hp
import matplotlib.pyplot as plt
import lsdb
from astropy.table import Table, join
from astropy.coordinates import Angle, Latitude, Longitude, SkyCoord
from mocpy import MOC
from tqdm import tqdm


class SkyMaskPipe:
    def __init__(self, **kwargs):
        defaults = {
            'order_cov':    4,            
            'order_foot':   13,
            'hipcat':       '/home/edonoso/hscmags/hscx_hectomap_gal/',
            'qafile':       '/home/edonoso/hsc_patches/patch_qa.csv',
            'order_patch':  13,
            'patchfile':    ['/home/edonoso/hsc_patches/tracts_patches_W-hectomap.parquet', 
                             '/home/edonoso/hsc_patches/tracts_patches_W-spring.parquet',
                             '/home/edonoso/hsc_patches/tracts_patches_W-AEGIS.parquet',
                             '/home/edonoso/hsc_patches/tracts_patches_W-autumn.parquet'],
            'order_holes':  15,
            'star_regs':    '/home/edonoso/hsc_bsregions/stars.reg.I.nodups.parquet',
            'box_regs':     '/home/edonoso/hsc_bsregions/box.reg.I.parquet',
            'order_user':   15,
            'circ_uregs':   '',
            'poly_uregs':   '',
            'order_extended': 15,
            'ellip_regs':   '',
            'order_out':    15
        }

        for (prop, val) in defaults.items():
            setattr(self, prop, kwargs.get(prop, val))
        
        self.nside_out   = 2**self.order_out
        self.nside_cov   = 2**self.order_cov
        self.foot        = None
        self.patchmap    = None
        self.holemap     = None
        self.usermap     = None
        self.extendedmap = None
        self.mask        = None
        

    @staticmethod
    def readQApatches(qafile):
        """
        Read contents of HSC QA patch list
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
    def filter_and_pixelate_patches(file, qatable, order=13):
        """
        Reads a file with HSC patches, matches it against the QA table containing quality measures
        of each patch, filter those that meet some depth/seeing/etc critera, and return their 
        pixels at a given order. Patches are pixelized as quadrangle polygons.
    
        Modify the patch aceptance criteria to suit your purposes, e.g. accepting only those above
        a minimum depth threshold in a given band, etc.
    
        Parameters
        ----------
        file : string
                  HSC patch file (defult parquet, but anything astropy can read)
        qatable : astropy table
                  Table of patches with QA measurements
        order : integer
                  Pixelization order
                
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
    
        # Apply uniformity criteria to patches based on minimum depth. This gets rids of border zones.
        idx = (table['gmag_psf_depth']>26.2) & (table['rmag_psf_depth']>25.9) & (table['imag_psf_depth']>25.7)
        print('    Patches with QA fulfilling conditions :', len(table[idx]))
    
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
        hsp_map
                 healsparse boolean map 
                
        Returns
        -------
        hsp_map
                 healsparse boolean map without empty single pixels
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
        hsp_map
                 healsparse boolean map 
                
        Returns
        -------
        hsp_map
                 healsparse boolean map with eroded borders
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
        file : string
                  Path to file (circular astropy-regions)
        fmt : string
                  Format of file, e.g. 'ascii', 'parquet', or any accepted by astropy.table
        columns : list of strings
                  Column names for ra, dec, radius of circles
        order : integer
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
        file : string
                  Path to file (circular astropy-regions)
        fmt : string
                  Format of file, e.g. 'ascii', 'parquet', or any accepted by astropy.table
        columns : list of strings
                  Column names for ra, dec, a, b and pa (position angle) of ellipses
        order : integer
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
        file : string
                  Path to file (box astropy-regions)
        fmt : string
                  Format of file, e.g. 'ascii', 'parquet', or any accepted by astropy.table
        columns : list of strings
                  Column names for ra_center, dec_center, width and height of boxes
        order : integer
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
        Input data must have 8 columns for the coordinates of the 4 vertices
        Coordinates and distances should be in degrees.
    
        Parameters
        ----------
        file : string
                  Path to file (polygon astropy-regions)
        fmt : string
                  Format of file, e.g. 'ascii', 'parquet', or any accepted by astropy.table
        columns : list of strings
                  Column names for ra_center, dec_center, width and height of boxes
        order : integer
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



    def build_holes_mask(self, star_regs=None, box_regs=None, fmt='parquet', value=True, order_holes=None, 
                         columns_circ=['ra','dec','radius'], 
                         columns_box=['ra_c','dec_c','width','height']):
        """
        Create a holes map corresponding to (circular and box regions) around bright stars
        (add reference [xxxx]).
        
        Note by default this map's value is set to True, as combine_mask() use it as negative while
        combining with other maps
    
        Parameters
        ----------
        star_regs : string
                  Path to file (circular astropy-regions)
        box_regs : bool
                  Path to file (box astropy-regions)
        fmt : string
                  Format of file, e.g. 'ascii', 'parquet', or any accepted by astropy.table
        value : bool
                  Set the value of pixels inside stars/boxes
        order_holes : integer
                  Pixelization order
        columns_circ : list of string
                  Column names for ra, dec, radius of circles                  
        columns_box : list of string
                  Column names for ra_center, dec_center, width, height of boxes 
        
        Returns
        -------
        hsp_map
                  healsparse boolean map 
        """
        if star_regs: self.star_regs=star_regs
        if box_regs: self.box_regs=box_regs
        print('BUILDING BRIGHT STAR HOLES MAP >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        
        star_pixels = self.pixelate_circles(self.star_regs, fmt=fmt, order=self.order_holes, columns=columns_circ) 
        box_pixels = self.pixelate_boxes(self.box_regs, fmt=fmt, order=self.order_holes, columns=columns_box)
    
        # Empty boolean map
        nside_holes = 2**self.order_holes
        self.holemap = hsp.HealSparseMap.make_empty(self.nside_cov, nside_holes, dtype=np.bool_)
        # Set the holes to True to invert them later
        self.holemap[star_pixels] = value
        self.holemap[box_pixels] = value
    
        print('--- Holes map area                        :', self.holemap.get_valid_area(degrees=True))



    def build_user_mask(self, circ_uregs=None, poly_uregs=None, fmt='ascii', value=True, order_user=None,
                        columns_ucirc=['ra','dec','radius'], 
                        columns_upoly=['ra0','ra1','ra2','ra3','dec0','dec1','dec2','dec3']):
        if circ_uregs: self.circ_uregs=circ_uregs
        if poly_uregs: self.poly_uregs=poly_uregs
        if order_user: self.order_user=order_user
        print('BUILDING USER DEFINED MAP >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')   
    
        # Empty boolean map
        nside_user = 2**self.order_user
        self.usermap = hsp.HealSparseMap.make_empty(self.nside_cov, nside_user, dtype=np.bool_)
        
        if self.circ_uregs:
            circle_pixels = self.pixelate_circles(self.circ_uregs, fmt=fmt, order=self.order_user, columns=columns_ucirc)
            self.usermap[circle_pixels] = value
    
        if self.poly_uregs:
            poly_pixels = self.pixelate_polys(self.poly_uregs, fmt=fmt, order=self.order_user, columns=columns_upoly)
            self.usermap[poly_pixels] = value
    
        print('--- User map area                         :', self.usermap.get_valid_area(degrees=True))



    def build_extended_mask(self, ellip_regs=None, fmt='ascii', value=True, order_extended=None,
                            columns_ellip=['ra','dec','a','b','pa']):
        if ellip_regs: self.ellip_regs=ellip_regs
        if order_extended: self.order_extended=order_extended
        print('BUILDING EXTENDED MAP >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')   
    
        # Empty boolean map
        nside_extended = 2**self.order_extended
        self.extendedmap = hsp.HealSparseMap.make_empty(self.nside_cov, nside_extended, dtype=np.bool_)
        
        if self.ellip_regs:
            ellip_pixels = self.pixelate_ellipses(self.ellip_regs, fmt=fmt, order=self.order_user, columns=columns_ellip)
            self.extendedmap[ellip_pixels] = value
    
        print('--- Extended map area                     :', self.extendedmap.get_valid_area(degrees=True))

        
    
    def build_patch_mask(self, patchfile=None, qafile=None, order_patch=None):

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
            fpx = self.filter_and_pixelate_patches(p, qatable, order=self.order_patch)
            self.patchmap[fpx] = True
    
        print('--- Patch map area                        :', self.patchmap.get_valid_area(degrees=True))
    

    
    def build_footprint_mask(self, hipcat=None, order_foot=None, columns=['ra_mag','dec_mag'],
                             remove_isopixels=False, erode_borders=False):
        if hipcat: self.hipcat=hipcat
        if order_foot: self.order_foot=order_foot
        colra, coldec = columns
        print('BUILDING FOOTPRINT MAP >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    
        # Create empty healsparse empty
        nside_foot   = 2**self.order_foot
        self.foot = hsp.HealSparseMap.make_empty(self.nside_cov, nside_foot, dtype=np.bool_)
        
        # Read (ra,dec) from a hipscatted catalog
        print('--- Pixelating hipscat catalog from:', self.hipcat)
        print('    Order ::',self.order_foot)
        srcs = lsdb.read_hipscat(self.hipcat, columns=columns).compute() 
    
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



    def combine_mask(self, apply_patchmap=True, apply_holemap=True, apply_extendedmap=True, apply_usermap=False):
        print('COMBINING MAPS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    
        # Upgrade maps if needed up to the desired resolution
        otmp = int(np.log2(self.foot.nside_sparse))
        if otmp < self.order_out:
            print('--- footprint order upgraded to:', self.order_out)
            self.foot = self.foot.upgrade(self.nside_out)
            
        if apply_patchmap and self.patchmap:
            otmp = int(np.log2(self.patchmap.nside_sparse))
            if otmp < self.order_out:
                print('--- patchmap order upgraded to: ', self.order_out)
                self.patchmap = self.patchmap.upgrade(self.nside_out)

        if apply_usermap and self.usermap:
            otmp = int(np.log2(self.usermap.nside_sparse))
            if otmp < self.order_out:
                print('--- usermap order upgraded to: ', self.order_out)
                self.usermap = self.usermap.upgrade(self.nside_out)

        if apply_extendedmap and self.extendedmap:
            otmp = int(np.log2(self.extendedmap.nside_sparse))
            if otmp < self.order_out:
                print('--- extendedmap order upgraded to: ', self.order_out)
                self.extendedmap = self.extendedmap.upgrade(self.nside_out)

    
        # Create empty map to contain the final mask and perform combination
        self.mask = hsp.HealSparseMap.make_empty(self.nside_cov, self.nside_out, dtype=np.bool_)
        self.mask |= self.foot
        
        if apply_patchmap:
            if self.patchmap:
                self.mask = self.intersect_boolmask(self.patchmap, self.mask)
            else:
                raise Exception('patchmap not defined')
        
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
                self.mask = self.intersect_boolmask(self.usermap, self.mask)
            else:
                raise Exception('usermap not defined')
        
        print('--- Combined map area                     :', self.mask.get_valid_area(degrees=True))



    @staticmethod
    def intersect_boolmask(mask1, mask2):
        """
        Intersect two arbitrary boolean masks in healsparse format.
    
        Parameters
        ----------
        mask1, mask2 : hsp_map
                 Healsparse boolean map
                
        Returns
        -------
        hsp_map
                 healsparse boolean map 
        """
        
        if mask1.nside_sparse != mask2.nside_sparse:
            raise Exception('Maps have different nside_sparse')
        
        tmp = mask1 & mask2
        msk = mask2 & tmp
        return msk
    

    def plot(self, stage='mask', nr=50_000, s=0.5, **kwargs):
        """
        Quickly visualize a mask by means of its randoms points
        
        Parameters
        ----------
        stage : string
                  Masking stage to use, e.g. 'mask', 'foot', 'holemap', etc.
        nr : integer
                  Number of randoms        
        s : float
                  Point size
        kwargs : [key=val]
                  Adittional keyword arguments passed to mataplolib.scatter()
        """
        if stage == 'foot':        mk = self.foot
        if stage == 'patchmap':    mk = self.patchmap
        if stage == 'holemap':     mk = self.holemap
        if stage == 'extendedmap': mk = self.extendedmap
        if stage == 'usermap':     mk = self.usermap
        if stage == 'mask':        mk = self.mask
        rra, rdec = hsp.make_uniform_randoms_fast(mk, nr)
        
        plt.figure(figsize=(12,8))
        plt.scatter(rra, rdec, s=s, **kwargs)
        plt.show()
        

    def makerans(self, stage='mask', nr=50_000, file=None, **kwargs):
        """
        Generate random points over a given mask and optionally save it to disk
    
        Parameters
        ----------
        stage : string
                  Masking stage to use, e.g. 'mask', 'foot', 'holemap', etc.
        nr : integer
                  Number of randoms
        file : string
                  Output file (fits table)  XXXX change to parquet!
                
        Returns
        -------
        dataframe/astropy_table
                  Input catalog with mask applied 
        """
        if stage == 'foot':        mk = self.foot
        if stage == 'patchmap':    mk = self.patchmap
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
        Apply a mask to catalog (dataframe/astropy_table) and optionally save it to disk
    
        Parameters
        ----------
        stage : string
                  Masking stage to use, e.g. 'mask', 'foot', 'holemap', etc.
        cat : dataframe/astropy_table
                  Input catalog
        columns : list of strings
                  Columns for RA and DEC coordinates
        file : string
                  Output file (parquet format)
                
        Returns
        -------
        dataframe/astropy_table
                  Input catalog with mask applied 
        """
        
        colra, coldec = columns
        if stage == 'foot':        mk = self.foot
        if stage == 'patchmap':    mk = self.patchmap
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



# # Subclassing HealSparseMap
# class SkyMask(hsp.HealSparseMap):
#     def __init__(self, *args, **kwargs):
#         # Call the superclass constructor
#         super().__init__(*args, **kwargs)

#     # Add custom methods
#     def plot(self, nr=50_000, s=1, **kwargs):
#         """
#         Plot the HealSparse map using matplotlib.
        
#         :param kwargs: Additional arguments to pass to the plot.
#         """
#         rra, rdec = hsp.make_uniform_randoms_fast(self, nr)
        
#         plt.figure()
#         plt.scatter(rra, rdec, s=s, **kwargs)
#         plt.show()

#     def intersect_boolmask(self, mask2):

#         tmp = self & mask2
#         msk = mask2 & tmp
#         return msk






# class SkyMask:
#     def __init__(self, **kwargs):
#         defaults = {
#             "order": 12,
#             "order_cov": 4
#         }

#         for (prop, val) in defaults.items():
#             setattr(self, prop, kwargs.get(prop, val))
        
#         self.nside = 2**self.order
#         self.nside_cov = 2**self.order_cov
#         self._hs = hsp.HealSparseMap.make_empty(self.nside_cov, self.nside, dtype=np.bool_)

#     # using property decorator --> a getter function 
#     @property
#     def hs(self): 
#          print("getter method called") 
#          return self._hs 

#     # a setter function 
#     @hs.setter 
#     def hs(self, hsmap): 
#         print("setter method called") 
#         self._hs = hsmap
#         self.nside = hsmap.nside_sparse 
#         self.nside_cov = hsmap.nside_coverage 
#         self.order = math.log2(hsmap.nside_sparse)
#         self.order_cov = math.log2(hsmap.nside_coverage)

#     def __str__(self):
#         return self.hs

#     def intersect(self, mask2):
#         pass

#     def apply(self, table, colra='ra_mag', coldec='dec_mag', file=''):
#         pass
    
#     def makerans(self, nr=100_000, file='', colra='ra', coldec='dec'):
#         pass

#     def plot(self, s=1):
#         pass
    




# class Neuron():

#     def __init__(self, **kwargs):
#         prop_defaults = {
#             "num_axon_segments": 0,
#             "apical_bifibrications": "fancy default"
#         }

#         for (prop, default) in prop_defaults.items():
#             setattr(self, prop, kwargs.get(prop, default))
