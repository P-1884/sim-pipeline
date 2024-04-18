import sys
import os
sim_pipeline_path = '/mnt/zfsusers/hollowayp/sim-pipeline/'
paltas_path = '/mnt/zfsusers/hollowayp/paltas/'
#sys.path.append('/global/homes/p/phil1884/')
sys.path.append(sim_pipeline_path)
sys.path.append(paltas_path)
os.chdir(sim_pipeline_path)

from slsim.Deflectors.elliptical_lens_galaxies import vel_disp_from_m_star
from Ellipticities_Translation import EllipticitiesTranslation
from scipy.stats import norm, truncnorm, uniform
from matplotlib.colors import Normalize, LogNorm
from Load_LensPop_LSST_db import db_LensPop_LSST
from slsim.Plots.lens_plots import LensingPlots
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import curve_fit
from slsim.lens_pop import LensPop
from astropy.units import Quantity
import matplotlib.pyplot as pl
import astropy.units as U
from tqdm import tqdm
import pandas as pd
import numpy as np
import arviz as az
import corner
import glob
import os
#from plot_image_collage import plot_collage

def set_max_pd(rows=None,columns=None):
    pd.set_option('display.max_columns', columns)
    pd.set_option('display.max_rows', rows)

set_max_pd(rows=30,columns=30)

#Adding in random, uniformly distributed lens position angles for LensPop, so ellipticities can be calculated:
np.random.seed(1)
def generate_LensPop_PA(db):
    return np.pi*np.random.random(size=len(db)) #Returning position angle, in radians

PA_LensPop_New = generate_LensPop_PA(db_LensPop_LSST)
db_LensPop_LSST['PA_lens'] = PA_LensPop_New
db_LensPop_e1,db_LensPop_e2 = EllipticitiesTranslation(db_LensPop_LSST['PA_lens'],db_LensPop_LSST['q_lens_flat'])
db_LensPop_LSST['e1_lens'] = db_LensPop_e1
db_LensPop_LSST['e2_lens'] = db_LensPop_e2

az.style.use("arviz-doc")

# define a cosmology
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# define a sky area
sky_area = Quantity(value=50, unit="deg2")

# define limits in the intrinsic deflector and source population (in addition to the skypy config
# file)
kwargs_deflector_cut = {"band": "i", "band_max": 23, "z_min": 0.01, "z_max": 1.26}
#magnitude limit set by median Lenspop magnification, and 5-sigma point-source limit:
kwargs_source_cut = {"band": "i", "band_max": 28.4, "z_min": 0.1, "z_max": 4.6}
#
kwargs_lens_cuts = {"mag_arc_limit": {"i": 25}, "min_magnification": 3}

print('Cuts:')
print('kwargs_deflector_cut',kwargs_deflector_cut)
print('kwargs_source_cut',kwargs_source_cut)
print('kwargs_lens_cuts',kwargs_lens_cuts)

print('Loading Lens Pop')
# run skypy pipeline and make galaxy-galaxy population class using GalaxyGalaxyLensPop
gg_lens_pop = LensPop(
    deflector_type="elliptical",
    source_type="galaxies",
    kwargs_deflector_cut=kwargs_deflector_cut,
    kwargs_source_cut=kwargs_source_cut,
    kwargs_mass2light=None,
    skypy_config=sim_pipeline_path+'/data/SkyPy/lsst-like.yml',
    sky_area=sky_area,
    cosmo=cosmo)

# drawing population
print('Drawing Population')
gg_lens_population = gg_lens_pop.draw_population(kwargs_lens_cuts=kwargs_lens_cuts)

def generate_lens_samples_dict(gg_lens_population_1):
    lens_samples = []
    lens_samples_dict = {'veldisp':[],'mstar':[],'tE':[],'zL':[],'zS':[],'mu_s':[],
                         'defl_e1_light':[], 'defl_e2_light':[],
                         'defl_e1_mass':[], 'defl_e2_mass':[],'e1_source':[],'e2_source':[],
                         'defl_mass_x':[],'defl_mass_y':[],'defl_gamma':[],'defl_gamma1':[],'defl_gamma2':[],'source_x':[],'source_y':[],
                         'source_Ns':[],'source_Rs':[],'defl_Ns':[],'defl_Rs':[],'defl_light_x':[],'defl_light_y':[],
                         'defl_mag_i_band':[],'source_mag_i_band':[]}
    labels = [
        r"$\sigma_v$",
        r"$\log(M_{*})$",
        r"$\theta_E$",
        r"$z_{\rm l}$",
        r"$z_{\rm s}$",
        r"$m_{\rm source}$",
        r"$m_{\rm lens}$"]
    for gg_lens in tqdm(gg_lens_population_1):
        vel_disp = gg_lens.deflector_velocity_dispersion()
        m_star = gg_lens.deflector_stellar_mass()
        theta_e = gg_lens.einstein_radius
        zl = gg_lens.deflector_redshift
        zs = gg_lens.source_redshift
        #source_mag = gg_lens.extended_source_magnitude(band="g", lensed=True)
        #deflector_mag = gg_lens.deflector_magnitude(band="g")
        magnification = gg_lens.extended_source_magnification()
        defl_e1_light, defl_e2_light, defl_e1_mass, defl_e2_mass = gg_lens.deflector_ellipticity()
        source_e1,source_e2 = gg_lens.source.ellipticity
        defl_gamma1 = gg_lens.deflector_mass_model_lenstronomy()[1][1]['gamma1']
        defl_gamma2 = gg_lens.deflector_mass_model_lenstronomy()[1][1]['gamma2']
        defl_mass_x = gg_lens.deflector_mass_model_lenstronomy()[1][0]['center_x']
        defl_mass_y = gg_lens.deflector_mass_model_lenstronomy()[1][0]['center_y']
        defl_light_x = gg_lens.deflector_light_model_lenstronomy()[1][0]['center_x']
        defl_light_y = gg_lens.deflector_light_model_lenstronomy()[1][0]['center_y']
        defl_Rs = gg_lens.deflector_light_model_lenstronomy()[1][0]['R_sersic']
        defl_Ns = gg_lens.deflector_light_model_lenstronomy()[1][0]['n_sersic']
        source_x,source_y = gg_lens.source_position
        source_Rs = gg_lens.source_light_model_lenstronomy()[1]['kwargs_source'][0]['R_sersic']
        source_Ns = gg_lens.source_light_model_lenstronomy()[1]['kwargs_source'][0]['n_sersic']
        defl_mag_i_band = gg_lens.deflector_magnitude('i')
        source_mag_i_band = gg_lens.extended_source_magnitude('i')
        lens_samples_dict['veldisp'].append(vel_disp)
        lens_samples_dict['mstar'].append(np.log10(m_star))
        lens_samples_dict['tE'].append(theta_e)
        lens_samples_dict['zL'].append(zl)
        lens_samples_dict['zS'].append(zs)
        lens_samples_dict['mu_s'].append(magnification)
        lens_samples_dict['defl_e1_light'].append(defl_e1_light)
        lens_samples_dict['defl_e2_light'].append(defl_e2_light)
        lens_samples_dict['defl_e1_mass'].append(defl_e1_mass)
        lens_samples_dict['defl_e2_mass'].append(defl_e2_mass)
        lens_samples_dict['e1_source'].append(source_e1)
        lens_samples_dict['e2_source'].append(source_e2)
        lens_samples_dict['defl_gamma'].append(2) #Hardcoded in slsim.lens.Lens.deflector_mass_model_lenstronomy
        lens_samples_dict['defl_gamma1'].append(defl_gamma1) 
        lens_samples_dict['defl_gamma2'].append(defl_gamma2) 
        lens_samples_dict['defl_mass_x'].append(defl_mass_x) 
        lens_samples_dict['defl_mass_y'].append(defl_mass_y) 
        lens_samples_dict['defl_light_x'].append(defl_light_x) 
        lens_samples_dict['defl_light_y'].append(defl_light_y) 
        lens_samples_dict['source_x'].append(source_x) 
        lens_samples_dict['source_y'].append(source_y) 
        lens_samples_dict['source_Rs'].append(source_Rs)
        lens_samples_dict['source_Ns'].append(source_Ns)
        lens_samples_dict['defl_Rs'].append(defl_Rs)
        lens_samples_dict['defl_Ns'].append(defl_Ns)
        lens_samples_dict['defl_mag_i_band'].append(defl_mag_i_band)
        lens_samples_dict['source_mag_i_band'].append(source_mag_i_band)
    return lens_samples_dict

lens_samples_dict = generate_lens_samples_dict(gg_lens_population)

def generate_simpipeline_db(lens_samples_dict_0):
    simpipeline_db = pd.DataFrame(lens_samples_dict_0)
    simpipeline_db = simpipeline_db.rename({'source_x':'xs','source_y':'ys','source_Rs':'Re_source','defl_Rs':'Re_lens',
                                          'source_mag_i_band':'i_source','defl_mag_i_band':'i_lens',
                                          'source_Ns':'Ns'},axis=1)
    simpipeline_db.to_csv(f'{sim_pipeline_path}/slsim_databases/trial_db.csv')
    return simpipeline_db

print('Saving:')
simpipeline_db = generate_simpipeline_db(lens_samples_dict)
print('Saved')
