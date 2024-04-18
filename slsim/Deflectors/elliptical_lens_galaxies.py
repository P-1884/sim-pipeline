import numpy as np
import numpy.random as random
from slsim.selection import deflector_cut
from slsim.Deflectors.velocity_dispersion import vel_disp_sdss
from slsim.Util import param_util
from slsim.Deflectors.deflectors_base import DeflectorsBase
import matplotlib.pyplot as pl

class EllipticalLensGalaxies(DeflectorsBase):
    """Class describing elliptical galaxies."""

    def __init__(self, galaxy_list, kwargs_cut, kwargs_mass2light, cosmo, sky_area):
        """

        :param galaxy_list: list of dictionary with galaxy parameters of
            elliptical galaxies (currently supporting skypy pipelines)
        :param kwargs_cut: cuts in parameters: band, band_mag, z_min, z_max
        :type kwargs_cut: dict
        :param kwargs_mass2light: mass-to-light relation
        :param cosmo: astropy.cosmology instance
        :type sky_area: `~astropy.units.Quantity`
        :param sky_area: Sky area over which galaxies are sampled. Must be in units of
            solid angle.
        """
        super().__init__(
            deflector_table=galaxy_list,
            kwargs_cut=kwargs_cut,
            cosmo=cosmo,
            sky_area=sky_area,
        )

        n = len(galaxy_list)
        column_names = galaxy_list.colnames
        if "vel_disp" not in column_names:
            galaxy_list["vel_disp"] = -np.ones(n)
        if "e1_light" not in column_names or "e2_light" not in column_names:
            galaxy_list["e1_light"] = -np.ones(n)
            galaxy_list["e2_light"] = -np.ones(n)
        if "e1_mass" not in column_names or "e2_mass" not in column_names:
            galaxy_list["e1_mass"] = -np.ones(n)
            galaxy_list["e2_mass"] = -np.ones(n)
        if "n_sersic" not in column_names:
            galaxy_list["n_sersic"] = -np.ones(n)

        num_total = len(galaxy_list)
        z_min, z_max = 0, np.max(galaxy_list["z"])
        redshift = np.arange(z_min, z_max, 0.1)
        vd_min=100
        hist_dict = {'bins':np.arange(50,350,25),'alpha':0.5,'density':False}
        z_list, vel_disp_list = vel_disp_sdss(sky_area, redshift, vd_min=vd_min, vd_max=500, cosmology=cosmo, noise=True)
        print('VAL',len(z_list),len(vel_disp_list))
        fig,ax = pl.subplots(1,2,figsize=(12,5))
        ax[0].hist(vel_disp_sdss(sky_area, redshift, vd_min=50, vd_max=500, cosmology=cosmo, noise=True)[1],
                **hist_dict,label='$\sigma_{min}$=50',color='purple')
        ax[0].hist(vel_disp_sdss(sky_area, redshift, vd_min=100, vd_max=500, cosmology=cosmo, noise=True)[1],
                **hist_dict,label='$\sigma_{min}$=100',color='darkblue')
        ax[0].hist(vel_disp_sdss(sky_area, redshift, vd_min=200, vd_max=500, cosmology=cosmo, noise=True)[1],
                **hist_dict,label='$\sigma_{min}$=200',color='darkorange')
        ax[0].set_xlabel('$\sigma$ km/s',fontsize=15)
        ax[0].set_ylabel('Counts',fontsize=15)
        ax[0].legend(fontsize=12)
        hist_dict = {'bins':np.arange(0,2,0.1),'alpha':0.5,'density':False}
        ax[1].hist(vel_disp_sdss(sky_area, redshift, vd_min=50, vd_max=500, cosmology=cosmo, noise=True)[0],
                **hist_dict,label='$\sigma_{min}$=50',color='purple')
        ax[1].hist(vel_disp_sdss(sky_area, redshift, vd_min=100, vd_max=500, cosmology=cosmo, noise=True)[0],
                **hist_dict,label='$\sigma_{min}$=100',color='darkblue')
        ax[1].hist(vel_disp_sdss(sky_area, redshift, vd_min=200, vd_max=500, cosmology=cosmo, noise=True)[0],
                **hist_dict,label='$\sigma_{min}$=200',color='darkorange')
        ax[1].set_xlabel('Redshift',fontsize=15)
        ax[1].set_ylabel('Counts',fontsize=15)
        ax[1].legend(fontsize=12)
        pl.show()
        print(f'Getting {n} galaxies from the LMF, but {len(vel_disp_list)} corresponding velocity dispersions, with ratio '+\
              f'{n/len(vel_disp_list)}, over {sky_area}, with vd_min={vd_min}') 
        print(f'{np.sum(np.array(vel_disp_list)>200)} have'+' $\sigma$>200')
        # sort for stellar masses in decreasing manner
        galaxy_list.sort("stellar_mass")
        galaxy_list.reverse()
        def abundance_match(galaxy_list,num_total,vel_disp_list):
            # sort velocity dispersion, largest values first
            vel_disp_list = np.flip(np.sort(vel_disp_list))
            num_vel_disp = len(vel_disp_list)
            # abundance match velocity dispersion with elliptical galaxy catalogue
            if num_vel_disp >= num_total:
                print('Cropping velocity dispersion list')
                #selection_indx_rand = np.random.choice(np.arange(0,num_vel_disp,1),size=num_total,replace=False).tolist()
                #selection_indx_rand.sort()
                #selection_indx = np.linspace(0,num_vel_disp-1,num_total).astype('int')
                #galaxy_list["vel_disp"] = vel_disp_list[selection_indx_rand]
                #galaxy_list["vel_disp"] = vel_disp_list[selection_indx]
                galaxy_list["vel_disp"] = vel_disp_list[:num_total]
                # randomly select
            else:
                print('NOT Cropping velocity dispersion list - cropping skypy catalogue instead')
                galaxy_list = galaxy_list[:num_vel_disp]
                galaxy_list["vel_disp"] = vel_disp_list
                num_total = num_vel_disp
            return galaxy_list
        def mstar_sigma_relation(galaxy_list):
            galaxy_list['vel_disp'] = vel_disp_from_m_star(galaxy_list['stellar_mass'])
            return galaxy_list
        #galaxy_list = abundance_match(galaxy_list,num_total,vel_disp_list)
        galaxy_list = mstar_sigma_relation(galaxy_list)
        self._galaxy_select = deflector_cut(galaxy_list, **kwargs_cut)
        self._num_select = len(self._galaxy_select)
        # TODO: random reshuffle of matched list

    def deflector_number(self):
        """

        :return: number of deflectors
        """
        number = self._num_select
        return number

    def draw_deflector(self):
        """

        :return: dictionary of complete parameterization of deflector
        """

        index = random.randint(0, self._num_select - 1)
        deflector = self._galaxy_select[index]
#        print('DEFL',deflector)
        if deflector["vel_disp"] == -1:
#            print('Setting vel_disp from m_star')
            stellar_mass = deflector["stellar_mass"]
            vel_disp = vel_disp_from_m_star(stellar_mass)
            deflector["vel_disp"] = vel_disp
        if deflector["e1_light"] == -1 or deflector["e2_light"] == -1:
            e1_light, e2_light, e1_mass, e2_mass = elliptical_projected_eccentricity(
                **deflector
            )
            deflector["e1_light"] = e1_light
            deflector["e2_light"] = e2_light
            deflector["e1_mass"] = e1_mass
            deflector["e2_mass"] = e2_mass
        if deflector["n_sersic"] == -1:
            deflector["n_sersic"] = 4  # TODO make a better estimate with scatter
        return deflector


def elliptical_projected_eccentricity(ellipticity, **kwargs):
    """Projected eccentricity of elliptical galaxies as a function of other deflector
    parameters.

    :param ellipticity: eccentricity amplitude
    :type ellipticity: float [0,1)
    :param kwargs: deflector properties
    :type kwargs: dict
    :return: e1_light, e2_light,e1_mass, e2_mass eccentricity components
    """
    e_light = param_util.epsilon2e(ellipticity)
    phi_light = np.random.uniform(0, np.pi)
    e1_light = e_light * np.cos(phi_light)
    e2_light = e_light * np.sin(phi_light)
    e_mass = 0.5 * ellipticity + np.random.normal(loc=0, scale=0.1)
    phi_mass = phi_light + np.random.normal(loc=0, scale=0.1)
    e1_mass = e_mass * np.cos(phi_mass)
    e2_mass = e_mass * np.sin(phi_mass)
    return e1_light, e2_light, e1_mass, e2_mass


def vel_disp_from_m_star(m_star,scatter=False):
    np.random.seed(1)
    """Function to calculate the velocity dispersion from the staller mass using
    empirical relation for elliptical galaxies.

    The power-law formula is given by:

    .. math::

         V_{\\mathrm{disp}} = 10^{2.32} \\left( \\frac{M_{\\mathrm{star}}}{10^{11}
         M_\\odot} \\right)^{0.24}

    2.32,0.24 is the parameters from [1] table 2
    2.34,0.18 and 0.04 are the parameters from [1] table 2, including scatter.
    [1]:Auger, M. W., et al. "The Sloan Lens ACS Survey. X. Stellar, dynamical, and
    total mass correlations of massive elliptical galaxies." The Astrophysical
    Journal 724.1 (2010): 511.

    :param m_star: stellar mass in the unit of solar mass
    :return: the velocity dispersion ("km/s")
    """
    if not scatter: v_disp = np.power(10, 2.32) * np.power(m_star / 1e11, 0.24)
    if scatter: v_disp = np.power(10,2.34)*np.power(m_star/1e11,0.18)*np.power(10,np.random.normal(0,0.04,size=len(m_star)))
    return v_disp
