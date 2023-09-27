import os
import numpy as np
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from sim_pipeline.galaxy_galaxy_lens import GalaxyGalaxyLens
from sim_pipeline.galaxy_galaxy_lens_pop import GalaxyGalaxyLensPop
from sim_pipeline.Sources.source_variability.variability import sinusoidal_variability
from astropy.units import Quantity
from sim_pipeline.image_simulation import (
    simulate_image,
    sharp_image,
    sharp_rgb_image,
    rgb_image_from_image_list, point_source_image_properties, point_source_image
)
import pytest

class TestImageSimulation(object):
    def setup_method(self):
        # path = os.path.dirname(sim_pipeline.__file__)

        path = os.path.dirname(__file__)
        module_path, _ = os.path.split(path)
        print(path, module_path)
        blue_one = Table.read(
            os.path.join(path, "TestData/blue_one_modified.fits"), format="fits"
        )
        red_one = Table.read(
            os.path.join(path, "TestData/red_one_modified.fits"), format="fits"
        )
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.source_dict = blue_one
        self.deflector_dict = red_one
        while True:
            gg_lens = GalaxyGalaxyLens(
                source_dict=self.source_dict,
                deflector_dict=self.deflector_dict,
                cosmo=cosmo,
            )
            if gg_lens.validity_test():
                self.gg_lens = gg_lens
                break

    def test_simulate_image(self):
        image = simulate_image(
            lens_class=self.gg_lens,
            band="g",
            num_pix=100,
            add_noise=True,
            observatory="LSST",
        )
        assert len(image) == 100

    def test_sharp_image(self):
        image = sharp_image(
            lens_class=self.gg_lens,
            band="g",
            mag_zero_point=30,
            delta_pix=0.05,
            num_pix=100,
            with_deflector=True,
        )
        assert len(image) == 100

    def test_sharp_rgb_image(self):
        image = sharp_rgb_image(
            lens_class=self.gg_lens,
            rgb_band_list=["r", "g", "i"],
            mag_zero_point=30,
            delta_pix=0.05,
            num_pix=100,
        )
        assert len(image) == 100

    def test_rgb_image_from_image_list(self):
        image_g = sharp_image(
            lens_class=self.gg_lens,
            band="g",
            mag_zero_point=30,
            delta_pix=0.05,
            num_pix=100,
            with_deflector=True,
        )
        image_r = sharp_image(
            lens_class=self.gg_lens,
            band="r",
            mag_zero_point=30,
            delta_pix=0.05,
            num_pix=100,
            with_deflector=True,
        )
        image_b = sharp_image(
            lens_class=self.gg_lens,
            band="i",
            mag_zero_point=30,
            delta_pix=0.05,
            num_pix=100,
            with_deflector=True,
        )
        image_list = [image_r, image_g, image_b]
        image = rgb_image_from_image_list(image_list, 0.5)
        assert len(image) == 100

@pytest.fixture
def quasar_lens_pop_instance():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=0.1, unit="deg2")
    return GalaxyGalaxyLensPop(deflector_type="all-galaxies",
    source_type="quasars",
    kwargs_deflector_cut=None,
    kwargs_source_cut=None,
    kwargs_mass2light=None,
    skypy_config=None,
    sky_area=sky_area,
    cosmo=cosmo)

def test_point_source_image_properties(quasar_lens_pop_instance):
    kwargs_lens_cut={'min_image_separation': 0.8, 'max_image_separation': 10}
    lens_class = quasar_lens_pop_instance.select_lens_at_random(**kwargs_lens_cut)
    result = point_source_image_properties(lens_class = lens_class, band = 'i', 
                        mag_zero_point = 27, delta_pix = 0.2, num_pix = 101)

    expected_columns = ["deflector_pix", "image_pix", "ra_image", "dec_image",
                            "image_amplitude", "image_magnitude", "radec_at_xy_0"]
    assert result.colnames[0] == expected_columns[0]
    assert result.colnames[1] == expected_columns[1]
    assert result.colnames[2] == expected_columns[2]
    assert result.colnames[3] == expected_columns[3]
    assert result.colnames[4] == expected_columns[4]
    assert result.colnames[5] == expected_columns[5]
    assert result.colnames[6] == expected_columns[6]

def test_point_source_image(quasar_lens_pop_instance):
    kwargs_lens_cut={'min_image_separation': 0.8, 'max_image_separation': 10}
    lens_class = quasar_lens_pop_instance.select_lens_at_random(**kwargs_lens_cut)
    image_data = point_source_image_properties(lens_class = lens_class, band = 'i', 
                        mag_zero_point = 27, delta_pix = 0.2, num_pix = 101)
    number = len(image_data['image_pix'])
    path = os.path.dirname(__file__)
    #module_path, _ = os.path.split(path)
    #print(path, module_path)
    psf_image_1 = [np.loadtxt(os.path.join(path, "TestData/psf_kernels_for_image_1.txt"), unpack = True)]
    psf_kernels = psf_image_1[:-1]
    psf_kernels.extend([psf_image_1[-1]] * number)
    
    #psf_image_2 = np.loadtxt('/data/dp0/psf_kernels_for_image_2.txt', unpack = True)
    #psf_deflector = np.loadtxt('/data/dp0/psf_kernels_for_deflector.txt',
                                #unpack = True)
    #psf_kernels = [psf_image_1, psf_image_2, psf_deflector]

    time = np.linspace(0, 10, 3)
    variability = {'time': time, 
        'function': sinusoidal_variability}
    # Call the function to get the result
    result1 = point_source_image(lens_class = lens_class, band = 'i', 
            mag_zero_point = 27, delta_pix = 0.2, num_pix = 101, 
            psf_kernels = psf_kernels, variability = variability,
                lensed=True)
    result2 = point_source_image(lens_class = lens_class, band = 'i', 
            mag_zero_point = 27, delta_pix = 0.2, num_pix = 101, 
            psf_kernels = psf_kernels, variability = None,
                lensed=True)
    assert len(result1[0]) == len(time)
    assert len(result2) == number

if __name__ == "__main__":
    pytest.main()