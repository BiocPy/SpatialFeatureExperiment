import os
import tempfile

import numpy as np
import pytest
from aicsimageio.writers import OmeTiffWriter

from spatialfeatureexperiment import BioFormatsImage, ExtImage, SpatialFeatureExperiment, SpatRasterImage


# Fixtures
@pytest.fixture
def sample_array():
    width, height = 200, 100
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)

    image_array = np.sin(xx * 10) * np.cos(yy * 10) * 127 + 128
    return image_array.astype(np.uint8)


@pytest.fixture
def sample_extent():
    return {"xmin": 0, "xmax": 100, "ymin": 0, "ymax": 50}


@pytest.fixture
def ext_image(sample_array, sample_extent):
    return ExtImage(sample_array, sample_extent)


@pytest.fixture
def sfe_object():
    counts = np.random.negative_binomial(5, 0.5, (100, 50))
    coords = np.random.uniform(0, 1000, (100, 2))
    return SpatialFeatureExperiment(counts, coords)


@pytest.fixture
def temp_ome_tiff():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a simple 3D array (TCZYX order)
        data = np.random.randint(0, 256, (1, 3, 100, 200, 1), dtype=np.uint8)

        # Save as OME-TIFF
        ome_tiff_path = os.path.join(temp_dir, "dummy.ome.tiff")
        OmeTiffWriter.save(data, ome_tiff_path, dim_order="TCZYX")

        yield ome_tiff_path


# Tests for ExtImage
def test_create_ext_image(sample_array, sample_extent):
    img = ExtImage(sample_array, sample_extent)
    assert img is not None
    assert img.shape == sample_array.shape
    assert img.extent == sample_extent


# Tests for SpatRasterImage
def test_create_spat_raster_image(sample_array, sample_extent):
    img = SpatRasterImage(sample_array, sample_extent)
    assert img is not None
    assert img.extent == sample_extent


# Tests for BioFormatsImage
def test_create_bioformats_image(temp_ome_tiff):
    img = BioFormatsImage(temp_ome_tiff)
    assert img is not None
    assert "xmin" in img.extent
    assert "xmax" in img.extent
    assert "ymin" in img.extent
    assert "ymax" in img.extent


def test_with_sfe(temp_ome_tiff):
    bfi = BioFormatsImage(temp_ome_tiff)

    sfe = SpatialFeatureExperiment()
    sfe.add_img(image_source=bfi, scale_factor=False, sample_id="sample1", image_id="ome_tiff", in_place=True)

    img = sfe.get_img(sample_id="sample1", image_id="ome_tiff")
    assert img is not None
    assert img.shape == bfi.shape
