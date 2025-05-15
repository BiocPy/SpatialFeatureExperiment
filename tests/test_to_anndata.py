import anndata as ad
import geopandas as gpd
from biocframe import BiocFrame

__author__ = "keviny2"
__copyright__ = "keviny2"
__license__ = "MIT"


def test_to_anndata(sfe):
    obj, alt_exps = sfe.to_anndata()

    assert isinstance(obj, ad.AnnData)

    assert obj.shape == (500, 200)

    assert isinstance(obj.uns['spatial']['col_geometries']['spot_polygons'], gpd.GeoDataFrame)
    assert isinstance(obj.uns['spatial']['row_geometries']['gene_points'], gpd.GeoDataFrame)
    assert isinstance(obj.uns['spatial']['annot_geometries']['tissue_boundaries'], gpd.GeoDataFrame)
    assert isinstance(obj.uns['spatial']['spatial_graphs'], BiocFrame)
    assert obj.uns['spatial']['unit'] == 'full_res_image_pixel'

