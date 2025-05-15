import pytest
from spatialfeatureexperiment import SpatialFeatureExperiment
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, Point
from biocframe import BiocFrame


@pytest.fixture
def sfe():
    nrows = 200
    ncols = 500
    counts = np.random.rand(nrows, ncols)

    spot_polys = gpd.GeoSeries(
        [
            Polygon([(1, -1), (1, 0), (0, 0)]),
            Polygon([(3, -1), (4, 0), (3, 1)]),
        ]
    )
    colgeoms = {
        "spot_polygons" : gpd.GeoDataFrame(
            {
                "geometry": spot_polys,
                "sample_id": ["sample01"] * len(spot_polys)
            }
        )
    }

    gene_points = gpd.GeoSeries(
        [
            Point(1.0, -1.0),
            Point(2.0, 0)
        ]
    )
    rowgeoms = {
        "gene_points": gpd.GeoDataFrame(
            {
                "geometry": gene_points,
                "sample_id": ["sample01"] * len(gene_points)
            }
        )
    }

    tissue_boundaries = gpd.GeoSeries(
        [
            Polygon([(2, 0), (1, 2), (1, 1)]),
            Polygon([(3, -1), (2, 3), (-2, 1)]),
        ]
    )
    annotgeoms = {
        "tissue_boundaries": gpd.GeoDataFrame(
            {
                "geometry": tissue_boundaries,
                "sample_id": ["sample01"] * len(tissue_boundaries)
            }
        )
    }

    # TODO: construct a valid spatial graph obj

    sfe_instance = SpatialFeatureExperiment(
        assays={"spots": counts},
        col_geometries=colgeoms,
        row_geometries=rowgeoms,
        annot_geometries=annotgeoms,
    )

    return sfe_instance
