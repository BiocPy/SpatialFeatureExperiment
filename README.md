[![PyPI-Server](https://img.shields.io/pypi/v/SpatialFeatureExperiment.svg)](https://pypi.org/project/SpatialFeatureExperiment/)
![Unit tests](https://github.com/BiocPy/SpatialFeatureExperiment/actions/workflows/run-tests.yml/badge.svg)

# SpatialFeatureExperiment

A Python package for storing and analyzing spatial-omics experimental data. This package provide the `SpatialFeatureExperiment` class, based on the [R package and class](https://github.com/pachterlab/SpatialFeatureExperiment).

## Install

To get started, install the package from [PyPI](https://pypi.org/project/SpatialFeatureExperiment/)

```bash
pip install spatialfeatureexperiment
```

## Quick Usage

This package uses shapely and geopandas to support the `*_geometries` slots.

```python
from spatialexperiment import SpatialFeatureExperiment
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon

nrows = 200
ncols = 500
counts = np.random.rand(nrows, ncols)
polys = gpd.GeoSeries(
    [
        Polygon([(1, -1), (1, 0), (0, 0)]),
        Polygon([(3, -1), (4, 0), (3, 1)]),
    ]
)

colgeoms = {"polygons" : gpd.GeoDataFrame({"geometry": polys})}
tspe = SpatialFeatureExperiment(assays={"spots": counts}, col_geometries=colgeoms)
```

<!-- biocsetup-notes -->

## Note

This project has been set up using [BiocSetup](https://github.com/biocpy/biocsetup)
and [PyScaffold](https://pyscaffold.org/).
