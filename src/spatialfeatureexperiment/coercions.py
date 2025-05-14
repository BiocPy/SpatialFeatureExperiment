from typing import Any, Dict, Literal, Optional, Union
from warnings import warn

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from biocframe import BiocFrame
from libpysal.graph import Graph
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from shapely.geometry import CAP_STYLE, MultiPoint, Point, Polygon
from spatialexperiment import LoadedSpatialImage, RemoteSpatialImage, SpatialExperiment, StoredSpatialImage

from .aligned_spatialimage import BioFormatsImage, ExtImage, SpatRasterImage

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


# TODO: need to test the polygon/multipoint
def dataframe_to_geopandas(
    coords_df: pd.DataFrame,
    spatial_coordinates_names: list = None,
    spot_diameter: float = None,
    buffer_radius: float = 1.0,
    vertices_col: Optional[str] = None,  # specifically for POLYGON/MULTIPOINT from lists of coords
    geometry_type: Literal["POINT", "POLYGON", "MULTIPOINT"] = "POINT",
    end_cap_style: Literal["ROUND", "FLAT", "SQUARE"] = "ROUND",
) -> gpd.GeoDataFrame:
    """Convert DataFrame coordinates to a Geopandas DataFrame.

    Args:
        coords_df:
            DataFrame with coordinates.

        spatial_coordinates_names:
            Names of the coordinate columns.

        spot_diameter:
            Diameter of spots for creating buffers.

        buffer_radius:
            Radius for buffering, used if geometry_type is 'POLYGON'
            and 'vertices_col' is not provided, or for 'POINT' if spot_diameter is given.

        vertices_col:
            Name of a column in `coords_df` that contains lists of
            coordinate tuples for creating POLYGON or MULTIPOINT geometries.
            For POLYGON: [(x1,y1), (x2,y2), ..., (xn,yn)]
            For MULTIPOINT: [(x1,y1), (x2,y2)] or list of Point objects.

        geometry_type:
            Type of geometry to create ('POINT', 'POLYGON', 'MULTIPOINT').

        end_cap_style:
            Style of the end cap for buffered geometries ("ROUND", "FLAT", "SQUARE").

    Returns:
        GeoDataFrame with geometries.
    """
    if coords_df.empty:
        return gpd.GeoDataFrame(geometry=[])

    if spatial_coordinates_names is None:
        if coords_df.shape[1] >= 2:
            spatial_coordinates_names = list(coords_df.columns[:2])
        elif coords_df.shape[1] == 1:
            # Single dimension, treat as x, y=0
            spatial_coordinates_names = [coords_df.columns[0], "_y_dummy"]
            coords_df["_y_dummy"] = 0.0
        else:
            spatial_coordinates_names = [f"V{i + 1}" for i in range(coords_df.shape[1])]

    x_col, y_col = spatial_coordinates_names[:2]

    geometries = []
    cap_style_map = {
        "ROUND": CAP_STYLE.round,
        "FLAT": CAP_STYLE.flat,
        "SQUARE": CAP_STYLE.square,
    }
    selected_cap_style = cap_style_map.get(end_cap_style.upper(), CAP_STYLE.round)

    if geometry_type == "POINT":
        geometries = [Point(x, y) for x, y in zip(coords_df[x_col], coords_df[y_col])]

        if spot_diameter is not None and not np.isnan(spot_diameter):
            radius = spot_diameter / 2
            geometries = [g.buffer(radius, cap_style=selected_cap_style) for g in geometries]
    elif geometry_type == "MULTIPOINT":
        if vertices_col and vertices_col in coords_df.columns:
            for nverts in coords_df[vertices_col]:
                if isinstance(nverts, list) and all(isinstance(p, (tuple, Point)) for p in nverts):
                    geometries.append(MultiPoint([p if isinstance(p, Point) else Point(p) for p in nverts]))
                else:
                    warn(f"Malformed vertices in column '{vertices_col}'. Creating empty multipoint.", UserWarning)
                    geometries.append(MultiPoint())
        else:
            warn(
                "Creating MULTIPOINT from single (x,y) pairs. For complex MultiPoints, use 'vertices_col'.", UserWarning
            )
            geometries = [MultiPoint([(x, y)]) for x, y in zip(coords_df[x_col], coords_df[y_col])]
    elif geometry_type == "POLYGON":
        if vertices_col and vertices_col in coords_df.columns:
            for nverts in coords_df[vertices_col]:
                if isinstance(nverts, list) and all(isinstance(p, tuple) and len(p) == 2 for p in nverts):
                    geometries.append(Polygon(nverts))
                else:
                    warn(f"Malformed vertices in column '{vertices_col}'. Creating empty polygon.", UserWarning)
                    geometries.append(Polygon())
        else:
            radius = buffer_radius
            if spot_diameter is not None and not np.isnan(spot_diameter):
                radius = spot_diameter / 2
            geometries = [
                Point(x, y).buffer(radius, cap_style=selected_cap_style)
                for x, y in zip(coords_df[x_col], coords_df[y_col])
            ]
    else:
        raise ValueError(f"Unsupported geometry type: '{geometry_type}'.")

    gdf = gpd.GeoDataFrame(coords_df.copy(), geometry=geometries)
    if "_y_dummy" in gdf.columns:
        gdf = gdf.drop(columns=["_y_dummy"])

    return gdf


def df_dict_to_gdf_dict(
    df_dict,
    spatial_coordinates_names: list = None,
    geometry_type: Literal["POINT", "POLYGON", "MULTIPOINT"] = "POINT",
    spot_diameter: float = None,
    buffer_radius: float = 1.0,
    vertices_col: Optional[str] = None,
    end_cap_style: Literal["ROUND", "FLAT", "SQUARE"] = "ROUND",
) -> Dict[str, gpd.GeoDataFrame]:
    """Convert a list of DataFrames to a list of GeoPandas DataFrames.

    Args:
        df_dict:
            Dictionary of DataFrames with coordinates.

        spatial_coordinates_names:
            Names of the coordinate columns.

        spot_diameter:
            Diameter of spots for creating buffers.

        geometry_type:
            Type of geometry to create ('POINT', 'POLYGON', 'MULTIPOINT').

        buffer_radius:
            Radius for buffering, used if geometry_type is 'POLYGON'
            and 'vertices_col' is not provided, or for 'POINT' if spot_diameter is given.

        vertices_col:
            Name of a column in `coords_df` that contains lists of
            coordinate tuples for creating POLYGON or MULTIPOINT geometries.
            For POLYGON: [(x1,y1), (x2,y2), ..., (xn,yn)]
            For MULTIPOINT: [(x1,y1), (x2,y2)] or list of Point objects.

        end_cap_style:
            Style of the end cap for buffered geometries.

    Returns:
        Dictionary of GeoDataFrames.
    """
    if not isinstance(df_dict, dict):
        raise ValueError("'df_dict' must be a dictionary.")

    result = {}
    for name, df in df_dict.items():
        result[name] = dataframe_to_geopandas(
            df,
            spatial_coordinates_names=spatial_coordinates_names,
            spot_diameter=spot_diameter,
            geometry_type=geometry_type,
            buffer_radius=buffer_radius,
            vertices_col=vertices_col,
            end_cap_style=end_cap_style,
        )

    return result


def spatial_coords_to_col_geometries(
    coords: BiocFrame, spot_diameter: float = None, end_cap_style: Literal["ROUND", "FLAT", "SQUARE"] = "ROUND"
):
    """Convert spatial coordinates to column geometries.

    Args:
        coords:
            Spatial coordinates DataFrame.

        spot_diameter:
            Diameter of spots.

        end_cap_style:
            Style for buffered spot polygons.

    Returns:
        GeoDataFrame with geometries.
    """

    if isinstance(coords, BiocFrame):
        coords = coords.to_pandas()
    elif isinstance(coords, np.ndarray):
        coords = pd.DataFrame(coords)
    elif isinstance(coords, pd.DataFrame):
        coords_df = coords.copy()
    else:
        raise TypeError("'coords' is an unsupported object. Must be a dataframe or ndarray.")

    if coords_df.columns.empty and coords_df.shape[1] > 0:
        coords_df.columns = [f"spatial{i + 1}" for i in range(coords_df.shape[1])]

    cg_sfc = dataframe_to_geopandas(
        coords,
        spatial_coords_names=list(coords.columns),
        spot_diameter=spot_diameter,
        geometry_type="POINT",
        end_cap_style=end_cap_style,
    )

    return cg_sfc


def spe_to_sfe(
    spe: SpatialExperiment,
    row_geometries: Optional[Dict[str, gpd.GeoDataFrame]] = None,
    column_geometries: Optional[Dict[str, gpd.GeoDataFrame]] = None,
    annotation_geometries: Optional[Dict[str, gpd.GeoDataFrame]] = None,
    spatial_coordinates_names: list = None,
    row_geometry_type: Literal["POINT", "POLYGON", "MULTIPOINT"] = "POINT",
    annotation_geometry_type: Literal["POINT", "POLYGON", "MULTIPOINT"] = "POLYGON",
    vertices_col_row: Optional[str] = None,
    vertices_col_annot: Optional[str] = None,
    buffer_radius_row: float = 1.0,
    buffer_radius_annot: float = 1.0,
    spatial_graphs: Optional[Dict[str, Union[Graph, Any]]] = None,
    spot_diameter: float = None,
    unit: str = None,
    end_cap_style: Literal["ROUND", "FLAT", "SQUARE"] = "ROUND",
    add_centroids_if_spots_are_polygons: bool = True,
):
    """Convert a SpatialExperiment to a SpatialFeatureExperiment.

    Args:
        spe:
            SpatialExperiment object.

        column_geometries:
            Column geometries.

        row_geometries:
            Row geometries.

        annotation_geometries:
            Annotation geometries.

        spatial_coordinates_names:
            Names of spatial coordinates.

        row_geometry_type:
            Default geometry type for
            row_geometries if converting from DataFrame.

        annotation_geometry_type:
            Default geometry type for annot_geometries
            if converting from DataFrame.

        vertices_col_row:
            Column name with vertex lists for complex row geometries.

        vertices_col_annot:
            Column name with vertex lists for complex annotation geometries.

        buffer_radius_row:
            Buffer radius for simple row polygons.

        buffer_radius_annot:
            Buffer radius for simple annotation polygons.

        spatial_graphs:
            Spatial graphs.

        spot_diameter:
            Diameter of spots.

        unit:
            Unit of measurement.

        end_cap_style:
            Style of end cap for buffered geometries.

        add_centroids:
            Whether to add centroids to column geometries.

        add_centroids_if_spots_are_polygons:
            If spot diameter is set, also add 'centroids'.

    Returns:
        SpatialFeatureExperiment object.
    """

    col_geometries_final = {}
    row_geometries_final = {}
    annot_geometries_final = {}

    if column_geometries:
        for name, geom_input in column_geometries.items():
            if isinstance(geom_input, gpd.GeoDataFrame):
                col_geometries_final[name] = geom_input
            elif isinstance(geom_input, pd.DataFrame):
                col_geometries_final[name] = dataframe_to_geopandas(
                    geom_input,
                    spatial_coordinates_names=spatial_coordinates_names or list(geom_input.columns[:2]),
                    geometry_type="POINT",
                    spot_diameter=spot_diameter if name == "spotPoly" else None,
                    end_cap_style=end_cap_style,
                )
            else:
                warn(
                    f"Skipping invalid column geometry input '{name}'. Must be DataFrame or GeoDataFrame.", UserWarning
                )

    # If no user provided col_geoms, derive from spe.spatial_coordinates
    if not col_geometries_final:
        se_spatial_coords = spe.get_spatial_coordinates()
        if se_spatial_coords is not None and not se_spatial_coords.empty:
            use_spot_diameter = spot_diameter if spot_diameter is not None and not np.isnan(spot_diameter) else None

            default_col_geom_name = "spotPoly" if use_spot_diameter else "centroids"

            col_geometries_final[default_col_geom_name] = spatial_coords_to_col_geometries(
                se_spatial_coords, spot_diameter=use_spot_diameter, end_cap_style=end_cap_style
            )

            if add_centroids_if_spots_are_polygons and default_col_geom_name == "spotPoly":
                if "centroids" not in col_geometries_final:
                    col_geometries_final["centroids"] = spatial_coords_to_col_geometries(
                        se_spatial_coords,
                        spot_diameter=None,
                    )
        else:
            warn("No user-provided column_geometries and SpatialExperiment has no spatial_coordinates.", UserWarning)

    if row_geometries:
        for name, geom_input in row_geometries.items():
            if isinstance(geom_input, gpd.GeoDataFrame):
                row_geometries_final[name] = geom_input
            elif isinstance(geom_input, pd.DataFrame):
                row_geometries_final[name] = dataframe_to_geopandas(
                    geom_input,
                    spatial_coordinates_names=spatial_coordinates_names,
                    geometry_type=row_geometry_type,
                    buffer_radius=buffer_radius_row,
                    vertices_col=vertices_col_row,
                    end_cap_style=end_cap_style,
                )
            else:
                warn(f"Skipping invalid row geometry input '{name}'. Must be DataFrame or GeoDataFrame.", UserWarning)

    if annotation_geometries:
        for name, geom_input in annotation_geometries.items():
            if isinstance(geom_input, gpd.GeoDataFrame):
                annot_geometries_final[name] = geom_input
            elif isinstance(geom_input, pd.DataFrame):
                annot_geometries_final[name] = dataframe_to_geopandas(
                    geom_input,
                    spatial_coordinates_names=spatial_coordinates_names,
                    geometry_type=annotation_geometry_type,
                    buffer_radius=buffer_radius_annot,
                    vertices_col=vertices_col_annot,
                    end_cap_style=end_cap_style,
                )
            else:
                warn(
                    f"Skipping invalid annotation geometry input '{name}'. Must be DataFrame or GeoDataFrame.",
                    UserWarning,
                )

    # Handle image data
    _images = spe.get_image_data()
    if len(_images) > 0:
        new_image_objects_list = []
        valid_img_indices = []

        if "data" not in _images.get_column_names():
            raise ValueError("SpatialExperiment img_data BiocFrame must contain a 'data' column with image objects.")

        if "scaleFactor" not in _images.get_column_names():
            warn(
                "'scaleFactor' column missing in SpatialExperiment.img_data. Assuming scale factor of 1.0 for all images.",
                UserWarning,
            )
            _images.set_column(column="scaleFactor", value=[1.0] * len(_images), in_place=True)

        if "image_id" not in _images.get_column_names():
            warn("'image_id' column missing in SpatialExperiment.img_data. Generating generic IDs.", UserWarning)
            _images["image_id"] = [f"image_{k}" for k in range(len(_images))]

        if "sample_id" not in _images.get_column_names():
            # Try to get sample_id from colData if all spots belong to one sample, or use a default
            if len(spe.get_column_data()["sample_id"].unique()) == 1:
                unique_sample_id = spe.get_column_data()["sample_id"].unique()[0]
                _images.set_column(column="sample_id", value=[unique_sample_id] * len(_images), in_place=True)
                warn(f"Using unique sample_id '{unique_sample_id}' for all images from colData.", UserWarning)
            else:
                warn(
                    "'sample_id' column missing in SpatialExperiment.img_data and multiple sample_ids in colData. Using default 'unknown_sample'.",
                    UserWarning,
                )
                _images.set_column(column="sample_id", value=["unknown_sample"] * len(_images), in_place=True)

        for i in range(len(_images)):
            img_obj_se = _images.get_column("data")[i]
            scale_factor = _images.get_column("scaleFactor")[i]
            image_id = _images.get_column("image_id")[i]
            sample_id = _images.get_column("sample_id")[i]

            sfe_img_obj = None

            try:
                # SFE image types
                if isinstance(img_obj_se, (SpatRasterImage, ExtImage, BioFormatsImage)):
                    sfe_img_obj = img_obj_se
                elif isinstance(img_obj_se, LoadedSpatialImage):
                    img_array_raw = np.array(img_obj_se.get_image())  # Typically HWC or HW

                    if img_array_raw.ndim == 2:  # Grayscale (H, W)
                        img_array_chw = img_array_raw[np.newaxis, :, :]  # Add channel dim -> (1, H, W)
                    elif img_array_raw.ndim == 3:  # Color (H, W, C)
                        img_array_chw = np.transpose(img_array_raw, (2, 0, 1))  # HWC to CHW
                    else:
                        warn(
                            f"Image '{image_id}' has unsupported array dimensions {img_array_raw.shape}. Dropping.",
                            UserWarning,
                        )
                        continue

                    # TODO: VERIFY THIS
                    img_transform = Affine.scale(scale_factor, scale_factor)

                    memfile = MemoryFile()
                    with memfile.open(
                        driver="GTiff",
                        height=img_array_chw.shape[1],
                        width=img_array_chw.shape[2],
                        count=img_array_chw.shape[0],
                        dtype=str(img_array_chw.dtype),
                        transform=img_transform,
                    ) as dataset:
                        dataset.write(img_array_chw)
                    # Store the MemoryFile itself
                    # sfe_img_obj = memfile
                    sfe_img_obj = SpatRasterImage(
                        image_id=image_id, sample_id=sample_id, data=memfile, transform=img_transform
                    )

                elif isinstance(img_obj_se, (RemoteSpatialImage, StoredSpatialImage)):
                    img_path = img_obj_se.img_source()
                    with rasterio.open(img_path) as src:
                        src_transform = src.transform
                        # asuming its what flip does from the R/Bioc implementation
                        # TODO: VERIFY THIS
                        final_transform = Affine(
                            scale_factor,
                            src_transform.b,
                            src_transform.c,
                            src_transform.d,
                            -scale_factor if src_transform.e < 0 else scale_factor,
                            src_transform.f,
                        )

                        profile = src.profile
                        profile["transform"] = final_transform
                        profile["driver"] = "GTiff"

                        img_array = src.read()

                    memfile = MemoryFile()
                    with memfile.open(**profile) as dataset:
                        dataset.write(img_array)
                    # sfe_img_obj = memfile
                    sfe_img_obj = SpatRasterImage(
                        image_id=image_id,
                        sample_id=sample_id,
                        data=memfile,
                        transform=final_transform,
                        crs=profile["crs"],
                    )

                else:
                    warn(f"Image '{image_id}' type {type(img_obj_se)} not convertible. Dropping.", UserWarning)
                    continue

                if sfe_img_obj is not None:
                    new_image_objects_list.append(sfe_img_obj)
                    valid_img_indices.append(i)

            except Exception as e:
                warn(f"Error processing image '{image_id}' (index {i}): {e}. Dropping.", RuntimeWarning)

        if valid_img_indices:
            sfe_img_data_bf = _images.get_slice(valid_img_indices, slice(None)).copy()

            sfe_img_data_bf.set_column(column="data", value=new_image_objects_list, in_place=True)
        else:
            sfe_img_data_bf = BiocFrame(nrows=0, column_names=_images.get_column_names())

    sfe_spatial_coords = spe.get_spatial_coordinates()

    alt_exps_final = spe.get_alternative_experiments()
    if len(alt_exps_final) > 0:
        warn("Ignoring alternative experiments in the spatial experiment object.", UserWarning)

    from .sfe import SpatialFeatureExperiment

    sfe = SpatialFeatureExperiment(
        assays=spe.get_assays(),
        row_ranges=spe.get_row_ranges(),
        row_data=spe.get_row_data(),
        column_data=spe.get_column_data(),
        row_names=spe.get_row_names(),
        column_names=spe.get_column_names(),
        metadata=spe.get_metadata(),
        reduced_dims=spe.get_reduced_dims(),
        main_experiment_name=spe.get_main_experiment_name(),
        alternative_experiments=alt_exps_final,
        spatial_coords=sfe_spatial_coords,
        img_data=sfe_img_data_bf,
        col_geometries=col_geometries_final,
        row_geometries=row_geometries_final,
        annot_geometries=annot_geometries_final,
        spatial_graphs=spatial_graphs,
        unit=unit,
    )

    return sfe
