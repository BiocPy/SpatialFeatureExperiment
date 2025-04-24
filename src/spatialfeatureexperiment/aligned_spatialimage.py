import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from warnings import warn

import biocutils as ut
import numpy as np
import rasterio
import rasterio.transform
from PIL import Image
from spatialexperiment.SpatialImage import VirtualSpatialImage

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def _validate_extent(extent):
    required_keys = ["xmin", "xmax", "ymin", "ymax"]
    if not all(k in extent for k in required_keys):
        raise ValueError(f"Extent must contain keys: {', '.join(required_keys)}.")

    if extent["xmin"] >= extent["xmax"] or extent["ymin"] >= extent["ymax"]:
        raise ValueError("Invalid extent: xmin must be < xmax and ymin must be < ymax.")


class AlignedSpatialImage(VirtualSpatialImage):
    """Base class for spatial images with extent.

    All images in `SpatialFeatureExperiment` have an extent in spatial coordinates.
    """

    def __init__(self, metadata: Optional[dict] = None):
        super().__init__(metadata=metadata)

    def get_extent(self) -> Dict[str, float]:
        """Get the extent of the image"""
        raise NotImplementedError("Subclasses must implement `get_extent`")

    def set_extent(self, extent: Dict[str, float], in_place: bool = False):
        """Set the extent of the image"""
        raise NotImplementedError("Subclasses must implement `set_extent`")

    @property
    def extent(self) -> Dict[str, float]:
        """Alias for :py:attr:`~get_extent`."""
        return self.get_extent()

    @extent.setter
    def extent(self, value: Dict[str, float]):
        """Alias for :py:attr:`~set_extent` with ``in_place = True``.

        As this mutates the original object, a warning is raised.
        """
        warn(
            "Setting property 'extent' is an in-place operation, use 'set_extent' instead",
            UserWarning,
        )
        self.set_extent(value, in_place=False)


class SpatRasterImage(AlignedSpatialImage):
    """`SpatRaster` representation of images in SpatialFeatureExperiment objects.

    This class is a wrapper around rasterio for handling GeoTIFF images.
    """

    def __init__(
        self,
        image: Union[rasterio.DatasetReader, np.ndarray],
        extent: Optional[Dict[str, float]] = None,
        metadata: Optional[dict] = None,
    ):
        """Initialize a `SpatRasterImage`.

        Args:
            image:
                The image data, either as a rasterio dataset or a numpy array.

            extent:
                The spatial extent of the image, by default None.
                Must have keys: 'xmin', 'xmax', 'ymin', 'ymax'.

            metadata:
                Additional image metadata. Defaults to None.
        """
        super().__init__(metadata=metadata)

        # Handle numpy array
        if isinstance(image, np.ndarray):
            if extent is None:
                extent = {"xmin": 0, "xmax": image.shape[1], "ymin": 0, "ymax": image.shape[0]}

            # Create a memory-based rasterio dataset
            self._src = self.numpy_array_to_rasterio(image, extent)
            self._in_memory = True
            self._img_source = None

        # Handle rasterio dataset
        elif isinstance(image, rasterio.DatasetReader):
            self._src = image
            self._in_memory = False

            # Get source path if available
            try:
                self._img_source = image.name
            except Exception as _:
                self._img_source = None

            # Get extent from transform if not provided
            if extent is None:
                bounds = image.bounds
                extent = {"xmin": bounds.left, "xmax": bounds.right, "ymin": bounds.bottom, "ymax": bounds.top}
        else:
            raise ValueError("img must be a rasterio dataset or numpy array.")

        # Store the extent
        self._extent = extent

    def numpy_array_to_rasterio(self, array, extent):
        """Convert a numpy array to a rasterio dataset."""
        from rasterio.io import MemoryFile

        # Handle different dimensions
        if len(array.shape) == 2:
            height, width = array.shape
            count = 1
        elif len(array.shape) == 3:
            height, width, count = array.shape
        else:
            raise ValueError("Array must be 2D or 3D")

        # Create transform
        xres = (extent["xmax"] - extent["xmin"]) / width
        yres = (extent["ymax"] - extent["ymin"]) / height
        transform = rasterio.transform.from_origin(extent["xmin"], extent["ymax"], xres, yres)

        # Create memory file
        memfile = MemoryFile()
        with memfile.open(
            driver="GTiff",
            height=height,
            width=width,
            count=count,
            dtype=array.dtype,
            transform=transform,
        ) as dataset:
            if count == 1:
                dataset.write(array, 1)
            else:
                for i in range(count):
                    dataset.write(array[:, :, i], i + 1)

        return memfile.open()

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, "_src") and self._src is not None:
            self._src.close()

    ##########################
    ######>> Printing <<######
    ##########################

    def __repr__(self):
        """String representation."""
        shape = self.shape

        output = f"{type(self).__name__}("

        if len(shape) == 3:
            output += f"{shape[1]} x {shape[0]} x {shape[2]} (width x height x channels)"
        else:
            output += f"{shape[1]} x {shape[0]} (width x height)"

        if len(self._metadata) > 0:
            output += ", metadata=" + ut.print_truncated_dict(self._metadata)
        output += ")"

        return output

    def __str__(self) -> str:
        """
        Returns:
            A pretty-printed string containing the contents of this object.
        """
        output = f"class: {type(self).__name__}\n"

        shape = self.shape
        if len(shape) == 3:
            output += f"{shape[1]} x {shape[0]} x {shape[2]} (width x height x channels)\n"
        else:
            output += f"{shape[1]} x {shape[0]} (width x height)\n"

        output += f"metadata({str(len(self.metadata))}): {ut.print_truncated_list(list(self.metadata.keys()), sep=' ', include_brackets=False, transform=lambda y: y)}\n"
        return output

    ###########################
    ######>> accessors <<######
    ###########################

    def img_source(self) -> str:
        """Get the source file path if available."""
        return self._img_source

    def get_extent(self) -> Dict[str, float]:
        """Get the extent of the image."""
        return self._extent

    def set_extent(self, extent: Dict[str, float], in_place: bool = False):
        """Set the extent of the image."""
        self._extent = extent

        # Update the transform if it's a rasterio dataset
        if hasattr(self, "_src") and self._src is not None:
            width = self._src.width
            height = self._src.height
            xres = (extent["xmax"] - extent["xmin"]) / width
            yres = (extent["ymax"] - extent["ymin"]) / height
            self._src.transform = rasterio.transform.from_origin(extent["xmin"], extent["ymax"], xres, yres)

    def img_raster(self, window: Optional[tuple] = None, out_shape: Optional[tuple] = None, resampling: int = 0):
        """Load the image.

        Args:
            window:
                Window to read, by default None (read entire image).

            out_shape:
                Output shape, by default None (no resampling).

            resampling:
                Resampling method, by default 0 (nearest).

        Returns:
            Image data.
        """
        if hasattr(self, "_src") and self._src is not None:
            return self._src.read(window=window, out_shape=out_shape, resampling=resampling)
        return None

    @property
    def shape(self):
        return self.get_dimensions()

    def get_dimensions(self):
        """Get the shape of the image (height, width, channels)."""
        if hasattr(self, "_src") and self._src is not None:
            return (self._src.height, self._src.width, self._src.count)
        return None

    @property
    def in_memory(self):
        """Check if the image is in memory."""
        return self._in_memory

    @property
    def array(self):
        """Get the image as a numpy array (loads into memory)"""
        if hasattr(self, "_src") and self._src is not None:
            if self._src.count == 1:
                return self._src.read(1)
            else:
                # Read all bands and stack
                return np.dstack([self._src.read(i + 1) for i in range(self._src.count)])
        return None


class BioFormatsImage(AlignedSpatialImage):
    """On-disk representation of BioFormats images in `SpatialFeatureExperiment` objects.

    This class is designed for OME-TIFF files and other formats supported by aicsimageio.
    """

    def __init__(
        self,
        path: Union[str, Path],
        extent: Dict[str, float] = None,
        is_full: bool = True,
        origin: List[float] = [0.0, 0.0],
        tranformation: dict = None,
        metadata: dict = None,
        validate: bool = False,
    ):
        """Initialize the image.

        Args:
            path:
                Path to the image file.

            extent:
                The spatial extent of the image, by default None.
                Must have keys: 'xmin', 'xmax', 'ymin', 'ymax'.

            is_full:
                Whether if this the full extent of the image.
                Defaults to True.

            origin:
                Origin of the image in spatial coordinates.
                Defaults to (0, 0).

            tranformation:
                Affine transformation.
                Defaults to None.

            metadata:
                Additional image metadata. Defaults to None.

            validate:
                Internal use only.
        """
        super().__init__(metadata)

        self._path = path

        if extent is None:
            self._extent = self._get_full_extent()
        else:
            self._extent = extent

        self._is_full = is_full
        self._origin = [0.0, 0.0] if origin is None else origin

        self._transformation = tranformation if tranformation is not None else {}

        if validate:
            self.validate()

    def validate(self):
        """Validate the object."""

        # Validate the path
        if not os.path.exists(self._path):
            raise FileNotFoundError(f"Image file not found: '{self._path}'.")

        # Validate extent
        _validate_extent(self._extent)

        # Validate origin
        if len(self.origin) != 2 or not all(isinstance(x, (int, float)) for x in self.origin):
            raise ValueError("origin must be a numeric vector of length 2.")

        # Validate transformation
        if self.transformation:
            warn("transformations are not validated.")

    def _get_pixel_size(self):
        """Get the physical pixel size from the metadata."""

        from aicsimageio import AICSImage

        try:
            with AICSImage(self.path) as img:
                physical_pixel_size = img.physical_pixel_sizes

                if physical_pixel_size and len(physical_pixel_size) >= 2:
                    return [physical_pixel_size[0], physical_pixel_size[1]]

            warn("Physical pixel size not found in metadata. Using pixel space.")
            return [1.0, 1.0]
        except Exception as e:
            warn(f"Error reading OME-TIFF metadata: {e}. Using pixel space.")
            return [1.0, 1.0]

    def _get_full_size(self):
        """Get the full size of the image in pixels"""

        from aicsimageio import AICSImage

        try:
            with AICSImage(self.path) as img:
                shape = img.shape

                if len(shape) >= 2:
                    # AICSImage returns shape as (t, z, y, x, c)
                    # We need x (width) and y (height)
                    return [shape[-2], shape[-3]]  # x, y (width, height)

            warn("Image size not found in metadata.")
            return [0, 0]
        except Exception as e:
            warn(f"Error reading OME-TIFF metadata: {e}.")
            return [0, 0]

    def _get_full_extent(self):
        """Get the full extent of the image."""
        scale_factors = self._get_pixel_size()
        sfx, sfy = scale_factors

        size_full = self._get_full_size()
        size_x, size_y = size_full

        return {"xmin": 0, "ymin": 0, "xmax": size_x / sfx, "ymax": size_y / sfy}

    ##########################
    ######>> Printing <<######
    ##########################

    def __repr__(self):
        """String representation"""
        dims = self.dimensions

        output = f"{type(self).__name__}("

        output += f"X: {dims[0]}, Y: {dims[1]}, C: {dims[2]}, Z: {dims[3]}, T: {dims[4]}"

        if len(self._metadata) > 0:
            output += ", metadata=" + ut.print_truncated_dict(self._metadata)
        output += ")"

        return output

    def __str__(self) -> str:
        """
        Returns:
            A pretty-printed string containing the contents of this object.
        """
        output = f"class: {type(self).__name__}\n"

        dims = self.dimensions
        output += f"X: {dims[0]}, Y: {dims[1]}, C: {dims[2]}, Z: {dims[3]}, T: {dims[4]}\n"

        output += f"metadata({str(len(self.metadata))}): {ut.print_truncated_list(list(self.metadata.keys()), sep=' ', include_brackets=False, transform=lambda y: y)}\n"
        return output

    ###########################
    ######>> accessors <<######
    ###########################

    def img_source(self) -> str:
        """Get the source file path"""
        return self.path

    def get_extent(self) -> Dict[str, float]:
        """Get the extent of the image, applying any transformations"""
        # if self.transformation:
        #     return self._transform_bbox(self._extent)
        return self._extent

    def set_extent(self, extent: Dict[str, float], in_place: bool = False):
        """Set the extent of the image."""
        _validate_extent(extent)
        self._extent = extent

    def to_ext_image(self, resolution=4, channel=None):
        raise NotImplementedError("Not yet implemented!")

    def img_raster(self, resolution=4, channel=None):
        """Load the image.

        Returns:
            Image data.
        """
        return self.to_ext_image(resolution=resolution, channel=channel)

    def get_dimensions(self):
        """Get the dimensions of the image (X, Y, C, Z, T)"""

        from aicsimageio import AICSImage

        try:
            with AICSImage(self.path) as img:
                shape = img.shape

                # AICSImage dimensions order is (T, Z, Y, X, C)
                # Convert to (X, Y, C, Z, T) for compatibility with R's BioFormatsImage
                if len(shape) == 5:
                    t, z, y, x, c = shape
                    return [x, y, c, z, t]
                elif len(shape) == 4:
                    # Handle 4D images (assume missing T dimension)
                    z, y, x, c = shape
                    return [x, y, c, z, 1]
                elif len(shape) == 3:
                    # Handle 3D images (assume missing Z and T dimensions)
                    y, x, c = shape
                    return [x, y, c, 1, 1]
                elif len(shape) == 2:
                    # Handle 2D images (assume single channel, Z and T)
                    y, x = shape
                    return [x, y, 1, 1, 1]
                else:
                    warn(f"Unexpected image shape: {shape}")
                    return [0, 0, 0, 0, 0]

        except Exception as e:
            warn(f"Error reading OME-TIFF metadata: {e}")
            return [0, 0, 0, 0, 0]

    @property
    def shape(self):
        return self.get_dimensions()


class ExtImage(AlignedSpatialImage):
    """Use the PIL/numpy arrays in SpatialFeatureExperiment objects.

    This class is a wrapper around `PIL.Image` or numpy arrays with spatial extent information.
    """

    def __init__(
        self,
        image: Union[Image.Image, np.ndarray],
        extent: Optional[Dict[str, float]] = None,
        metadata: Optional[dict] = None,
    ):
        """Initialize an ExtImage.

        Args:
            image:
                The image data.

            extent:
                The spatial extent of the image, by default None.
                Must have keys: 'xmin', 'xmax', 'ymin', 'ymax'.

            metadata:
                Additional image metadata. Defaults to None.
        """
        super().__init__(metadata=metadata)

        if isinstance(image, np.ndarray):
            self._array = image
        elif isinstance(image, Image.Image):
            self._array = np.array(image)
        else:
            raise ValueError("img must be a PIL Image or numpy array")

        if extent is None:
            raise ValueError("Extent must be specified for ExtImage.")

        _validate_extent(extent)
        self._extent = extent

    ##########################
    ######>> Printing <<######
    ##########################

    def __repr__(self):
        """String representation."""
        shape = self.shape

        output = f"{type(self).__name__}("

        if len(shape) == 3:
            output += f"{shape[1]} x {shape[0]} x {shape[2]} (width x height x channels)"
        else:
            output += f"{shape[1]} x {shape[0]} (width x height)"

        if len(self._metadata) > 0:
            output += ", metadata=" + ut.print_truncated_dict(self._metadata)
        output += ")"

        return output

    def __str__(self) -> str:
        """
        Returns:
            A pretty-printed string containing the contents of this object.
        """
        output = f"class: {type(self).__name__}\n"

        shape = self.shape
        if len(shape) == 3:
            output += f"{shape[1]} x {shape[0]} x {shape[2]} (width x height x channels)\n"
        else:
            output += f"{shape[1]} x {shape[0]} (width x height)\n"

        output += f"metadata({str(len(self.metadata))}): {ut.print_truncated_list(list(self.metadata.keys()), sep=' ', include_brackets=False, transform=lambda y: y)}\n"
        return output

    ###########################
    ######>> accessors <<######
    ###########################

    def img_source(self) -> str:
        """Get the source file path."""
        return None

    def get_extent(self) -> Dict[str, float]:
        """Get the extent of the image."""
        return self._extent

    def set_extent(self, extent: Dict[str, float], in_place: bool = False):
        """Set the extent of the image."""
        _validate_extent(extent)
        self._extent = extent

    @property
    def array(self):
        """Get the image as a numpy array."""
        return self._array

    @property
    def shape(self):
        """Get the shape of the image (height, width, channels)."""
        return self.get_dimensions()

    def get_dimensions(self):
        """Get the shape of the image (height, width, channels)."""
        return self._array.shape

    def to_pil(self):
        """Convert to PIL Image."""
        return Image.fromarray(self._array)

    def img_raster(self) -> Image.Image:
        """Load the image.

        Returns:
            Image data.
        """
        return self.to_pil()
