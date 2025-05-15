import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from warnings import warn

import biocutils as ut
import numpy as np
import rasterio
import rasterio.transform
from PIL import Image
from spatialexperiment import VirtualSpatialImage

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def _validate_extent(extent: Dict[str, float]):
    required_keys = ["xmin", "xmax", "ymin", "ymax"]
    if not all(k in extent for k in required_keys):
        raise ValueError(f"Extent must contain keys: {', '.join(required_keys)}.")

    if extent["xmin"] >= extent["xmax"] or extent["ymin"] >= extent["ymax"]:
        raise ValueError("Invalid extent: xmin must be < xmax and ymin must be < ymax.")


def _transform_extent(extent: Dict[str, float], affine_matrix: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Transforms an extent (bounding box) by an affine matrix.

    If no matrix is provided, returns the original extent.

    Args:
        extent:
            The extent dictionary {'xmin', 'xmax', 'ymin', 'ymax'}.

        affine_matrix:
            A 2x3 or 3x3 numpy array representing the affine transformation.
            If None, original extent is returned.

    Returns:
        The transformed extent.
    """
    if affine_matrix is None:
        return extent.copy()

    xmin, xmax, ymin, ymax = extent["xmin"], extent["xmax"], extent["ymin"], extent["ymax"]
    # define the four corners of the bounding box
    corners = np.array([[xmin, ymin, 1], [xmax, ymin, 1], [xmin, ymax, 1], [xmax, ymax, 1]])

    # affine_matrix is 3x3 for homogeneous coordinates
    if affine_matrix.shape == (2, 3):
        mat_3x3 = np.vstack([affine_matrix, [0, 0, 1]])
    elif affine_matrix.shape == (3, 3):
        mat_3x3 = affine_matrix
    elif affine_matrix.shape == (2, 2):
        # rotation/scale/shear only
        mat_3x3 = np.eye(3)
        mat_3x3[:2, :2] = affine_matrix
    else:
        raise ValueError("Affine matrix must be 2x3, 3x3, or 2x2.")

    # Apply the transformation to the corners
    # We need to transpose corners to be (3, N) for matrix multiplication, then transpose back
    transformed_corners = (mat_3x3 @ corners.T).T

    # Find the new min and max coordinates
    new_xmin = np.min(transformed_corners[:, 0])
    new_xmax = np.max(transformed_corners[:, 0])
    new_ymin = np.min(transformed_corners[:, 1])
    new_ymax = np.max(transformed_corners[:, 1])

    return {"xmin": new_xmin, "xmax": new_xmax, "ymin": new_ymin, "ymax": new_ymax}


class AlignedSpatialImage(VirtualSpatialImage):
    """Base class for spatial images with extent.

    All images in `SpatialFeatureExperiment` have an extent in spatial coordinates.
    """

    def __init__(self, metadata: Optional[dict] = None):
        """Initializes the AlignedSpatialImage.

        Args:
            metadata:
                Optional dictionary of metadata.
        """
        super().__init__(metadata=metadata)
        self._extent = {}

    def get_extent(self) -> Dict[str, float]:
        """Get the spatial extent of the image.

        Subclasses must implement this to return their specific extent.

        Returns:
            A dictionary with keys 'xmin', 'xmax', 'ymin', 'ymax'.
        """
        raise NotImplementedError("Subclasses must implement `get_extent`")

    def set_extent(self, extent: Dict[str, float], in_place: bool = False) -> "AlignedSpatialImage":
        """Set the spatial extent of the image.

        Subclasses must implement this.

        Args:
            extent:
                A dictionary with keys 'xmin', 'xmax', 'ymin', 'ymax'.

            in_place:
                If True, modifies the object in place.
                Otherwise, returns a new instance.

        Returns:
            The modified AlignedSpatialImage.
        """
        raise NotImplementedError("Subclasses must implement `set_extent`")

    @property
    def extent(self) -> Dict[str, float]:
        """Alias for :py:meth:`~get_extent`."""
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
        self.set_extent(value, in_place=True)


class SpatRasterImage(AlignedSpatialImage):
    """`SpatRaster` representation of images in SpatialFeatureExperiment objects.

    This class is a wrapper around rasterio for handling GeoTIFF images,
    aligning with R's `SpatRasterImage` which uses `terra::SpatRaster`.
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
                The spatial extent of the image. Required if `image` is a numpy array.

                If `image` is a `rasterio.DatasetReader`,
                `extent` is inferred if not provided.

                Must have keys: 'xmin', 'xmax', 'ymin', 'ymax'.
            metadata: Additional image metadata. Defaults to None.
        """
        super().__init__(metadata=metadata)
        self._src: Optional[rasterio.DatasetReader] = None
        self._in_memory: bool = False
        self._img_source: Optional[str] = None

        if isinstance(image, np.ndarray):
            if extent is None:
                # In R, SpatRaster from array would typically require extent.
                # For consistency, let's default to pixel coordinates if not given,
                # but ideally, it should be provided.
                warn("Extent not provided for numpy array; defaulting to pixel coordinates (0,0) origin.", UserWarning)
                height, width = image.shape[0], image.shape[1]
                self._extent = {"xmin": 0, "xmax": width, "ymin": 0, "ymax": height}
            else:
                _validate_extent(extent)
                self._extent = extent.copy()

            # Create a memory-based rasterio dataset
            self._src = self._numpy_array_to_rasterio(image, self._extent)
            self._in_memory = True
            self._img_source = "memory"

        elif isinstance(image, rasterio.DatasetReader):
            self._src = image
            self._in_memory = image.name.startswith("MEM") if hasattr(image, "name") else False

            try:
                if not self._in_memory and image.name and os.path.exists(image.name):
                    self._img_source = image.name
                else:
                    self._img_source = "rasterio_dataset_reader"
            except Exception:
                self._img_source = "rasterio_dataset_reader"

            if extent is None:
                bounds = image.bounds
                self._extent = {"xmin": bounds.left, "xmax": bounds.right, "ymin": bounds.bottom, "ymax": bounds.top}
            else:
                _validate_extent(extent)
                self._extent = extent.copy()
                # If extent is provided, we should update the transform of the rasterio object
                width = self._src.width
                height = self._src.height
                x_res = (self._extent["xmax"] - self._extent["xmin"]) / width
                y_res = (self._extent["ymax"] - self._extent["ymin"]) / height
                new_transform = rasterio.transform.from_origin(
                    self._extent["xmin"], self._extent["ymax"], x_res, abs(y_res)
                )
                # rasterio DatasetReader's transform is read-only.
                # To truly set it, we'd need to create a new dataset or operate on a copy in memory.
                # For now, we store the desired extent and use it.
                # If the image is modified, a new rasterio object might be needed.
                if self._src.transform != new_transform:
                    warn(
                        "Provided extent differs from rasterio source's transform. The provided extent will be used. "
                        "Transform operations may reflect this new extent.",
                        RuntimeWarning,
                    )
                    self._src.transform = new_transform

        else:
            raise ValueError("img must be a rasterio.DatasetReader or numpy.ndarray.")

    def _numpy_array_to_rasterio(self, array: np.ndarray, extent: Dict[str, float]) -> rasterio.io.MemoryFile:
        """Converts a numpy array to an in-memory rasterio dataset."""
        from rasterio.io import MemoryFile

        # Grayscale
        if array.ndim == 2:
            height, width = array.shape
            count = 1
            # Add channel dimension for rasterio
            array_for_rasterio = array[np.newaxis, :, :]
        elif array.ndim == 3:  # Channels last (H, W, C)
            height, width, count = array.shape
            # Transpose to (C, H, W) for rasterio
            array_for_rasterio = np.transpose(array, (2, 0, 1))
        else:
            raise ValueError("Array must be 2D (H, W) or 3D (H, W, C)")

        xres = (extent["xmax"] - extent["xmin"]) / width
        yres = (extent["ymax"] - extent["ymin"]) / height
        transform = rasterio.transform.from_origin(extent["xmin"], extent["ymax"], xres, abs(yres))

        memfile = MemoryFile()
        with memfile.open(
            driver="GTiff",
            height=height,
            width=width,
            count=count,
            dtype=str(array.dtype),
            transform=transform,
        ) as dataset:
            dataset.write(array_for_rasterio)

        return memfile.open()

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, "_src") and self._src is not None and not self._src.closed:
            try:
                self._src.close()
            except Exception:
                pass

    ##########################
    ######>> Printing <<######
    ##########################

    def __repr__(self):
        """String representation."""
        dims = self.get_dimensions()
        shape_str = (
            f"{dims[1]} x {dims[0]} x {dims[2]} (width x height x channels)"
            if len(dims) == 3 and dims[2] > 0
            else f"{dims[1]} x {dims[0]} (width x height)"
        )

        output = f"{type(self).__name__}({shape_str}"
        if self._img_source:
            output += f", source='{self._img_source}'"
        if len(self.metadata) > 0:
            output += ", metadata=" + ut.print_truncated_dict(self.metadata)
        output += ")"

        return output

    def __str__(self) -> str:
        """
        Returns:
            A pretty-printed string containing the contents of this object.
        """
        output = f"class: {type(self).__name__}\n"
        dims = self.get_dimensions()
        shape_str = (
            f"{dims[1]} x {dims[0]} x {dims[2]} (width x height x channels)"
            if len(dims) == 3 and dims[2] > 0
            else f"{dims[1]} x {dims[0]} (width x height)"
        )
        output += f"dimensions: {shape_str}\n"
        output += f"extent: xmin={self._extent['xmin']:.2f}, xmax={self._extent['xmax']:.2f}, ymin={self._extent['ymin']:.2f}, ymax={self._extent['ymax']:.2f}\n"
        output += f"in_memory: {self._in_memory}\n"
        if self._img_source:
            output += f"img_source: {self._img_source}\n"
        output += f"metadata({str(len(self.metadata))}): {ut.print_truncated_list(list(self.metadata.keys()), sep=' ', include_brackets=False, transform=lambda y: y)}\n"

        return output

    ###########################
    ######>> accessors <<######
    ###########################

    def img_source(self, as_path: bool = False) -> Optional[str]:
        """Get the source file path if available."""
        if (
            self._in_memory
            or not self._img_source
            or self._img_source == "memory"
            or self._img_source == "rasterio_dataset_reader"
        ):
            return None

        return self._img_source

    def get_extent(self) -> Dict[str, float]:
        """Get the extent of the image."""
        return self._extent.copy()

    def set_extent(self, extent: Dict[str, float], in_place: bool = False) -> "SpatRasterImage":
        """Set the extent of the image."""
        _validate_extent(extent)

        obj = self if in_place else self.copy()
        obj._extent = extent.copy()

        # If the underlying rasterio object exists, its transform might need update.
        # DatasetReader.transform is read-only.
        if obj._src:
            width = obj._src.width
            height = obj._src.height
            x_res = (obj._extent["xmax"] - obj._extent["xmin"]) / width
            y_res = (obj._extent["ymax"] - obj._extent["ymin"]) / height

            warn(
                f"Extent set. Note: The underlying rasterio.DatasetReader's transform is not modified in-place. "
                f"The new extent will be used for future operations originating from this {type(self).__name__} object.",
                UserWarning,
            )
            obj._src.transform = rasterio.transform.from_origin(
                obj._extent["xmin"], obj._extent["ymax"], x_res, abs(y_res)
            )

        return obj

    def copy(self) -> "SpatRasterImage":
        """Creates a copy of the SpatRasterImage."""

        new_metadata = self.metadata.copy()
        new_extent = self._extent.copy()

        if self._img_source and os.path.exists(self._img_source) and not self._in_memory:
            new_src = rasterio.open(self._img_source)
            return SpatRasterImage(image=new_src, extent=new_extent, metadata=new_metadata)
        elif self._in_memory and self._src:
            np_array = self.array
            return SpatRasterImage(image=np_array, extent=new_extent, metadata=new_metadata)
        elif self._src:
            warn(
                "Copying a SpatRasterImage with a complex rasterio.DatasetReader. "
                "The underlying reader is not deeply copied, changes to it might affect the copy."
            )
            return SpatRasterImage(image=self._src, extent=new_extent, metadata=new_metadata)
        else:
            raise RuntimeError("Cannot copy SpatRasterImage: source data is unavailable.")

    def img_raster(
        self,
        window: Optional[rasterio.windows.Window] = None,
        out_shape: Optional[Tuple[int, int, int]] = None,
        resampling_method_str: str = "nearest",
    ) -> np.ndarray:
        """Load the image data as a numpy array.

        Args:
            window:
                A rasterio.windows.Window object to read a subset.

            out_shape:
                Tuple (bands, height, width) for the output array. If None, native shape.

            resampling_method_str:
                Resampling method string (e.g., "nearest", "bilinear").

        Returns:
            Image data as a numpy array (bands, height, width) or (height, width) if single band.
        """
        if self._src is not None:
            from rasterio.enums import Resampling

            resampling = Resampling[resampling_method_str]

            data = self._src.read(window=window, out_shape=out_shape, resampling=resampling)
            if data.shape[0] == 1:
                return data.squeeze(axis=0)
            return data

        raise RuntimeError("Image source (_src) is not available.")

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get the shape of the image (height, width, channels/bands).

        This matches common numpy/PIL dimension order after loading.
        """
        return self.get_dimensions()

    def get_dimensions(self) -> Tuple[int, int, int]:
        """Get the dimensions of the image (height, width, channels/count).

        Returns:
            This method returns (height, width, channels).
        """
        if self._src is not None:
            return (self._src.height, self._src.width, self._src.count)

        raise RuntimeError("Image source (_src) is not available to get dimensions.")

    @property
    def in_memory(self) -> bool:
        """Check if the image is primarily in memory."""
        return self._in_memory

    @property
    def array(self) -> np.ndarray:
        """Get the image as a numpy array (loads into memory if not already).

        Returns:
            NumPy array in (height, width, channels) or (height, width) format.
        """
        if self._src is not None:
            img_data_bands_first = self._src.read()  # (channels, height, width)
            if self._src.count == 1:
                return img_data_bands_first.squeeze(axis=0)  # (height, width)
            else:
                return np.transpose(img_data_bands_first, (1, 2, 0))  # (height, width, channels)

        raise RuntimeError("Image source (_src) is not available.")

    def to_ext_image(
        self, maxcell: Optional[int] = None, channel: Optional[Union[int, List[int]]] = None
    ) -> "ExtImage":
        """Convert this `SpatRasterImage` to an `ExtImage` (in-memory PIL/numpy based).

        Args:
            maxcell:
                Maximum number of pixels for the output `ExtImage`.
                If the original image is larger, it will be downsampled.

            channel:
                Specific channel index or list of indices to select.
                If None, all channels are used.

        Returns:
            An `ExtImage` instance.
        """
        if self._src is None:
            raise RuntimeError("Image source (_src) is not available for conversion.")

        current_height, current_width, num_channels = self.get_dimensions()
        current_pixels = current_height * current_width

        resampling_method = rasterio.enums.Resampling.nearest  # Default for downsampling

        target_height, target_width = current_height, current_width
        if maxcell is not None and current_pixels > maxcell:
            scale_factor = math.sqrt(maxcell / current_pixels)
            target_height = int(current_height * scale_factor)
            target_width = int(current_width * scale_factor)
            warn(
                f"Image downsampled from {current_width}x{current_height} to {target_width}x{target_height} to meet maxcell={maxcell}"
            )

        out_shape: Optional[Tuple[int, int, int]] = None
        if target_height != current_height or target_width != current_width:
            out_shape = (num_channels, target_height, target_width)

        # Select channels
        bands_to_read: Optional[Union[int, List[int]]] = None
        if channel is not None:
            if isinstance(channel, int):
                bands_to_read = channel + 1
                if not (1 <= bands_to_read <= num_channels):
                    raise ValueError(f"Channel index {channel} out of bounds for {num_channels} channels.")
                if out_shape:
                    out_shape = (1, out_shape[1], out_shape[2])
            elif isinstance(channel, list):
                bands_to_read = [c + 1 for c in channel]
                if not all(1 <= b <= num_channels for b in bands_to_read):
                    raise ValueError(f"One or more channel indices in {channel} are out of bounds.")
                if out_shape:
                    out_shape = (len(bands_to_read), out_shape[1], out_shape[2])
            else:
                raise TypeError("Channel must be an int or list of ints.")

        img_data_bands_first = self._src.read(indexes=bands_to_read, out_shape=out_shape, resampling=resampling_method)

        if img_data_bands_first.ndim == 3:
            img_array_hwc = np.transpose(img_data_bands_first, (1, 2, 0))
        elif img_data_bands_first.ndim == 2:
            img_array_hwc = img_data_bands_first
        else:
            raise ValueError("Unexpected image data shape after read.")

        return ExtImage(image=img_array_hwc, extent=self.get_extent(), metadata=self.metadata.copy())


class BioFormatsImage(AlignedSpatialImage):
    """On-disk representation of BioFormats images (e.g., OME-TIFF) in SFE objects.

    This class uses `aicsimageio` for reading, aligning with R's `BioFormatsImage`.

    Transformations are stored and applied lazily.
    """

    def __init__(
        self,
        path: Union[str, Path],
        extent: Optional[Dict[str, float]] = None,
        is_full: bool = True,
        origin: Optional[List[float]] = None,
        transformation: Optional[Union[List[Dict[str, Any]], np.ndarray]] = None,
        metadata: Optional[dict] = None,
        validate: bool = True,
    ):
        """Initialize the BioFormatsImage.

        Args:
            path:
                Path to the image file (e.g., OME-TIFF).

            extent:
                The spatial extent of the image IF KNOWN and potentially different from full.
                If None, it's inferred from metadata (full extent).

            is_full:
                Whether the provided/inferred extent is the full image.

            origin:
                Spatial coordinates [x, y] of the image's own origin (often [0,0] in its coordinate system).

            transformation:
                Stored transformation(s) to be applied.
                Can be a list of dicts (e.g. [{'type': 'rotate', 'degrees': 90}]) or a single 2x3 or 3x3 affine numpy matrix.

            metadata:
                Additional image metadata.

            validate:
                Internal use only.
        """
        super().__init__(metadata)

        self._path = Path(path)
        if validate and not self._path.exists():
            raise FileNotFoundError(f"Image file not found: '{self._path}'.")

        self._is_full = is_full
        self._origin = [0.0, 0.0] if origin is None else origin

        self._transformation_list: List[Dict[str, Any]] = []
        self._combined_affine_matrix: Optional[np.ndarray] = None

        if transformation is not None:
            if isinstance(transformation, np.ndarray):
                if transformation.shape == (2, 3) or transformation.shape == (3, 3) or transformation.shape == (2, 2):
                    self._combined_affine_matrix = transformation
                else:
                    raise ValueError("Transformation matrix must be 2x3, 3x3, or 2x2 numpy array.")
            elif isinstance(transformation, list):
                self._transformation_list = transformation
            else:
                raise TypeError("Transformation must be a list of operations or a numpy affine matrix.")

        if extent is None:
            self._base_extent = self._infer_full_extent()
            self._is_full = True
        else:
            _validate_extent(extent)
            self._base_extent = extent.copy()

        if not (
            isinstance(self._origin, list)
            and len(self._origin) == 2
            and all(isinstance(x, (int, float)) for x in self._origin)
        ):
            raise ValueError("Origin must be a list/tuple of two numbers [x, y].")

    def _get_aicsimage(self):
        """Helper to get AICSImage object. Requires aicsimageio."""
        try:
            from aicsimageio import AICSImage

            return AICSImage(self._path)
        except ImportError:
            raise ImportError("aicsimageio is required for BioFormatsImage. Please install it.")
        except Exception as e:
            raise RuntimeError(f"Error initializing AICSImage for {self._path}: {e}")

    #  method written by llm
    def _infer_full_extent(self) -> Dict[str, float]:
        """Infers the full spatial extent from image metadata using aicsimageio."""
        try:
            img = self._get_aicsimage()
            # Physical pixel sizes (usually [Z, Y, X] or [Y,X] if 2D)
            # We need X and Y pixel sizes. AICSImage physical_pixel_sizes are (Z,Y,X)
            # If C is present, it might be (C,Z,Y,X) or similar. We need to map to spatial X,Y.
            # AICSImage dims are typically 'TCZYX' or 'CZYX' etc.
            # Let's assume metadata gives pixel sizes in order of dimensions.

            # Try to get pixel sizes for X and Y dimensions
            ps_x, ps_y = 1.0, 1.0  # Default to 1.0 (pixel space)
            if img.physical_pixel_sizes.X is not None:
                ps_x = img.physical_pixel_sizes.X
            if img.physical_pixel_sizes.Y is not None:
                ps_y = img.physical_pixel_sizes.Y

            if ps_x == 0.0:
                ps_x = 1.0
                warn("Physical pixel size X is 0, using 1.0.")
            if ps_y == 0.0:
                ps_y = 1.0
                warn("Physical pixel size Y is 0, using 1.0.")

            # Image dimensions (shape)
            # AICSImage.dims.shape gives shape in order of AICSImage.dims.order (e.g., TCZYX)
            size_x = img.dims.X
            size_y = img.dims.Y

            # Extent in physical units (e.g., microns)
            # Assuming origin of image data is (0,0) at top-left pixel
            # Spatial extent: xmin, xmax, ymin, ymax
            # If origin is top-left, y increases downwards.
            # If ps_x, ps_y are pixel sizes, then total width = size_x * ps_x
            xmin = self._origin[0]
            ymin = self._origin[1]
            xmax = xmin + (size_x * ps_x)
            ymax = ymin + (size_y * ps_y)

            return {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}

        except Exception as e:
            warn(
                f"Could not infer full extent from metadata for {self._path}: {e}. Defaulting to pixel space extent [0,width,0,height].",
                RuntimeWarning,
            )
            # Fallback: try to get pixel dimensions and assume pixel space
            try:
                with Image.open(self._path) as pil_img:  # Basic PIL open for size
                    width, height = pil_img.size
                    return {
                        "xmin": self._origin[0],
                        "xmax": self._origin[0] + width,
                        "ymin": self._origin[1],
                        "ymax": self._origin[1] + height,
                    }
            except Exception:  # Ultimate fallback
                warn(f"PIL could not open {self._path} for size fallback. Using dummy extent.", RuntimeWarning)
                return {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}

    def __repr__(self):
        dims = self.get_dimensions()  # X, Y, C, Z, T
        dim_str = f"X:{dims[0]}, Y:{dims[1]}, C:{dims[2]}, Z:{dims[3]}, T:{dims[4]}"
        output = f"{type(self).__name__}(path='{str(self._path)}', dims=({dim_str})"

        if len(self.metadata) > 0:
            output += ", metadata=" + ut.print_truncated_dict(self.metadata)
        output += ")"

        return output

    def __str__(self) -> str:
        output = f"class: {type(self).__name__}\n"
        output += f"path: {str(self._path)}\n"
        dims = self.get_dimensions()
        output += f"dimensions (X,Y,C,Z,T): {dims[0]}, {dims[1]}, {dims[2]}, {dims[3]}, {dims[4]}\n"

        current_extent = self.get_extent()
        output += f"base_extent: xmin={self._base_extent['xmin']:.2f}, xmax={self._base_extent['xmax']:.2f}, ymin={self._base_extent['ymin']:.2f}, ymax={self._base_extent['ymax']:.2f}\n"
        output += f"transformed_extent: xmin={current_extent['xmin']:.2f}, xmax={current_extent['xmax']:.2f}, ymin={current_extent['ymin']:.2f}, ymax={current_extent['ymax']:.2f}\n"
        output += f"is_full: {self.is_full}\n"
        output += f"origin: {self.origin}\n"

        if self._transformation_list:
            output += f"transformations_list: {self._transformation_list}\n"

        if self._combined_affine_matrix is not None:
            output += f"combined_affine_matrix: {self._combined_affine_matrix.tolist()}\n"
        output += f"metadata({str(len(self.metadata))}): {ut.print_truncated_list(list(self.metadata.keys()), sep=' ', include_brackets=False, transform=lambda y: y)}\n"

        return output

    def copy(self) -> "BioFormatsImage":
        """Creates a copy of the BioFormatsImage."""
        return BioFormatsImage(
            path=self._path,
            extent=self._base_extent.copy(),
            is_full=self._is_full,
            origin=self._origin.copy(),
            transformation=self._combined_affine_matrix.copy()
            if self._combined_affine_matrix is not None
            else [t.copy() for t in self._transformation_list],
            metadata=self.metadata.copy(),
            validate=False,
        )

    @property
    def path(self) -> Path:
        """Get the path to the image file."""
        return self._path

    def img_source(self, as_path: bool = False) -> str:
        """Get the source file path."""
        return str(self._path)

    def get_extent(self) -> Dict[str, float]:
        """Get the spatial extent of the image, applying stored transformations to the base extent."""
        if self._combined_affine_matrix is not None:
            return _transform_extent(self._base_extent, self._combined_affine_matrix)
        elif self._transformation_list:
            warn(
                "Transformation list present but no combined affine matrix; returning base extent. Implement list processing."
            )
            return self._base_extent.copy()

        return self._base_extent.copy()

    def set_extent(self, extent: Dict[str, float], in_place: bool = False) -> "BioFormatsImage":
        """Set the base spatial extent of the image (pre-transformation).

        To change transformations, use the `transformation` property or specific methods.
        """
        _validate_extent(extent)
        obj = self if in_place else self.copy()
        obj._base_extent = extent.copy()

        return obj

    @property
    def is_full(self) -> bool:
        """Is the current base_extent considered the full extent of the image?"""
        return self._is_full

    @is_full.setter
    def is_full(self, value: bool):
        """Set the `is_full` flag."""
        if not isinstance(value, bool):
            raise TypeError("'is_full' must be a boolean.")

        self._is_full = value

    @property
    def origin(self) -> List[float]:
        """Spatial coordinates [x, y] of the image's own origin."""
        return self._origin

    @origin.setter
    def origin(self, value: List[float]):
        """Set the spatial origin [x,y]."""
        if not (isinstance(value, list) and len(value) == 2 and all(isinstance(x, (int, float)) for x in value)):
            raise ValueError("Origin must be a list/tuple of two numbers [x, y].")

        self._origin = value

    @property
    def transformation(self) -> Optional[Union[List[Dict[str, Any]], np.ndarray]]:
        """Stored transformation(s) to be applied.

        Returns:
            The combined affine matrix if available, else the list of operations.
        """
        if self._combined_affine_matrix is not None:
            return self._combined_affine_matrix

        return self._transformation_list

    # TODO: implement a transformation setter.
    @transformation.setter
    def transformation(self):
        raise NotImplementedError("Setting transformations are not supported.")

    def get_dimensions(self) -> Tuple[int, int, int, int, int]:
        """Get the dimensions of the image (X, Y, C, Z, T) from metadata.

        This refers to the dimensions of the source image file, not affected by transformations.
        """
        try:
            img = self._get_aicsimage()
            # AICSImage.dims provides an ordered dictionary
            # of dimensions like ('T', 'C', 'Z', 'Y', 'X')
            size_x = img.dims.X if "X" in img.dims.order else 1
            size_y = img.dims.Y if "Y" in img.dims.order else 1
            size_c = img.dims.C if "C" in img.dims.order else 1
            size_z = img.dims.Z if "Z" in img.dims.order else 1
            size_t = img.dims.T if "T" in img.dims.order else 1
            return (size_x, size_y, size_c, size_z, size_t)
        except Exception as e:
            warn(
                f"Error reading OME-TIFF metadata for dimensions from {self._path}: {e}. Returning dummy dimensions.",
                UserWarning,
            )
            return (0, 0, 0, 0, 0)

    @property
    def shape(self) -> Tuple[int, int, int, int, int]:
        """Alias for get_dimensions, returning (X,Y,C,Z,T)."""
        return self.get_dimensions()

    def img_raster(
        self,
        resolution: Optional[int] = None,
        scene: Optional[int] = 0,
        channel: Optional[Union[int, List[int]]] = None,
        **kwargs,
    ) -> Image.Image:
        """Load the image data as a PIL Image, applying transformations.

        This calls `to_ext_image` and then extracts the PIL image.
        `resolution` here might map to `scene` in aicsimageio if multi-resolution is stored as scenes.

        Defaulting to scene 0.
        """
        raise NotImplementedError("method not implemented!")

    def to_ext_image(self) -> "ExtImage":
        raise NotImplementedError("method not implemented!")


class ExtImage(AlignedSpatialImage):
    """In-memory image using PIL/numpy arrays with spatial extent information."""

    def __init__(
        self,
        image: Union[Image.Image, np.ndarray],
        extent: Optional[Dict[str, float]] = None,
        metadata: Optional[dict] = None,
    ):
        """Initialize an ExtImage.

        Args:
            image:
                The image data (PIL Image or numpy array).

            extent:
                The spatial extent of the image. Must have keys: 'xmin', 'xmax', 'ymin', 'ymax'.
                If None, and image is numpy array, defaults to pixel extent.

            metadata:
                Additional image metadata. Defaults to None.
        """
        super().__init__(metadata=metadata)

        if isinstance(image, np.ndarray):
            self._array: np.ndarray = image.copy()
            self._pil_image_cache: Optional[Image.Image] = None
        elif isinstance(image, Image.Image):
            self._array: np.ndarray = np.array(image)
            self._pil_image_cache = image.copy()
        else:
            raise ValueError("image must be a PIL.Image.Image or numpy.ndarray.")

        if extent is None:
            if isinstance(image, np.ndarray):
                height, width = self._array.shape[0], self._array.shape[1]
                self._extent = {"xmin": 0, "xmax": width, "ymin": 0, "ymax": height}
                warn("Extent not provided for ExtImage with numpy array; defaulting to pixel coordinates (0,0) origin.")
            elif isinstance(image, Image.Image):
                width, height = image.size
                self._extent = {"xmin": 0, "xmax": width, "ymin": 0, "ymax": height}
                warn("Extent not provided for ExtImage with PIL Image; defaulting to pixel coordinates (0,0) origin.")
            else:
                raise ValueError("Extent must be specified for ExtImage if type is unknown for default.")
        else:
            _validate_extent(extent)
            self._extent = extent.copy()

    ##########################
    ######>> Printing <<######
    ##########################

    def __repr__(self):
        """String representation."""
        dims = self.get_dimensions()
        shape_str = (
            f"{dims[1]} x {dims[0]} x {dims[2]} (width x height x channels)"
            if len(dims) == 3 and dims[2] > 0
            else f"{dims[1]} x {dims[0]} (width x height)"
        )

        output = f"{type(self).__name__}({shape_str}"
        if len(self.metadata) > 0:
            output += ", metadata=" + ut.print_truncated_dict(self.metadata)
        output += ")"

        return output

    def __str__(self) -> str:
        """
        Returns:
            A pretty-printed string containing the contents of this object.
        """
        output = f"class: {type(self).__name__}\n"
        dims = self.get_dimensions()
        shape_str = (
            f"{dims[1]} x {dims[0]} x {dims[2]} (width x height x channels)"
            if len(dims) == 3 and dims[2] > 0
            else f"{dims[1]} x {dims[0]} (width x height)"
        )
        output += f"dimensions: {shape_str}\n"
        output += f"extent: xmin={self._extent['xmin']:.2f}, xmax={self._extent['xmax']:.2f}, ymin={self._extent['ymin']:.2f}, ymax={self._extent['ymax']:.2f}\n"
        output += f"metadata({str(len(self.metadata))}): {ut.print_truncated_list(list(self.metadata.keys()), sep=' ', include_brackets=False, transform=lambda y: y)}\n"

        return output

    def copy(self) -> "ExtImage":
        """Creates a copy of the ExtImage."""
        return ExtImage(
            image=self._array.copy(),
            extent=self._extent.copy(),
            metadata=self.metadata.copy(),
        )

    def img_source(self, as_path: bool = False) -> None:
        """Get the source file path (always None for in-memory ExtImage)."""
        return None

    def get_extent(self) -> Dict[str, float]:
        """Get the extent of the image."""
        return self._extent.copy()

    def set_extent(self, extent: Dict[str, float], in_place: bool = False) -> "ExtImage":
        """Set the extent of the image."""
        _validate_extent(extent)
        obj = self if in_place else self.copy()
        obj._extent = extent.copy()
        return obj

    @property
    def array(self) -> np.ndarray:
        """Get the image as a numpy array (height, width, channels) or (height, width)."""
        return self._array

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the image array (height, width, channels) or (height, width)."""
        return self._array.shape

    def get_dimensions(self) -> Tuple[int, ...]:
        """Get the dimensions of the image array (height, width, channels) or (height, width)."""
        return self._array.shape

    def to_pil(self) -> Image.Image:
        """Convert to PIL Image."""
        if self._pil_image_cache is None or not np.array_equal(self._array, np.array(self._pil_image_cache)):
            self._pil_image_cache = Image.fromarray(self._array)

        return self._pil_image_cache

    def img_raster(self) -> Image.Image:
        """Get the image as a PIL Image object."""
        return self.to_pil()
