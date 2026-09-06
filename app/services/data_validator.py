import rasterio
from rasterio.io import MemoryFile
from rasterio.errors import RasterioIOError
from pydantic import BaseModel
from typing import List, Dict, Tuple, Optional

class ImageMetadata(BaseModel):
    filename: str
    crs: Optional[str]
    bounds: Dict[str, float]
    width: int
    height: int
    count: int # number of bands

class ValidationResult(BaseModel):
    is_valid: bool
    message: str
    metadata: List[ImageMetadata]
    is_coregistered: Optional[bool] = None

def check_coregistration(metadata_list: List[ImageMetadata]) -> bool:
    """
    Check if a list of images are co-registered.
    For simplicity, we check if they have the exact same CRS and identical bounding boxes.
    In a real-world scenario, you might allow for minor tolerances or just overlapping bounds.
    """
    if len(metadata_list) < 2:
        return True
        
    base_meta = metadata_list[0]
    for meta in metadata_list[1:]:
        if meta.crs != base_meta.crs:
            return False
        # Check bounding box equivalence (could use math.isclose for floating point tolerance)
        if meta.bounds != base_meta.bounds:
            return False
            
    return True

def validate_and_extract_metadata(files: List[Tuple[str, bytes]]) -> ValidationResult:
    """
    Validates uploaded GeoTIFF files, extracts metadata using rasterio,
    and checks if multiple files are co-registered.
    """
    metadata_list = []
    
    for filename, file_bytes in files:
        try:
            with MemoryFile(file_bytes) as memfile:
                with memfile.open() as dataset:
                    crs = dataset.crs.to_string() if dataset.crs else None
                    bounds = {
                        "left": dataset.bounds.left,
                        "bottom": dataset.bounds.bottom,
                        "right": dataset.bounds.right,
                        "top": dataset.bounds.top
                    }
                    
                    meta = ImageMetadata(
                        filename=filename,
                        crs=crs,
                        bounds=bounds,
                        width=dataset.width,
                        height=dataset.height,
                        count=dataset.count
                    )
                    metadata_list.append(meta)
        except RasterioIOError:
            return ValidationResult(
                is_valid=False,
                message=f"File {filename} is not a valid GeoTIFF or could not be opened.",
                metadata=[]
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                message=f"Error processing {filename}: {str(e)}",
                metadata=[]
            )
            
    # Check co-registration if there are multiple images
    is_coregistered = None
    if len(metadata_list) > 1:
        is_coregistered = check_coregistration(metadata_list)
        if not is_coregistered:
            return ValidationResult(
                is_valid=False,
                message="Multiple images uploaded but they are not co-registered (mismatched CRS or bounding boxes).",
                metadata=metadata_list,
                is_coregistered=False
            )
            
    return ValidationResult(
        is_valid=True,
        message="All files valid and processed successfully.",
        metadata=metadata_list,
        is_coregistered=is_coregistered if is_coregistered is not None else True
    )
