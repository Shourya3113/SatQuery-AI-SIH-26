import rasterio
from rasterio.transform import from_origin
import numpy as np
import httpx
import asyncio

def create_dummy_geotiff(filename, crs="EPSG:4326", width=10, height=10):
    """Creates a small, valid GeoTIFF for testing."""
    data = np.random.randint(0, 255, (1, height, width), dtype=np.uint8)
    transform = from_origin(0, 0, 1, 1) # West, North, xsize, ysize
    
    with rasterio.open(
        filename, 'w', driver='GTiff',
        height=height, width=width,
        count=1, dtype=data.dtype,
        crs=crs, transform=transform,
    ) as dst:
        dst.write(data)

async def main():
    print("Creating dummy GeoTIFF files...")
    create_dummy_geotiff("test_image_1.tif")
    create_dummy_geotiff("test_image_2.tif")
    
    url = "http://127.0.0.1:8000/api/query"
    query_text = "Are there any new buildings between these two images?"
    
    print(f"Sending request with query: '{query_text}'")
    
    # httpx works well for async testing and is likely installed with FastAPI dependencies
    async with httpx.AsyncClient() as client:
        # Open files in binary mode
        with open("test_image_1.tif", "rb") as f1, open("test_image_2.tif", "rb") as f2:
            files = [
                ("files", ("test_image_1.tif", f1, "image/tiff")),
                ("files", ("test_image_2.tif", f2, "image/tiff"))
            ]
            data = {"query": query_text}
            
            try:
                response = await client.post(url, data=data, files=files)
                print(f"Status Code: {response.status_code}")
                import json
                print("Response JSON:")
                print(json.dumps(response.json(), indent=2))
            except Exception as e:
                print(f"Failed to send request: {e}")

if __name__ == "__main__":
    asyncio.run(main())
