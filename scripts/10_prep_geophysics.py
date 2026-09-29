"""
10_prep_geophysics.py

Converts the BC aeromagnetic compilation (GSC Open File 9222 / BCGS Open File
2024-08, 100 m) from Geosoft grids into numeric GeoTIFFs in BC Albers, aligned
with the other province-wide covariates, and derives two standard edge filters.

    rtf — residual total magnetic field (nT): magnetite-rich rock, intrusions
    1vd — first vertical derivative (nT/m): sharpens shallow sources
    as  — analytic signal (nT/m): peaks over the edges of magnetic bodies,
          largely independent of magnetisation direction
    tdr — tilt derivative (degrees, -90..90): angle between vertical and
          horizontal gradients — balances weak and strong anomalies so faint
          contacts show as clearly as strong ones (0° ≈ source edge)

Inputs are a manual download — geophysical-data.canada.ca/Portal has no
scriptable API. Choose the Geosoft GRD format, not TIF (TIF is a coloured map
image with no measured values):
    data/geophysical/*Residual Total Field*/*.GRD
    data/geophysical/*1st Vertical Derivative*/*.GRD

Usage:
    pixi run python scripts/10_prep_geophysics.py

Outputs (all on one shared 100 m BC Albers grid):
    outputs/geophysics/rtf.tif
    outputs/geophysics/1vd.tif
    outputs/geophysics/as.tif
    outputs/geophysics/tdr.tif
"""

import struct
import zlib
from pathlib import Path

import numpy as np
import rasterio
from loguru import logger
from rasterio.transform import from_origin
from rasterio.warp import Resampling, calculate_default_transform, reproject

SOURCE_CRS = "EPSG:26910"  # NAD83 / UTM 10N, per the .GRD.xml metadata
OUTPUT_CRS = "EPSG:3005"
RESOLUTION = 100  # metres — the compilation's native cell size
GEOSOFT_DUMMY = -1e32

root_dir = Path(__file__).resolve().parents[1]
input_dir = root_dir / "data" / "geophysical"
output_dir = root_dir / "outputs" / "geophysics"
output_dir.mkdir(parents=True, exist_ok=True)


def find_grd(pattern):
    matches = sorted(input_dir.glob(f"{pattern}/*.GRD"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one .GRD matching {pattern} in {input_dir}, found {len(matches)}")
    return matches[0]


def read_geosoft_grid(path):
    """Read a compressed float32 Geosoft .grd into a north-up array + affine transform."""
    raw = path.read_bytes()
    es, sf, ne, nv, kx = struct.unpack_from("<5i", raw, 0)
    de, dv, x0, y0, rot = struct.unpack_from("<5d", raw, 20)
    zbase, zmult = struct.unpack_from("<2d", raw, 60)
    if (es, sf, kx, rot) != (1024 + 4, 2, 1, 0.0):
        raise ValueError(f"unsupported Geosoft layout in {path.name}: ES={es} SF={sf} KX={kx} ROT={rot}")

    body = raw[512:]
    n_blocks = struct.unpack_from("<i", body, 8)[0]
    offsets = struct.unpack_from(f"<{n_blocks}q", body, 16)
    sizes = struct.unpack_from(f"<{n_blocks}i", body, 16 + 8 * n_blocks)
    # each block has an undocumented 16-byte prefix before its zlib stream
    decompressed = b"".join(
        zlib.decompress(body[off - 512 + 16 : off - 512 + size]) for off, size in zip(offsets, sizes)
    )
    grid = np.frombuffer(decompressed, dtype="<f4").reshape(nv, ne)
    grid = np.where(grid <= GEOSOFT_DUMMY, np.nan, grid / zmult + zbase).astype("float32")
    grid = np.ascontiguousarray(grid[::-1])  # stored south-to-north

    # Geosoft's origin is the centre of the south-west cell
    transform = from_origin(x0 - de / 2, y0 + (nv - 0.5) * dv, de, dv)
    return grid, transform


def albers_grid_for(grid, transform):
    height, width = grid.shape
    left, top = transform.c, transform.f
    right, bottom = left + width * transform.a, top + height * transform.e
    dst_transform, dst_width, dst_height = calculate_default_transform(
        SOURCE_CRS, OUTPUT_CRS, width, height, left, bottom, right, top, resolution=RESOLUTION
    )
    return dst_transform, (dst_height, dst_width)


def reproject_onto(grid, transform, dst_transform, dst_shape):
    out = np.full(dst_shape, np.nan, dtype="float32")
    reproject(
        grid,
        out,
        src_transform=transform,
        src_crs=SOURCE_CRS,
        src_nodata=np.nan,
        dst_transform=dst_transform,
        dst_crs=OUTPUT_CRS,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
        num_threads=4,
    )
    return out


def save(array, name, transform):
    path = output_dir / f"{name}.tif"
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "width": array.shape[1],
        "height": array.shape[0],
        "crs": OUTPUT_CRS,
        "transform": transform,
        "nodata": np.nan,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype("float32"), 1)
    p1, p50, p99 = np.nanpercentile(array, [1, 50, 99])
    logger.info(f"  saved {path.name} — p1 {p1:.3f}, median {p50:.3f}, p99 {p99:.3f}")


# --- residual total field defines the shared output grid ---
logger.info("rtf: reading residual total field...")
src, src_transform = read_geosoft_grid(find_grd("*Residual Total Field*"))
logger.info(f"  {src.shape[1]:,} x {src.shape[0]:,} cells, {np.isfinite(src).mean():.1%} valid")
dst_transform, dst_shape = albers_grid_for(src, src_transform)
logger.info(f"  reprojecting {SOURCE_CRS} -> {OUTPUT_CRS} at {RESOLUTION} m ({dst_shape[1]:,} x {dst_shape[0]:,})...")
rtf = reproject_onto(src, src_transform, dst_transform, dst_shape)
del src
save(rtf, "rtf", dst_transform)

# --- 1vd is reprojected onto the same grid so every pixel lines up with rtf ---
logger.info("1vd: reading first vertical derivative...")
src, src_transform = read_geosoft_grid(find_grd("*1st Vertical Derivative*"))
vd = reproject_onto(src, src_transform, dst_transform, dst_shape)
del src
save(vd, "1vd", dst_transform)

# --- derived edge filters ---
logger.info("deriving analytic signal and tilt derivative...")
dy, dx = np.gradient(rtf, RESOLUTION)  # nT/m; row axis first
del rtf
horizontal = np.hypot(dx, dy)
del dx, dy
save(np.hypot(horizontal, vd), "as", dst_transform)
save(np.degrees(np.arctan2(vd, horizontal)), "tdr", dst_transform)

logger.info("done — geophysics covariates ready in outputs/geophysics/")
