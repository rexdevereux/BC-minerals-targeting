"""
07_get_satellite_imagery.py

Downloads Landsat 8/9 scenes covering an AOI via Microsoft Planetary Computer STAC,
builds a median composite over a summer window, and computes porphyry-relevant
alteration indices. Outputs GeoTIFFs ready for QGIS and a PNG overview plot.

Usage:
    pixi run python scripts/07_get_satellite_imagery.py

Outputs:
    data/composite.zarr               raw band composite (reload without redownloading)
    outputs/indices/ndvi.tif
    outputs/indices/ndwi.tif
    outputs/indices/clay_ratio.tif
    outputs/indices/iron_oxide_ratio.tif
    outputs/indices/ferrous_ratio.tif
    outputs/porphyry_indices.png
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import planetary_computer
import pystac_client
import odc.stac
import rioxarray
import xarray as xr

# ── config ────────────────────────────────────────────────────────────────────

aoi_path        = Path('data/stikinia_aoi_2.gpkg')     # path to your AOI file
date_range      = '2019-07-01/2023-09-30'   # summer window across multiple years
cloud_cover_max = 20                         # percent
resolution      = 30                         # metres
ndvi_threshold  = 0.3                        # mask pixels above (vegetation)
ndwi_threshold  = 0.0                        # mask pixels above (water)
max_aoi_km2     = 5000                       # hard limit — increase resolution if larger

output_dir  = Path('outputs/indices')
zarr_path   = Path('data/composite.zarr')
output_dir.mkdir(parents=True, exist_ok=True)

# ── load aoi ─────────────────────────────────────────────────────────────────

print(f'loading aoi: {aoi_path}')
assert aoi_path.exists(), f'aoi file not found: {aoi_path}'

aoi = gpd.read_file(aoi_path)
aoi_wgs84 = aoi.to_crs('EPSG:4326')

# size check
area_km2 = aoi.to_crs('EPSG:6933').geometry.area.sum() / 1e6
print(f'aoi area: {area_km2:,.0f} km²')

if area_km2 > max_aoi_km2:
    raise ValueError(
        f'aoi is {area_km2:,.0f} km² which exceeds the {max_aoi_km2:,.0f} km² limit at {resolution}m resolution. '
        f'clip to a smaller area in qgis or set a coarser resolution in config.'
    )

aoi_geom = aoi_wgs84.geometry.iloc[0].__geo_interface__

# ── stac search ──────────────────────────────────────────────────────────────

print('searching planetary computer...')
catalog = pystac_client.Client.open(
    'https://planetarycomputer.microsoft.com/api/stac/v1',
    modifier=planetary_computer.sign_inplace,
)

search = catalog.search(
    collections=['landsat-c2-l2'],
    intersects=aoi_geom,
    datetime=date_range,
    query={'eo:cloud_cover': {'lt': cloud_cover_max}},
)

items = list(search.items())
items = [i for i in items if i.properties.get('platform') != 'landsat-7']

# filter to july-september only
items = [
    i for i in items
    if i.datetime.month in (7, 8, 9)
]
print(f'scenes found: {len(items)} (landsat-7 excluded, jul-sep only)')

if not items:
    raise RuntimeError(
        f'no scenes found for aoi in {date_range} with cloud cover < {cloud_cover_max}%. '
        f'try widening the date range or increasing cloud_cover_max.'
    )

for item in items[:5]:
    print(f"  {item.datetime.date()}  cloud={item.properties.get('eo:cloud_cover', '?')}%  platform={item.properties.get('platform', '?')}")
if len(items) > 5:
    print(f'  ... and {len(items) - 5} more')

# ── load ─────────────────────────────────────────────────────────────────────

bands = ['blue', 'green', 'red', 'nir08', 'swir16', 'swir22']

# re-sign immediately before load to avoid token expiry during compute
print('signing items...')
items_signed = [planetary_computer.sign(item) for item in items]

print('loading (lazy)...')
ds = odc.stac.load(
    items_signed,
    bands=bands,
    geopolygon=aoi_wgs84.geometry.iloc[0],
    resolution=resolution,
    groupby='solar_day',
    chunks={'time': 1, 'x': 512, 'y': 512},
)
print(f'time steps: {len(ds.time)}  grid: {ds.dims["y"]}x{ds.dims["x"]}')

# ── scale to surface reflectance ─────────────────────────────────────────────

scale  = 0.0000275
offset = -0.2

sr = ds.astype('float32') * scale + offset
sr = sr.clip(0, 1)
sr = sr.where(ds != 0)  # mask nodata (raw value 0 = fill)

# ── median composite ─────────────────────────────────────────────────────────

print('computing median composite (downloading data now)...')
composite = sr.median(dim='time').compute()
print('done')

# clip to exact aoi boundary
composite = composite.rio.write_crs(ds.rio.crs)
composite = composite.rio.clip(
    aoi_wgs84.to_crs(ds.rio.crs).geometry,
    drop=True,
)
print(f'clipped: {composite.dims}')

# ── save to zarr ─────────────────────────────────────────────────────────────

composite.to_zarr(zarr_path, mode='w')
print(f'saved composite to {zarr_path}')
print(f'reload with: composite = xr.open_zarr("{zarr_path}")')

# ── compute indices ───────────────────────────────────────────────────────────

print('computing indices...')

def safe_ratio(a, b):
    return a / b.where(b != 0)

blue  = composite['blue']
green = composite['green']
red   = composite['red']
nir   = composite['nir08']
swir1 = composite['swir16']
swir2 = composite['swir22']

ndvi       = safe_ratio(nir - red,   nir + red)
ndwi       = safe_ratio(green - nir, green + nir)
clay       = safe_ratio(swir1, swir2)
iron_oxide = safe_ratio(red,   blue)
ferrous    = safe_ratio(swir1, nir)

# apply vegetation and water mask
mask         = (ndvi > ndvi_threshold) | (ndwi > ndwi_threshold)
clay_m       = clay.where(~mask)
iron_oxide_m = iron_oxide.where(~mask)
ferrous_m    = ferrous.where(~mask)

masked_pct = float(mask.mean()) * 100
print(f'masked (veg + water): {masked_pct:.1f}% of aoi')

print(f'masked (veg + water): {masked_pct:.1f}% of aoi')
print(f'ndvi masked: {float((ndvi > ndvi_threshold).mean()) * 100:.1f}%')
print(f'ndwi masked: {float((ndwi > ndwi_threshold).mean()) * 100:.1f}%')
print(f'clay_m valid pixels: {int(clay_m.notnull().sum())}')

# ── export geotiffs ───────────────────────────────────────────────────────────

print('exporting geotiffs...')
crs = composite.rio.crs

def export_index(da, name):
    da_out = da.rio.write_crs(crs).rio.write_nodata(np.nan)
    path = output_dir / f'{name}.tif'
    da_out.rio.to_raster(path, dtype='float32')
    print(f'  {path}')

export_index(ndvi,         'ndvi')
export_index(ndwi,         'ndwi')
export_index(clay_m,       'clay_ratio')
export_index(iron_oxide_m, 'iron_oxide_ratio')
export_index(ferrous_m,    'ferrous_ratio')

# ── plot ─────────────────────────────────────────────────────────────────────

print('plotting...')
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('BC Porphyry Mapping — Landsat Composite', fontsize=14)

def plot_band(ax, data, title, cmap='RdYlGn', vmin=None, vmax=None):
    im = ax.imshow(data.values, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')
    ax.set_title(title, fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

rgb = np.stack([red.values, green.values, blue.values], axis=-1)
rgb_display = np.clip(rgb / 0.3, 0, 1)
axes[0, 0].imshow(rgb_display, interpolation='none')
axes[0, 0].set_title('True Colour (RGB)', fontsize=11)
axes[0, 0].axis('off')

plot_band(axes[0, 1], ndvi,        'NDVI',                               cmap='RdYlGn', vmin=-0.2, vmax=0.8)
plot_band(axes[0, 2], ndwi,        'NDWI (water)',                       cmap='Blues',  vmin=-0.5, vmax=0.5)
plot_band(axes[1, 0], clay_m,      'Clay Ratio\n(phyllic/argillic)',     cmap='hot',    vmin=0.8,  vmax=1.4)
plot_band(axes[1, 1], iron_oxide_m,'Iron Oxide Ratio\n(hematite/goethite)', cmap='OrRd', vmin=0.5, vmax=2.5)
plot_band(axes[1, 2], ferrous_m,   'Ferrous Ratio\n(chlorite/amphibole)', cmap='YlOrBr', vmin=0.3, vmax=1.2)

plt.tight_layout()
plot_path = Path('outputs/porphyry_indices.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'saved {plot_path}')

print('07_imagery.py complete')