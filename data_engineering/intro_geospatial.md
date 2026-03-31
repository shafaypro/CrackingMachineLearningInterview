# Geospatial AI Systems — Google Solar API, ArcGIS & Production ML

## Why Geospatial Matters in AI/ML (2026)

Geospatial data is everywhere in production ML:
- **Energy**: Solar potential assessment, grid optimization, EV charging placement
- **Logistics**: Route optimization, demand forecasting, last-mile delivery
- **Climate Tech**: Carbon mapping, deforestation detection, flood risk modeling
- **Urban Planning**: Zoning optimization, infrastructure planning
- **Agriculture**: Crop yield prediction, irrigation optimization
- **Insurance**: Risk modeling by location, catastrophe modeling

---

## Coordinate Systems (EPSG Basics)

Understanding coordinate systems is essential before building any geospatial ML pipeline.

### Common EPSG Codes

| EPSG | Name | Type | Use Case |
|------|------|------|----------|
| **4326** | WGS84 | Geographic (lat/lon degrees) | GPS, most APIs, GeoJSON |
| **3857** | Web Mercator | Projected (meters) | Web maps (Google Maps, OSM) |
| **32632** | UTM Zone 32N | Projected (meters, local) | Accurate local distance calculations (Europe) |
| **4269** | NAD83 | Geographic | US government data |

```python
import pyproj
from pyproj import Transformer

# Transform GPS coordinates (WGS84) → meters (UTM)
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)

# San Francisco: (lon, lat) in WGS84
lon, lat = -122.4194, 37.7749
x_meters, y_meters = transformer.transform(lon, lat)
print(f"UTM: ({x_meters:.0f}, {y_meters:.0f}) meters")

# Always use projected CRS for distance/area calculations
# Always use WGS84 for API inputs and GeoJSON
```

### Critical Rule for ML

> **Never compute distances with lat/lon directly.** 1 degree of longitude ≠ same distance at different latitudes. Always project to meters first.

```python
import geopandas as gpd
from shapely.geometry import Point

# Create GeoDataFrame in WGS84
gdf = gpd.GeoDataFrame(
    {"name": ["NYC", "LA"], "value": [100, 200]},
    geometry=[Point(-74.0060, 40.7128), Point(-118.2437, 34.0522)],
    crs="EPSG:4326"
)

# Reproject to meters for distance calculation
gdf_projected = gdf.to_crs("EPSG:3857")
distance_meters = gdf_projected.geometry.iloc[0].distance(gdf_projected.geometry.iloc[1])
distance_km = distance_meters / 1000
print(f"Distance NYC→LA: {distance_km:.0f} km")
```

---

## Google Solar API

### What It Does

Google Solar API analyzes satellite imagery to provide:
- **Roof segment detection**: Area, orientation, tilt of each roof section
- **Solar panel layout**: Optimal panel placement
- **Annual solar energy potential**: kWh/year estimate
- **Financial modeling**: Payback period, savings

### API Overview

```python
import requests
import json

API_KEY = "your-google-solar-api-key"
BASE_URL = "https://solar.googleapis.com/v1"

def get_building_insights(lat: float, lon: float) -> dict:
    """Get solar potential for a building at given coordinates"""
    url = f"{BASE_URL}/buildingInsights:findClosest"
    params = {
        "key": API_KEY,
        "location.latitude": lat,
        "location.longitude": lon,
        "requiredQuality": "HIGH"  # HIGH, MEDIUM, or LOW
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def analyze_solar_potential(building: dict) -> dict:
    """Extract key metrics from Solar API response"""
    solar_potential = building.get("solarPotential", {})

    return {
        "max_array_area_m2": solar_potential.get("maxArrayAreaMeters2"),
        "max_sunshine_hours_year": solar_potential.get("maxSunshineHoursPerYear"),
        "carbon_offset_kg": solar_potential.get("carbonOffsetFactorKgPerMwh"),
        "panel_configs": [
            {
                "panels": cfg["panelsCount"],
                "yearly_energy_kwh": cfg["yearlyEnergyDcKwh"],
                "roof_segments_used": len(cfg.get("roofSegmentSummaries", []))
            }
            for cfg in solar_potential.get("solarPanelConfigs", [])[:5]  # Top 5 configs
        ]
    }

# Usage
building = get_building_insights(37.7749, -122.4194)
analysis = analyze_solar_potential(building)
print(json.dumps(analysis, indent=2))
```

### ML Integration: Solar Potential Scoring at Scale

```python
import pandas as pd
import asyncio
import aiohttp
from sklearn.ensemble import GradientBoostingRegressor

async def batch_solar_analysis(addresses: list[dict]) -> pd.DataFrame:
    """Analyze solar potential for many buildings in parallel"""

    async def fetch_one(session, lat, lon, address_id):
        url = f"{BASE_URL}/buildingInsights:findClosest"
        params = {"key": API_KEY, "location.latitude": lat, "location.longitude": lon}
        async with session.get(url, params=params) as resp:
            data = await resp.json()
            return {"id": address_id, "lat": lat, "lon": lon, **analyze_solar_potential(data)}

    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_one(session, addr["lat"], addr["lon"], addr["id"])
            for addr in addresses
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    return pd.DataFrame([r for r in results if not isinstance(r, Exception)])

# ML Model: Predict solar adoption probability
def train_solar_adoption_model(df: pd.DataFrame) -> GradientBoostingRegressor:
    features = [
        "max_array_area_m2",
        "max_sunshine_hours_year",
        "num_panels_optimal",
        "home_value",           # From external source
        "electricity_rate",     # From utility data
        "roof_age_years"        # From permit data
    ]
    target = "solar_adoption_within_2_years"  # Historical data

    X = df[features].fillna(df[features].median())
    y = df[target]

    model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1)
    model.fit(X, y)
    return model
```

---

## ArcGIS in ML Pipelines

### ArcGIS Python API

```python
from arcgis.gis import GIS
from arcgis.features import FeatureLayer, GeoAccessor
from arcgis.geometry import Geometry
import pandas as pd

# Connect to ArcGIS Online or Enterprise
gis = GIS("https://www.arcgis.com", "username", "password")
# Or: gis = GIS("pro")  # Use ArcGIS Pro credentials

# Load a feature layer
url = "https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/USA_Counties/FeatureServer/0"
layer = FeatureLayer(url)

# Query features as Spatially Enabled DataFrame (SEDF)
query_result = layer.query(
    where="STATE_NAME = 'California'",
    out_fields=["NAME", "POPULATION", "AREA"],
    return_geometry=True
)
df = query_result.sdf  # Spatially Enabled DataFrame (GeoDataFrame-like)

print(df.head())
print(f"Total CA counties: {len(df)}")
print(f"Total population: {df['POPULATION'].sum():,.0f}")
```

### Spatial Analysis for ML Feature Engineering

```python
from arcgis.features import GeoAccessor, GeoSeriesAccessor
import pandas as pd
import numpy as np

# Load parcel data with solar potential scores
parcels_df = pd.DataFrame({
    "parcel_id": range(1000),
    "lat": np.random.uniform(37.0, 38.0, 1000),
    "lon": np.random.uniform(-122.5, -121.5, 1000),
    "solar_score": np.random.uniform(0, 1, 1000)
})

# Convert to SEDF
sedf = pd.DataFrame.spatial.from_xy(parcels_df, x_column="lon", y_column="lat", sr=4326)

# Spatial join: add demographic features from census layer
census_layer = gis.content.get("census-layer-id").layers[0]
sedf_with_demographics = sedf.spatial.join(
    right_df=census_layer.query().sdf,
    how="left",
    op="within"  # Join parcels within census tracts
)

# Buffer analysis: count competitors within 5km
sedf["competitors_5km"] = sedf.spatial.relationship(
    other=competitor_locations_sedf,
    spatial_rel="within",
    distance=5000,
    units="meters"
).groupby("parcel_id").size()
```

---

## Spatial Queries with PostGIS

### Why PostGIS?

PostGIS extends PostgreSQL with geospatial types and operations — perfect for production ML pipelines that need spatial joins, buffering, and proximity queries at scale.

```sql
-- Enable PostGIS
CREATE EXTENSION postgis;

-- Create table with geometry
CREATE TABLE buildings (
    id SERIAL PRIMARY KEY,
    address TEXT,
    geom GEOMETRY(Point, 4326),  -- WGS84 lat/lon
    solar_score FLOAT,
    annual_kwh FLOAT,
    metadata JSONB
);

-- Index for spatial queries (CRITICAL for performance)
CREATE INDEX buildings_geom_idx ON buildings USING GIST(geom);

-- Find buildings within 1km of a charging station
SELECT b.id, b.address, b.solar_score,
       ST_Distance(
           ST_Transform(b.geom, 3857),  -- Convert to meters
           ST_Transform(c.geom, 3857)
       ) / 1000.0 AS distance_km
FROM buildings b
JOIN charging_stations c ON ST_DWithin(
    ST_Transform(b.geom, 3857),
    ST_Transform(c.geom, 3857),
    1000  -- 1000 meters
)
WHERE b.solar_score > 0.7
ORDER BY distance_km;

-- Aggregate solar potential by neighborhood polygon
SELECT
    n.name AS neighborhood,
    COUNT(b.id) AS building_count,
    AVG(b.solar_score) AS avg_solar_score,
    SUM(b.annual_kwh) AS total_annual_kwh
FROM neighborhoods n
JOIN buildings b ON ST_Within(b.geom, n.geom)
GROUP BY n.name
ORDER BY total_annual_kwh DESC;
```

### Python Integration

```python
import geopandas as gpd
from sqlalchemy import create_engine

engine = create_engine("postgresql://user:pass@localhost/geospatial_db")

# Load spatial data with PostGIS
gdf = gpd.read_postgis("""
    SELECT id, address, solar_score, annual_kwh, geom
    FROM buildings
    WHERE solar_score > 0.5
    ORDER BY solar_score DESC
    LIMIT 10000
""", engine, geom_col="geom", crs="EPSG:4326")

# Write ML predictions back to PostGIS
gdf["adoption_probability"] = model.predict(gdf[feature_cols])
gdf.to_postgis("ml_predictions", engine, if_exists="replace", index=False)
```

---

## Combining Geospatial Data with ML Pipelines

### Feature Engineering from Geospatial Data

```python
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

def engineer_spatial_features(buildings_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Create ML features from spatial data"""

    features = pd.DataFrame()

    # 1. Distance to amenities (schools, hospitals, transit)
    for amenity_name, amenity_gdf in amenity_layers.items():
        features[f"dist_{amenity_name}_m"] = buildings_gdf.geometry.apply(
            lambda g: amenity_gdf.geometry.distance(g).min() * 111000  # deg to meters approx
        )

    # 2. Population density (from census polygons)
    joined = gpd.sjoin(buildings_gdf, census_gdf, how="left", predicate="within")
    features["population_density"] = joined["pop_per_sqkm"]

    # 3. Elevation features (from DEM raster)
    features["elevation_m"] = extract_raster_values(buildings_gdf.geometry, dem_raster)
    features["slope_degrees"] = extract_raster_values(buildings_gdf.geometry, slope_raster)

    # 4. Cluster-based features
    from sklearn.cluster import KMeans
    coords = np.column_stack([
        buildings_gdf.geometry.x,
        buildings_gdf.geometry.y
    ])
    kmeans = KMeans(n_clusters=50, random_state=42)
    features["spatial_cluster"] = kmeans.fit_predict(coords)

    # 5. H3 hexagonal grid features (Uber H3)
    import h3
    features["h3_index_res8"] = buildings_gdf.apply(
        lambda row: h3.latlng_to_cell(row.geometry.y, row.geometry.x, 8),
        axis=1
    )

    return features
```

### H3 Hexagonal Indexing for Spatial ML

```python
import h3
import pandas as pd

# H3 is Uber's hierarchical hexagonal spatial index
# Perfect for spatial feature engineering and aggregation

def aggregate_by_h3(df: pd.DataFrame, resolution: int = 8) -> pd.DataFrame:
    """
    H3 Resolution Guide:
    Res 4: ~170km² (country-level)
    Res 7: ~5km² (neighborhood)
    Res 8: ~0.7km² (block)
    Res 10: ~0.015km² (building)
    """
    df["h3_index"] = df.apply(
        lambda row: h3.latlng_to_cell(row["lat"], row["lon"], resolution),
        axis=1
    )

    return df.groupby("h3_index").agg({
        "solar_score": ["mean", "max", "count"],
        "annual_kwh": "sum",
        "home_value": "median"
    }).reset_index()
```

---

## Real-World ML System: Solar Lead Scoring Pipeline

```
                    ┌─────────────────────────────────────────────┐
                    │        SOLAR LEAD SCORING SYSTEM             │
                    │                                              │
  Address List → [Geocoder] → [Google Solar API] → [Feature Eng] │
                     │              │                    │         │
              {lat, lon}    {solar_potential}    {spatial feats}  │
                     └──────────────┴────────────────────┘        │
                                    │                             │
                              [ML Model]                          │
                         (GBM Classifier)                         │
                                    │                             │
                              [Lead Score]                        │
                                    │                             │
                            [PostGIS DB]                         │
                                    │                             │
                        [Sales Dashboard / CRM]                   │
                    └─────────────────────────────────────────────┘
```

```python
from dataclasses import dataclass
import pandas as pd

@dataclass
class SolarLead:
    address: str
    lat: float
    lon: float
    solar_score: float
    adoption_probability: float
    estimated_savings_annual: float
    priority: str  # "hot", "warm", "cold"

async def score_leads(addresses: list[str]) -> list[SolarLead]:
    # Step 1: Geocode addresses
    geocoded = await batch_geocode(addresses)

    # Step 2: Fetch solar data
    solar_data = await batch_solar_analysis(geocoded)

    # Step 3: Add spatial features
    spatial_features = engineer_spatial_features(
        gpd.GeoDataFrame(solar_data, geometry=gpd.points_from_xy(
            solar_data["lon"], solar_data["lat"]
        ))
    )

    # Step 4: Merge and predict
    X = pd.concat([solar_data, spatial_features], axis=1)[FEATURE_COLS]
    probabilities = adoption_model.predict_proba(X)[:, 1]

    # Step 5: Create lead objects
    leads = []
    for i, (_, row) in enumerate(solar_data.iterrows()):
        prob = probabilities[i]
        leads.append(SolarLead(
            address=row["address"],
            lat=row["lat"],
            lon=row["lon"],
            solar_score=row["solar_score"],
            adoption_probability=prob,
            estimated_savings_annual=row["annual_kwh"] * 0.12,  # $0.12/kWh
            priority="hot" if prob > 0.7 else "warm" if prob > 0.4 else "cold"
        ))

    return sorted(leads, key=lambda x: x.adoption_probability, reverse=True)
```

---

## Interview Questions

**Q: What coordinate system would you use for computing distances in a geospatial ML pipeline?**
> Always project to a metric CRS (e.g., UTM or Web Mercator EPSG:3857) before computing distances. Never use lat/lon degrees for distance — 1 degree of longitude has different real-world distances at different latitudes. Use pyproj or GeoPandas `.to_crs()` to reproject.

**Q: How would you design a system to score solar potential for 1 million addresses?**
> Batch geocode addresses → parallel async calls to Google Solar API (respect rate limits with semaphores) → feature engineering with PostGIS spatial joins → batch ML inference → store results in PostGIS with spatial index for downstream queries. Monitor API costs carefully — Solar API is metered.

**Q: What is H3 and why is it useful for ML?**
> H3 is Uber's hexagonal hierarchical spatial index. Each hex cell has a unique ID at multiple resolutions. For ML: use H3 as a groupby key for spatial aggregation features, encode spatial location without leaking exact coordinates, and enable efficient neighbor lookups. H3 hexagons have equal area (unlike lat/lon grids), making them better for spatial statistics.

**Q: How do you prevent data leakage in geospatial ML models?**
> Don't use future data for historical predictions. Don't use spatial proximity as a feature if your train/test split is random (nearby points will be in both sets — use spatial cross-validation: block holdout by region). Avoid using census data that's derived from the same population you're predicting.

**Q: How would you combine ArcGIS data with a Python ML pipeline?**
> Use the ArcGIS Python API (`arcgis` package) to query feature layers as Spatially Enabled DataFrames or use the REST API directly. Convert to GeoPandas for sklearn-compatible processing. Alternatively, export to PostGIS for complex spatial joins at scale. Always manage CRS transformations explicitly.
