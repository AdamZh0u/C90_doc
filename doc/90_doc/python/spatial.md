## Spatial problem solving

[python - How to generate edge index after Delaunay triangulation? - Stack Overflow](https://stackoverflow.com/questions/69512972/how-to-generate-edge-index-after-delaunay-triangulation)


# Geopandas

```python
# df_limit = df[["lng","lat","type"]]

# gdf_poi = df_to_gdf(df_limit,lon="lng", lat="lat",scproj='epsg:4326',proj='epsg:3857')

# gdf_poi["code"] = gdf_poi["type"].astype("category").cat.codes

# gdf_poi.to_file(result_dir + "gdf_poix.geojson",driver='GeoJSON', encoding='utf-8')
```