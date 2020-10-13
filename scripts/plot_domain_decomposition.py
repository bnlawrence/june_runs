import sys
import geopandas as gpd
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from collections import defaultdict

from june import paths

england_map = gpd.read_file(
    paths.data_path
    / "plotting/super_area_boundaries/cdee800e-60d5-479a-b8bb-ee8c8f850645202043-1-ifo1t9.r0z0m.shp"
)
england_map = england_map.rename(columns={"msoa11cd": "super_area"})
residents_per_super_area = pd.read_csv(
    paths.data_path / "plotting/demography/residents_per_super_area.csv"
)
residents_per_super_area = residents_per_super_area.rename(
    columns={"Variable: All usual residents; measures: Value": "n_residents"}
)
residents_per_super_area = residents_per_super_area.loc[
    :, ["geography code", "n_residents"]
]
england_map = pd.merge(
    england_map,
    residents_per_super_area,
    left_on="super_area",
    right_on="geography code",
)
england_map.set_index("super_area", inplace=True)

super_area_centroids = pd.read_csv(
    paths.data_path / "input/geography/super_area_centroids.csv", index_col=0
)


with open(sys.argv[1], "r") as f:
    super_areas_to_domain = json.load(f)

domain_to_super_areas = defaultdict(list)
for super_area, domain in super_areas_to_domain.items():
    domain_to_super_areas[domain].append(super_area)

domain_centroids = []
for domain, super_areas in domain_to_super_areas.items():
    centroids = super_area_centroids.loc[super_areas, ["X", "Y"]]
    domain_centroids.append(np.mean(centroids, axis=0))
domain_centroids = np.array(domain_centroids)

n_residents_per_domain = {}
for domain in domain_to_super_areas:
    n_residents_domain = england_map.loc[
        domain_to_super_areas[domain], "n_residents"
    ].values.sum()
    for super_area in domain_to_super_areas[domain]:
        n_residents_per_domain[super_area] = n_residents_domain

england_map.loc[n_residents_per_domain.keys(), "n_residents_domain"] = np.array(
    list(n_residents_per_domain.values())
)

england_map.loc[super_areas_to_domain.keys(), "label"] = super_areas_to_domain.values()
fig, ax = plt.subplots()

# # create the colorbar
norm = colors.Normalize(
    vmin=england_map.n_residents_domain.min(), vmax=england_map.n_residents_domain.max()
)
cbar = plt.cm.ScalarMappable(norm=norm, cmap="viridis")

england_map.plot("n_residents_domain", ax=ax, cmap="viridis", legend=False)

ax.scatter(domain_centroids[:, 0], domain_centroids[:, 1], color="black", s=1.5)

# add colorbar
ax_cbar = fig.colorbar(cbar, ax=ax)
# add label for the colorbar
ax_cbar.set_label("population / mean population", rotation=-90, labelpad=15)
fig.savefig("domain_decomposition.png", dpi=300)
plt.show()
