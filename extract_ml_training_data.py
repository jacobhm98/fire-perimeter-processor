import geopandas as gpd
from geopandas import GeoDataFrame
from typing import cast, List, Tuple
import matplotlib.pyplot as plt


def read_parquet() -> GeoDataFrame:
    return gpd.read_parquet("data/processed/fire_perimeters.parquet")


def extract_pairs(gdf: GeoDataFrame) -> List[List[Tuple[GeoDataFrame, GeoDataFrame]]]:
    processed_groups = [
        process_group(cast(GeoDataFrame, group))
        for _, group in gdf.groupby("incident_id")
        if len(group) > 1
    ]

    return processed_groups


def process_group(group: GeoDataFrame) -> List[Tuple[GeoDataFrame, GeoDataFrame]]:
    with_dates = group.dropna(subset=["perimeter_date"])
    sorted_group = with_dates.sort_values("perimeter_date")

    pairs = []

    for i in range(len(sorted_group) - 1):
        curr, next = sorted_group.iloc[i], sorted_group.iloc[i + 1]
        if assert_gap(curr["perimeter_date"], next["perimeter_date"]):
            pairs.append((curr, next))

    return pairs


def assert_gap(date1, date2) -> bool:
    if date1 == date2:
        return False
    gap = (date2 - date1).days
    return gap <= 6


def overlaps(perim1, perim2) -> bool:
    """
    Check if two perimeters overlap or if one is contained in the other.

    Returns True if:
    - The perimeters intersect (share any area)
    - One perimeter is contained within the other

    Args:
        perim1: First perimeter (GeoDataFrame, Series, or Polygon)
        perim2: Second perimeter (GeoDataFrame, Series, or Polygon)

    Returns:
        True if perimeters overlap or one contains the other, False otherwise
    """
    # Extract geometry objects
    if isinstance(perim1, GeoDataFrame):
        geom1 = perim1.geometry.iloc[0]
    elif hasattr(perim1, "geometry"):
        # It's a Series with geometry attribute
        geom1 = perim1.geometry
    else:
        # It's already a geometry object
        geom1 = perim1

    if isinstance(perim2, GeoDataFrame):
        geom2 = perim2.geometry.iloc[0]
    elif hasattr(perim2, "geometry"):
        # It's a Series with geometry attribute
        geom2 = perim2.geometry
    else:
        # It's already a geometry object
        geom2 = perim2

    # Check if they intersect (overlap)
    if geom1.intersects(geom2):
        return True

    # Check if one contains the other
    if geom1.contains(geom2) or geom2.contains(geom1):
        return True

    return False


def plot_incident_by_id(
    sequences: List[List[Tuple[GeoDataFrame, GeoDataFrame]]], incident_id: str
) -> None:
    """Plot perimeter growth for a specific incident by ID."""

    # Find the incident
    target_sequence = None
    for seq in sequences:
        if seq:  # Check if sequence has pairs
            first_input = seq[0][0]
            seq_incident_id = (
                first_input.iloc[0]["incident_id"]
                if hasattr(first_input.iloc[0], "incident_id")
                else first_input["incident_id"]
            )
            if seq_incident_id == incident_id:
                target_sequence = seq
                break

    if target_sequence is None:
        print(f"Incident {incident_id} not found in filtered sequences")
        return

    # Plot the incident
    fig, ax = plt.subplots(figsize=(14, 12))

    # Get incident info
    first_input = target_sequence[0][0]
    incident_name = (
        first_input.iloc[0]["incident_name"]
        if hasattr(first_input.iloc[0], "incident_name")
        else first_input["incident_name"]
    )

    # Collect all unique perimeters
    all_perimeters = []
    dates = []
    seen_dates = set()

    for input_perim, label_perim in target_sequence:
        input_date = (
            input_perim.iloc[0]["perimeter_date"]
            if hasattr(input_perim.iloc[0], "perimeter_date")
            else input_perim["perimeter_date"]
        )
        if input_date not in seen_dates:
            all_perimeters.append(input_perim)
            dates.append(input_date)
            seen_dates.add(input_date)

    # Add last label
    last_label = target_sequence[-1][1]
    last_date = (
        last_label.iloc[0]["perimeter_date"]
        if hasattr(last_label.iloc[0], "perimeter_date")
        else last_label["perimeter_date"]
    )
    if last_date not in seen_dates:
        all_perimeters.append(last_label)
        dates.append(last_date)

    # Plot with color gradient
    cmap = plt.cm.YlOrRd
    for i, (perim, date) in enumerate(zip(all_perimeters, dates)):
        color = cmap(i / max(len(all_perimeters) - 1, 1))

        # Convert to GeoDataFrame if it's a Series
        if not isinstance(perim, GeoDataFrame):
            perim = gpd.GeoDataFrame([perim], geometry="geometry")

        # Plot filled polygon
        perim.plot(ax=ax, color=color, alpha=0.4, edgecolor="black", linewidth=2)

    # Only label every Nth perimeter to avoid clutter
    label_frequency = max(1, len(all_perimeters) // 15)  # Show ~15 labels max
    for i, (perim, date) in enumerate(zip(all_perimeters, dates)):
        if i % label_frequency == 0 or i == len(all_perimeters) - 1:
            # Convert to GeoDataFrame if needed
            if not isinstance(perim, GeoDataFrame):
                perim = gpd.GeoDataFrame([perim], geometry="geometry")

            centroid = perim.geometry.centroid.iloc[0]
            ax.text(
                centroid.x,
                centroid.y,
                date.strftime("%m/%d"),
                fontsize=9,
                ha="center",
                va="center",
                weight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    alpha=0.8,
                    edgecolor="black",
                ),
            )

    first_date = dates[0]
    last_date = dates[-1]

    ax.set_title(
        f"{incident_name} ({incident_id})\nPerimeter Growth: {first_date.date()} to {last_date.date()}\n{len(target_sequence)} training pairs, {len(all_perimeters)} unique perimeters",
        fontsize=14,
        weight="bold",
    )
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)

    # Add colorbar
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max(len(all_perimeters) - 1, 1))
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Time progression", rotation=270, labelpad=20, fontsize=12)

    plt.tight_layout()
    output_file = f"data/processed/{incident_id}_filtered_sequence.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_file}")
    plt.show()


def main():
    gdf = read_parquet()
    sequences: List[List[Tuple[GeoDataFrame, GeoDataFrame]]] = extract_pairs(gdf)

    print(f"\nExtracted {len(sequences)} incident sequences")
    total_pairs = sum(len(seq) for seq in sequences)
    print(f"Total training pairs: {total_pairs}")

    ## Plot specific incident - REVERSE fire from 1976 (has overlapping perimeters!)
    # print("\nPlotting REVERSE Fire (1976-NA-000000) - 82 overlapping perimeters...")
    # plot_incident_by_id(sequences, "1976-NA-000000")

    intersections = 0
    for sequence in sequences:
        for perim1, perim2 in sequence:
            if overlaps(perim1, perim2):
                intersections += 1

    print(
        f"Found {intersections} intersecting polygons out of {total_pairs} training pairs"
    )


if __name__ == "__main__":
    main()
