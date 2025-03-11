#!/usr/bin/env python3

from datetime import datetime
from typing import Literal

import click
import pandas as pd

from chainscope.typing import DATA_DIR, Properties


def date_to_int(date_str: str) -> int:
    """Convert date string to YYYYMMDD integer format."""
    dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    return int(dt.strftime("%Y%m%d"))


def get_year(date_str: str) -> int:
    """Extract year from date string."""
    dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    return dt.year


def create_us_place_properties(
    df: pd.DataFrame,
    coord_type: Literal["long", "lat"],
    entity_type: str,
    # bucket: int,
    # metric: Literal["views", "pop"],
    # n_buckets: int,
) -> Properties:
    """Create properties for US places based on longitude or latitude."""
    # Filter by entity type
    type_df = df[df["entity_type"] == entity_type].copy()

    # Get the metric column
    # metric_col = "page_views" if metric == "views" else "population"

    # Filter by bucket - reverse the bucket numbering
    # type_df["bucket"] = pd.qcut(
    #     type_df[metric_col], q=n_buckets, labels=range(n_buckets, 0, -1)
    # )
    # bucket_df = type_df[type_df["bucket"] == bucket]

    # Get the coordinate values
    coord_col = "longitude" if coord_type == "long" else "latitude"
    value_by_name = {
        f"{row['name']}, {row['state_id']}": float(row[coord_col])
        for _, row in type_df.iterrows()
    }

    # Sort by coordinate value
    value_by_name = dict(sorted(value_by_name.items(), key=lambda x: x[1]))

    ent_plural_str_map = {
        "city": "cities",
        "college": "colleges",
        "zip": "zip codes",
        "county": "counties",
        "structure": "structures",
        "natural_place": "natural places",
    }
    # Create appropriate questions based on coordinate type
    q_prefix = f"about US {ent_plural_str_map[entity_type]}:\n\n"
    if coord_type == "long":
        props = Properties(
            gt_question=q_prefix + "Is {x} located east of {y}?",
            lt_question=q_prefix + "Is {x} located west of {y}?",
            gt_open_ended_question="Which place is located more to the east: {x} or {y}?",
            lt_open_ended_question="Which place is located more to the west: {x} or {y}?",
            value_by_name=value_by_name,  # type: ignore
        )
    else:  # lat
        props = Properties(
            gt_question=q_prefix + "Is {x} located north of {y}?",
            lt_question=q_prefix + "Is {x} located south of {y}?",
            gt_open_ended_question="Which place is located more to the north: {x} or {y}?",
            lt_open_ended_question="Which place is located more to the south: {x} or {y}?",
            value_by_name=value_by_name,  # type: ignore
        )

    return props


def create_world_place_properties(
    df: pd.DataFrame,
    coord_type: Literal["long", "lat"],
    entity_type: str,
) -> Properties:
    """Create properties for world places based on longitude or latitude."""
    # Filter by entity type
    type_df = df[df["entity_type"] == entity_type].copy()

    # Get the coordinate values
    coord_col = "longitude" if coord_type == "long" else "latitude"
    value_by_name = {
        str(row["name"]): float(row[coord_col]) for _, row in type_df.iterrows()
    }

    # Sort by coordinate value
    value_by_name = dict(sorted(value_by_name.items(), key=lambda x: x[1]))

    # Create appropriate questions based on coordinate type
    ent_plural_str_map = {
        "structure": "structures",
        "populated_place": "places",
        "natural_place": "natural places",
    }
    q_prefix = f"about world {ent_plural_str_map[entity_type]}:\n\n"
    if coord_type == "long":
        props = Properties(
            gt_question=q_prefix + "Is {x} located east of {y}?",
            lt_question=q_prefix + "Is {x} located west of {y}?",
            gt_open_ended_question="Which place is located more to the east: {x} or {y}?",
            lt_open_ended_question="Which place is located more to the west: {x} or {y}?",
            value_by_name=value_by_name,  # type: ignore
        )
    else:  # lat
        props = Properties(
            gt_question=q_prefix + "Is {x} located north of {y}?",
            lt_question=q_prefix + "Is {x} located south of {y}?",
            gt_open_ended_question="Which place is located more to the north: {x} or {y}?",
            lt_open_ended_question="Which place is located more to the south: {x} or {y}?",
            value_by_name=value_by_name,  # type: ignore
        )

    return props


def create_art_release_properties(df: pd.DataFrame) -> dict[str, Properties]:
    """Create properties for art based on release date, separated by entity type."""
    properties_by_type = {}

    ent_plural_str_map = {"book": "books", "movie": "movies", "song": "songs"}
    for entity_type in df["entity_type"].unique():
        type_df = df[df["entity_type"] == entity_type].copy()

        # Create mapping of art to release dates and sort by date
        value_by_name = {
            f"{row['creator']}'s {row['title']}": date_to_int(row["release_date"])
            for _, row in type_df.iterrows()
        }

        # Skip if no valid dates
        if not value_by_name:
            continue

        # Sort by release date
        value_by_name = dict(sorted(value_by_name.items(), key=lambda x: x[1]))

        q_prefix = f"about {ent_plural_str_map[entity_type]}:\n\n"
        props = Properties(
            gt_question=q_prefix + "Was {x} released later than {y}?",
            lt_question=q_prefix + "Was {x} released earlier than {y}?",
            gt_open_ended_question="Which work was released later: {x} or {y}?",
            lt_open_ended_question="Which work was released earlier: {x} or {y}?",
            value_by_name=value_by_name,  # type: ignore
        )

        properties_by_type[entity_type] = props

    return properties_by_type


def create_art_length_properties(df: pd.DataFrame) -> dict[str, Properties]:
    """Create properties for art based on length, separated by entity type."""
    properties_by_type = {}

    ent_plural_str_map = {"book": "books", "movie": "movies"}
    for entity_type in df["entity_type"].unique():
        type_df = df[df["entity_type"] == entity_type].copy()

        # Create mapping of art to lengths and sort by length
        value_by_name = {
            f"{row['creator']}'s {row['title']}": float(row["length"])
            for _, row in type_df.iterrows()
            if pd.notna(row["length"])
            and float(row["length"]) > 0  # Filter out negative lengths
        }

        # Skip if no valid lengths
        if not value_by_name:
            continue

        # Sort by length
        value_by_name = dict(sorted(value_by_name.items(), key=lambda x: x[1]))

        q_prefix = f"about {ent_plural_str_map[entity_type]}:\n\n"
        props = Properties(
            gt_question=q_prefix + "Is {x} longer than {y}?",
            lt_question=q_prefix + "Is {x} shorter than {y}?",
            gt_open_ended_question="Which work is longer: {x} or {y}?",
            lt_open_ended_question="Which work is shorter: {x} or {y}?",
            value_by_name=value_by_name,  # type: ignore
        )

        properties_by_type[entity_type] = props

    return properties_by_type


def create_us_place_metric_properties(
    df: pd.DataFrame,
    metric: Literal["population", "density"],
    entity_type: str,
) -> Properties | None:
    """Create properties for US places based on population or density."""
    # Filter by entity type
    type_df = df[df["entity_type"] == entity_type].copy()

    # Get the metric values
    value_by_name = {
        f"{row['name']}, {row['state_id']}": float(row[metric])
        for _, row in type_df.iterrows()
        if pd.notna(row[metric])
        and float(row[metric]) > 0  # Filter out negative values
    }

    # Skip if no valid values
    if not value_by_name:
        return None

    ent_plural_str_map = {
        "city": "cities",
        "zip": "zip codes",
        "county": "counties",
    }

    q_prefix = f"about US {ent_plural_str_map[entity_type]}:\n\n"
    if metric == "population":
        props = Properties(
            gt_question=q_prefix + "Is {x} more populous than {y}?",
            lt_question=q_prefix + "Is {x} less populous than {y}?",
            gt_open_ended_question="Which place is more populous: {x} or {y}?",
            lt_open_ended_question="Which place is less populous: {x} or {y}?",
            value_by_name=value_by_name,  # type: ignore
        )
    else:  # density
        props = Properties(
            gt_question=q_prefix + "Is {x} more densely populated than {y}?",
            lt_question=q_prefix + "Is {x} less densely populated than {y}?",
            gt_open_ended_question="Which place is more densely populated: {x} or {y}?",
            lt_open_ended_question="Which place is less densely populated: {x} or {y}?",
            value_by_name=value_by_name,  # type: ignore
        )

    return props


def create_world_place_metric_properties(
    df: pd.DataFrame,
    metric: Literal["total_area", "population"],
    entity_type: str,
) -> Properties | None:
    """Create properties for world places based on total area or population."""
    # Filter by entity type
    type_df = df[df["entity_type"] == entity_type].copy()

    # Get the metric values
    value_by_name = {
        str(row["name"]): float(row[metric])
        for _, row in type_df.iterrows()
        if pd.notna(row[metric])
        and float(row[metric]) > 0  # Filter out negative values
    }

    # Skip if no valid values
    if not value_by_name:
        return None

    # Sort by metric value
    value_by_name = dict(sorted(value_by_name.items(), key=lambda x: x[1]))

    ent_plural_str_map = {
        "structure": "structures",
        "populated_place": "places",
        "natural_place": "natural places",
    }

    q_prefix = f"about world {ent_plural_str_map[entity_type]}:\n\n"
    if metric == "population":
        props = Properties(
            gt_question=q_prefix + "Is {x} more populous than {y}?",
            lt_question=q_prefix + "Is {x} less populous than {y}?",
            gt_open_ended_question="Which place is more populous: {x} or {y}?",
            lt_open_ended_question="Which place is less populous: {x} or {y}?",
            value_by_name=value_by_name,  # type: ignore
        )
    else:  # total_area
        props = Properties(
            gt_question=q_prefix + "Does {x} have larger area than {y}?",
            lt_question=q_prefix + "Does {x} have smaller area than {y}?",
            gt_open_ended_question="Which place has larger area: {x} or {y}?",
            lt_open_ended_question="Which place has smaller area: {x} or {y}?",
            value_by_name=value_by_name,  # type: ignore
        )

    return props


def create_historical_figure_birth_properties(df: pd.DataFrame) -> Properties:
    """Create properties for historical figures based on birth year."""
    # Create mapping of figures to birth years and sort
    value_by_name = {
        str(row["name"]): float(row["birth_year"])
        for _, row in df.iterrows()
        if pd.notna(row["birth_year"])
    }

    q_prefix = "about historical figures:\n\n"
    props = Properties(
        gt_question=q_prefix + "Was {x} born later than {y}?",
        lt_question=q_prefix + "Was {x} born earlier than {y}?",
        gt_open_ended_question="Who was born later: {x} or {y}?",
        lt_open_ended_question="Who was born earlier: {x} or {y}?",
        value_by_name=value_by_name,  # type: ignore
    )

    return props


def create_historical_figure_death_properties(df: pd.DataFrame) -> Properties:
    """Create properties for historical figures based on death year."""
    # Create mapping of figures to death years and sort
    value_by_name = {
        str(row["name"]): float(row["death_year"])
        for _, row in df.iterrows()
        if pd.notna(row["death_year"])
    }

    # Sort by death year
    value_by_name = dict(sorted(value_by_name.items(), key=lambda x: x[1]))

    q_prefix = "about historical figures:\n\n"
    props = Properties(
        gt_question=q_prefix + "Did {x} die later than {y}?",
        lt_question=q_prefix + "Did {x} die earlier than {y}?",
        gt_open_ended_question="Who died later: {x} or {y}?",
        lt_open_ended_question="Who died earlier: {x} or {y}?",
        value_by_name=value_by_name,  # type: ignore
    )

    return props


def create_historical_figure_age_properties(df: pd.DataFrame) -> Properties:
    """Create properties for historical figures based on age at death."""
    # Create mapping of figures to ages and sort
    value_by_name = {
        str(row["name"]): float(row["age"])
        for _, row in df.iterrows()
        if pd.notna(row["age"]) and float(row["age"]) >= 0  # Filter out negative ages
    }

    # Sort by age
    value_by_name = dict(sorted(value_by_name.items(), key=lambda x: x[1]))

    q_prefix = "about historical figures:\n\n"
    props = Properties(
        gt_question=q_prefix + "Did {x} live longer than {y}?",
        lt_question=q_prefix + "Did {x} die younger than {y}?",
        gt_open_ended_question="Who lived longer: {x} or {y}?",
        lt_open_ended_question="Who died younger: {x} or {y}?",
        value_by_name=value_by_name,  # type: ignore
    )

    return props


def create_nyc_place_properties(
    df: pd.DataFrame,
    coord_type: Literal["long", "lat"],
) -> Properties:
    """Create properties for NYC places based on longitude or latitude."""
    # Map coord_type to actual column name
    coord_col = "longitude" if coord_type == "long" else "latitude"

    # Count occurrences of each NAME, BOROUGH combination
    df["place_key"] = df["name"] + ", " + df["borough_name"]
    duplicate_keys = set(df[df["place_key"].duplicated(keep=False)]["place_key"])

    # Create mapping of places to coordinates, excluding duplicates
    value_by_name = {
        f"{row['name']}, {row['borough_name']}": float(row[coord_col])
        for _, row in df.iterrows()
        if pd.notna(row[coord_col]) and row["place_key"] not in duplicate_keys
    }

    # Sort by coordinate value
    value_by_name = dict(sorted(value_by_name.items(), key=lambda x: x[1]))

    q_prefix = "about places in NYC:\n\n"
    if coord_type == "long":
        props = Properties(
            gt_question=q_prefix + "Is {x} located east of {y}?",
            lt_question=q_prefix + "Is {x} located west of {y}?",
            gt_open_ended_question="Which place is located more to the east: {x} or {y}?",
            lt_open_ended_question="Which place is located more to the west: {x} or {y}?",
            value_by_name=value_by_name,  # type: ignore
        )
    else:  # lat
        props = Properties(
            gt_question=q_prefix + "Is {x} located north of {y}?",
            lt_question=q_prefix + "Is {x} located south of {y}?",
            gt_open_ended_question="Which place is located more to the north: {x} or {y}?",
            lt_open_ended_question="Which place is located more to the south: {x} or {y}?",
            value_by_name=value_by_name,  # type: ignore
        )

    return props


@click.command()
@click.option(
    "--dataset",
    "-d",
    type=click.Choice(
        [
            "headlines",
            "us_place",
            "world_place",
            "art-release",
            "art-length",
            "historical-figure",
            "nyc-place",
            "ALL",
        ]
    ),
    default="ALL",
)
def main(dataset: str) -> None:
    """Convert world models CSV data to property files."""
    if dataset == "nyc-place" or dataset == "ALL":
        # Read NYC places data
        places_path = DATA_DIR / "world_models" / "nyc_place.csv"
        df = pd.read_csv(places_path)

        # Process each coordinate type
        coord_types: list[Literal["long", "lat"]] = ["long", "lat"]
        for coord_type in coord_types:
            props = create_nyc_place_properties(df, coord_type)

            # Skip if no places
            if not props.value_by_name:
                continue

            # Save to file
            out_path = DATA_DIR / "properties" / f"wm-nyc-place-{coord_type}.yaml"
            props.to_yaml_file(out_path)

            # Print info
            print(
                f"Created property {out_path.stem} with {len(props.value_by_name)} entities"
            )

    if dataset == "historical-figure" or dataset == "ALL":
        # Read historical figures data
        figures_path = DATA_DIR / "world_models" / "historical_figure.csv"
        df = pd.read_csv(figures_path)

        # Process birth year properties
        birth_props = create_historical_figure_birth_properties(df)
        birth_out_path = DATA_DIR / "properties" / "wm-person-birth.yaml"
        birth_props.to_yaml_file(birth_out_path)
        print(
            f"Created property {birth_out_path.stem} with {len(birth_props.value_by_name)} entities"
        )

        # Process death year properties
        death_props = create_historical_figure_death_properties(df)
        death_out_path = DATA_DIR / "properties" / "wm-person-death.yaml"
        death_props.to_yaml_file(death_out_path)
        print(
            f"Created property {death_out_path.stem} with {len(death_props.value_by_name)} entities"
        )

        # Process age properties
        age_props = create_historical_figure_age_properties(df)
        age_out_path = DATA_DIR / "properties" / "wm-person-age.yaml"
        age_props.to_yaml_file(age_out_path)
        print(
            f"Created property {age_out_path.stem} with {len(age_props.value_by_name)} entities"
        )

    if dataset == "headlines" or dataset == "ALL":
        # Read headlines data
        headlines_path = DATA_DIR / "world_models" / "headline.csv"
        df = pd.read_csv(headlines_path, index_col=0)

        # Add year column
        df["year"] = df["pub_date"].apply(get_year)

        # Filter for year 2013 and page 3
        group_df = df[(df["year"] == 2013) & (df["print_page"] == 3)]

        # Create mapping of headlines to dates and sort by date
        value_by_name: dict[str, int | float] = dict(
            sorted(
                {
                    str(row["headline"]): date_to_int(row["pub_date"])
                    for _, row in group_df.iterrows()
                }.items(),
                key=lambda x: x[1],
            )
        )

        # Skip if no articles
        if value_by_name:
            q_prefix = "about NYT articles:\n\n"
            # Create properties object
            props = Properties(
                gt_question=q_prefix + 'Was "{x}" published later than "{y}"?',
                lt_question=q_prefix + 'Was "{x}" published earlier than "{y}"?',
                gt_open_ended_question='Which article was published later: "{x}" or "{y}"?',
                lt_open_ended_question='Which article was published earlier: "{x}" or "{y}"?',
                value_by_name=value_by_name,  # type: ignore
            )

            # Save to file
            out_path = DATA_DIR / "properties" / "wm-nyt-pubdate.yaml"
            props.to_yaml_file(out_path)
            print(
                f"Created property {out_path.stem} with {len(value_by_name)} entities"
            )

    if dataset == "us_place" or dataset == "ALL":
        # Read US places data
        places_path = DATA_DIR / "world_models" / "us_place.csv"
        df = pd.read_csv(places_path)

        # Process each entity type and coordinate type
        for entity_type in df["entity_type"].unique():
            # Process coordinates
            coord_types: list[Literal["long", "lat"]] = ["long", "lat"]
            for coord_type in coord_types:
                props = create_us_place_properties(df, coord_type, entity_type)

                # Skip if no places
                if not props.value_by_name:
                    continue

                entity_str = (
                    "natural" if entity_type == "natural_place" else entity_type
                )
                # Save to file
                out_path = (
                    DATA_DIR / "properties" / f"wm-us-{entity_str}-{coord_type}.yaml"
                )
                props.to_yaml_file(out_path)

                # Print info
                print(
                    f"Created property {out_path.stem} with {len(props.value_by_name)} entities"
                )

            # Process metrics
            us_metrics_map = {
                "population": "popu",
                "density": "dens",
            }
            for metric, metric_str in us_metrics_map.items():
                props = create_us_place_metric_properties(df, metric, entity_type)  # type: ignore

                # Skip if no valid data
                if props is None or not props.value_by_name:
                    continue

                # Save to file
                out_path = (
                    DATA_DIR / "properties" / f"wm-us-{entity_type}-{metric_str}.yaml"
                )
                props.to_yaml_file(out_path)

                # Print info
                print(
                    f"Created property {out_path.stem} with {len(props.value_by_name)} entities"
                )

    if dataset == "world_place" or dataset == "ALL":
        # Read world places data
        places_path = DATA_DIR / "world_models" / "world_place.csv"
        df = pd.read_csv(places_path)

        # Filter out rows with missing coordinates
        df = df.dropna(subset=["latitude", "longitude"])

        entity_type_map = {
            "natural_place": "natural",
            "populated_place": "populated",
            "structure": "structure",
        }

        # Process each entity type and coordinate type
        for entity_type in df["entity_type"].unique():
            # Process coordinates
            coord_types: list[Literal["long", "lat"]] = ["long", "lat"]
            for coord_type in coord_types:
                props = create_world_place_properties(df, coord_type, entity_type)

                # Skip if no places
                if not props.value_by_name:
                    continue

                entity_type_str = entity_type_map[entity_type]
                # Save to file
                out_path = (
                    DATA_DIR
                    / "properties"
                    / f"wm-world-{entity_type_str}-{coord_type}.yaml"
                )
                props.to_yaml_file(out_path)

                # Print info
                print(
                    f"Created property {out_path.stem} with {len(props.value_by_name)} entities"
                )

            # Process metrics
            world_metrics: list[Literal["total_area", "population"]] = [
                "total_area",
                "population",
            ]
            for metric in world_metrics:
                props = create_world_place_metric_properties(df, metric, entity_type)

                # Skip if no valid data
                if props is None or not props.value_by_name:
                    continue

                # Convert entity type to shorter name
                entity_type_str = entity_type_map[entity_type]

                metric_str = metric.replace("total_area", "area")
                # Save to file
                out_path = (
                    DATA_DIR
                    / "properties"
                    / f"wm-world-{entity_type_str}-{metric_str}.yaml"
                )
                props.to_yaml_file(out_path)

                # Print info
                print(
                    f"Created property {out_path.stem} with {len(props.value_by_name)} entities"
                )

    if dataset in ["art-release", "art-length"] or dataset == "ALL":
        # Read art data
        art_path = DATA_DIR / "world_models" / "art.csv"
        df = pd.read_csv(art_path)

        if dataset == "art-release" or dataset == "ALL":
            props_by_type = create_art_release_properties(df)

            # Save each entity type to a separate file
            for entity_type, props in props_by_type.items():
                out_path = DATA_DIR / "properties" / f"wm-{entity_type}-release.yaml"
                props.to_yaml_file(out_path)

                # Print info
                print(
                    f"Created property {out_path.stem} with {len(props.value_by_name)} entities"
                )

        if dataset == "art-length" or dataset == "ALL":
            props_by_type = create_art_length_properties(df)

            # Save each entity type to a separate file
            for entity_type, props in props_by_type.items():
                out_path = DATA_DIR / "properties" / f"wm-{entity_type}-length.yaml"
                props.to_yaml_file(out_path)

                # Print info
                print(
                    f"Created property {out_path.stem} with {len(props.value_by_name)} entities"
                )


if __name__ == "__main__":
    main()
