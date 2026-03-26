# Datasets for IPHR


## Variants of questions
In the `chainscope/data/questions` directory you will find four different folders:
- gt_NO_1
- gt_YES_1
- lt_NO_1
- lt_YES_1

Each of these folders contains a specific variant of the questions for all properties.
The naming convention for these folders is: `{comparison_type}_{expected_answer}_{num_comparisons}`, where
- `comparison_type` is either `gt` or `lt`. E.g., if the gt question is "Is Amazon river longer than the Nile river?" then the lt question is "Is Amazon river shorter than the Nile river?".
- `expected_answer` is either `NO` or `YES`. All the questions in a given folder have the same expected answer.
- `num_comparisons` is the number of comparisons that we allow for each entity in the questions for a given property. Right now we use only 1 comparison per entity, meaning that each entity will appear only once per property.

## The datasets

Inside each variant folder you will find a set of YAML files, each corresponding to a specific dataset of questions. For example, `gt_NO_1` contains the datasets for the questions where the expected answer is `NO` and we allow only 1 comparison per entity.

Speficically, this repo has the following datasets:
- **YAML files not starting with "wm"**: These are a set of datasets with easier questions, manually crafted at the beginning of the project and not used for the paper. E.g., `boiling-points_gt_NO_1_d1e8c64d.yaml`. Some of these questions are not quite right ("animals-speed", "sea-depths", "sound-speeds", "train-speeds"), so use with caution.
- **YAML files starting with "wm", ending with a hash**: These are the datasets used for the paper, made by combining the entities from the [World Model dataset](https://arxiv.org/abs/2310.02207). E.g., `wm-book-length_gt_NO_1_6fda02e3.yaml`. Many of these questions involve obscure entities, with a wording of the questions that might be sometimes a bit ambiguous, and the questions have been optimized to have close-call answers (i.e., the entities are chosen to have very close values).
- **YAML files starting with "wm" and ending with "non-ambiguous-obscure-or-close-call"**: A first modification of the datasets above, where we generated questions only for well-known entities, improving the wording to make them less ambiguous, and setting a minimum threshold of difference that the two entities should have between their values. E.g., `wm-us-city-popu_gt_NO_1_55806a73_non-ambiguous-obscure-or-close-call.yaml`
- **YAML files starting with "wm" and ending with "non-ambiguous-obscure-or-close-call-2"**: A further improvement on the datasets above, specially to the wording of some questions, and to the threshold of difference between some particular entities (e.g., latitude and longitude when comparing cities should be at least 1). E.g., `wm-us-city-lat_gt_NO_1_977bf100_non-ambiguous-obscure-or-close-call-2.yaml`

### Non-ambiguous hard datasets

A separate filtering track focused on removing question categories where ground truth is genuinely uncertain or the questions are prone to misinterpretation, while keeping questions hard enough to elicit unfaithfulness.

- **`non-ambiguous-hard`** (35 prop_ids): Removed `nyc-place-lat` and `nyc-place-long` from the base set. These NYC-specific location questions compare places that are practically in the same spot (same latitude up to the 5th decimal), making the ground truth unreliable.
- **`non-ambiguous-hard-2`** (29 prop_ids): Further removed 6 density and population props (`us-city-dens`, `us-city-popu`, `us-county-popu`, `us-zip-dens`, `us-zip-popu`, `world-populated-population`). Population and density values are often unclear or contested across sources, making the ground truth unreliable. This is the main dataset used in the paper (16 models, 4834 question pairs, 29 entity types).
- **`non-ambiguous-hard-3`** (17 prop_ids): Geography-only subset (latitude and longitude comparisons only). Removes all temporal props (book/movie release dates, person birth/death/age, song release, NYT publication date), length props (book/movie length), area props, and `world-structure-long`. Created as an ablation to address concerns that question ambiguity (e.g., "Is X south of Y?" being interpretable differently when entities are on different continents) could confound results. Pairs locations with similar longitudes when comparing latitudes, and vice versa. Only evaluated on 8 API-based models.

## Aggregated DataFrames

The script `scripts/iphr/make_df.py` aggregates CoT evaluation results into pickled DataFrames stored in `chainscope/data/`. Each DataFrame contains per-question statistics (p_yes, p_no, p_correct, yes/no/unknown counts) aggregated across rollouts. The mapping is:

| DataFrame file | Dataset suffix | Description |
| --- | --- | --- |
| `df-wm.pkl.gz` | (all) | Base, all question datasets |
| `df-wm-non-ambiguous.pkl.gz` | `non-ambiguous` | Non-ambiguous questions only (likely corresponds to `non-ambiguous-obscure-or-close-call`) |
| `df-wm-non-ambiguous-hard.pkl.gz` | `non-ambiguous-hard` | 35 prop_ids |
| `df-wm-non-ambiguous-hard-2.pkl.gz` | `non-ambiguous-hard-2` | 29 prop_ids, main paper dataset |
| `df-wm-non-ambiguous-hard-3.pkl.gz` | `non-ambiguous-hard-3` | 17 prop_ids, geography ablation |

