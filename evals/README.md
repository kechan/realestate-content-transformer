# City-page rewrite evals

Rough plan for a repeatable check of city-level housing rewrites. Nothing here is automated yet.

## Why

Rewrite quality can change without any change to our data:

1. **Prompt changes** in `realestate_spam` (`LocalLogicGPTRewriter`) or in the data points `rewrite_city` passes.
2. **Model drift**: the same model producing worse output over time, or a model switch (e.g. gpt-4o to GPT-6 Luna).

Until now these were caught by ad hoc sampled reviews of PROD after a run. This directory holds a fixed set of geographies to re-run before a change ships and periodically afterwards.

## Priority

Accuracy over variation. A rewrite that stays close to the original wording but is correct is acceptable; a varied rewrite that changes a fact is not.

## Files

- `eval_geog_ids.txt`: the geographies to evaluate, one per line as `<geog_id>, <brief note>`. The note is for a quick human glance only: what had or can go wrong there, or why the geography is in the set. An entry does not need a known issue.

  The description contains commas, so this file is not a valid `--geog_ids_file` input as is. Take the first field of each line.

## When to run

- Before merging any prompt change, or any change to the metrics injected into the prompt.
- Before switching model, and on the new model before its first PROD run.
- Periodically (e.g. monthly, alongside the regular run) to catch model drift.

## How to run

Locally, never against PROD content:

- Reads of source data (original LocalLogic text, `rlp_mkt_trends_current`) may come from PROD, read-only.
- Generate rewrites for each geog_id in the list with the real code path, so the listing-count threshold and plain-rewrite fallback apply exactly as in a PROD run.
- Do not write rewrites back to any PROD index. Keep outputs as local files.
- Generate each case more than once. At `temperature=1.0` a single pass proves little.

## What to check

For each output, against the original text and the source metrics:

**Statistics**
- Opening sentence names the correct place and province, with the correct month, listing count and overall median.
- Property-type prices match the source, in the order detached, semi-detached, townhouse, condominium.
- A property type below the listing-count threshold (11) or with no data has no price, and none is invented.
- If the overall count is below threshold or missing, there is no statistical opening at all.

**Meaning**
- Figures are not attributed to a larger area (metro, region) than the page's own geography.
- Qualifiers keep their strength (mainly, roughly, about, nearly, most, around one third).
- Each figure keeps its subject: shares of residents stay about people, shares of dwellings stay about dwellings.
- "Rented" stays occupied-by-renters, not "available for rent".

**Inclusiveness**
- Numeric facts from the original (ownership/rental share, construction-era breakdown) are retained.
- Descriptive content (dwelling mix, bedroom sizes) may be compressed or dropped.

**Form**
- Past tense for the two statistical sentences only; present tense for the rest.
- Plain wording, no added commentary.
- Output wrapped in `<housing></housing>`.

## Maintaining the list

The list starts with geographies that had mistakes in PROD reviews. Planned coverage, per province:

- The 5 largest geographies by listing inventory. These are the most important and visible pages.
- 5 sampled medium-size geographies.
- 5 sampled rural geographies.

Also:

- Add a geog_id whenever a review finds a real mistake, with a terse note of what went wrong.
- Keep cases that now pass. They are the regression set.
- Aim to cover both paths (statistical and plain rewrite), each property-type gating outcome, places whose original text does not name them, and eventually French.

## Not done yet

- A script to generate outputs for the list and save them locally.
- Automated checks. The statistics and form checks are deterministic and can be coded; the meaning checks are manual for now.
- A stored baseline of accepted outputs to diff against.
- The largest / medium / rural geographies per province described above.
- Known-issue cases from ON and QC, and French pages.
