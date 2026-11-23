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
- After every monthly PROD run, on what was actually published (see "After a PROD run" below).

## How to run

### Before a change ships

Locally, never against PROD content:

- Reads of source data (original LocalLogic text, `rlp_mkt_trends_current`) may come from PROD, read-only.
- Generate rewrites for each geog_id in the list with the real code path, so the listing-count threshold and plain-rewrite fallback apply exactly as in a PROD run.
- Do not write rewrites back to any PROD index. Keep outputs as local files.
- Generate each case more than once. At `temperature=1.0` a single pass proves little.

### After a PROD run

Check what customers see, read-only:

1. Read the published page from PROD ES (`rlp_content_geo_overrides_current`, keyed by the geog's longId) and check it against the original text and the source metrics, as in "What to check" below. Confirm its version is the current `%Y%m`.
2. Open the live page on royallepage.ca and corroborate it with the ES record:
   - The live housing text matches what ES holds (no stale or cached content).
   - The sections we do not rewrite (transport, services, character) come from LocalLogic for whichever place the site renders. They should match the eval geog_id's text in `rlp_geo_details_en`. If they match a different place, the page is not about the geog_id we evaluated.

Step 2 matters most for longIds shared by several geog_ids, where the pipeline can write another place's content onto the page. See `docs/shared_longid_issue.md`.

### Shared-longId entries

Four entries share their longId with another level-30 geography. Their notes in `eval_geog_ids.txt` contain `SHARED LONGID` (`grep "SHARED LONGID"`). Whichever geography is processed last in a province run writes the page, so a failure on these is usually the shared-longId issue, not the rewrite. Report them separately from rewrite results.

| Entry | Shares longId with | Status (2026-11 run) |
|---|---|---|
| Hamilton ON (`on_hamilton`) | Hamilton Township (`g30_drb6s2f7`) | Shows the city. Passes. Can flip on any ON run. |
| Lunenburg NS (`ns_lunenburg`) | the town (`g30_dxfk2qef`) | Shows the town (22 listings, $599,950), not the district. Expected to fail until fixed. |
| Portage la Prairie MB (`mb_portage-la-prairie`) | the rural municipality (`g30_cbfhkydn`) | Shows the city. Low risk: the rural municipality has no English housing profile. |
| Moncton NB (`nb_moncton`) | the parish (`g30_f81yemc4`) | Housing text is the city's, but `localLogicId[0]` is the parish, so the LocalLogic sections are the parish's. Both have English text, so the page can flip. Its statistics are also wrong for a separate reason (see "Board category-code entries"). |

Only level-30 geographies count: the pipeline rewrites level 30 only. Some eval places also share a longId with a level-40 metropolitan area (e.g. Red Deer, Canmore, Greater Sudbury); that does not affect the page.

A failure on any other entry can still come from this issue if its page is written by another geography, so check the page's `localLogicId` and the source geog when a statistics check fails.

How to build the live page URL from a longId is still to be written down.

### Board category-code entries

A separate problem from shared longIds, in the source metrics rather than in our pipeline. The per-type counts and medians in `rlp_mkt_trends_current` come from each listing's numeric `searchCategoryType` (101 condo, 102 townhouse, 103 detached, 104 semi). Some listing boards use these codes differently:

- Residential homes get `999` (uncategorized), so they fall out of all four types and the overall count.
- On the SAINTJOHN board (NB), vacant land gets `103`, so it is counted and priced as detached.

The page then shows a count that is too low and medians that do not describe homes, even though the rewrite reproduces the source correctly. The listing itself still says what it is (`listingType` RES / CONDO / VACANT), so the fix belongs in realestate-analytics' code mapping, not here.

Entries affected are marked `BOARD CATEGORY CODES` in `eval_geog_ids.txt` (`grep "BOARD CATEGORY CODES"`). Judge their rewrites against the source metrics as usual, and report the source problem separately.

| Entry | Board | Evidence (sale listings in `rlp_listing_current`, 2026-11-10) |
|---|---|---|
| Moncton NB (`g30_f81ym5kb`) | SAINTJOHN | 308 of 374 residential listings are `999`. 12 of 16 `103` "detached" listings are vacant land. Published: 16 listings, $375,000 overall, $395,000 detached. The parish (`g30_f81yemc4`) is worse: all 60 `103` listings are land (median $109,995). |
| Victoria BC (`g30_c2878bj4`) | VICTORIA | 172 of 517 residential listings are `999` (asking about $669,000 to $3M, no property type) and only 1 is `103`, so there is no detached price and condo is the only eligible type. |

To check a place, run a read-only aggregation of its sale listings (`guid` contains the geog_id) by `boardType`, `searchCategoryType` and `listingType`, and compare with a board that maps correctly (e.g. Halifax on NOVASCOTIA: 1,107 residential in `103`, land in its own code `501`). A low share of residential listings in 101-104 is not by itself this problem: multi-unit listings (201-204) and some mobile-home codes are excluded by design (e.g. Amherst NS).

How widespread this is on other boards, in NB and elsewhere, has not been checked.

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
- 5 medium-size geographies: hand-picked from the places ranked just below the top 5 by listing inventory, with an English profile. "Medium" is relative to the province: about 150-1,100 listings in AB, BC and ON, 50-100 in QC, 30-65 in SK, about 10 in MB, where only eight places reach the threshold, and 30-95 in NS. Entries already in the list for a known issue count towards the five.
- Rural geographies: small towns, villages, townships, rural municipalities, electoral areas and reserves away from the big metro areas, with an English profile. Most fall below the listing threshold and take the plain-rewrite path, so 3 per province is enough (not 5): a mix of places just above the threshold and places below it.

Done so far for AB, BC, ON, QC, MB, SK and NS: largest, medium and rural. Only geographies with an English profile are picked; in NS many of the top-ranked places (e.g. Queens, East Hants) have none. The other provinces and territories are not done.

Also:

- Add a geog_id whenever a review finds a real mistake, with a terse note of what went wrong.
- Keep cases that now pass. They are the regression set.
- Aim to cover both paths (statistical and plain rewrite), each property-type gating outcome, places whose original text does not name them, and eventually French.

## Not done yet

- A script to generate outputs for the list and save them locally.
- Automated checks. The statistics and form checks are deterministic and can be coded; the meaning checks are manual for now.
- A stored baseline of accepted outputs to diff against.
- Any geographies for provinces and territories outside AB, BC, ON, QC, MB, SK and NS.
- Known-issue cases from ON and QC, and French pages.
- The royallepage.ca URL pattern for a longId (city page and property-type subpages), for the live-page check.
