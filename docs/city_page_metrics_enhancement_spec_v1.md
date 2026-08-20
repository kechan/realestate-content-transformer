# City Page Metrics Enhancement — Requirement & Tech Scope Spec (v1)

**Status:** Draft — scope confirmed via email, implementation not started
**Author:** Kelvin Chan (compiled from email exchange, 2026-07)
**Scope:** v1 only. Rollout is expected to happen in phases; later phases are out of scope for this document.

## 1. Background

City-level page content (the `housing` section rewritten by `LocallogicContentRewriter.rewrite_cities()` in [`realestate_content_transformer/data/pipeline.py`](../realestate_content_transformer/data/pipeline.py)) currently presents a single non-dated **average** listing price, computed live from the active-listings index (`rlp_listing_current`) via `get_avg_price_and_active_pct()`.

This is being replaced with a set of **dated, median-based market metrics** sourced from the `rlp_mkt_trends_current` index, matching the style already used elsewhere in the product for market-trend reporting.

## 2. Requirement (from email)

Requirement was communicated informally via email, in two passes:

1. **Initial ask:** present 4 metrics instead of the generic average — as-of month, active listing count, overall median list price, and a property-type breakdown for **DETACHED** and **CONDO**.
2. **Follow-up (confirmed):** the property-type breakdown should also include **SEMI-DETACHED** and **TOWNHOUSE** — i.e. all 4 mkt_trends property types that have a direct pipeline analog, not just the original 2.

No formal PRD/ticket exists for this feature — this document is the first structured spec, written retroactively from the email thread and follow-on discussion.

### Sample v1 target text (Barrie, ON — as supplied by stakeholder)

> As of June 2026, there were 412 residential properties for sale in Barrie, Ontario, with a median MLS® list price of $779,022.
> By property type, the median was $842,000 for a single-family detached home and $434,800 for a condominium. Barrie is largely an owner-occupied market, about 70% of residents own their home, and the housing stock leans toward detached houses, with smaller apartment buildings and townhomes filling out the mix, most with three or more bedrooms. To see what your budget covers here, browse current Barrie listings below, or connect with a local Royal LePage agent who works this market day to day.

**Note:** the numbers in this sample are illustrative, not a literal live pull — confirmed by cross-checking Barrie's actual current data (see §4), which doesn't match this example numerically. The *structure and tone* of the sample is the actual spec; the numbers are not. The sample also predates the SEMI-DETACHED/TOWNHOUSE follow-up, so a v1 output will have a longer property-type breakdown sentence than shown above.

## 3. Metrics to Add

6 numeric metrics, plus one piece of date context (the as-of month) used to frame all of them in the rewritten text:

| # | Metric | Description |
|---|---|---|
| 1 | Active listing count | Count of residential properties for sale, city-wide, all property types |
| 2 | Overall median list price | Median MLS® list (asking) price, city-wide, all property types |
| 3–6 | Median list price by property type | One value each for DETACHED, SEMI-DETACHED, TOWNHOUSE, CONDO |

**As-of month** (not counted above — it's the date label, not a measured metric): always the last **completed** calendar month at run time (e.g. a run in July 2026 reports June 2026 data). All 6 metrics above share this same as-of month.

**Explicitly confirmed with stakeholder:**
- All price figures are **list/asking price**, never sold price.
- "Median" is intentionally accepted as `mkt_trends`'s `last_mth_median_asking_price` field — defined as the median asking price of listings *newly added* in that month (a flow metric), not a median recomputed over the full active inventory (a stock metric). This distinction was raised and the stakeholder confirmed the flow-metric definition is what's intended; no further semantic work needed here.
- The generic average-price line is **removed** from city-level output entirely, replaced by these metrics.

**Explicitly out of scope for v1 (confirmed by omission / non-response, not to be assumed available):**
- LUXURY, INVESTMENT, RENTAL breakdowns on the city page — `mkt_trends` has no matching `propertyType` bucket for these (see §4), and the stakeholder ask never named them. If a future phase needs them, that's new work, not a v1 gap to fix.
- The existing property-type **subpages** (`rewrite_property_types()`) are unaffected by this spec — see §6.

## 4. Data Availability — Verified

Investigated directly against **PROD** Elasticsearch (`esp.royallepage.ca:443`, index `rlp_mkt_trends_current`), using Barrie, ON (`geog_id = g30_dpzk6g9n`, `longId = on_barrie`) as the worked example. Verified 2026-07-20.

All 6 target metrics are available. `mkt_trends` documents are keyed `{geog_id}_{propertyType}` — one document per property type, not a single combined document. Confirmed working docs:

- `g30_dpzk6g9n_ALL`
- `g30_dpzk6g9n_DETACHED`
- `g30_dpzk6g9n_SEMI-DETACHED`
- `g30_dpzk6g9n_TOWNHOUSE`
- `g30_dpzk6g9n_CONDO`

Relevant fields (both are `nested` time-series arrays of `{month, value}`; take the last entry for "last completed month"):

| Field | Maps to |
|---|---|
| `metrics.mth_end_snapshot_listing_count` | Active listing count (metric #2, `ALL` propertyType doc only) |
| `metrics.last_mth_median_asking_price` | Median list price (metrics #3–7, one value per propertyType doc) |

### Barrie worked example — June 2026 (last completed month as of verification date)

| propertyType | Active listings | Median list price |
|---|---|---|
| ALL | 588 | $725,000 |
| DETACHED | 384 | $799,900 |
| SEMI-DETACHED | 6 | $617,499.50 |
| TOWNHOUSE | 99 | $589,450 |
| CONDO | 99 | $450,000 |

### Known data-quality risk: low sample size for some property types

Barrie's SEMI-DETACHED bucket has only **6** active listings for the month. A median over 6 data points is noisy and will fluctuate month to month more than the other categories. This isn't a Barrie-specific quirk — smaller/rural geog_ids processed by this pipeline (e.g. `mb_grey`, `pe_lot-48`) will hit this far more severely, potentially with 0–2 active listings for a given property type in a given month.

**Precedent already in the codebase:** `get_avg_price_and_active_pct()` ([pipeline.py:986](../realestate_content_transformer/data/pipeline.py)) already defines `MIN_LISTING_THRESHOLD = 10` and suppresses the "% of active listings" stat below that threshold for exactly this reason.

**Decision (2026-07, user):** the new mkt_trends-based fetch does **not** apply a similar threshold. Any low-sample-size value (e.g. Barrie's n=6 SEMI-DETACHED) is reported verbatim, as returned by `mkt_trends`, with no suppression logic in `pipeline.py`. Genuinely missing/null values (e.g. a property type with zero listings in the month, likely returning `None`/absent from the mkt_trends doc) are still expected to be omitted — but that's left to the LLM prompt instruction ("if a data point is genuinely missing, omit that property type"), not pre-filtered in code. This was tested and confirmed working against real Barrie data (§8).

### Two candidate "current price" fields — resolved

`mkt_trends` docs also expose `metrics.current_metrics.median_asking_price`, a second price field that does **not** match `last_mth_median_asking_price` for the same month (e.g. Barrie ALL: $699,900 vs. $725,000). Its exact time window is undocumented/unclear (likely a rolling/live figure rather than locked to a closed calendar month). **Resolved:** use `last_mth_median_asking_price` only — this is the field explicitly confirmed with the stakeholder (§3) and it aligns with the "as of last month" framing. `current_metrics` is not used by this feature.

## 5. Property-Type Taxonomy Mismatch

`mkt_trends`'s `propertyType` enum (`ALL, DETACHED, SEMI-DETACHED, TOWNHOUSE, CONDO, OTHER`) does not fully align with the pipeline's existing property-type subpage set (`LUXURY, CONDO, SEMI-DETACHED, TOWNHOUSE, INVESTMENT, RENTAL`):

| Pipeline property type | mkt_trends equivalent |
|---|---|
| CONDO | ✅ CONDO |
| SEMI-DETACHED | ✅ SEMI-DETACHED |
| TOWNHOUSE | ✅ TOWNHOUSE |
| LUXURY | ❌ none |
| INVESTMENT | ❌ none |
| RENTAL | ❌ none |

This is fine for v1 as scoped — the city-level breakdown only needs DETACHED/SEMI-DETACHED/TOWNHOUSE/CONDO, all of which have a direct mkt_trends match. It becomes relevant only if a future phase asks for LUXURY/INVESTMENT/RENTAL breakdowns on the city page, which would need a different data source (likely the existing `rlp_listing_current` query pattern used by `get_avg_price_and_active_pct()`, filtered by `carriageTrade`/`searchCategoryType`/`transactionType` as documented in the ES query skill's listing schema reference).

## 6. Implementation Constraints

Confirmed with stakeholder: implement as **least-invasive as possible**, with **zero behavior change** to the property-type subpages.

- **`get_avg_price_and_active_pct()`** ([pipeline.py:986](../realestate_content_transformer/data/pipeline.py)) is **not modified or removed**. It remains the data source for `rewrite_property_types()` / `rewrite_property_type()`, unchanged.
- A **new, separate fetch method** is added to `LocallogicContentRewriter` to query `rlp_mkt_trends_current` and return the 6 new values (count + 5 medians) for a given `geog_id`.
- **`realestate_content_transformer/data/pipeline.py`** — only `rewrite_city()`'s `params_dict` construction changes (currently builds a single `{'Average price on MLS®': avg_price}` entry; this becomes the new metric set). This is scoped to the `property_type=None` (city-level) code path only.
- **`realestate_spam/realestate_spam/llm/chatgpt.py`** — `LocalLogicGPTRewriter.construct_openai_user_prompt()`'s `property_type is None` guideline branch (city-level prompt, roughly L464–486) is updated: new opening-sentence template (replacing "always start with 'Homes for sale in {city}'" with something reflecting the as-of-month/count/median framing), plus a new guideline line covering how to phrase the multi-value property-type breakdown sentence. The `else` branch (property-type-specific prompts, used by subpages, roughly L487–507) is **not touched**.
- No changes anticipated to `archive.py`, `data_models.py`, or the CLI script beyond what's naturally needed to pass the new data through.

## 7. Open Questions / Not Yet Decided

- Whether French (`overrides_fr`) output needs the same treatment in v1 or is deferred to a later phase (not addressed in the email thread).
- Whether the "70% owner-occupied / housing stock mix" portion of the sample text (unrelated to mkt_trends) stays exactly as-is from the current LocalLogic content, or also changes — nothing in the email thread suggests it changes; treating as unchanged unless told otherwise.

**Resolved since first draft of this doc:**
- ~~Exact prompt wording for the opening sentence~~ — RESOLVED. SEO engineer feedback (email + mockup image) confirmed: `"As of {{month_year}}, there are {{active_listing_count}} homes for sale in {{city}}, {{province_full}}, with a median MLS® list price of {{median_price_all}}."` User confirmed this was the *only* actual feedback in that email — the mockup's asking-price range, "average" framing, Royal LePage House Price Survey block, computed price gap, and narrower detached/condo-only breakdown were not intentional asks and are explicitly out of scope. See `city_page_metrics_enhancement_prompt_v1.txt` for the full current draft.
- ~~Minimum listing-count threshold~~ — RESOLVED. No threshold/suppression logic in `pipeline.py` (see §4). Low-sample values are passed through verbatim; the LLM prompt handles genuinely-missing data.
- New requirement surfaced by the confirmed opening sentence: `{{province_full}}` (full province name, e.g. "Ontario") is needed. The pipeline currently only carries `prov_code` ("ON") — a code-to-full-name mapping needs to be added to `pipeline.py` before implementation.

## 8. Prompt Testing (informal, pre-implementation)

The draft prompt (`city_page_metrics_enhancement_prompt_v1.txt`) was manually tested against `gpt-5.6-luna` (OpenAI's current cost-tier-optimized model, replacing the now-superseded `gpt-4o` referenced in §1/§6 model configuration — `gpt-4o` itself is not deprecated in the API as of this writing, but `gpt-5.6-luna` was selected as a suitable low-cost model for this straightforward rewrite task) by pasting the fully-filled Barrie prompt directly into the model, ahead of any code changes. Two consecutive runs both correctly followed the (then-hybrid) opening structure, the property-type breakdown ordering, avoided the word "average," and retained all original housing content without truncating for word count. This is a good signal the prompt design is sound, though it predates the now-finalized pure "As of {{month_year}}..." opening and should be re-tested against that exact wording before implementation is considered final.
