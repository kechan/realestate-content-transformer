# Empty "By property type" sentence on city pages

Status: open, not fixed. Found 2026-11-10 in the post-run check of the 2026-11 PROD run (version 202611).

## Symptom

A city page with enough listings overall for the statistical opening, but no property type with enough listings for its own price, gets a second sentence that says nothing:

| Page | geog_id | Listings (all / detached / semi / townhouse / condo) | Published second sentence |
|---|---|---|---|
| Montague PE (`pe_montague`) | `g30_f85qwk5e` | 11 / 10 / 1 / - / - | "By property type, the median was not provided for any property type." |
| Lot 15 PE (`pe_lot-15`) | `g30_f862nj4y` | 14 / 7 / 1 / 4 / 2 | "By property type, the median was not provided for detached, semi-detached, townhouse, or condominium properties." |

The opening sentence and the numbers are correct. The sentence is filler, and it reads as if data is missing from the site.

Frequency: 2 of 639 pages checked this month (eval places for AB, BC, ON, QC, SK, NS and MB; every level-30 place for NL, NB and PE). It can happen in any province and any month: it depends only on how listings split across types.

## Cause

The listing-count gate (11) is applied separately to the overall count and to each property type. When the overall count passes and every type fails, the prompt still gets the full city-page guidelines.

Guideline 3 in `LocalLogicGPTRewriter`'s city-page prompt (`realestate_spam/llm/chatgpt.py`, around line 478, the `include_start_with_guideline` / SALE branch) always asks for the sentence:

> 3. Immediately follow with a second sentence starting 'By property type, the median was' that reports the median list price for each property-type data point provided above, in the order detached, semi-detached, townhouse, condominium. If a data point is genuinely missing, omit that property type rather than inventing a value.

With no property-type data points, the model still writes the sentence it was told to write, and fills it with "not provided".

## Proposed fix (not started)

Include guideline 3 only when at least one property-type median is passed. When none is, the opening sentence stands alone.

This is a prompt change in `realestate_spam`. Per `evals/README.md`, test it locally on the eval list before it ships, and add Montague and Lot 15 to the list as regression cases for this path. Check that pages with at least one eligible type still get the sentence.

## Check after a PROD run

For a page whose overall count is at least 11 and whose four type counts are all below 11 (or missing), the published text should not contain "By property type" or "not provided".
