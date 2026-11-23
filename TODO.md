# TODO / Open Issues

## 1. Several geog_ids share one longId (pages can show the wrong place)

Status: open, not fixed on purpose. Found 2026-10-16. Details, affected pages, the
full list and the ad-hoc patch: `docs/shared_longid_issue.md`.

- The pipeline assumes geog_id and longId are 1:1. 121 longIds are each shared by
  2 geog_ids (QC 59, NB 40, MB 9, BC 5, NS 4, ON 2, AB 1, YT 1).
- Pages are written by longId but versions are read by `localLogicId[0]`, so on a
  shared page both places write and the last one processed wins.
- Known wrong pages now: 12 in QC (e.g. qc_lange-gardien, qc_saint-donat) and
  ns_lunenburg (shows the town, 20 listings, instead of the district, 178; the largest
  wrong page so far).
  on_hamilton is correct now but at risk on every run; it is in the eval list.
- Next step, not started: confirm which geog_id each shared page is meant for
  (compare the live page's transport/services text with each candidate).
- If someone complains: patch with a forced `--geog_id` run (see the doc). The patch
  is undone by the next province run.

## 2. Empty "By property type" sentence when no type has a price

Status: open, not fixed. Found 2026-11-10. Details and proposed fix:
`docs/empty_property_type_sentence_issue.md`.

- When a page's overall listing count passes the threshold (11) but no property type
  does, the prompt still asks for the "By property type, the median was ..." sentence,
  and the model writes filler ("... was not provided for any property type").
- Seen on 2 pages in the 2026-11 run: pe_montague, pe_lot-15.
- Fix (not started): send guideline 3 of the city-page prompt (`realestate_spam`) only
  when at least one property-type median is passed; eval before shipping.
