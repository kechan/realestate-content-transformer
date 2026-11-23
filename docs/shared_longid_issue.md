# Several geog_ids share one longId

Status: open, not fixed on purpose. Found 2026-10-16 while looking at the QC run (run 10).
Do not fix until it is fully understood (see "Before any fix" below).

## What is wrong

The pipeline assumes geog_id and longId are 1:1. They are not. Several geog_ids can map
to one longId. A geog_id never maps to more than one longId.

- A geo record in `rlp_content_geo_current` is keyed by geog_id (the ES `_id`). Its
  `longId` is just a field in `_source`.
- The longId is built from the province and the name slug, so two different places
  with the same name get the same longId. Example: Saint-Lambert near Montreal
  (`g30_f25dz3cu`) and Saint-Lambert in Abitibi (`g30_f0xw2pzv`) are both
  `qc_saint-lambert`.
- The page store `rlp_content_geo_overrides_current` is keyed by longId, so the two
  places share one page record (one customer-facing page). That record's
  `localLogicId` usually names one of them (`['g30_f25dz3cu']`).

### How the pipeline handles it

The run loops over geog_ids, not longIds. It never works out which geog_id a page
belongs to.

1. The set of geog_ids is every level-30 geo record in the province
   (`_get_geo`, `pipeline.py:1613-1641`), inner-joined with `rlp_geo_details_en`
   (`pipeline.py:394`):

   ```python
   terms = {"level": 30}   # only level 30 needed for rewrite
   ...
   geo_df = pd.DataFrame(sources, index=doc_ids)
   geo_df.index.name = 'geog_id'
   ```

2. Each row carries the geo record's own longId (`pipeline.py:312`). The overrides
   table's longId becomes `longId_y`, because `join_df` keeps the left side
   unsuffixed (`realestate_core/common/utils.py:63`).
3. Writes go to the page record by that longId (`_handle_ES_update`,
   `pipeline.py:1022`):

   ```python
   response = self.es_client.update(index=self.geo_overrides_index_name, id=longId, body=update_script)
   ```

4. Stored versions and existing overrides are read through a join on
   `localLogicId[0]` (`_get_geo_overrides`, `pipeline.py:1584-1587`; join in
   `extract_content`, `pipeline.py:398`):

   ```python
   locallogic_ids = [src['localLogicId'][0] if src.get('localLogicId') is not None else None for src in sources]
   geo_overrides_df = pd.DataFrame(sources, index=locallogic_ids)
   ```

| Step | Key |
|---|---|
| Write a page | the geo record's longId |
| Read the stored version back | `localLogicId[0]` on the page record, joined to geog_id |

So for the place that is not `localLogicId[0]`:
- Its subpage versions never read back, so its subpages look out of date on every
  run. `rewrite_property_type` checks the archive before calling GPT
  (`pipeline.py:835`), finds this month's rewrite for the longId, and writes the same
  text back with no GPT call. This is why QC rewrites 25 subpages per property type
  on every run, and ON 2.
- City pages have no version check, so both places rewrite the same page on every
  run. The loop order is the order ES `scan` returns the geo records, which is not
  guaranteed, so whichever is processed last wins and that can change between runs.
- The archive key (`rewrite:{longId}:...`) has the same collision.

## Why it is not fixed yet

The fix touches keys: which place a page belongs to, how versions are read, how the
archive is keyed. It also needs a product decision: which place should a shared page
represent? Too risky to change without a plan and testing.

## What a reader sees (adverse impact)

The description and the stats on one page always come from the same place, because
both come from the same rewrite. So a page is internally consistent. But it can be
about the other place with the same name: its LocalLogic description, and its listing
count and median if it has enough listings.

A page is only at risk when both places have a LocalLogic housing description (the
pipeline skips a place with no description). If only one place has a description, the
page always shows that place.

## Affected pages in ON and QC (checked against PROD 2026-10-16, after the ON and QC runs)

Method: for each page, compare its text with each place's LocalLogic description (word
overlap) and match the listing count in the opening sentence. "Shown" is the place
whose content the page has now. "Owner" is `localLogicId[0]` on the page record.
Subpage matches are weaker than city-page matches, because subpage text is paraphrased
more.

Shows the other place now (city page; most subpages also match the other place):

| longId | Owner (localLogicId) | Shown now | Notes |
|---|---|---|---|
| qc_lange-gardien | g30_f2m7j6y6 (near Quebec City, 15 listings) | g30_f247dgff (Outaouais, 11 listings) | Stats and description both from the other town, far apart |
| qc_saint-donat | g30_f2x7z53k (Bas-Saint-Laurent, no listing data) | g30_f25pwbv2 (Lanaudiere, 18 listings) | Stats and description both from the other town, far apart |
| qc_saint-isidore | g30_f2m3n05b (Chaudiere-Appalaches) | g30_f259emdx (Monteregie) | Far apart; both below the listing threshold (plain rewrite) |
| qc_saint-severin | g30_f2k3qq10 (Mauricie) | g30_f2jrwfhh (Beauce) | Far apart; plain rewrite |
| qc_bedford | g30_f2h0ejzx | g30_f2h0ek2h | Neighbouring places (likely town vs township); plain rewrite |
| qc_disraeli | g30_f2jjpw2s | g30_f2jm07p7 | Neighbouring; city and rental pages only, other subpages match the owner |
| qc_hatley | g30_f2hc47ep | g30_f2hcdymg | Neighbouring; plain rewrite |
| qc_lepiphanie | g30_f25szfdu | g30_f25sz63v | Neighbouring; plain rewrite |
| qc_notre-dame-du-bon-conseil | g30_f2ht9wjj | g30_f2ht931j | Neighbouring; plain rewrite |
| qc_plessisville | g30_f2hyywr3 | g30_f2hzp175 | Neighbouring; plain rewrite |
| qc_stanstead | g30_f2h8p4zn | g30_f2h8thv1 | Neighbouring; city and investment pages only |
| qc_valcourt | g30_f2hdf6me | g30_f2hdcbbz | Neighbouring; plain rewrite |

Correct now, but at risk on every run:

| longId | Places | Notes |
|---|---|---|
| on_hamilton | g30_dpxnh9by (City of Hamilton, 3,199 listings), g30_drb6s2f7 (Hamilton Township, 74 listings) | Shows the city now. Both are in localLogicId, township first. Nothing guarantees the city wins on a future run. Highest-traffic page affected. |

Not at risk from the pipeline:
- QC 13 shared longIds where only the owner has a description: the page always shows
  the owner.
- QC 28 shared longIds where neither place has a description now: the pipeline skips
  them. Their existing pages were written earlier and were not checked.
- QC 6 and ON 1 shared longIds with no pages.

The other provinces (AB, BC, MB, NB, NS, YT: 60 shared longIds) have not been checked
for wrong content yet, except ns_lunenburg below.

## Confirmed case outside ON and QC: ns_lunenburg (checked 2026-10-18)

Found by the post-run eval after the full gpt-6-luna rerun of all provinces. Lunenburg NS
is in `evals/eval_geog_ids.txt` as the municipal district. It is the only eval page in AB,
BC, SK, NS and MB (68 checked) that failed, and the largest wrong page found so far.

The two places (read-only PROD reads):

| | g30_dxfhegge | g30_dxfk2qef |
|---|---|---|
| levelName | municipal district | town |
| LocalLogic intro | "Lunenburg is a municipal district located within Nova Scotia" | "Lunenburg is a town located in Nova Scotia" |
| Listings (Sept 2026, ALL) | 178, median $499,900 | 20, median $599,900 |
| Property types with a price | detached only | detached only |
| In page record's `localLogicId` | yes (`['g30_dxfhegge']`) | no |

Both geo records have the same `city` ("Lunenburg"), `longId` (`ns_lunenburg`) and
`path` (`ns/lunenburg`), so the site has one URL for both places.

What the page shows now (all 7 pages at version 202610): the town, on every page.
- City page: "At the end of September 2026, there were 20 homes for sale in Lunenburg,
  Nova Scotia, with a median MLS® list price of $599,900 ...". These are the town's
  figures. The description is the town's too: row houses, "built before 1960 ... 1960s and
  1980s", ~90% owner-occupied dwellings. The district's text mentions mobile homes and
  "this municipal district"; neither appears. Word overlap with the LocalLogic text:
  town 0.73, district 0.50.
- Subpages (luxury, condo, semi-detached, townhouse, investment, rental): all say "this
  town" or "the town" and start from the town's pre-1960 / 1960s-1980s construction
  sentence. Word overlap is closer to the town on 5 of 6 (investment is even: 0.47 vs 0.50).

Not a model problem. The rewrite is faithful to its input: the figures match the town's
mkt_trends exactly and the description follows the town's LocalLogic text. The pipeline
gave the model the town's data for a page whose `localLogicId` is the district. That is
the shared-longId mechanism above: both geog_ids write to `ns_lunenburg` on every NS run,
and the town was processed last this time.

How visible it is:
- The district is the 3rd largest NS place in the eval list by listings (after Halifax
  1,394 and Cape Breton 296; ahead of Colchester Subd. B 118 and Chester 99). It is in the
  NS top 5 by listing inventory.
- The page understates the market about 9x (20 vs 178 homes) and overstates the median by
  $100,000.
- Compare on_hamilton (much bigger, but correct now) and the QC cases (11-18 listings at
  most).

Open, for the follow-up thread:
- Confirm which place the live royallepage.ca page renders in the sections we do not
  rewrite (transport, services, character); see "Before any fix" below. `localLogicId`
  says the district.
- Whether the page flips between town and district from run to run (no earlier version
  was checked).
- If a patch is wanted before a fix, the ad-hoc patch below with
  `--geog_id='g30_dxfhegge'` would put the district back, until the next NS run.

## If someone complains: ad-hoc patch

Rewrite the page for the intended place with a forced single-geog run:

```bash
python scripts/run_locallogic_content_rewriter.py --geog_id='<intended geog_id>' --lang='en' --force_rewrite=True
```

`--force_rewrite=True` matters. Without it, the subpages are skipped (already at this
month's version) or reuse the archived rewrite, which may be the other place's text.

The patch does not last. The next province run rewrites the city page for both places
again, and the last one processed wins. Re-check (and re-patch) the affected page after
every province run until this issue is fixed.

## Before any fix: confirm which geog_id each shared page is meant for

Not started. Each shared longId is one URL and one page record, so it can show only one
place. The other same-named place has no page of its own; it can only overwrite the
shared one. So each shared longId needs one intended geog_id, but which one has to be
confirmed, not assumed. Evidence, strongest first:

1. The live page's sections we do not rewrite. The pipeline only overrides `housing`.
   Transport, services and character come straight from LocalLogic for whichever
   geog_id the site renders. Compare the live page's transport text with each
   candidate's text in `rlp_geo_details_en`. Map boundary and listings on the page are
   further clues.
2. `localLogicId` on the page record. This is probably what the site uses, so the
   "owner" above is most likely right, but this is not confirmed. If (1) matches the
   owner, the tables above are correct as written.
3. Common sense or product input. For far-apart pairs (qc_lange-gardien,
   qc_saint-donat) the better-known place may be what users expect, even if the record
   says otherwise.

"Only one is right" may not hold everywhere:
- on_hamilton, bc_langley and nb_moncton list both geog_ids in `localLogicId`, so the
  site may treat each as one merged page. For Hamilton, the City of Hamilton is
  clearly what users expect.
- nb_caraquet lists neither of its two geog_ids.
- Neighbouring pairs (e.g. Bedford town vs township) are two real places sharing one
  page. Whichever is picked, the other has no page; the content difference is small.

Suggested first checks: qc_lange-gardien (far apart, wrong stats) and on_hamilton.

## Monitoring through the eval list

Hamilton ON (`g30_dpxnh9by`, the City of Hamilton) is in `evals/eval_geog_ids.txt`. The
eval check reads the page by longId (`on_hamilton`), so it reads this shared page. If
Hamilton Township ever wins, the eval shows it: the opening sentence would have the
township's count (74 in Sept 2026) instead of the city's (3,199). Check this on every ON
run until the issue is fixed.

## Possible fix directions (not decided)

- Decide which place each shared longId should represent (SEO/product).
- Read versions by longId (the write key) instead of localLogicId.
- Skip the non-intended geog_ids in the rewrite loop (an exclusion list is the
  lowest-risk first step).
- Re-key the archive, which has the same collision.

## Full list of shared longIds (PROD, 2026-10-16)

All level-30 geo records in `rlp_content_geo_current`, all provinces: 5,155 records, 5,034 distinct longIds, 121 shared by 2 geog_ids (AB 1, BC 5, MB 9, NB 40, NS 4, ON 2, QC 59, YT 1).
Every shared longId has exactly 2 geog_ids. PE, NL, SK, NT and NU have none.

Format: `longId -> geog_id, geog_id`. `*` marks a geog_id listed in the page record's
`localLogicId`. A line with no `*` (nb_caraquet) has a page record whose
`localLogicId` names neither.

```
## AB
ab_taber -> g30_c8b7214e*, g30_c8bk1bsx

## BC
bc_alert-bay -> g30_c0vxzbhc*, g30_c0vxz6kr
bc_esquimalt -> g30_c285x8kz*, g30_c285x711
bc_langley -> g30_c28z2f1u*, g30_c28z6c3b*
bc_north-vancouver -> g30_c2b2x3mx*, g30_c2b8bczz
bc_sechelt-part -> g30_c0ze4tnm*, g30_c2bhbrgy

## MB
mb_dauphin -> g30_cc0fgr78*, g30_cc0g6bru
mb_fairford-part-50 -> g30_cc1vxvu4*, g30_cc1ve9sf
mb_gambler-63-part -> g30_c8zzu3pw*, g30_c8zzv50s
mb_gillam -> g30_cfhcphb7*, g30_cfh87r5n
mb_lac-du-bonnet -> g30_cbgtyh38*, g30_cbgtutne
mb_morris -> g30_cbf9jfbu*, g30_cbfbb20b
mb_portage-la-prairie -> g30_cbfh72u9*, g30_cbfhkydn
mb_ste-anne -> g30_cbg674e0*, g30_cbg689yf
mb_thompson -> g30_ccfmjw6h*, g30_cbf0u69e

## NB
nb_alma -> g30_f81gdw6t*, g30_f81g2zk7
nb_baker-brook -> g30_f2rm1hr7*, g30_f2rjrvvk
nb_balmoral -> g30_f888z5rr*, g30_f82xqqn1
nb_bathurst -> g30_f83qfqry*, g30_f83nnuhx
nb_beresford -> g30_f83r2g6u*, g30_f83py6z4
nb_blackville -> g30_f831vuhy*, g30_f831w8q6
nb_canterbury -> g30_f80j0dx2*, g30_f2puzz3s
nb_caraquet -> g30_f83zdp2j, g30_f83zecgu
nb_chipman -> g30_f81nsv60*, g30_f81nym4k
nb_clair -> g30_f2rhykrb*, g30_f2rjjbzn
nb_dalhousie -> g30_f889qbrn*, g30_f88c35rn
nb_dorchester -> g30_f84jhh84*, g30_f84j7s3h
nb_drummond -> g30_f2rgefuc*, g30_f2rvnkne
nb_gagetown -> g30_f80uqmnu*, g30_f80uhncf
nb_grand-manan -> g30_dxbw24mx*, g30_dxbtf0c3
nb_harvey -> g30_f80k5k8h*, g30_f81gvgqe
nb_hillsborough -> g30_f84j0zdf*, g30_f81vnhz2
nb_lac-baker -> g30_f2rjmfvp*, g30_f2rjkcuu
nb_mcadam -> g30_f805k5z8*, g30_f805g161
nb_moncton -> g30_f81yemc4*, g30_f81ym5kb*
nb_new-maryland -> g30_f80t49mp*, g30_f807zjk0
nb_norton -> g30_f8178w1j*, g30_f815r1pj
nb_paquetville -> g30_f83xnmy7*, g30_f83wuz3h
nb_richibucto -> g30_f83ce9ge*, g30_f83cm00m
nb_riviere-verte -> g30_f2rt1r4g*, g30_f2rwk82b
nb_rogersville -> g30_f833z5u3*, g30_f836n3h6
nb_rothesay -> g30_f814kbtr*, g30_f8144vdm
nb_sackville -> g30_f84jnybu*, g30_f84jvtu1
nb_saint-andre -> g30_f2rgf5ew*, g30_f2ru4mmv
nb_saint-andrews -> g30_f8023zp9*, g30_f802d4gj
nb_saint-isidore -> g30_f83wxcjr*, g30_f83ww804
nb_saint-leonard -> g30_f2rsq7q1*, g30_f2ruc1x8
nb_saint-quentin -> g30_f82n68bs*, g30_f2rz8z74
nb_salisbury -> g30_f81tzuct*, g30_f81whk55
nb_shediac -> g30_f84neyd1*, g30_f84nds2c
nb_shippagan -> g30_f83zx1d3*, g30_f8d01twn
nb_simonds -> g30_f2pztxun*, g30_f811wxqm
nb_stanley -> g30_f80x33z2*, g30_f8298u4z
nb_sussex -> g30_f817sysj*, g30_f81kj7e0
nb_woodstock -> g30_f2pyjc07*, g30_f2pytckp

## NS
ns_digby -> g30_dxcjzed5*, g30_dxchtvn5
ns_lunenburg -> g30_dxfhegge*, g30_dxfk2qef
ns_shelburne -> g30_dxc918h1*, g30_dxc96ey5
ns_yarmouth -> g30_dxbcrsms*, g30_dxc1c0hk

## ON
on_fort-albany-part-67 -> g30_f1mcqmrp*, g30_f1mceutz
on_hamilton -> g30_dpxnh9by*, g30_drb6s2f7*

## QC
qc_akulivik -> g30_f6bdqk9g*, g30_f6b6nhmz
qc_aupaluk -> g30_f6w7j32x*, g30_f6w92f2b
qc_bedford -> g30_f2h0ejzx*, g30_f2h0ek2h
qc_chisasibi -> g30_f1xcq95x*, g30_f381r6ne
qc_clermont -> g30_f2qp3qxp*, g30_f0xwkgc9
qc_disraeli -> g30_f2jjpw2s*, g30_f2jm07p7
qc_eastmain -> g30_f32349dv*, g30_f322u4zk
qc_hatley -> g30_f2hc47ep*, g30_f2hcdymg
qc_hemmingford -> g30_f258m889*, g30_f258m4fv
qc_inukjuak -> g30_f62ey083*, g30_f62kwn6u
qc_ivujivik -> g30_f70eh0pz*, g30_f70en6pr
qc_kangiqsualujjuaq -> g30_fd3njgd8*, g30_fd3jgn96
qc_kangiqsujuaq -> g30_f6uy31tw*, g30_f6uy6uxc
qc_kangirsuk -> g30_f6wjfbb5*, g30_f6wjw36n
qc_kawawachikamach -> g30_f9b2rcym*, g30_f9b3ynzh
qc_kuujjuaq -> g30_f6r76yzg*, g30_f6r6u0e6
qc_kuujjuarapik -> g30_f3cxc9r6*, g30_f3bdwfm1
qc_lac-simon -> g30_f24t6cns*, g30_f28crv1z
qc_lange-gardien -> g30_f2m7j6y6*, g30_f247dgff
qc_lepiphanie -> g30_f25szfdu*, g30_f25sz63v
qc_mistissini -> g30_f2gxeq19*, g30_f2gqtkn8
qc_natashquan -> g30_f8uj1xm2*, g30_f8ujkxs3
qc_nemaska -> g30_f31tyyj8*, g30_f31wnygh
qc_notre-dame-de-lourdes -> g30_f25y0wv7*, g30_f2hzmx7u
qc_notre-dame-du-bon-conseil -> g30_f2ht9wjj*, g30_f2ht931j
qc_plessisville -> g30_f2hyywr3*, g30_f2hzp175
qc_quaqtaq -> g30_f6yd3trh*, g30_f6ye0xjw
qc_riviere-saint-jean -> g30_f89vcn5y*, g30_f8fjy5bt
qc_saint-augustin -> g30_f2svejfx*, g30_f9n7mj74
qc_saint-celestin -> g30_f2hqztng*, g30_f2hwb7d8
qc_saint-cyprien -> g30_f2wbqdve*, g30_f2np9myj
qc_saint-damase -> g30_f2h4fw9y*, g30_f2xuc1pz
qc_saint-donat -> g30_f2x7z53k*, g30_f25pwbv2
qc_saint-elzear -> g30_f2jryw68*, g30_f896nfst
qc_saint-isidore -> g30_f2m3n05b*, g30_f259emdx
qc_saint-lambert -> g30_f25dz3cu*, g30_f0xw2pzv
qc_saint-leon-le-grand -> g30_f2xgptq7*, g30_f2hpeddm
qc_saint-louis-de-gonzague -> g30_f253hj5k*, g30_f2jzpwvp
qc_saint-sebastien -> g30_f25bzdzg*, g30_f2js92ct
qc_saint-severin -> g30_f2k3qq10*, g30_f2jrwfhh
qc_saint-simeon -> g30_f893ebnk*, g30_f2w26p6x
qc_saint-simon -> g30_f2hhn6fu*, g30_f2wfec0b
qc_saint-stanislas -> g30_f2k90py9*, g30_f2sxj31e
qc_sainte-felicite -> g30_f88n9n5f*, g30_f2q70msw
qc_sainte-francoise -> g30_f2kb4r77*, g30_f2wcmp5v
qc_sainte-jeanne-darc -> g30_f2xetp2c*, g30_f2sy1pjc
qc_sainte-monique -> g30_f2hqt8uq*, g30_f2sv51sq
qc_sainte-perpetue -> g30_f2q7c3vx*, g30_f2hqpk28
qc_sainte-sabine -> g30_f2h166wm*, g30_f2mbx0v1
qc_salluit -> g30_f741qy8t*, g30_f741ygmh
qc_senneterre -> g30_f2950swr*, g30_f29epnh4
qc_stanstead -> g30_f2h8p4zn*, g30_f2h8thv1
qc_tasiujaq -> g30_f6qm3014*, g30_f6qq0s1v
qc_umiujaq -> g30_f619sep5*, g30_f619ejvu
qc_valcourt -> g30_f2hdf6me*, g30_f2hdcbbz
qc_waskaganish -> g30_f30h0zpk*, g30_f1pgx75f
qc_waswanipi -> g30_f2cgpxem*, g30_f2cffr63
qc_wemindji -> g30_f32j8p8v*, g30_f32j07m3
qc_whapmagoostui -> g30_f3bfjxzf*, g30_f3bdqc44

## YT
yt_teslin -> g30_c49wk40s*, g30_c49wdgd0
```
