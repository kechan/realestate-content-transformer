# Dual-Model Support: gpt-4o + gpt-5.6-luna — Scope, Findings & Solution (v1.5)

**Status:** Research complete, solution identified, NOT YET IMPLEMENTED.
**Prerequisite:** v1 (`city_page_metrics_enhancement_spec_v1.md`) has run cleanly across all provinces and is considered bug-free. This phase (v1.5) is the next step after that, before any v2 work.
**Goal:** A single codebase (`raa`, `realestate-spam`, `realestate-content-transformer`) that works correctly with both `gpt-4o` (what v1 ships with) and `gpt-5.6-luna` (a ~10x cheaper model OpenAI now recommends for cost-sensitive workloads), with no per-model conditional/branching logic required.

If you're a fresh thread picking this up: read this whole document before touching code. It contains everything discovered so far — you should not need to re-derive any of it.

---

## 1. Why v1 shipped on gpt-4o, not Luna

During v1 testing, calling `rewrite_city()` with `llm_model='gpt-5.6-luna'` failed with:

```
Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.
```

Root cause: the `raa` package (`/Users/kechan/Developer/raa`, a sibling repo — `RAA` class wraps all actual OpenAI API calls for this codebase) hardcodes the parameter name `max_tokens` in three separate places (see §4). `gpt-5.6-luna`'s backend rejects that parameter outright and requires `max_completion_tokens` instead.

Given `gpt-4o` is not deprecated, has no announced removal date, and is actually on cheaper *grandfathered legacy pricing* ($2.50/$10 per 1M input/output tokens vs. new integrations defaulting to GPT-4.1 at $5/$15), the decision was made to ship v1 on `gpt-4o` and treat the Luna migration as separate, later work — this document.

---

## 2. Key environment fact: which OpenAI SDK is actually running

The `openai` Python package installed in `~/Developer/python39_env` (the env used for local dev/notebook work — confirm this hasn't changed before relying on it) is **0.27.2** — a legacy, pre-1.0 SDK. Confirmed via `pip show openai` in that env.

This matters because `raa.py`'s `RAA` class branches its internal implementation based on SDK version:
```python
if pkg_resources.parse_version(openai_version) >= pkg_resources.parse_version("1.3.0"):
    self.openai_version = ">=1.3.0"   # uses client.chat.completions.create(...)
else:
    self.openai_version = "<1.3.0"    # uses openai.ChatCompletion.create(...) — legacy module-level call
```

With `openai==0.27.2` installed, the `<1.3.0` branch is what actually executes today — specifically `_openai_ChatCompletion_older_than_1d3d0()` in `raa.py`. Don't assume the `>=1.3.0` branch is live without re-checking the installed version first.

Also confirmed: `gptcache` is the only other package in that env depending on `openai`, and it does **not** pin a version — so upgrading `openai` later (for unrelated reasons) carries low risk, though it's **not required** for this fix (see §3).

---

## 3. Findings — empirically verified, live API calls

All of the following was confirmed by calling the real OpenAI API directly (bypassing `raa`/the project code, standalone scratch scripts only — no project files were modified during this research phase, aside from one already-applied change noted in §3.5).

### 3.1 `max_completion_tokens` is universal — works for both models

| Model | Parameter | Result |
|---|---|---|
| `gpt-4o` | `max_tokens` | ✅ works (today's behavior) |
| `gpt-4o` | `max_completion_tokens` | ✅ also works |
| `gpt-5.6-luna` | `max_completion_tokens` | ✅ works |
| `gpt-5.6-luna` | `max_tokens` | ❌ fails with the error quoted in §1 |

This means **no per-model branching is needed** — the fix is a single, universal parameter rename, not a "detect model, pick param name" mechanism. This directly satisfies the stated goal.

### 3.2 No SDK upgrade required

Tested with the actual installed `openai==0.27.2`. The legacy `openai.ChatCompletion.create(**kwargs)` call forwards kwargs directly to the REST API with no client-side schema validation — so it happily accepts `max_completion_tokens` even though that parameter name postdates this SDK version. Confirmed both `gpt-4o` and `gpt-5.6-luna` work correctly through this exact legacy call style with `max_completion_tokens`.

### 3.3 Reasoning-token nuance — same numeric cap, less headroom for Luna

`max_completion_tokens` includes **hidden reasoning tokens** for models that do internal reasoning (like `gpt-5.6-luna`); `gpt-4o` has zero reasoning-token overhead. Tested with a realistic-length prompt (~150-word target, comparable to our actual rewrite task) and a budget of 1000:

| Model | Total completion tokens used | Of which reasoning tokens | Visible output | Truncated? |
|---|---|---|---|---|
| `gpt-4o` | 181 | 0 | 181 tokens, ~152 words | No (`finish_reason: stop`) |
| `gpt-5.6-luna` | 822 | 615 (75%) | ~207 tokens, ~149 words | No (`finish_reason: stop`) |

Both succeeded at this length, but Luna used 75% of its budget on invisible reasoning, leaving much thinner headroom than gpt-4o at the identical numeric cap. Our actual production content is comfortably within this range (Barrie's real v1 output was ~150 words), so this hasn't caused truncation in testing — but it's the reason for the token-cap bump in §4.4. If output length grows in the future (e.g. a v2 feature adds more required content), re-verify Luna isn't hitting `finish_reason: 'length'`.

### 3.4 `test_health()` — a separate, easy-to-miss call site

`LocalLogicGPTRewriter.test_openai_health()` (`realestate-spam/realestate_spam/llm/chatgpt.py:314-315`) is a thin wrapper:
```python
def test_openai_health(self) -> bool:
    return self.raa.test_health()
```
`RAA.test_health()` (`raa.py:54-86`) is a **completely separate method** from `get_completion_from_messages()`/`get_completion()` — it does not route through `_openai_ChatCompletion_older_than_1d3d0()`/`_newer_than_1d3d0()` at all. It has its own independently-hardcoded `max_tokens=2` (two occurrences, one per SDK-version branch).

This matters a lot: `test_openai_health()` is called at the **start of every CLI run** — `main_single_location()` and `rerun_to_recover()` in `scripts/run_locallogic_content_rewriter.py`, and `app/main.py`. If this specific call site isn't fixed, switching the model to `gpt-5.6-luna` would make **every single run abort immediately at the health check**, before any actual rewrite logic ever executes — a much worse failure mode than the one caught during v1 testing (which was in `rewrite()`, not the health check). Easy to miss if you only look at `rewrite()`'s call path.

### 3.5 Already applied this session (not part of the pending fix)

`raa/raa/raa.py`'s `AVAILABLE_GPT_MODELS` allowlist already includes `'gpt-5.6-luna'` — this was a real edit made during v1 testing (previously, constructing `RAA(llm_model='gpt-5.6-luna', ...)` raised `ValueError` immediately). This is done; no action needed here.

---

## 4. The fix — exact changes needed (not yet applied)

Four small edits, three in `raa.py`, one in `chatgpt.py`. No per-model conditional logic anywhere.

### 4.1 `raa/raa/raa.py` — `RAA.test_health()` (~lines 68 and 76)
Two occurrences of `max_tokens=2` → `max_completion_tokens=2` (one inside the `>=1.3.0` branch, one inside the `<1.3.0` branch of this method). **Do not skip this one** — see §3.4 for why it's the most consequential of the three.

### 4.2 `raa/raa/raa.py` — `_openai_ChatCompletion_newer_than_1d3d0()` (~line 164)
`max_tokens=max_tokens` → `max_completion_tokens=max_tokens` inside the `client.chat.completions.create(...)` call. Not the path exercised today (see §2), but fix for consistency/forward-compatibility in case the SDK is ever upgraded.

### 4.3 `raa/raa/raa.py` — `_openai_ChatCompletion_older_than_1d3d0()` (~line 215)
`max_tokens=max_tokens` → `max_completion_tokens=max_tokens` inside the `openai.ChatCompletion.create(...)` call (the `max_tokens is None` branch above it, ~line 205-209, doesn't pass the param at all and needs no change). **This is the path actually exercised today** given the installed SDK version — this is the fix that directly resolves the error seen during v1 testing.

Note: only the actual keyword argument name passed to the OpenAI API call needs to change, in these three spots. `RAA.get_completion_from_messages()`'s own parameter (`max_tokens=500` default) and `chatgpt.py`'s call sites (e.g. `self.raa.get_completion_from_messages(messages, temperature=..., max_tokens=1000, ...)`) can keep using `max_tokens` as their own internal Python parameter name — that's just a variable name at that layer, not what ultimately gets sent to OpenAI. No need to rename it end-to-end; only the final API-call kwarg matters.

### 4.4 `realestate-spam/realestate_spam/llm/chatgpt.py` — `LocalLogicGPTRewriter.rewrite()` (line 326)
Bump `max_tokens=1000` → `max_tokens=1500`. Rationale: extra headroom for Luna's reasoning-token overhead (§3.3). This is a cap, not a cost — raising it doesn't increase spend unless output actually grows to use the extra room, so it's effectively free for `gpt-4o` (which has never come close to 1000 in testing) while giving `gpt-5.6-luna` meaningfully more margin.

**Out of scope for this fix:** other `max_tokens=`/`max_completion_tokens=` call sites in `chatgpt.py` (`DisplayNameOriginAgent` line 49, `FirstTouchNoteSpamDetectorAgent` line 135, `DataAugmentationAgent` line 232) belong to unrelated agent classes, not `LocalLogicGPTRewriter`. They will incidentally also work correctly for Luna once the shared `raa.py` fixes (§4.1-4.3) land, since those are the shared, class-agnostic layer — but their own `max_tokens=500` cap values are not being touched here; that's a separate decision if/when those agents are also migrated.

---

## 5. Explicitly ruled out — don't do these

- **No per-model conditional/branching logic** for the parameter name. `max_completion_tokens` is universal (§3.1) — a `switch`/`if model == ...` mechanism would be unnecessary complexity.
- **No `openai` pip package upgrade** in `python39_env` or anywhere else. Confirmed unnecessary (§3.2).
- **No changes to `pipeline.py`** or any v1 feature logic (province mapping, `get_market_trend_metrics()`, prompt guideline text) — all of that is already model-agnostic; this phase is purely about the OpenAI API call layer.
- **`AVAILABLE_GPT_MODELS`** — already done (§3.5), don't re-touch.

---

## 6. How to test — including via the notebook

### 6.1 Environment prerequisites (should already be true; verify, don't redo blindly)

This testing was done locally via `~/Library/CloudStorage/GoogleDrive-kelvin@jumptools.com/My Drive/ChatGPT/notebooks/Explore_Locallogic_ES.ipynb`, against a local Docker Elasticsearch. Several durable fixes were made to that local environment during v1 testing — these should still be in place, but verify rather than assume if starting fresh:

- **Local ES** runs on `localhost:9200` (Docker). Cluster health should be `green`. If `red`, see `docs/ES_ADMIN.md` in the `realestate-analytics` repo (disk watermark / segment bloat runbook) — this was hit and fixed once already; the persistent watermark settings (`low: 95%`, `high: 97%`, `flood_stage: 99%`) should already be applied to that cluster, but Docker restarts don't necessarily reset ES data-dir settings, so this is usually a non-issue on restart.
- **`rlp_geo_details_fr`** locally was empty and has since been fully seeded (9,708 docs, mirrored from prod) — needed because `extract_content()` unconditionally queries both `en` and `fr` regardless of which language you're testing.
- **`python39_env`'s `setuptools`** was pinned to `69.0.2` (matching prod) after a `pkg_resources` removal broke imports (`setuptools` 82+ dropped `pkg_resources` entirely). If you hit `ModuleNotFoundError: No module named 'pkg_resources'` again, re-pin: `~/Developer/python39_env/bin/pip install "setuptools==69.0.2"`.
- **`pipeline.py`'s ES client construction** (`LocallogicContentRewriter.__init__`) was fixed so that `es_host='localhost', es_port=9200` correctly uses the no-auth local path, while `localhost` on any other port (e.g. an SSH-tunneled 9201/9202) still correctly uses the ApiKey/HTTPS path. This is now a permanent part of `pipeline.py`, not something to redo.

### 6.2 Standalone parameter verification (do this first, cheapest signal)

Before touching any project code, re-run (or write fresh) a scratch script like the one used in §3.1/3.2 research — call both models with both parameter names directly via `openai.ChatCompletion.create(...)` (matching whatever SDK version is actually installed — check first per §2), confirm the table in §3.1 still holds. This validates the OpenAI-side behavior hasn't changed since this document was written, independent of any code in this repo.

### 6.3 Apply the fix, then test via notebook

1. Apply all four changes from §4.
2. In the notebook, construct `LocallogicContentRewriter` as usual (see `city_page_metrics_enhancement_prompt_v1.txt` and the v1 spec doc for the general pattern), but set `llm_model='gpt-5.6-luna'`.
3. **Test the health check specifically first** (this is the one most likely to have been missed — §3.4):
   ```python
   ll_rewriter.llm_model = 'gpt-5.6-luna'
   # however test_openai_health() is invoked in your test context — confirm it returns True,
   # not an exception about max_tokens.
   ```
4. Run `rewrite_city()` against known-good test cases from v1 validation:
   - Barrie (`geog_id='g30_dpzk6g9n'`, `longId='on_barrie'`) — the main validated case.
   - Coquitlam (`geog_id='g30_c2b8qq29'`, `longId='bc_coquitlam'`) — the missing-property-type (SEMI-DETACHED) edge case.
5. For each, verify against the same checklist used in v1 validation:
   - Opening sentence matches exactly: "As of {month}, there are {count} homes for sale in {city}, {province}, with a median MLS® list price of {price}."
   - Property-type breakdown sentence correct, with genuinely-missing types cleanly omitted (Coquitlam case).
   - Never says "average".
   - Original content (ownership %, housing mix, construction era) retained, not truncated mid-sentence.
   - Query local ES afterward (`rlp_content_geo_overrides_current/_doc/{longId}`) to confirm the write succeeded and inspect the actual persisted text.
6. **Regression-check `gpt-4o` still works** after the `raa.py` changes, since they touch shared code used by both models — re-run the same Barrie/Coquitlam tests with `llm_model='gpt-4o'` and confirm output is unchanged in quality/structure from v1's original validation.
7. If you want to inspect actual token usage/reasoning-token consumption for a specific real case (not just the synthetic prompt used in §3.3's research), the OpenAI response object's `usage.completion_tokens_details.reasoning_tokens` field has this — not currently surfaced anywhere in `raa.py`'s return value (`get_completion_from_messages()` only returns the string content), so you'd need a temporary standalone script or a `raa.py` debug tweak to see it, not something available through the normal `rewrite()` path today.

---

## 7. Open questions — not decided, flag before proceeding

- **`LIGHT_WEIGHT_LLM`** (used for the health-check-only writer in `run_locallogic_content_rewriter.py`) is currently hardcoded to `'gpt-4o'` separately from `LLM`. Should it also move to `'gpt-5.6-luna'` for consistency, or is there a reason to keep the health check on a different (possibly more reliable/simpler) model than the actual rewrite model? Not discussed/decided.
- **Model configurability**: currently the model is a hardcoded source constant with no CLI flag or `config.yaml` key (raised during v1 work, deliberately deferred). If Luna migration happens, is this still deferred, or does it become worth doing at the same time (so switching models doesn't require a code edit + redeploy)? Not decided.
- Whether `DisplayNameOriginAgent`, `FirstTouchNoteSpamDetectorAgent`, `DataAugmentationAgent` (other `chatgpt.py` classes, unrelated to this project's actual use of `LocalLogicGPTRewriter`) should also be deliberately migrated to Luna, or just incidentally left working via the shared `raa.py` fix without anyone actively re-validating them. Out of scope for this project's phase either way, but worth being aware the shared fix touches their code path too.
