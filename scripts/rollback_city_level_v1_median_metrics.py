"""
ONE-OFF ROLLBACK SCRIPT -- city-level v1 (median metrics) feature only.

If the v1 city-level rewrite (dated median metrics opening sentence) turns out to be
badly broken after a real PROD run, this restores the affected city-level `housing`
(and any other rewritten section: transport/services/character) content back to the
last archived pre-v1 version, using the existing ChatGPTRewriteArchiver history file
(archive_rewrites.txt) -- NOT git, since the ES documents already published to PROD
are what actually need fixing, not the code.

Scope: city-level content only (archive records with property_type == 'None').
Property-type subpages are never touched by this script, since v1 never changed them.

This script is fully standalone: it does not modify or call rerun_to_recover() or any
other pipeline.py method beyond read-only reuse of the ES client / archiver / the
existing _handle_ES_update() write helper. It always targets whatever es_host/es_port
you pass in -- there is no local-vs-prod branching baked in, so you must pass PROD's
real host/port explicitly. The ES API key is picked up the same way the main pipeline
already does, automatically, from ~/.rlp_es_env or ~/.env (see LocallogicContentRewriter
.__init__) -- nothing extra to configure here.

How scope is determined:
  1. Every (longId, lang) that has a city-level archive record (property_type == 'None')
     at --v1_version is treated as "touched by v1".
  2. For each, the archived record at --restore_version (prefix match, e.g. '202607'
     matches a stored '202607' or '20260701') is looked up as the restore target.
  3. If no restore-version record exists for a given longId/lang, it is reported as
     UNRESTORABLE and skipped -- never silently dropped.
  4. For everything restorable, the archived <housing>/<transport>/<services>/<character>
     tags are parsed back out of the archived chatgpt_response and written to
     rlp_content_geo_overrides_current using the exact same ES update-script shape
     rewrite_city() itself uses -- so the resulting doc is indistinguishable from one
     that never received the v1 rewrite.

Usage:
  # 1. Dry run first -- always. Prints the full restore plan, writes nothing.
  python scripts/rollback_city_level_v1_median_metrics.py \\
      --es_host <prod_es_host> --es_port <prod_es_port> \\
      --archiver_file /path/to/archive_rewrites.txt \\
      --v1_version 202608 --restore_version 202607

  # 2. Once the plan looks right, actually write to PROD ES:
  python scripts/rollback_city_level_v1_median_metrics.py \\
      --es_host <prod_es_host> --es_port <prod_es_port> \\
      --archiver_file /path/to/archive_rewrites.txt \\
      --v1_version 202608 --restore_version 202607 \\
      --execute

  # Optional: restrict to a handful of longIds first (smoke test before full rollback)
  --longids_file ./rollback_test_longids.txt

  # Optional: restrict language(s), default is both
  --lang en fr
"""
import argparse
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

from realestate_content_transformer.data.pipeline import LocallogicContentRewriter

SECTION_TAGS = ['housing', 'transport', 'services', 'character']
CITY_LEVEL_PROPERTY_TYPE = 'None'   # literal string, matches how rewrite_city() archives records


def parse_profiles_from_archive(chatgpt_response: str) -> dict:
  """Reconstruct the {section: content} dict from an archived <tag>...</tag> response string."""
  profiles = {}
  for tag in SECTION_TAGS:
    match = re.search(rf'<{tag}>(.*?)</{tag}>', chatgpt_response, re.DOTALL)
    if match:
      profiles[tag] = match.group(1)
  return profiles


def build_update_script(lang: str, version: str, profiles: dict) -> dict:
  """
  Mirrors the ES update-script shape used in LocallogicContentRewriter.rewrite_city()
  (pipeline.py) exactly, so a restored doc is indistinguishable from one rewrite_city()
  itself wrote. The upsert branch is realistically dead here -- scope determination
  guarantees the target doc already exists (v1 wrote it) -- but is included for parity
  with the original code path.
  """
  return {
    "script": {
      "source": f"""
          if (ctx._source.containsKey('overrides_{lang}')) {{
              ctx._source['overrides_{lang}'].data.version = params.version;
              ctx._source['overrides_{lang}'].data.profiles = params.profiles;
          }} else {{
              def profiles = params.profiles;
              def data = ['version': params.version, 'profiles': profiles];
              ctx._source['overrides_{lang}'] = ['data': data];
          }}
      """,
      "params": {
        "version": version,
        "profiles": profiles,
      },
    },
    "upsert": {
      f"overrides_{lang}": {
        "data": {
          "version": version,
          "profiles": profiles,
        }
      }
    },
  }


def load_longids_filter(file_path: str):
  path = Path(file_path)
  if not path.exists():
    raise FileNotFoundError(f"longids file not found: {file_path}")
  content = path.read_text()
  longids = [x.strip() for x in content.replace('\n', ',').split(',') if x.strip()]
  if not longids:
    raise ValueError(f"No longIds found in file: {file_path}")
  return set(longids)


def determine_scope(cached_df, v1_version: str, langs, longids_filter=None):
  touched_df = cached_df[
    (cached_df['property_type'] == CITY_LEVEL_PROPERTY_TYPE) &
    (cached_df['version'] == v1_version) &
    (cached_df['lang'].isin(langs))
  ]
  if longids_filter is not None:
    touched_df = touched_df[touched_df['longId'].isin(longids_filter)]
  return touched_df


def restore(rewriter, touched_df, restore_version: str, dry_run: bool, logger) -> dict:
  results = {'restored': [], 'unrestorable': [], 'failed': []}

  for _, row in touched_df.iterrows():
    long_id = row['longId']
    lang = row['lang']

    archived = rewriter.archiver.get_record(
      longId=long_id, property_type=CITY_LEVEL_PROPERTY_TYPE, version=restore_version,
      lang=lang, use_cache=True,
    )
    if not archived:
      results['unrestorable'].append((long_id, lang))
      logger.error(f"[UNRESTORABLE] no archive record at version~{restore_version} for longId={long_id} lang={lang}")
      continue

    profiles = parse_profiles_from_archive(archived['chatgpt_response'])
    if not profiles:
      results['unrestorable'].append((long_id, lang))
      logger.error(f"[UNRESTORABLE] archive record found but no parseable section tags for longId={long_id} lang={lang}")
      continue

    restored_version = archived['version']   # the actual archived version string, e.g. '202607'
    update_script = build_update_script(lang=lang, version=restored_version, profiles=profiles)

    if dry_run:
      logger.info(f"[DRY_RUN] would restore longId={long_id} lang={lang} -> version={restored_version} sections={list(profiles.keys())}")
      results['restored'].append((long_id, lang))
      continue

    ok = rewriter._handle_ES_update(update_script, long_id)
    if ok:
      logger.info(f"[RESTORED] longId={long_id} lang={lang} -> version={restored_version}")
      results['restored'].append((long_id, lang))
    else:
      logger.error(f"[FAILED] longId={long_id} lang={lang} -> version={restored_version}")
      results['failed'].append((long_id, lang))

  return results


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--es_host', required=True, help="PROD Elasticsearch host. No default -- must be explicit.")
  parser.add_argument('--es_port', required=True, type=int, help="PROD Elasticsearch port. No default -- must be explicit.")
  parser.add_argument('--archiver_file', required=True, help="Path to PROD's archive_rewrites.txt")
  parser.add_argument('--v1_version', required=True, help="Version string (e.g. 202608) of the v1 run to roll back.")
  parser.add_argument(
    '--restore_version', default='202512',
    help=(
      "Version string/prefix to restore back to. Defaults to '202512', confirmed from PROD's "
      "archive_rewrites.txt as the last pre-v1 version (see CLAUDE.md, 'Rollback Reference'). "
      "Double-check this is still accurate before relying on the default -- if any PROD run "
      "(e.g. a hotfix) happened between now and when v1 actually ships, override this explicitly."
    ),
  )
  parser.add_argument('--lang', nargs='+', default=['en', 'fr'], help="Language(s) to restore. Default: en fr")
  parser.add_argument('--longids_file', default=None, help="Optional: restrict rollback to these longIds only (smoke test before full run).")
  parser.add_argument('--execute', action='store_true', help="Actually write to ES. Without this flag, runs as a dry run only.")
  parser.add_argument('--yes', action='store_true', help="Skip the interactive confirmation prompt before --execute.")
  args = parser.parse_args()

  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
  log_filename = f'rollback_city_level_v1_{timestamp}.log'
  logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_filename)],
  )
  logger = logging.getLogger('rollback_city_level_v1')

  dry_run = not args.execute

  logger.info(f"TARGETING ES {args.es_host}:{args.es_port}  (dry_run={dry_run})")
  logger.info(f"archiver_file={args.archiver_file}  v1_version={args.v1_version}  restore_version={args.restore_version}  lang={args.lang}")

  if args.execute and not args.yes:
    confirm = input(
      f"About to WRITE rollback changes to PROD ES at {args.es_host}:{args.es_port}.\n"
      f"Type ROLLBACK to proceed: "
    )
    if confirm.strip() != 'ROLLBACK':
      logger.info("Confirmation not given. Aborting, nothing written.")
      sys.exit(1)

  longids_filter = load_longids_filter(args.longids_file) if args.longids_file else None

  rewriter = LocallogicContentRewriter(
    es_host=args.es_host, es_port=args.es_port,
    archiver_filepath=args.archiver_file,
  )

  touched_df = determine_scope(
    rewriter.archiver.cached_df, v1_version=args.v1_version, langs=args.lang, longids_filter=longids_filter,
  )
  logger.info(f"Scope: {len(touched_df)} (longId, lang) city-level record(s) touched by v1_version={args.v1_version}")

  if len(touched_df) == 0:
    logger.error("Nothing in scope -- check --v1_version and --archiver_file. Aborting.")
    sys.exit(1)

  results = restore(rewriter, touched_df, restore_version=args.restore_version, dry_run=dry_run, logger=logger)

  logger.info(
    f"SUMMARY: restored={len(results['restored'])}  "
    f"unrestorable={len(results['unrestorable'])}  failed={len(results['failed'])}"
  )
  if results['unrestorable']:
    logger.error(f"UNRESTORABLE (no {args.restore_version} archive record found): {results['unrestorable']}")
  if results['failed']:
    logger.error(f"FAILED ES writes (see log for detail): {results['failed']}")

  if dry_run:
    logger.info("This was a DRY RUN -- nothing was written. Re-run with --execute to apply.")

  logger.info(f"Full log written to {log_filename}")


if __name__ == '__main__':
  main()
