# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real Estate Content Transformer is a Python package that uses GPT models to rewrite and enhance real estate geographic content stored in Elasticsearch. The system processes Canadian province/city data and property-type-specific content using LocalLogic data, with capabilities for bulk processing, error recovery, and content archiving.

## Core Architecture

### Main Components

1. **LocallogicContentRewriter** (`realestate_content_transformer/data/pipeline.py`)
   - Central orchestrator for content rewriting operations
   - Manages Elasticsearch connections and bulk updates via `BulkUpserter`
   - Handles two rewriting modes:
     - City-level content: Rewrites geographic content for cities/regions
     - Property-type-specific content: Generates content for LUXURY, CONDO, SEMI-DETACHED, TOWNHOUSE, INVESTMENT, RENTAL
   - Uses `simple_append` flag to control GPT usage at city level (if False, uses GPT for rewrites; if True, simple stat append)
   - Integrates with `LocalLogicGPTRewriter` from `realestate_spam` package for GPT-powered content generation
   - Supports RAG (Retrieval-Augmented Generation) for better context-aware rewrites

2. **Data Models** (`realestate_content_transformer/data/data_models.py`)
   - `GeoDoc`: Base geographic document with geog_id, longId, city info
   - `Profile`: Content sections (housing, transport, services, character)
   - `Data`: Contains name, province, and profile data
   - `Overrides`: Wrapper for override data
   - `GeoDetailDoc`: Geographic detail document
   - `GeoOverridesDoc`: Override document with localLogicId and language-specific overrides

3. **ChatGPTRewriteArchiver** (`realestate_content_transformer/data/archive.py`)
   - Stores GPT rewrite history for recovery and versioning
   - Supports two storage backends:
     - Redis: For production distributed systems
     - Plain text: For development and testing
   - Key format: `rewrite:{longId}:{property_type}:{version}:{lang}`
   - Stores user prompts and GPT responses
   - Enables version-based recovery with date prefix matching (e.g., '202312')

4. **BulkUpserter** (`realestate_content_transformer/data/pipeline.py`)
   - Handles bulk Elasticsearch updates efficiently
   - Batches updates (default: 100 documents)
   - Maps longId to geog_id for detailed error logging
   - Automatic flushing when batch size reached

### External Dependencies

The package depends on two sibling packages:
- `realestate_core`: Provides common utilities and class extensions
- `realestate_spam`: Contains `LocalLogicGPTRewriter` for GPT-based content generation

## Running the Application

### Main Script

Run content rewriting for provinces or specific locations. The three location arguments (`--prov_code`, `--geog_id`, `--geog_ids_file`) are mutually exclusive:

```bash
# Process entire province
python scripts/run_locallogic_content_rewriter.py --prov_code='BC' --lang='en'

# Process specific location
python scripts/run_locallogic_content_rewriter.py --geog_id='g30_dxbcrsms' --lang='en'

# Process multiple locations from a file (batch mode)
python scripts/run_locallogic_content_rewriter.py --geog_ids_file='./my_geog_ids.txt' --lang='en'

# Use config file
python scripts/run_locallogic_content_rewriter.py --config scripts/config.yaml

# Process specific property type only
python scripts/run_locallogic_content_rewriter.py --prov_code='ON' --property_type='LUXURY'

# Force rewrite regardless of version
python scripts/run_locallogic_content_rewriter.py --prov_code='AB' --force_rewrite=True
```

### Batch Processing via geog_ids_file

The `--geog_ids_file` flag accepts a path to a file containing multiple `geog_id`s. Supported formats:

```
# Comma-delimited single line
g30_dxbcrsms, g30_f2m673bs, g30_c2b2379d

# One per line
g30_dxbcrsms
g30_f2m673bs
g30_c2b2379d

# Mixed
g30_dxbcrsms, g30_f2m673bs
g30_c2b2379d
```

Batch runs are logged with `location_identifier = batch_{N}_geogs`. Failures on individual geog_ids are caught and logged, and processing continues with the remaining ids.

### Configuration

Configuration via YAML file (`scripts/config.yaml`):
```yaml
es_host: localhost
es_port: 9201
prov_code: "PE"          # or use geog_id or geog_ids_file instead
lang: en
log_level: INFO
archiver_file: ./archive_rewrites.txt
force_rewrite: false
# geog_ids_file: ./my_geog_ids.txt   # uncomment to use batch mode
```

### Error Recovery

Recover from failed runs using the rerun mechanism:

```bash
python scripts/run_locallogic_content_rewriter.py --rerun --run_num=5 --prov_code='BC' --lang='en'
```

Or use the error rerun script to automatically retry failed geog_ids:

```bash
python scripts/error_rerun.py --year=2024 --month=9 --config=config.yaml
```

This script:
- Parses log files for ERROR entries with geog_id
- Extracts unique failed geog_ids
- Reruns each failed location with 40-second delays

### Logging

Logs are automatically created with format: `{timestamp}_run_{run_number}_{location}_{lang}.log`

Run entries are tracked in CSV files:
- `run_entry_table.csv`: Normal runs with timestamp, run_number, prov_code, lang, duration, rewrites_count
- `rerun_entry_table.csv`: Recovery runs

## Testing

### Environment Setup

```bash
export ES_HOST=localhost
export ES_PORT=9201
```

### Running Tests

```bash
# Run all tests
python -m unittest tests/test_LocallogicContentRewriter.py

# Run specific test
python -m unittest tests/test_LocallogicContentRewriter.TestLocallogicContentRewriter.test_or_setup_extract_all_context

# Run archiver tests
python -m unittest tests/test_archiver.py
```

Test configuration is in `tests/test_config.yaml`.

## Utility Scripts

### Archive Logs

Compress and archive old log files:

```bash
python scripts/archive_logs.py
```

Groups logs by province and run number ranges, creating `.tar.gz` archives.

### Parse Logs

Extract information from log files:

```bash
python scripts/parse_log.py <log_file>
```

## Model Configuration

Current models (as of latest version):
- `LIGHT_WEIGHT_LLM`: `gpt-4o` (for health checks and simple operations)
- `LLM`: `gpt-4o` (for content rewriting)

Models are configured at the top of:
- `realestate_content_transformer/data/pipeline.py`
- `scripts/run_locallogic_content_rewriter.py`
- `app/main.py`

## FastAPI Application

A FastAPI service is available in `app/main.py` for running the rewriter as a web service. This is currently experimental and includes Celery integration for async task processing.

## Property Types

Supported property types:
- LUXURY
- CONDO
- SEMI-DETACHED
- TOWNHOUSE
- INVESTMENT
- RENTAL

These are validated in the archiver and processed sequentially by the main rewriter.

## Province Codes

Valid Canadian province/territory codes:
AB, BC, MB, NB, NL, NT, NS, NU, ON, PE, QC, SK, YT

## Key Files to Check When Modifying

- Content generation logic: `realestate_content_transformer/data/pipeline.py` (specifically `rewrite_cities` and `rewrite_property_types` methods)
- Data structure changes: `realestate_content_transformer/data/data_models.py`
- Archive format changes: `realestate_content_transformer/data/archive.py`
- Script CLI options: `scripts/run_locallogic_content_rewriter.py`
- Recovery logic: Look for `rerun_to_recover` method in pipeline.py

## Installation

```bash
pip install -e .
```

The package is named `realestate_content_transformer` version 1.0.1.
