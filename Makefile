PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Configuration of extension
EXT_NAME=duckboost
EXT_CONFIG=${PROJ_DIR}extension_config.cmake

# Match duckdb/community-extensions `.github/workflows/build.yml` defaults
# for DuckDB 2.0.0 LTS (“Cinnamon Teal”). Label local builds so extension
# footers match what INSTALL FROM community expects. DuckDB 2.0 reads the
# engine label from DUCKDB_VERSION / OVERRIDE_GIT_DESCRIBE.
DUCKDB_COMMUNITY_COMMIT ?= d54cbbe3df17e8e5ce68312f483b4155836112aa
DUCKDB_COMMUNITY_TAG ?= v2.0.0
OVERRIDE_GIT_DESCRIBE ?= $(DUCKDB_COMMUNITY_TAG)
export DUCKDB_VERSION ?= $(DUCKDB_COMMUNITY_TAG)
export DUCKDB_COMMIT ?= $(DUCKDB_COMMUNITY_COMMIT)

# Include the Makefile from extension-ci-tools
include extension-ci-tools/makefiles/duckdb_extension.Makefile

.PHONY: check-extension-metadata
check-extension-metadata:
	python3 $(PROJ_DIR)scripts/check_extension_metadata.py \
		--expect-tag $(DUCKDB_COMMUNITY_TAG)
