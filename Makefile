PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Configuration of extension
EXT_NAME=duckboost
EXT_CONFIG=${PROJ_DIR}extension_config.cmake

# Target the DuckDB 2.0.1 LTS release. Label local builds so extension
# footers match what INSTALL FROM community expects. DuckDB 2.0 reads the
# engine label from DUCKDB_VERSION / OVERRIDE_GIT_DESCRIBE.
DUCKDB_COMMUNITY_COMMIT ?= 8d78114e90a53823803660bf821f6073ac87da8c
DUCKDB_COMMUNITY_TAG ?= v2.0.1
OVERRIDE_GIT_DESCRIBE ?= $(DUCKDB_COMMUNITY_TAG)
export DUCKDB_VERSION ?= $(DUCKDB_COMMUNITY_TAG)
export DUCKDB_COMMIT ?= $(DUCKDB_COMMUNITY_COMMIT)

# Include the Makefile from extension-ci-tools
include extension-ci-tools/makefiles/duckdb_extension.Makefile

.PHONY: check-extension-metadata
check-extension-metadata:
	python3 $(PROJ_DIR)scripts/check_extension_metadata.py \
		--expect-tag $(DUCKDB_COMMUNITY_TAG)
