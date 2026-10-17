PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Configuration of extension
EXT_NAME=duckboost
EXT_CONFIG=${PROJ_DIR}extension_config.cmake

# Match duckdb/community-extensions `.github/workflows/build.yml` defaults
# (as of 2026-10-15). The Discord prep note mentioned v2.0.0-alpha44357; the
# live community pin is v2.0.0-alpha46057 on commit b15358c. Label local builds
# with that tag so extension footers match what INSTALL FROM community expects.
# DuckDB 2.0 reads the engine label from DUCKDB_VERSION / OVERRIDE_GIT_DESCRIBE
# rather than deriving it from git tags (see duckdb/extension-ci-tools#439).
DUCKDB_COMMUNITY_COMMIT ?= b15358c371daf10dad799a43b6167f6ccdce84d8
DUCKDB_COMMUNITY_TAG ?= v2.0.0-alpha46057
OVERRIDE_GIT_DESCRIBE ?= $(DUCKDB_COMMUNITY_TAG)
export DUCKDB_VERSION ?= $(DUCKDB_COMMUNITY_TAG)
export DUCKDB_COMMIT ?= $(DUCKDB_COMMUNITY_COMMIT)

# Include the Makefile from extension-ci-tools
include extension-ci-tools/makefiles/duckdb_extension.Makefile
