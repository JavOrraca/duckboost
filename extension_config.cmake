# Out-of-tree / community-style load config for duckboost.
# From a DuckDB checkout:
#   EXTENSION_CONFIGS=/path/to/duckboost/extension_config.cmake DUCKDB_EXTENSIONS='duckboost' make reldebug
# Or after extracting duckboost next to DuckDB as extension_external/duckboost:
#   duckdb_extension_load(duckboost SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR} LOAD_TESTS)

duckdb_extension_load(duckboost
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}
    LOAD_TESTS
)

# duckdb_extension_load only builds the extension; this links it into the DuckDB shell,
# library and tests. Older DuckDB versions link by default and lack this command.
if(COMMAND duckdb_extension_statically_link)
    duckdb_extension_statically_link(duckboost)
endif()
