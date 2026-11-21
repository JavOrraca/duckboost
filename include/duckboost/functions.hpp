//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckboost/functions.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {
namespace duckboost {

void RegisterDuckBoostFunctions(ExtensionLoader &loader);
void RegisterIntervalFunctions(ExtensionLoader &loader);

unordered_map<string, string> MapVectorToOptions(Vector &map_vector, idx_t row);
vector<double> ReadFeatureList(Vector &list_vector, idx_t row);
vector<string> ReadVarcharList(Vector &list_vector, idx_t row);

} // namespace duckboost
} // namespace duckdb
