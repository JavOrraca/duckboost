//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckboost/functions.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {
class AggregateFunctionSet;
struct AggregateFinalizeInputData;
class TaskScheduler;
namespace duckboost {

void ConfigureTrainingScheduler(AggregateFunctionSet &functions, ExtensionLoader &loader);
TaskScheduler &GetTrainingScheduler(const AggregateFinalizeInputData &input);
void RegisterDuckBoostFunctions(ExtensionLoader &loader);
void RegisterIntervalFunctions(ExtensionLoader &loader);

unordered_map<string, string> MapVectorToOptions(Vector &map_vector, idx_t row);
vector<double> ReadFeatureList(Vector &list_vector, idx_t row);
vector<string> ReadVarcharList(Vector &list_vector, idx_t row);

} // namespace duckboost
} // namespace duckdb
