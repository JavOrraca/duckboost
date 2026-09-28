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

} // namespace duckboost
} // namespace duckdb
