#pragma once

#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {
namespace duckboost {

void RegisterRawFunctions(ExtensionLoader &loader);

} // namespace duckboost
} // namespace duckdb
