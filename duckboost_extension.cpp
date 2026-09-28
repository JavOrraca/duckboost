#define DUCKDB_EXTENSION_MAIN

#include "duckboost_extension.hpp"
#include "duckboost/functions.hpp"

namespace duckdb {

static void LoadInternal(ExtensionLoader &loader) {
	duckboost::RegisterDuckBoostFunctions(loader);
}

void DuckboostExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}

std::string DuckboostExtension::Name() {
	return "duckboost";
}

std::string DuckboostExtension::Version() const {
	return DefaultVersion();
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(duckboost, loader) {
	duckdb::LoadInternal(loader);
}
}
