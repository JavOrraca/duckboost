//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckboost/import.hpp
//
// Vendor model-dump → duckboost converters.
//===----------------------------------------------------------------------===//

#pragma once

#include "duckboost/model.hpp"

namespace duckdb {
namespace duckboost {

struct ImportOptions {
	BoostTask task = BoostTask::REGRESSION;
	bool task_set = false;
	double base_score = 0;
	bool base_score_set = false;
	double learning_rate = 1.0;
	bool learning_rate_set = false;
	idx_t n_classes = 0;
	bool n_classes_set = false;
	vector<string> feature_names;

	static ImportOptions FromMap(const unordered_map<string, string> &options);
};

//! Import a vendor dump (or native duckboost JSON) into a BoostModel.
BoostModel ImportModel(BoostBackend backend, const string &dump, const ImportOptions &options = ImportOptions());

BoostModel ImportXGBoostJSON(const string &dump, const ImportOptions &options);
BoostModel ImportLightGBMText(const string &dump, const ImportOptions &options);
BoostModel ImportCatBoostJSON(const string &dump, const ImportOptions &options);

bool LooksLikeDuckBoostJSON(const string &dump);

} // namespace duckboost
} // namespace duckdb
