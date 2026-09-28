//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckboost/model.hpp
//
// Unified tree-ensemble model format shared by all duckboost backends.
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string.hpp"
#include "duckdb/common/typedefs.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/common/vector.hpp"

namespace duckdb {
namespace duckboost {

enum class BoostBackend : uint8_t { REFERENCE = 0, XGBOOST = 1, LIGHTGBM = 2, CATBOOST = 3 };

enum class BoostTask : uint8_t { REGRESSION = 0, BINARY = 1, MULTICLASS = 2 };

//! LESS: feature < threshold → left (default). EQUAL: feature == threshold → right (CatBoost OneHot).
enum class SplitCompare : uint8_t { LESS = 0, EQUAL = 1 };

struct TreeNode {
	idx_t feature = 0;
	double threshold = 0;
	idx_t left = 0;
	idx_t right = 0;
	double value = 0;
	bool is_leaf = true;
	SplitCompare compare = SplitCompare::LESS;
};

struct BoostTree {
	vector<TreeNode> nodes;
};

enum class CtrElementKind : uint8_t { CAT_FEATURE_VALUE = 0, FLOAT_FEATURE = 1, CAT_FEATURE_EXACT_VALUE = 2 };

//! One component of a CatBoost CTR combination hash.
struct CtrCombineElement {
	CtrElementKind kind = CtrElementKind::CAT_FEATURE_VALUE;
	idx_t feature_index = 0; // flat feature index
	double border_or_value = 0;
};

//! CatBoost OnlineCtr feature evaluated at predict time into a synthetic feature slot.
struct CtrFeatureSpec {
	idx_t feature_index = 0;
	string ctr_type = "Counter";
	double prior_numerator = 0;
	double prior_denominator = 1;
	double scale = 1;
	double shift = 0;
	int64_t counter_denominator = 0;
	//! Combination elements in CatBoost order (cat / float bin / one-hot).
	vector<CtrCombineElement> elements;
	//! Legacy/simple path: flat categorical indices only (used when elements empty).
	vector<idx_t> cat_feature_indices;
	//! Counter: hash → count. Borders: hash → (failures, successes) packed as pair in parallel maps.
	vector<uint64_t> hash_keys;
	vector<int64_t> hash_values;     // Counter counts, or Borders failures
	vector<int64_t> hash_values_alt; // Borders successes (empty for Counter)
};

struct BoostModel {
	idx_t duckboost_version = 1;
	BoostBackend backend = BoostBackend::REFERENCE;
	BoostTask task = BoostTask::REGRESSION;
	double base_score = 0;
	//! Per-class biases for multiclass; empty means use base_score for every class.
	vector<double> base_scores;
	double learning_rate = 0.1;
	idx_t n_features = 0;
	//! Number of caller-supplied features before synthetic CTR slots.
	idx_t n_raw_features = 0;
	//! 1 for regression/binary; >= 2 for multiclass.
	idx_t n_classes = 1;
	vector<string> feature_names;
	//! For multiclass: trees laid out as [round][class] → index round * n_classes + class.
	vector<BoostTree> trees;
	vector<CtrFeatureSpec> ctr_features;

	string ToJSON() const;
	static BoostModel FromJSON(const string &json);

	double ClassBias(idx_t class_idx) const;
	vector<double> MaterializeFeatures(const vector<double> &features) const;
	double EvalTree(const BoostTree &tree, const vector<double> &features) const;
	double PredictRaw(const vector<double> &features) const;
	vector<double> PredictRawMulti(const vector<double> &features) const;
	//! Regression/binary probability, or multiclass argmax class index.
	double Predict(const vector<double> &features) const;
	vector<double> PredictProba(const vector<double> &features) const;
};

struct TrainOptions {
	BoostBackend backend = BoostBackend::REFERENCE;
	BoostTask task = BoostTask::REGRESSION;
	idx_t n_estimators = 10;
	idx_t max_depth = 3;
	double learning_rate = 0.1;
	idx_t min_samples_leaf = 1;
	idx_t max_bins = 16;
	//! Multiclass only. 0 means max(label) + 1.
	idx_t n_classes = 0;
	vector<string> feature_names;

	static TrainOptions FromMap(const unordered_map<string, string> &options);
};

//! Class count for a multiclass train; labels must be integer class indices in [0, n_classes).
idx_t ResolveClassCount(const vector<double> &y, const TrainOptions &options);

struct EvalOptions {
	string metric = "auto"; // auto | rmse | mae | accuracy | logloss
	static EvalOptions FromMap(const unordered_map<string, string> &options);
};

struct SqlExportOptions {
	bool separate_trees = true;
	string prediction_alias = "prediction";
	static SqlExportOptions FromMap(const unordered_map<string, string> &options);
};

string BackendToString(BoostBackend backend);
BoostBackend BackendFromString(const string &name);
string TaskToString(BoostTask task);
BoostTask TaskFromString(const string &name);

bool BackendTrainingSupported(BoostBackend backend);
string BackendCapabilityNote(BoostBackend backend);

BoostModel TrainModel(const vector<double> &y, const vector<vector<double>> &x, const TrainOptions &options);
double EvaluateModel(const BoostModel &model, const vector<double> &y, const vector<vector<double>> &x,
                     const EvalOptions &options);
string ExportModelSQL(const BoostModel &model, const string &table_name, const vector<string> &feature_columns,
                      const SqlExportOptions &options);

} // namespace duckboost
} // namespace duckdb
