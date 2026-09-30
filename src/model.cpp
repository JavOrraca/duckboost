#include "duckboost/model.hpp"
#include "duckboost/json_util.hpp"
#include "duckboost/native_train.hpp"

#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/string_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <unordered_map>

namespace duckdb {
namespace duckboost {

bool NodeGoesLeft(const TreeNode &node, double value) {
	if (node.compare != SplitCompare::LESS && std::isnan(value)) {
		return node.default_left;
	}
	if (node.compare == SplitCompare::EQUAL) {
		return value != node.threshold;
	}
	if (node.compare == SplitCompare::IN) {
		return !std::binary_search(node.categories.begin(), node.categories.end(), value);
	}
	if (node.missing_type == SplitMissingType::ZERO && (std::isnan(value) || value == 0.0)) {
		return node.default_left;
	}
	if (node.missing_type == SplitMissingType::NAN_VALUE && std::isnan(value)) {
		return node.default_left;
	}
	if (node.missing_type == SplitMissingType::NONE && std::isnan(value)) {
		value = 0.0;
	}
	return value < node.threshold;
}

namespace {

string QuoteIdent(const string &name) {
	return "\"" + StringUtil::Replace(name, "\"", "\"\"") + "\"";
}

string TreeToSQL(const BoostTree &tree, const vector<string> &feature_columns, idx_t node_idx) {
	if (node_idx >= tree.nodes.size()) {
		throw InvalidInputException("duckboost: invalid tree node index in SQL export");
	}
	auto &node = tree.nodes[node_idx];
	if (node.is_leaf) {
		return FormatDouble(node.value);
	}
	if (node.feature >= feature_columns.size()) {
		throw InvalidInputException("duckboost: feature index out of range in SQL export");
	}
	auto feature = QuoteIdent(feature_columns[node.feature]);
	auto left = TreeToSQL(tree, feature_columns, node.left);
	auto right = TreeToSQL(tree, feature_columns, node.right);
	auto missing_branch = node.default_left ? left : right;
	if (node.compare == SplitCompare::IN) {
		if (node.categories.empty()) {
			return "CASE WHEN " + feature + " IS NULL OR isnan(" + feature + ") THEN " + missing_branch + " ELSE " +
			       left + " END";
		}
		std::ostringstream categories;
		for (idx_t i = 0; i < node.categories.size(); i++) {
			if (i > 0) {
				categories << ',';
			}
			categories << FormatDouble(node.categories[i]);
		}
		return "CASE WHEN " + feature + " IS NULL OR isnan(" + feature + ") THEN " + missing_branch + " WHEN " +
		       feature + " IN (" + categories.str() + ") THEN " + right + " ELSE " + left + " END";
	}
	if (node.compare == SplitCompare::EQUAL) {
		// OneHot True (equal) → right; NULL/NaN follow learned default_left.
		return "CASE WHEN " + feature + " IS NULL OR isnan(" + feature + ") THEN " + missing_branch + " WHEN " +
		       feature + " = " + FormatDouble(node.threshold) + " THEN " + right + " ELSE " + left + " END";
	}
	auto missing_test = feature + " IS NULL OR isnan(" + feature + ")";
	if (node.missing_type == SplitMissingType::ZERO) {
		missing_test += " OR " + feature + " = 0";
	} else if (node.missing_type == SplitMissingType::NONE) {
		missing_branch = 0.0 < node.threshold ? left : right;
	}
	return "CASE WHEN " + missing_test + " THEN " + missing_branch + " WHEN " + feature + " < " +
	       FormatDouble(node.threshold) + " THEN " + left + " ELSE " + right + " END";
}

static constexpr uint64_t CTR_MAGIC_MULT = 0x4906ba494954cb65ULL;

uint64_t FeatureHashU64(double value) {
	// CatBoost passes CityHash as signed/unsigned 32-bit; preserve bit pattern via int32.
	auto as_i = static_cast<int64_t>(std::llround(value));
	return static_cast<uint64_t>(static_cast<uint32_t>(static_cast<int32_t>(as_i)));
}

uint64_t CtrElementValue(const CtrCombineElement &element, const vector<double> &features) {
	if (element.feature_index >= features.size()) {
		throw InvalidInputException("duckboost: CTR combination feature index out of range");
	}
	switch (element.kind) {
	case CtrElementKind::CAT_FEATURE_VALUE:
		return FeatureHashU64(features[element.feature_index]);
	case CtrElementKind::FLOAT_FEATURE:
		// CatBoost combination float bin: feature >= border → 1 else 0.
		return features[element.feature_index] >= element.border_or_value ? 1 : 0;
	case CtrElementKind::CAT_FEATURE_EXACT_VALUE:
		return features[element.feature_index] == element.border_or_value ? 1 : 0;
	default:
		throw InvalidInputException("duckboost: unknown CTR combination element kind");
	}
}

uint64_t CombineCtrHash(const CtrFeatureSpec &ctr, const vector<double> &features) {
	uint64_t hash = 0;
	if (!ctr.elements.empty()) {
		for (auto &element : ctr.elements) {
			auto value = CtrElementValue(element, features);
			hash = CTR_MAGIC_MULT * (hash + CTR_MAGIC_MULT * value);
		}
		return hash;
	}
	for (auto idx : ctr.cat_feature_indices) {
		if (idx >= features.size()) {
			throw InvalidInputException("duckboost: CTR cat feature index out of range");
		}
		auto value = FeatureHashU64(features[idx]);
		hash = CTR_MAGIC_MULT * (hash + CTR_MAGIC_MULT * value);
	}
	return hash;
}

double CtrValueFromCounts(const CtrFeatureSpec &ctr, int64_t count_or_failures, int64_t successes) {
	auto type = StringUtil::Lower(ctr.ctr_type);
	if (type == "counter" || type == "featurefreq" || type == "freq") {
		auto denom = static_cast<double>(ctr.counter_denominator) + ctr.prior_denominator;
		if (denom == 0) {
			return ctr.shift;
		}
		return ctr.shift + ctr.scale * ((ctr.prior_numerator + static_cast<double>(count_or_failures)) / denom);
	}
	if (type == "borders" || type == "buckets" || type == "border") {
		auto denom = ctr.prior_numerator + static_cast<double>(successes) + ctr.prior_denominator +
		             static_cast<double>(count_or_failures);
		if (denom == 0) {
			return ctr.shift;
		}
		return ctr.shift + ctr.scale * ((ctr.prior_numerator + static_cast<double>(successes)) / denom);
	}
	throw NotImplementedException("duckboost: unsupported CTR type '%s'", ctr.ctr_type);
}

double EvaluateCtrValue(const CtrFeatureSpec &ctr, const vector<double> &features) {
	auto key = CombineCtrHash(ctr, features);
	unordered_map<uint64_t, idx_t> lookup;
	lookup.reserve(ctr.hash_keys.size());
	for (idx_t i = 0; i < ctr.hash_keys.size(); i++) {
		lookup[ctr.hash_keys[i]] = i;
	}
	auto it = lookup.find(key);
	int64_t primary = 0;
	int64_t secondary = 0;
	if (it != lookup.end()) {
		primary = ctr.hash_values[it->second];
		if (it->second < ctr.hash_values_alt.size()) {
			secondary = ctr.hash_values_alt[it->second];
		}
	}
	return CtrValueFromCounts(ctr, primary, secondary);
}

string CtrElementKindToString(CtrElementKind kind) {
	switch (kind) {
	case CtrElementKind::FLOAT_FEATURE:
		return "float_feature";
	case CtrElementKind::CAT_FEATURE_EXACT_VALUE:
		return "cat_feature_exact_value";
	case CtrElementKind::CAT_FEATURE_VALUE:
	default:
		return "cat_feature_value";
	}
}

CtrElementKind CtrElementKindFromString(const string &name) {
	auto lower = StringUtil::Lower(name);
	if (lower == "float_feature") {
		return CtrElementKind::FLOAT_FEATURE;
	}
	if (lower == "cat_feature_exact_value") {
		return CtrElementKind::CAT_FEATURE_EXACT_VALUE;
	}
	if (lower == "cat_feature_value") {
		return CtrElementKind::CAT_FEATURE_VALUE;
	}
	throw InvalidInputException("duckboost: unknown CTR combination_element '%s'", name);
}

string SqlCtrElementValueExpr(const CtrCombineElement &element, const vector<string> &feature_columns) {
	if (element.feature_index >= feature_columns.size()) {
		throw InvalidInputException("duckboost: CTR SQL export missing feature column for index %llu",
		                            (unsigned long long)element.feature_index);
	}
	auto feature = QuoteIdent(feature_columns[element.feature_index]);
	switch (element.kind) {
	case CtrElementKind::CAT_FEATURE_VALUE:
		// Feature is already a CityHash; cast through INT then UINT32 bit pattern.
		return "CAST(CAST(CAST(ROUND(" + feature + ") AS BIGINT) AS INTEGER) AS UINTEGER)::UHUGEINT";
	case CtrElementKind::FLOAT_FEATURE:
		return "(CASE WHEN " + feature + " >= " + FormatDouble(element.border_or_value) +
		       " THEN 1::UHUGEINT ELSE 0::UHUGEINT END)";
	case CtrElementKind::CAT_FEATURE_EXACT_VALUE:
		return "(CASE WHEN " + feature + " = " + FormatDouble(element.border_or_value) +
		       " THEN 1::UHUGEINT ELSE 0::UHUGEINT END)";
	default:
		throw InvalidInputException("duckboost: unknown CTR combination element in SQL export");
	}
}

string SqlCombineCtrHashExpr(const CtrFeatureSpec &ctr, const vector<string> &feature_columns) {
	const string magic = "5260239421824346981::UHUGEINT"; // 0x4906ba494954cb65
	const string mod = "18446744073709551616::UHUGEINT";
	string hash = "0::UHUGEINT";
	auto append_value = [&](const string &value_expr) {
		hash = "((" + magic + " * ((" + hash + " + ((" + magic + " * (" + value_expr + ")) % " + mod + ")) % " + mod +
		       ")) % " + mod + ")";
	};
	if (!ctr.elements.empty()) {
		for (auto &element : ctr.elements) {
			append_value(SqlCtrElementValueExpr(element, feature_columns));
		}
	} else {
		for (auto idx : ctr.cat_feature_indices) {
			CtrCombineElement element;
			element.kind = CtrElementKind::CAT_FEATURE_VALUE;
			element.feature_index = idx;
			append_value(SqlCtrElementValueExpr(element, feature_columns));
		}
	}
	return "CAST(" + hash + " AS UBIGINT)";
}

string SqlCtrValueExpr(const CtrFeatureSpec &ctr, const vector<string> &feature_columns) {
	auto hash_expr = SqlCombineCtrHashExpr(ctr, feature_columns);
	double default_value = CtrValueFromCounts(ctr, 0, 0);
	if (ctr.hash_keys.empty()) {
		return FormatDouble(default_value);
	}
	string expr = "CASE " + hash_expr;
	for (idx_t i = 0; i < ctr.hash_keys.size(); i++) {
		int64_t secondary = i < ctr.hash_values_alt.size() ? ctr.hash_values_alt[i] : 0;
		auto value = CtrValueFromCounts(ctr, ctr.hash_values[i], secondary);
		expr += " WHEN " + std::to_string(ctr.hash_keys[i]) + "::UBIGINT THEN " + FormatDouble(value);
	}
	expr += " ELSE " + FormatDouble(default_value) + " END";
	return expr;
}

double Sigmoid(double x) {
	return 1.0 / (1.0 + std::exp(-x));
}

} // namespace

string BackendToString(BoostBackend backend) {
	switch (backend) {
	case BoostBackend::REFERENCE:
		return "reference";
	case BoostBackend::XGBOOST:
		return "xgboost";
	case BoostBackend::LIGHTGBM:
		return "lightgbm";
	case BoostBackend::CATBOOST:
		return "catboost";
	default:
		return "unknown";
	}
}

BoostBackend BackendFromString(const string &name) {
	auto lower = StringUtil::Lower(name);
	if (lower == "reference" || lower == "duckboost" || lower == "gbdt") {
		return BoostBackend::REFERENCE;
	}
	if (lower == "xgboost" || lower == "xgb") {
		return BoostBackend::XGBOOST;
	}
	if (lower == "lightgbm" || lower == "lgbm" || lower == "lgb") {
		return BoostBackend::LIGHTGBM;
	}
	if (lower == "catboost" || lower == "cb") {
		return BoostBackend::CATBOOST;
	}
	throw InvalidInputException("duckboost: unknown backend '%s' (expected reference, xgboost, lightgbm, or catboost)",
	                            name);
}

string TaskToString(BoostTask task) {
	switch (task) {
	case BoostTask::BINARY:
		return "binary";
	case BoostTask::MULTICLASS:
		return "multiclass";
	case BoostTask::REGRESSION:
	default:
		return "regression";
	}
}

BoostTask TaskFromString(const string &name) {
	auto lower = StringUtil::Lower(name);
	if (lower == "regression" || lower == "regressor" || lower == "mse") {
		return BoostTask::REGRESSION;
	}
	if (lower == "binary" || lower == "classification" || lower == "logistic" || lower == "binomial") {
		return BoostTask::BINARY;
	}
	if (lower == "multiclass" || lower == "multi_class" || lower == "multi-class" || lower == "softmax") {
		return BoostTask::MULTICLASS;
	}
	throw InvalidInputException("duckboost: unknown task '%s' (expected regression, binary, or multiclass)", name);
}

bool BackendTrainingSupported(BoostBackend backend) {
	if (backend == BoostBackend::REFERENCE) {
		return true;
	}
	return NativeTrainerLinked(backend);
}

string BackendCapabilityNote(BoostBackend backend) {
	switch (backend) {
	case BoostBackend::REFERENCE:
		return "in-process reference GBDT (train/predict/evaluate/to_sql)";
	case BoostBackend::XGBOOST:
		if (NativeTrainerLinked(backend)) {
			return "native train via XGBoost C API (dump→import); also duckboost_import dump_model JSON";
		}
		if (NativeTrainerCompiled(backend)) {
			return "import dump_model JSON via duckboost_import; native train compiled as stub";
		}
		return "import dump_model JSON via duckboost_import; native train requires DUCKBOOST_WITH_XGBOOST";
	case BoostBackend::LIGHTGBM:
		if (NativeTrainerLinked(backend)) {
			return "native train via LightGBM C API (dump→import); also duckboost_import save_model text";
		}
		if (NativeTrainerCompiled(backend)) {
			return "import save_model text via duckboost_import; native train compiled as stub";
		}
		return "import save_model text via duckboost_import; native train requires DUCKBOOST_WITH_LIGHTGBM";
	case BoostBackend::CATBOOST:
		if (NativeTrainerCompiled(backend)) {
			return "import save_model JSON via duckboost_import; no public CatBoost train C API (use dump import)";
		}
		return "import save_model JSON via duckboost_import; native train requires dump import (no public C API)";
	default:
		return "unknown";
	}
}

TrainOptions TrainOptions::FromMap(const unordered_map<string, string> &options) {
	TrainOptions result;
	for (auto &entry : options) {
		auto key = StringUtil::Lower(entry.first);
		auto &value = entry.second;
		if (key == "backend") {
			result.backend = BackendFromString(value);
		} else if (key == "task" || key == "objective") {
			// Legacy: objective doubled as task name (regression/binary/multiclass).
			result.task = TaskFromString(value);
		} else if (key == "n_estimators" || key == "num_boost_round" || key == "iterations") {
			result.n_estimators = static_cast<idx_t>(std::stoull(value));
		} else if (key == "max_depth" || key == "depth") {
			result.max_depth = static_cast<idx_t>(std::stoull(value));
		} else if (key == "learning_rate" || key == "eta" || key == "lr") {
			result.learning_rate = std::stod(value);
		} else if (key == "min_samples_leaf" || key == "min_data_in_leaf") {
			result.min_samples_leaf = static_cast<idx_t>(std::stoull(value));
		} else if (key == "min_child_weight") {
			result.min_child_weight = std::stod(value);
		} else if (key == "max_bins") {
			result.max_bins = static_cast<idx_t>(std::stoull(value));
		} else if (key == "reg_lambda" || key == "lambda" || key == "lambda_l2") {
			result.reg_lambda = std::stod(value);
		} else if (key == "reg_alpha" || key == "alpha" || key == "lambda_l1") {
			result.reg_alpha = std::stod(value);
		} else if (key == "min_split_gain" || key == "gamma") {
			result.min_split_gain = std::stod(value);
		} else if (key == "subsample" || key == "bagging_fraction") {
			result.subsample = std::stod(value);
		} else if (key == "colsample_bytree" || key == "colsample" || key == "feature_fraction") {
			result.colsample_bytree = std::stod(value);
		} else if (key == "validation_fraction" || key == "valid_fraction") {
			result.validation_fraction = std::stod(value);
		} else if (key == "early_stopping_rounds" || key == "early_stopping") {
			result.early_stopping_rounds = static_cast<idx_t>(std::stoull(value));
		} else if (key == "seed" || key == "random_seed") {
			result.seed = static_cast<uint64_t>(std::stoull(value));
		} else if (key == "n_classes" || key == "num_class" || key == "num_classes") {
			result.n_classes = static_cast<idx_t>(std::stoull(value));
			if (result.n_classes < 2) {
				throw InvalidInputException("duckboost: n_classes must be >= 2");
			}
		} else if (key == "class_weight" || key == "class_weights") {
			result.class_weight = value;
		} else if (key == "feature_names") {
			result.feature_names = StringUtil::Split(value, ',');
			for (auto &name : result.feature_names) {
				StringUtil::Trim(name);
			}
		} else if (key == "categorical_features" || key == "cat_features" || key == "categorical_feature") {
			result.categorical_features = StringUtil::Split(value, ',');
			for (auto &feature : result.categorical_features) {
				StringUtil::Trim(feature);
			}
		} else {
			throw InvalidInputException("duckboost: unknown train option '%s'", entry.first);
		}
	}
	if (result.n_estimators == 0) {
		throw InvalidInputException("duckboost: n_estimators must be > 0");
	}
	if (result.max_depth == 0) {
		throw InvalidInputException("duckboost: max_depth must be > 0");
	}
	if (!(result.learning_rate > 0)) {
		throw InvalidInputException("duckboost: learning_rate must be > 0");
	}
	if (!(result.min_child_weight >= 0)) {
		throw InvalidInputException("duckboost: min_child_weight must be >= 0");
	}
	if (!(result.reg_lambda >= 0) || !(result.reg_alpha >= 0) || !(result.min_split_gain >= 0)) {
		throw InvalidInputException("duckboost: reg_lambda, reg_alpha, and min_split_gain must be >= 0");
	}
	if (!(result.subsample > 0 && result.subsample <= 1.0)) {
		throw InvalidInputException("duckboost: subsample must be in (0, 1]");
	}
	if (!(result.colsample_bytree > 0 && result.colsample_bytree <= 1.0)) {
		throw InvalidInputException("duckboost: colsample_bytree must be in (0, 1]");
	}
	if (!(result.validation_fraction >= 0 && result.validation_fraction < 1.0)) {
		throw InvalidInputException("duckboost: validation_fraction must be in [0, 1)");
	}
	if (result.max_bins == 0) {
		throw InvalidInputException("duckboost: max_bins must be > 0");
	}
	if (result.early_stopping_rounds > 0 && result.validation_fraction == 0) {
		result.validation_fraction = 0.2;
	}
	return result;
}

EvalOptions EvalOptions::FromMap(const unordered_map<string, string> &options) {
	EvalOptions result;
	result.metric = "auto";
	for (auto &entry : options) {
		auto key = StringUtil::Lower(entry.first);
		if (key == "metric") {
			result.metric = StringUtil::Lower(entry.second);
		} else {
			throw InvalidInputException("duckboost: unknown evaluate option '%s'", entry.first);
		}
	}
	return result;
}

SqlExportOptions SqlExportOptions::FromMap(const unordered_map<string, string> &options) {
	SqlExportOptions result;
	for (auto &entry : options) {
		auto key = StringUtil::Lower(entry.first);
		auto value = StringUtil::Lower(entry.second);
		if (key == "separate_trees") {
			if (value == "true" || value == "1" || value == "yes") {
				result.separate_trees = true;
			} else if (value == "false" || value == "0" || value == "no") {
				result.separate_trees = false;
			} else {
				throw InvalidInputException("duckboost: separate_trees must be true/false");
			}
		} else if (key == "prediction_alias" || key == "alias") {
			result.prediction_alias = entry.second;
		} else {
			throw InvalidInputException("duckboost: unknown to_sql option '%s'", entry.first);
		}
	}
	return result;
}

static bool ParseBoolOption(const string &value, const char *name) {
	auto lower = StringUtil::Lower(value);
	if (lower == "true" || lower == "1" || lower == "yes") {
		return true;
	}
	if (lower == "false" || lower == "0" || lower == "no") {
		return false;
	}
	throw InvalidInputException("duckboost: %s must be true/false", name);
}

ImportanceOptions ImportanceOptions::FromMap(const unordered_map<string, string> &options) {
	ImportanceOptions result;
	for (auto &entry : options) {
		auto key = StringUtil::Lower(entry.first);
		auto value = StringUtil::Lower(entry.second);
		if (key == "metric") {
			if (value != "gain" && value != "cover" && value != "frequency" && value != "weight" && value != "count") {
				throw InvalidInputException("duckboost: importance metric must be gain, cover, or frequency (got '%s')",
				                            entry.second);
			}
			if (value == "weight" || value == "count") {
				value = "frequency";
			}
			result.metric = value;
		} else if (key == "normalize") {
			result.normalize = ParseBoolOption(entry.second, "normalize");
		} else if (key == "include_unused" || key == "include_zero" || key == "all_features") {
			result.include_unused = ParseBoolOption(entry.second, "include_unused");
		} else if (key == "sort" || key == "order_by") {
			if (value != "importance" && value != "gain" && value != "cover" && value != "frequency" &&
			    value != "weight" && value != "count" && value != "name" && value != "variable" && value != "index") {
				throw InvalidInputException(
				    "duckboost: importance sort must be importance, gain, cover, frequency, name, or index");
			}
			if (value == "weight" || value == "count") {
				value = "frequency";
			}
			if (value == "variable") {
				value = "name";
			}
			result.sort = value;
		} else {
			throw InvalidInputException("duckboost: unknown importance option '%s'", entry.first);
		}
	}
	return result;
}

static string FeatureNameAt(const BoostModel &model, idx_t feature_idx) {
	if (feature_idx < model.feature_names.size() && !model.feature_names[feature_idx].empty()) {
		return model.feature_names[feature_idx];
	}
	for (auto &ctr : model.ctr_features) {
		if (ctr.feature_index == feature_idx) {
			return "_duckboost_ctr_" + std::to_string(feature_idx);
		}
	}
	return "f" + std::to_string(feature_idx);
}

vector<FeatureImportance> ComputeFeatureImportance(const BoostModel &model, const ImportanceOptions &options) {
	const idx_t n_features = model.n_features > 0 ? model.n_features : model.feature_names.size();
	vector<FeatureImportance> rows(n_features);
	for (idx_t i = 0; i < n_features; i++) {
		rows[i].feature_index = i;
		rows[i].variable = FeatureNameAt(model, i);
	}
	idx_t max_seen = n_features;
	for (auto &tree : model.trees) {
		for (auto &node : tree.nodes) {
			if (node.is_leaf) {
				continue;
			}
			if (node.feature >= max_seen) {
				rows.resize(node.feature + 1);
				for (idx_t i = max_seen; i <= node.feature; i++) {
					rows[i].feature_index = i;
					rows[i].variable = FeatureNameAt(model, i);
				}
				max_seen = node.feature + 1;
			}
			rows[node.feature].gain += std::max(0.0, node.gain);
			rows[node.feature].cover += std::max(0.0, node.cover);
			rows[node.feature].frequency += 1;
		}
	}

	if (!options.include_unused) {
		vector<FeatureImportance> used;
		used.reserve(rows.size());
		for (auto &row : rows) {
			if (row.frequency > 0 || row.gain > 0 || row.cover > 0) {
				used.push_back(row);
			}
		}
		rows = std::move(used);
	}

	double total_gain = 0;
	double total_cover = 0;
	double total_frequency = 0;
	for (auto &row : rows) {
		total_gain += row.gain;
		total_cover += row.cover;
		total_frequency += static_cast<double>(row.frequency);
	}
	string effective_metric = options.metric;
	if (effective_metric == "gain" && total_gain <= 0) {
		effective_metric = "frequency";
	}

	auto metric_value = [&](const FeatureImportance &row) -> double {
		if (effective_metric == "cover") {
			return row.cover;
		}
		if (effective_metric == "frequency") {
			return static_cast<double>(row.frequency);
		}
		return row.gain;
	};
	double total = 0;
	if (effective_metric == "cover") {
		total = total_cover;
	} else if (effective_metric == "frequency") {
		total = total_frequency;
	} else {
		total = total_gain;
	}
	for (auto &row : rows) {
		auto value = metric_value(row);
		row.importance = options.normalize ? (total > 0 ? value / total : 0) : value;
	}

	std::sort(rows.begin(), rows.end(), [&](const FeatureImportance &a, const FeatureImportance &b) {
		int order = 0;
		if (options.sort == "gain") {
			order = (a.gain > b.gain) - (a.gain < b.gain);
		} else if (options.sort == "cover") {
			order = (a.cover > b.cover) - (a.cover < b.cover);
		} else if (options.sort == "frequency") {
			order = (a.frequency > b.frequency) - (a.frequency < b.frequency);
		} else if (options.sort == "name") {
			if (a.variable < b.variable) {
				order = 1;
			} else if (a.variable > b.variable) {
				order = -1;
			}
		} else if (options.sort == "index") {
			order = (a.feature_index < b.feature_index) - (a.feature_index > b.feature_index);
		} else {
			order = (a.importance > b.importance) - (a.importance < b.importance);
		}
		if (order != 0) {
			return order > 0;
		}
		if (a.feature_index != b.feature_index) {
			return a.feature_index < b.feature_index;
		}
		return a.variable < b.variable;
	});
	return rows;
}

string BoostModel::ToJSON() const {
	std::ostringstream out;
	out << "{\"duckboost_version\":" << duckboost_version;
	out << ",\"backend\":\"" << EscapeJSON(BackendToString(backend)) << "\"";
	out << ",\"task\":\"" << EscapeJSON(TaskToString(task)) << "\"";
	out << ",\"base_score\":" << FormatDouble(base_score);
	if (!base_scores.empty()) {
		out << ",\"base_scores\":[";
		for (idx_t i = 0; i < base_scores.size(); i++) {
			if (i > 0) {
				out << ',';
			}
			out << FormatDouble(base_scores[i]);
		}
		out << ']';
	}
	out << ",\"learning_rate\":" << FormatDouble(learning_rate);
	out << ",\"n_features\":" << n_features;
	out << ",\"n_classes\":" << n_classes;
	out << ",\"feature_names\":[";
	for (idx_t i = 0; i < feature_names.size(); i++) {
		if (i > 0) {
			out << ',';
		}
		out << '"' << EscapeJSON(feature_names[i]) << '"';
	}
	out << "],\"trees\":[";
	for (idx_t t = 0; t < trees.size(); t++) {
		if (t > 0) {
			out << ',';
		}
		out << "{\"nodes\":[";
		auto &tree = trees[t];
		for (idx_t n = 0; n < tree.nodes.size(); n++) {
			if (n > 0) {
				out << ',';
			}
			auto &node = tree.nodes[n];
			out << "{\"is_leaf\":" << (node.is_leaf ? "true" : "false");
			out << ",\"feature\":" << node.feature;
			out << ",\"threshold\":" << FormatDouble(node.threshold);
			out << ",\"left\":" << node.left;
			out << ",\"right\":" << node.right;
			out << ",\"value\":" << FormatDouble(node.value);
			if (node.compare == SplitCompare::EQUAL) {
				out << ",\"compare\":\"equal\"";
			} else if (node.compare == SplitCompare::IN) {
				out << ",\"compare\":\"in\",\"categories\":[";
				for (idx_t i = 0; i < node.categories.size(); i++) {
					if (i > 0) {
						out << ',';
					}
					out << FormatDouble(node.categories[i]);
				}
				out << ']';
			}
			if (!node.is_leaf) {
				out << ",\"default_left\":" << (node.default_left ? "true" : "false");
				if (node.missing_type == SplitMissingType::ZERO) {
					out << ",\"missing_type\":\"zero\"";
				} else if (node.missing_type == SplitMissingType::NONE) {
					out << ",\"missing_type\":\"none\"";
				}
				if (node.gain > 0) {
					out << ",\"gain\":" << FormatDouble(node.gain);
				}
				if (node.cover > 0) {
					out << ",\"cover\":" << FormatDouble(node.cover);
				}
			}
			out << '}';
		}
		out << "]}";
	}
	out << ']';
	if (n_raw_features > 0 || !ctr_features.empty()) {
		out << ",\"n_raw_features\":" << (n_raw_features > 0 ? n_raw_features : n_features);
	}
	if (!ctr_features.empty()) {
		out << ",\"ctr_features\":[";
		for (idx_t c = 0; c < ctr_features.size(); c++) {
			if (c > 0) {
				out << ',';
			}
			auto &ctr = ctr_features[c];
			out << "{\"feature_index\":" << ctr.feature_index;
			out << ",\"ctr_type\":\"" << EscapeJSON(ctr.ctr_type) << "\"";
			out << ",\"prior_numerator\":" << FormatDouble(ctr.prior_numerator);
			out << ",\"prior_denominator\":" << FormatDouble(ctr.prior_denominator);
			out << ",\"scale\":" << FormatDouble(ctr.scale);
			out << ",\"shift\":" << FormatDouble(ctr.shift);
			out << ",\"counter_denominator\":" << ctr.counter_denominator;
			out << ",\"elements\":[";
			for (idx_t i = 0; i < ctr.elements.size(); i++) {
				if (i > 0) {
					out << ',';
				}
				auto &el = ctr.elements[i];
				out << "{\"kind\":\"" << EscapeJSON(CtrElementKindToString(el.kind)) << "\"";
				out << ",\"feature_index\":" << el.feature_index;
				out << ",\"border_or_value\":" << FormatDouble(el.border_or_value) << '}';
			}
			out << "],\"cat_feature_indices\":[";
			for (idx_t i = 0; i < ctr.cat_feature_indices.size(); i++) {
				if (i > 0) {
					out << ',';
				}
				out << ctr.cat_feature_indices[i];
			}
			out << "],\"hash_keys\":[";
			for (idx_t i = 0; i < ctr.hash_keys.size(); i++) {
				if (i > 0) {
					out << ',';
				}
				out << '"' << std::to_string(ctr.hash_keys[i]) << '"';
			}
			out << "],\"hash_values\":[";
			for (idx_t i = 0; i < ctr.hash_values.size(); i++) {
				if (i > 0) {
					out << ',';
				}
				out << ctr.hash_values[i];
			}
			out << "],\"hash_values_alt\":[";
			for (idx_t i = 0; i < ctr.hash_values_alt.size(); i++) {
				if (i > 0) {
					out << ',';
				}
				out << ctr.hash_values_alt[i];
			}
			out << "]}";
		}
		out << ']';
	}
	out << '}';
	return out.str();
}

BoostModel BoostModel::FromJSON(const string &json) {
	JsonParser p(json);
	p.Expect('{');
	BoostModel model;
	bool first = true;
	while (!p.TryConsume('}')) {
		if (!first) {
			p.Expect(',');
		}
		first = false;
		auto key = p.ParseString();
		p.Expect(':');
		if (key == "duckboost_version") {
			model.duckboost_version = static_cast<idx_t>(p.ParseNumber());
		} else if (key == "backend") {
			model.backend = BackendFromString(p.ParseString());
		} else if (key == "task") {
			model.task = TaskFromString(p.ParseString());
		} else if (key == "base_score") {
			model.base_score = p.ParseNumber();
		} else if (key == "base_scores") {
			p.Expect('[');
			bool first_score = true;
			while (!p.TryConsume(']')) {
				if (!first_score) {
					p.Expect(',');
				}
				first_score = false;
				model.base_scores.push_back(p.ParseNumber());
			}
		} else if (key == "learning_rate") {
			model.learning_rate = p.ParseNumber();
		} else if (key == "n_features") {
			model.n_features = static_cast<idx_t>(p.ParseNumber());
		} else if (key == "n_classes") {
			model.n_classes = static_cast<idx_t>(p.ParseNumber());
		} else if (key == "feature_names") {
			p.Expect('[');
			bool first_name = true;
			while (!p.TryConsume(']')) {
				if (!first_name) {
					p.Expect(',');
				}
				first_name = false;
				model.feature_names.push_back(p.ParseString());
			}
		} else if (key == "trees") {
			p.Expect('[');
			bool first_tree = true;
			while (!p.TryConsume(']')) {
				if (!first_tree) {
					p.Expect(',');
				}
				first_tree = false;
				p.Expect('{');
				BoostTree tree;
				bool first_field = true;
				while (!p.TryConsume('}')) {
					if (!first_field) {
						p.Expect(',');
					}
					first_field = false;
					auto tree_key = p.ParseString();
					p.Expect(':');
					if (tree_key != "nodes") {
						p.SkipValue();
						continue;
					}
					p.Expect('[');
					bool first_node = true;
					while (!p.TryConsume(']')) {
						if (!first_node) {
							p.Expect(',');
						}
						first_node = false;
						p.Expect('{');
						TreeNode node;
						bool first_node_field = true;
						while (!p.TryConsume('}')) {
							if (!first_node_field) {
								p.Expect(',');
							}
							first_node_field = false;
							auto node_key = p.ParseString();
							p.Expect(':');
							if (node_key == "is_leaf") {
								node.is_leaf = p.ParseBool();
							} else if (node_key == "feature") {
								node.feature = static_cast<idx_t>(p.ParseNumber());
							} else if (node_key == "threshold") {
								node.threshold = p.ParseNumber();
							} else if (node_key == "left") {
								node.left = static_cast<idx_t>(p.ParseNumber());
							} else if (node_key == "right") {
								node.right = static_cast<idx_t>(p.ParseNumber());
							} else if (node_key == "value") {
								node.value = p.ParseNumber();
							} else if (node_key == "compare") {
								auto cmp = StringUtil::Lower(p.ParseString());
								if (cmp == "less" || cmp == "lt" || cmp == "<") {
									node.compare = SplitCompare::LESS;
								} else if (cmp == "equal" || cmp == "eq" || cmp == "==") {
									node.compare = SplitCompare::EQUAL;
								} else if (cmp == "in" || cmp == "set") {
									node.compare = SplitCompare::IN;
								} else {
									throw InvalidInputException("duckboost: unknown split compare '%s'", cmp);
								}
							} else if (node_key == "categories") {
								p.Expect('[');
								bool first_category = true;
								while (!p.TryConsume(']')) {
									if (!first_category) {
										p.Expect(',');
									}
									first_category = false;
									node.categories.push_back(p.ParseNumber());
								}
							} else if (node_key == "default_left") {
								node.default_left = p.ParseBool();
							} else if (node_key == "missing_type") {
								auto missing = StringUtil::Lower(p.ParseString());
								if (missing == "nan") {
									node.missing_type = SplitMissingType::NAN_VALUE;
								} else if (missing == "zero") {
									node.missing_type = SplitMissingType::ZERO;
								} else if (missing == "none") {
									node.missing_type = SplitMissingType::NONE;
								} else {
									throw InvalidInputException("duckboost: unknown split missing_type '%s'", missing);
								}
							} else if (node_key == "gain") {
								node.gain = p.ParseNumber();
							} else if (node_key == "cover") {
								node.cover = p.ParseNumber();
							} else {
								p.SkipValue();
							}
						}
						if (node.compare == SplitCompare::IN) {
							std::sort(node.categories.begin(), node.categories.end());
							node.categories.erase(std::unique(node.categories.begin(), node.categories.end()),
							                      node.categories.end());
						}
						tree.nodes.push_back(node);
					}
				}
				if (tree.nodes.empty()) {
					throw InvalidInputException("duckboost: tree has no nodes");
				}
				model.trees.push_back(std::move(tree));
			}
		} else if (key == "n_raw_features") {
			model.n_raw_features = static_cast<idx_t>(p.ParseNumber());
		} else if (key == "ctr_features") {
			p.Expect('[');
			bool first_ctr = true;
			while (!p.TryConsume(']')) {
				if (!first_ctr) {
					p.Expect(',');
				}
				first_ctr = false;
				p.Expect('{');
				CtrFeatureSpec ctr;
				bool first_field = true;
				while (!p.TryConsume('}')) {
					if (!first_field) {
						p.Expect(',');
					}
					first_field = false;
					auto ck = p.ParseString();
					p.Expect(':');
					if (ck == "feature_index") {
						ctr.feature_index = static_cast<idx_t>(p.ParseNumber());
					} else if (ck == "ctr_type") {
						ctr.ctr_type = p.ParseString();
					} else if (ck == "prior_numerator") {
						ctr.prior_numerator = p.ParseNumber();
					} else if (ck == "prior_denominator") {
						ctr.prior_denominator = p.ParseNumber();
					} else if (ck == "scale") {
						ctr.scale = p.ParseNumber();
					} else if (ck == "shift") {
						ctr.shift = p.ParseNumber();
					} else if (ck == "counter_denominator") {
						ctr.counter_denominator = static_cast<int64_t>(p.ParseNumber());
					} else if (ck == "elements") {
						p.Expect('[');
						bool first_el = true;
						while (!p.TryConsume(']')) {
							if (!first_el) {
								p.Expect(',');
							}
							first_el = false;
							p.Expect('{');
							CtrCombineElement element;
							bool first_efield = true;
							while (!p.TryConsume('}')) {
								if (!first_efield) {
									p.Expect(',');
								}
								first_efield = false;
								auto ek = p.ParseString();
								p.Expect(':');
								if (ek == "kind" || ek == "combination_element") {
									element.kind = CtrElementKindFromString(p.ParseString());
								} else if (ek == "feature_index") {
									element.feature_index = static_cast<idx_t>(p.ParseNumber());
								} else if (ek == "border_or_value" || ek == "border" || ek == "value") {
									element.border_or_value = p.ParseNumber();
								} else {
									p.SkipValue();
								}
							}
							ctr.elements.push_back(element);
						}
					} else if (ck == "cat_feature_indices") {
						p.Expect('[');
						bool first_idx = true;
						while (!p.TryConsume(']')) {
							if (!first_idx) {
								p.Expect(',');
							}
							first_idx = false;
							ctr.cat_feature_indices.push_back(static_cast<idx_t>(p.ParseNumber()));
						}
					} else if (ck == "hash_keys") {
						p.Expect('[');
						bool first_key = true;
						while (!p.TryConsume(']')) {
							if (!first_key) {
								p.Expect(',');
							}
							first_key = false;
							if (p.Peek() == '"') {
								ctr.hash_keys.push_back(std::stoull(p.ParseString()));
							} else {
								ctr.hash_keys.push_back(static_cast<uint64_t>(p.ParseNumber()));
							}
						}
					} else if (ck == "hash_values") {
						p.Expect('[');
						bool first_val = true;
						while (!p.TryConsume(']')) {
							if (!first_val) {
								p.Expect(',');
							}
							first_val = false;
							ctr.hash_values.push_back(static_cast<int64_t>(p.ParseNumber()));
						}
					} else if (ck == "hash_values_alt") {
						p.Expect('[');
						bool first_val = true;
						while (!p.TryConsume(']')) {
							if (!first_val) {
								p.Expect(',');
							}
							first_val = false;
							ctr.hash_values_alt.push_back(static_cast<int64_t>(p.ParseNumber()));
						}
					} else {
						p.SkipValue();
					}
				}
				model.ctr_features.push_back(std::move(ctr));
			}
		} else {
			p.SkipValue();
		}
	}
	if (model.n_features == 0 && !model.feature_names.empty()) {
		model.n_features = model.feature_names.size();
	}
	if (model.n_raw_features == 0) {
		model.n_raw_features = model.n_features;
	}
	if (model.task == BoostTask::MULTICLASS) {
		if (model.n_classes < 2) {
			model.n_classes = MaxValue<idx_t>(model.base_scores.size(), 2);
		}
	} else if (model.n_classes == 0) {
		model.n_classes = 1;
	}
	return model;
}

double BoostModel::ClassBias(idx_t class_idx) const {
	if (class_idx < base_scores.size()) {
		return base_scores[class_idx];
	}
	return base_score;
}

vector<double> BoostModel::MaterializeFeatures(const vector<double> &features) const {
	auto required_raw = n_raw_features > 0 ? n_raw_features : n_features;
	if (!ctr_features.empty()) {
		required_raw = n_raw_features > 0 ? n_raw_features : required_raw;
	}
	if (features.size() < required_raw) {
		throw InvalidInputException("duckboost: expected at least %llu features, got %llu",
		                            (unsigned long long)required_raw, (unsigned long long)features.size());
	}
	vector<double> materialized = features;
	if (materialized.size() < n_features) {
		materialized.resize(n_features, 0);
	}
	for (auto &ctr : ctr_features) {
		if (ctr.feature_index >= materialized.size()) {
			materialized.resize(ctr.feature_index + 1, 0);
		}
		materialized[ctr.feature_index] = EvaluateCtrValue(ctr, features);
	}
	return materialized;
}

double BoostModel::EvalTree(const BoostTree &tree, const vector<double> &features) const {
	idx_t node_idx = 0;
	while (true) {
		if (node_idx >= tree.nodes.size()) {
			throw InvalidInputException("duckboost: corrupt tree during prediction");
		}
		auto &node = tree.nodes[node_idx];
		if (node.is_leaf) {
			return node.value;
		}
		if (node.feature >= features.size()) {
			throw InvalidInputException("duckboost: feature index out of range during prediction");
		}
		node_idx = NodeGoesLeft(node, features[node.feature]) ? node.left : node.right;
	}
}

double BoostModel::PredictRaw(const vector<double> &features) const {
	if (task == BoostTask::MULTICLASS) {
		throw InvalidInputException("duckboost: PredictRaw is not defined for multiclass; use Predict / PredictProba");
	}
	auto feats = MaterializeFeatures(features);
	double score = ClassBias(0);
	for (auto &tree : trees) {
		score += learning_rate * EvalTree(tree, feats);
	}
	return score;
}

vector<double> BoostModel::PredictRawMulti(const vector<double> &features) const {
	auto feats = MaterializeFeatures(features);
	if (task != BoostTask::MULTICLASS) {
		double score = ClassBias(0);
		for (auto &tree : trees) {
			score += learning_rate * EvalTree(tree, feats);
		}
		return {score};
	}
	if (n_classes < 2) {
		throw InvalidInputException("duckboost: multiclass model requires n_classes >= 2");
	}
	if (trees.size() % n_classes != 0) {
		throw InvalidInputException("duckboost: multiclass tree count %llu is not divisible by n_classes %llu",
		                            (unsigned long long)trees.size(), (unsigned long long)n_classes);
	}
	vector<double> scores(n_classes);
	for (idx_t c = 0; c < n_classes; c++) {
		scores[c] = ClassBias(c);
	}
	const idx_t n_rounds = trees.size() / n_classes;
	for (idx_t round = 0; round < n_rounds; round++) {
		for (idx_t c = 0; c < n_classes; c++) {
			scores[c] += learning_rate * EvalTree(trees[round * n_classes + c], feats);
		}
	}
	return scores;
}

double BoostModel::Predict(const vector<double> &features) const {
	if (task == BoostTask::MULTICLASS) {
		auto scores = PredictRawMulti(features);
		idx_t best = 0;
		for (idx_t c = 1; c < scores.size(); c++) {
			if (scores[c] > scores[best]) {
				best = c;
			}
		}
		return static_cast<double>(best);
	}
	auto raw = PredictRaw(features);
	if (task == BoostTask::BINARY) {
		return Sigmoid(raw);
	}
	return raw;
}

vector<double> BoostModel::PredictProba(const vector<double> &features) const {
	if (task == BoostTask::REGRESSION) {
		return {Predict(features)};
	}
	if (task == BoostTask::BINARY) {
		auto p = Predict(features);
		return {1.0 - p, p};
	}
	auto scores = PredictRawMulti(features);
	double max_score = scores[0];
	for (idx_t i = 1; i < scores.size(); i++) {
		max_score = MaxValue(max_score, scores[i]);
	}
	vector<double> proba(scores.size());
	double sum = 0;
	for (idx_t i = 0; i < scores.size(); i++) {
		proba[i] = std::exp(scores[i] - max_score);
		sum += proba[i];
	}
	if (sum > 0) {
		for (auto &p : proba) {
			p /= sum;
		}
	}
	return proba;
}

string ExportModelSQL(const BoostModel &model, const string &table_name, const vector<string> &feature_columns,
                      const SqlExportOptions &options) {
	if (table_name.empty()) {
		throw InvalidInputException("duckboost: table_name must not be empty");
	}
	vector<string> raw_columns = feature_columns;
	if (raw_columns.empty()) {
		raw_columns = model.feature_names;
	}
	const idx_t required_raw = model.n_raw_features > 0 ? model.n_raw_features : model.n_features;
	if (raw_columns.size() < required_raw) {
		throw InvalidInputException("duckboost: need %llu feature column names for SQL export",
		                            (unsigned long long)required_raw);
	}

	vector<string> columns = raw_columns;
	if (columns.size() < model.n_features) {
		columns.resize(model.n_features);
	}
	for (auto &ctr : model.ctr_features) {
		if (ctr.feature_index >= columns.size()) {
			columns.resize(ctr.feature_index + 1);
		}
		if (columns[ctr.feature_index].empty()) {
			columns[ctr.feature_index] = "_duckboost_ctr_" + std::to_string(ctr.feature_index);
		}
	}

	string from_sql = QuoteIdent(table_name);
	if (!model.ctr_features.empty()) {
		std::ostringstream src;
		src << "(SELECT *";
		for (auto &ctr : model.ctr_features) {
			src << ", (" << SqlCtrValueExpr(ctr, raw_columns) << ") AS " << QuoteIdent(columns[ctr.feature_index]);
		}
		src << " FROM " << QuoteIdent(table_name) << ") AS " << QuoteIdent("_duckboost_src");
		from_sql = src.str();
	}

	auto alias = QuoteIdent(options.prediction_alias);

	if (model.task == BoostTask::MULTICLASS) {
		if (model.n_classes < 2 || model.trees.size() % model.n_classes != 0) {
			throw InvalidInputException("duckboost: invalid multiclass model for SQL export");
		}
		const idx_t n_rounds = model.trees.size() / model.n_classes;
		std::ostringstream inner;
		inner << "SELECT ";
		for (idx_t c = 0; c < model.n_classes; c++) {
			if (c > 0) {
				inner << ", ";
			}
			string expr = FormatDouble(model.ClassBias(c));
			for (idx_t round = 0; round < n_rounds; round++) {
				expr += " + " + FormatDouble(model.learning_rate) + " * (" +
				        TreeToSQL(model.trees[round * model.n_classes + c], columns, 0) + ")";
			}
			inner << "(" << expr << ") AS " << QuoteIdent("score_" + std::to_string(c));
		}
		inner << " FROM " << from_sql;

		// Argmax over class scores (ties → lowest class index).
		string argmax = "0";
		for (idx_t c = 1; c < model.n_classes; c++) {
			string cond;
			for (idx_t prev = 0; prev < c; prev++) {
				if (prev > 0) {
					cond += " AND ";
				}
				cond += QuoteIdent("score_" + std::to_string(c)) + " > " + QuoteIdent("score_" + std::to_string(prev));
			}
			argmax = "CASE WHEN " + cond + " THEN " + std::to_string(c) + " ELSE " + argmax + " END";
		}
		return "SELECT (" + argmax + ") AS " + alias + " FROM (" + inner.str() + ") AS " +
		       QuoteIdent("_duckboost_scores");
	}

	if (!options.separate_trees) {
		string expr = FormatDouble(model.ClassBias(0));
		for (auto &tree : model.trees) {
			expr += " + " + FormatDouble(model.learning_rate) + " * (" + TreeToSQL(tree, columns, 0) + ")";
		}
		if (model.task == BoostTask::BINARY) {
			expr = "1.0 / (1.0 + EXP(-(" + expr + ")))";
		}
		return "SELECT (" + expr + ") AS " + alias + " FROM " + from_sql;
	}

	std::ostringstream inner;
	inner << "SELECT ";
	for (idx_t t = 0; t < model.trees.size(); t++) {
		if (t > 0) {
			inner << ", ";
		}
		inner << "(" << TreeToSQL(model.trees[t], columns, 0) << ") AS " << QuoteIdent("tree_" + std::to_string(t));
	}
	if (model.trees.empty()) {
		inner << FormatDouble(model.ClassBias(0)) << " AS " << QuoteIdent("base_score");
	} else {
		inner << ", " << FormatDouble(model.ClassBias(0)) << " AS " << QuoteIdent("base_score");
	}
	inner << " FROM " << from_sql;

	string sum_expr = QuoteIdent("base_score");
	for (idx_t t = 0; t < model.trees.size(); t++) {
		sum_expr += " + " + FormatDouble(model.learning_rate) + " * " + QuoteIdent("tree_" + std::to_string(t));
	}
	if (model.task == BoostTask::BINARY) {
		sum_expr = "1.0 / (1.0 + EXP(-(" + sum_expr + ")))";
	}
	return "SELECT (" + sum_expr + ") AS " + alias + " FROM (" + inner.str() + ") AS " + QuoteIdent("_duckboost_trees");
}

} // namespace duckboost
} // namespace duckdb
