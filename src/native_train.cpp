#include "duckboost/native_train.hpp"
#include "duckboost/import.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/string_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cctype>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#if defined(DUCKBOOST_WITH_XGBOOST) && !defined(DUCKBOOST_NATIVE_STUB)
extern "C" {
typedef void *DMatrixHandle;
typedef void *BoosterHandle;
typedef uint64_t bst_ulong;

const char *XGBGetLastError();
int XGDMatrixCreateFromMat(const float *data, bst_ulong nrow, bst_ulong ncol, float missing, DMatrixHandle *out);
int XGDMatrixSetFloatInfo(DMatrixHandle handle, const char *field, const float *array, bst_ulong len);
int XGDMatrixFree(DMatrixHandle handle);
int XGBoosterCreate(const DMatrixHandle dmats[], bst_ulong len, BoosterHandle *out);
int XGBoosterFree(BoosterHandle handle);
int XGBoosterSetParam(BoosterHandle handle, const char *name, const char *value);
int XGBoosterUpdateOneIter(BoosterHandle handle, int iter, DMatrixHandle dtrain);
int XGBoosterEvalOneIter(BoosterHandle handle, int iter, const DMatrixHandle dmats[], const char *evnames[],
                         bst_ulong len, const char **out_result);
int XGBoosterDumpModelEx(BoosterHandle handle, const char *fmap, int with_stats, const char *format, bst_ulong *out_len,
                         const char ***out_dump_array);
int XGBoosterDumpModelExWithFeatures(BoosterHandle handle, int fnum, const char **fname, const char **ftype,
                                     int with_stats, const char *format, bst_ulong *out_len, const char ***out_models);
int XGBoosterSaveJsonConfig(BoosterHandle handle, bst_ulong *out_len, const char **out_str);
}
#endif

#if defined(DUCKBOOST_WITH_LIGHTGBM) && !defined(DUCKBOOST_NATIVE_STUB)
extern "C" {
typedef void *DatasetHandle;
typedef void *BoosterHandle;

const char *LGBM_GetLastError();
int LGBM_DatasetCreateFromMat(const void *data, int data_type, int32_t nrow, int32_t ncol, int is_row_major,
                              const char *parameters, const DatasetHandle reference, DatasetHandle *out);
int LGBM_DatasetSetField(DatasetHandle handle, const char *field_name, const void *field_data, int num_element,
                         int type);
int LGBM_DatasetSetFeatureNames(DatasetHandle handle, const char **feature_names, int num_feature_names);
int LGBM_DatasetFree(DatasetHandle handle);
int LGBM_BoosterCreate(const DatasetHandle train_data, const char *parameters, BoosterHandle *out);
int LGBM_BoosterAddValidData(BoosterHandle handle, const DatasetHandle valid_data);
int LGBM_BoosterUpdateOneIter(BoosterHandle handle, int *is_finished);
int LGBM_BoosterSaveModelToString(BoosterHandle handle, int start_iteration, int num_iteration,
                                  int feature_importance_type, int64_t buffer_len, int64_t *out_len, char *out_str);
int LGBM_BoosterFree(BoosterHandle handle);
}
#ifndef C_API_DTYPE_FLOAT64
#define C_API_DTYPE_FLOAT64 1
#endif
#ifndef C_API_DTYPE_FLOAT32
#define C_API_DTYPE_FLOAT32 0
#endif
#endif

namespace duckdb {
namespace duckboost {

namespace {

void EnsureRectangular(const vector<double> &y, const vector<vector<double>> &x) {
	if (y.empty()) {
		throw InvalidInputException("duckboost: cannot train native model on empty dataset");
	}
	if (y.size() != x.size()) {
		throw InvalidInputException("duckboost: y/x row count mismatch during native train");
	}
	const idx_t n_features = x[0].size();
	if (n_features == 0) {
		throw InvalidInputException("duckboost: native train requires at least one feature");
	}
	for (idx_t i = 0; i < x.size(); i++) {
		if (x[i].size() != n_features) {
			throw InvalidInputException("duckboost: jagged feature matrix at row %llu", (unsigned long long)i);
		}
	}
}

ImportOptions ImportOptionsFromTrain(const TrainOptions &options) {
	ImportOptions import_options;
	import_options.task = options.task;
	import_options.task_set = true;
	import_options.feature_names = options.feature_names;
	return import_options;
}

struct RowSplit {
	vector<idx_t> train_rows;
	vector<idx_t> valid_rows;
};

RowSplit SplitTrainValidRows(idx_t n_rows, const TrainOptions &options, const vector<bool> &is_validation) {
	RowSplit split;
	ResolveTrainValidRows(n_rows, options, is_validation, split.train_rows, split.valid_rows);
	return split;
}

double ParseEvalMetric(const char *eval_result) {
	if (!eval_result) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	const char *colon = std::strrchr(eval_result, ':');
	if (!colon || !*(colon + 1)) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	return std::strtod(colon + 1, nullptr);
}

string ObjectiveForXGBoost(BoostTask task, idx_t n_classes) {
	switch (task) {
	case BoostTask::BINARY:
		return "binary:logistic";
	case BoostTask::MULTICLASS:
		if (n_classes < 2) {
			throw InvalidInputException("duckboost: xgboost multiclass train requires n_classes >= 2");
		}
		return "multi:softprob";
	case BoostTask::REGRESSION:
	default:
		return "reg:squarederror";
	}
}

string ObjectiveForLightGBM(const TrainOptions &options, idx_t n_classes) {
	if (options.task == BoostTask::BINARY) {
		return "binary";
	}
	if (options.task == BoostTask::MULTICLASS) {
		if (n_classes < 2) {
			throw InvalidInputException("duckboost: lightgbm multiclass train requires n_classes >= 2");
		}
		return "multiclass";
	}
	switch (options.loss) {
	case RegressionLoss::ABSOLUTE_ERROR:
		return "regression_l1";
	case RegressionLoss::QUANTILE:
		return "quantile";
	case RegressionLoss::EXPECTILE:
		throw NotImplementedException("duckboost: expectile is supported by the reference backend only");
	case RegressionLoss::SQUARED_ERROR:
	default:
		return "regression";
	}
}

idx_t InferClassCount(const vector<double> &y, const TrainOptions &options) {
	if (options.task != BoostTask::MULTICLASS) {
		return 1;
	}
	return ResolveClassCount(y, options);
}

#if defined(DUCKBOOST_WITH_XGBOOST) && !defined(DUCKBOOST_NATIVE_STUB)

[[noreturn]] void ThrowXGBoostError(const char *context) {
	const char *err = XGBGetLastError();
	throw InvalidInputException("duckboost: xgboost %s failed: %s", context, err ? err : "unknown error");
}

void PackXGBoostRows(const vector<double> &y, const vector<vector<double>> &x, const vector<double> &weights,
                     const vector<idx_t> &rows, vector<float> &flat, vector<float> &labels, vector<float> &weight_f) {
	const idx_t ncol = x[0].size();
	flat.resize(rows.size() * ncol);
	labels.resize(rows.size());
	weight_f.clear();
	if (!weights.empty()) {
		weight_f.resize(rows.size());
	}
	for (idx_t i = 0; i < rows.size(); i++) {
		const idx_t row = rows[i];
		labels[i] = static_cast<float>(y[row]);
		if (!weight_f.empty()) {
			weight_f[i] = static_cast<float>(weights[row]);
		}
		for (idx_t j = 0; j < ncol; j++) {
			flat[i * ncol + j] = static_cast<float>(x[row][j]);
		}
	}
}

DMatrixHandle MakeXGBoostDMatrix(const vector<float> &flat, const vector<float> &labels, const vector<float> &weight_f,
                                 idx_t ncol) {
	DMatrixHandle dmat = nullptr;
	const idx_t nrow = labels.size();
	if (XGDMatrixCreateFromMat(flat.data(), static_cast<bst_ulong>(nrow), static_cast<bst_ulong>(ncol),
	                           std::numeric_limits<float>::quiet_NaN(), &dmat) != 0) {
		ThrowXGBoostError("XGDMatrixCreateFromMat");
	}
	if (XGDMatrixSetFloatInfo(dmat, "label", labels.data(), static_cast<bst_ulong>(nrow)) != 0) {
		XGDMatrixFree(dmat);
		ThrowXGBoostError("XGDMatrixSetFloatInfo(label)");
	}
	if (!weight_f.empty()) {
		if (XGDMatrixSetFloatInfo(dmat, "weight", weight_f.data(), static_cast<bst_ulong>(nrow)) != 0) {
			XGDMatrixFree(dmat);
			ThrowXGBoostError("XGDMatrixSetFloatInfo(weight)");
		}
	}
	return dmat;
}

BoostModel TrainWithXGBoost(const vector<double> &y, const vector<vector<double>> &x, const TrainOptions &options,
                            const vector<double> &weights, const vector<bool> &is_validation) {
	EnsureRectangular(y, x);
	const idx_t nrow = y.size();
	const idx_t ncol = x[0].size();
	const idx_t n_classes = InferClassCount(y, options);
	if (!weights.empty() && weights.size() != nrow) {
		throw InvalidInputException("duckboost: sample weight count must match row count for xgboost train");
	}

	const auto split = SplitTrainValidRows(nrow, options, is_validation);
	vector<float> train_flat, train_labels, train_weights;
	PackXGBoostRows(y, x, weights, split.train_rows, train_flat, train_labels, train_weights);
	DMatrixHandle dtrain = MakeXGBoostDMatrix(train_flat, train_labels, train_weights, ncol);
	DMatrixHandle dvalid = nullptr;
	vector<float> valid_flat, valid_labels, valid_weights;
	if (!split.valid_rows.empty()) {
		PackXGBoostRows(y, x, weights, split.valid_rows, valid_flat, valid_labels, valid_weights);
		dvalid = MakeXGBoostDMatrix(valid_flat, valid_labels, valid_weights, ncol);
	}

	BoosterHandle booster = nullptr;
	const DMatrixHandle dmats[] = {dtrain};
	if (XGBoosterCreate(dmats, 1, &booster) != 0) {
		XGDMatrixFree(dtrain);
		if (dvalid) {
			XGDMatrixFree(dvalid);
		}
		ThrowXGBoostError("XGBoosterCreate");
	}

	auto cleanup = [&]() {
		XGBoosterFree(booster);
		XGDMatrixFree(dtrain);
		if (dvalid) {
			XGDMatrixFree(dvalid);
		}
	};

	auto set_param = [&](const char *name, const string &value) {
		if (XGBoosterSetParam(booster, name, value.c_str()) != 0) {
			cleanup();
			ThrowXGBoostError((string("XGBoosterSetParam(") + name + ")").c_str());
		}
	};

	set_param("verbosity", "0");
	set_param("max_depth", std::to_string(options.max_depth));
	set_param("eta", std::to_string(options.learning_rate));
	set_param("min_child_weight",
	          std::to_string(options.min_child_weight > 0 ? options.min_child_weight : options.min_samples_leaf));
	set_param("lambda", std::to_string(options.reg_lambda));
	set_param("alpha", std::to_string(options.reg_alpha));
	set_param("gamma", std::to_string(options.min_split_gain));
	set_param("subsample", std::to_string(options.subsample));
	set_param("colsample_bytree", std::to_string(options.colsample_bytree));
	set_param("objective", ObjectiveForXGBoost(options.task, n_classes));
	set_param("seed", std::to_string(options.seed));
	set_param("max_bin", std::to_string(MaxValue<idx_t>(options.max_bins, 2)));
	if (options.task == BoostTask::MULTICLASS) {
		set_param("num_class", std::to_string(n_classes));
		set_param("eval_metric", "mlogloss");
	} else if (options.task == BoostTask::BINARY) {
		set_param("eval_metric", "logloss");
	} else {
		set_param("eval_metric", "rmse");
	}

	idx_t best_rounds = 0;
	idx_t rounds_since_improve = 0;
	double best_valid = std::numeric_limits<double>::infinity();
	idx_t trained_rounds = 0;
	for (idx_t iter = 0; iter < options.n_estimators; iter++) {
		if (XGBoosterUpdateOneIter(booster, static_cast<int>(iter), dtrain) != 0) {
			cleanup();
			ThrowXGBoostError("XGBoosterUpdateOneIter");
		}
		trained_rounds = iter + 1;
		if (!dvalid || options.early_stopping_rounds == 0) {
			continue;
		}
		const DMatrixHandle eval_mats[] = {dvalid};
		const char *eval_names[] = {"valid"};
		const char *eval_result = nullptr;
		if (XGBoosterEvalOneIter(booster, static_cast<int>(iter), eval_mats, eval_names, 1, &eval_result) != 0) {
			cleanup();
			ThrowXGBoostError("XGBoosterEvalOneIter");
		}
		const double metric = ParseEvalMetric(eval_result);
		if (std::isfinite(metric) && metric < best_valid - 1e-12) {
			best_valid = metric;
			best_rounds = trained_rounds;
			rounds_since_improve = 0;
		} else {
			rounds_since_improve++;
			if (rounds_since_improve >= options.early_stopping_rounds) {
				break;
			}
		}
	}

	bst_ulong out_len = 0;
	const char **out_dump = nullptr;
	// Always dump with f0..fN names so feature indices stay positional; attach names on import.
	if (XGBoosterDumpModelEx(booster, "", 0, "json", &out_len, &out_dump) != 0 || !out_dump) {
		cleanup();
		ThrowXGBoostError("XGBoosterDumpModelEx");
	}

	const idx_t trees_per_round = options.task == BoostTask::MULTICLASS ? n_classes : 1;
	idx_t keep = out_len;
	if (dvalid && options.early_stopping_rounds > 0 && best_rounds > 0) {
		keep = MinValue<idx_t>(out_len, best_rounds * trees_per_round);
	}

	string dump = "[";
	for (bst_ulong i = 0; i < keep; i++) {
		if (i > 0) {
			dump += ",";
		}
		dump += out_dump[i] ? out_dump[i] : "{}";
	}
	dump += "]";

	// The dump omits the learned intercept (boost_from_average) and objective; the config carries both.
	bst_ulong config_len = 0;
	const char *config_str = nullptr;
	if (XGBoosterSaveJsonConfig(booster, &config_len, &config_str) != 0 || !config_str) {
		cleanup();
		ThrowXGBoostError("XGBoosterSaveJsonConfig");
	}
	string config(config_str, static_cast<size_t>(config_len));

	cleanup();

	auto import_options = ImportOptionsFromTrain(options);
	import_options.config = std::move(config);
	if (options.task == BoostTask::MULTICLASS) {
		import_options.n_classes = n_classes;
		import_options.n_classes_set = true;
	}
	return ImportXGBoostJSON(dump, import_options);
}

#endif // XGBoost linked

#if defined(DUCKBOOST_WITH_LIGHTGBM) && !defined(DUCKBOOST_NATIVE_STUB)

[[noreturn]] void ThrowLightGBMError(const char *context) {
	const char *err = LGBM_GetLastError();
	throw InvalidInputException("duckboost: lightgbm %s failed: %s", context, err ? err : "unknown error");
}

string LightGBMDatasetParams(const TrainOptions &options, int32_t ncol) {
	string dataset_params = "max_bin=" + std::to_string(MaxValue<idx_t>(options.max_bins, 2));
	if (options.categorical_features.empty()) {
		return dataset_params;
	}
	dataset_params += " categorical_feature=";
	for (idx_t i = 0; i < options.categorical_features.size(); i++) {
		if (i > 0) {
			dataset_params += ",";
		}
		auto token = options.categorical_features[i];
		bool all_digits = !token.empty();
		for (char c : token) {
			if (!std::isdigit(static_cast<unsigned char>(c))) {
				all_digits = false;
				break;
			}
		}
		if (all_digits) {
			dataset_params += token;
			continue;
		}
		idx_t found = static_cast<idx_t>(ncol);
		for (idx_t f = 0; f < options.feature_names.size(); f++) {
			if (options.feature_names[f] == token) {
				found = f;
				break;
			}
		}
		if (found >= static_cast<idx_t>(ncol)) {
			throw InvalidInputException("duckboost: unknown categorical feature '%s'", token);
		}
		dataset_params += std::to_string(found);
	}
	return dataset_params;
}

void PackLightGBMRows(const vector<double> &y, const vector<vector<double>> &x, const vector<double> &weights,
                      const vector<idx_t> &rows, vector<double> &flat, vector<float> &labels, vector<float> &weight_f) {
	const idx_t ncol = x[0].size();
	flat.resize(rows.size() * ncol);
	labels.resize(rows.size());
	weight_f.clear();
	if (!weights.empty()) {
		weight_f.resize(rows.size());
	}
	for (idx_t i = 0; i < rows.size(); i++) {
		const idx_t row = rows[i];
		labels[i] = static_cast<float>(y[row]);
		if (!weight_f.empty()) {
			weight_f[i] = static_cast<float>(weights[row]);
		}
		for (idx_t j = 0; j < ncol; j++) {
			flat[i * ncol + j] = x[row][j];
		}
	}
}

DatasetHandle MakeLightGBMDataset(const vector<double> &flat, const vector<float> &labels,
                                  const vector<float> &weight_f, int32_t ncol, const string &dataset_params,
                                  DatasetHandle reference, const TrainOptions &options) {
	DatasetHandle dataset = nullptr;
	const auto nrow = NumericCast<int32_t>(labels.size());
	if (LGBM_DatasetCreateFromMat(flat.data(), C_API_DTYPE_FLOAT64, nrow, ncol, 1, dataset_params.c_str(), reference,
	                              &dataset) != 0) {
		ThrowLightGBMError("LGBM_DatasetCreateFromMat");
	}
	if (LGBM_DatasetSetField(dataset, "label", labels.data(), nrow, C_API_DTYPE_FLOAT32) != 0) {
		LGBM_DatasetFree(dataset);
		ThrowLightGBMError("LGBM_DatasetSetField(label)");
	}
	if (!weight_f.empty()) {
		if (LGBM_DatasetSetField(dataset, "weight", weight_f.data(), nrow, C_API_DTYPE_FLOAT32) != 0) {
			LGBM_DatasetFree(dataset);
			ThrowLightGBMError("LGBM_DatasetSetField(weight)");
		}
	}
	if (!options.feature_names.empty() && options.feature_names.size() == static_cast<idx_t>(ncol)) {
		vector<const char *> fnames(static_cast<idx_t>(ncol));
		for (int32_t i = 0; i < ncol; i++) {
			fnames[static_cast<idx_t>(i)] = options.feature_names[static_cast<idx_t>(i)].c_str();
		}
		if (LGBM_DatasetSetFeatureNames(dataset, fnames.data(), ncol) != 0) {
			LGBM_DatasetFree(dataset);
			ThrowLightGBMError("LGBM_DatasetSetFeatureNames");
		}
	}
	return dataset;
}

BoostModel TrainWithLightGBM(const vector<double> &y, const vector<vector<double>> &x, const TrainOptions &options,
                             const vector<double> &weights, const vector<bool> &is_validation) {
	EnsureRectangular(y, x);
	const auto ncol = NumericCast<int32_t>(x[0].size());
	const idx_t n_classes = InferClassCount(y, options);
	if (!weights.empty() && weights.size() != y.size()) {
		throw InvalidInputException("duckboost: sample weight count must match row count for lightgbm train");
	}

	const auto split = SplitTrainValidRows(y.size(), options, is_validation);
	const string dataset_params = LightGBMDatasetParams(options, ncol);
	vector<double> train_flat;
	vector<float> train_labels, train_weights;
	PackLightGBMRows(y, x, weights, split.train_rows, train_flat, train_labels, train_weights);
	DatasetHandle train_dataset =
	    MakeLightGBMDataset(train_flat, train_labels, train_weights, ncol, dataset_params, nullptr, options);
	DatasetHandle valid_dataset = nullptr;
	vector<double> valid_flat;
	vector<float> valid_labels, valid_weights;
	if (!split.valid_rows.empty()) {
		PackLightGBMRows(y, x, weights, split.valid_rows, valid_flat, valid_labels, valid_weights);
		valid_dataset =
		    MakeLightGBMDataset(valid_flat, valid_labels, valid_weights, ncol, dataset_params, train_dataset, options);
	}

	idx_t num_leaves = MaxValue<idx_t>(2, 1ULL << MinValue<idx_t>(options.max_depth == 0 ? 10 : options.max_depth, 10));
	int64_t max_depth = static_cast<int64_t>(options.max_depth);
	if (options.growth_policy == GrowthPolicy::LOSSGUIDE || options.max_leaves_set) {
		num_leaves = MaxValue<idx_t>(options.max_leaves, 2);
		if (!options.max_depth_set || options.max_depth == 0) {
			max_depth = -1;
		}
	}
	string params = StringUtil::Format(
	    "objective=%s learning_rate=%g num_leaves=%llu max_depth=%lld min_data_in_leaf=%llu "
	    "lambda_l2=%g lambda_l1=%g min_gain_to_split=%g bagging_fraction=%g feature_fraction=%g "
	    "verbosity=-1 force_col_wise=true seed=%llu deterministic=true",
	    ObjectiveForLightGBM(options, n_classes), options.learning_rate, (unsigned long long)num_leaves,
	    (long long)max_depth, (unsigned long long)MaxValue<idx_t>(options.min_samples_leaf, 1), options.reg_lambda,
	    options.reg_alpha, options.min_split_gain, options.subsample, options.colsample_bytree,
	    (unsigned long long)options.seed);
	if (options.task == BoostTask::MULTICLASS) {
		params += " num_class=" + std::to_string(n_classes);
		params += " metric=multi_logloss";
	} else if (options.task == BoostTask::BINARY) {
		params += " metric=binary_logloss";
	} else if (options.loss == RegressionLoss::QUANTILE) {
		params += " metric=quantile";
	} else if (options.loss == RegressionLoss::ABSOLUTE_ERROR) {
		params += " metric=l1";
	} else {
		params += " metric=l2";
	}
	if (options.loss == RegressionLoss::QUANTILE) {
		params += " alpha=" + std::to_string(options.objective_alpha);
	}
	if (options.min_child_weight > 0) {
		params += " min_sum_hessian_in_leaf=" + std::to_string(options.min_child_weight);
	}
	if (options.subsample < 1.0) {
		params += " bagging_freq=1";
	}
	if (valid_dataset && options.early_stopping_rounds > 0) {
		params += " early_stopping_round=" + std::to_string(options.early_stopping_rounds);
	}

	BoosterHandle booster = nullptr;
	if (LGBM_BoosterCreate(train_dataset, params.c_str(), &booster) != 0) {
		LGBM_DatasetFree(train_dataset);
		if (valid_dataset) {
			LGBM_DatasetFree(valid_dataset);
		}
		ThrowLightGBMError("LGBM_BoosterCreate");
	}
	if (valid_dataset) {
		if (LGBM_BoosterAddValidData(booster, valid_dataset) != 0) {
			LGBM_BoosterFree(booster);
			LGBM_DatasetFree(train_dataset);
			LGBM_DatasetFree(valid_dataset);
			ThrowLightGBMError("LGBM_BoosterAddValidData");
		}
	}

	for (idx_t iter = 0; iter < options.n_estimators; iter++) {
		int is_finished = 0;
		if (LGBM_BoosterUpdateOneIter(booster, &is_finished) != 0) {
			LGBM_BoosterFree(booster);
			LGBM_DatasetFree(train_dataset);
			if (valid_dataset) {
				LGBM_DatasetFree(valid_dataset);
			}
			ThrowLightGBMError("LGBM_BoosterUpdateOneIter");
		}
		if (is_finished) {
			break;
		}
	}

	int64_t out_len = 0;
	if (LGBM_BoosterSaveModelToString(booster, 0, -1, 0, 0, &out_len, nullptr) != 0) {
		LGBM_BoosterFree(booster);
		LGBM_DatasetFree(train_dataset);
		if (valid_dataset) {
			LGBM_DatasetFree(valid_dataset);
		}
		ThrowLightGBMError("LGBM_BoosterSaveModelToString(size)");
	}
	vector<char> buffer(static_cast<idx_t>(out_len) + 1);
	if (LGBM_BoosterSaveModelToString(booster, 0, -1, 0, out_len, &out_len, buffer.data()) != 0) {
		LGBM_BoosterFree(booster);
		LGBM_DatasetFree(train_dataset);
		if (valid_dataset) {
			LGBM_DatasetFree(valid_dataset);
		}
		ThrowLightGBMError("LGBM_BoosterSaveModelToString");
	}

	LGBM_BoosterFree(booster);
	LGBM_DatasetFree(train_dataset);
	if (valid_dataset) {
		LGBM_DatasetFree(valid_dataset);
	}

	string dump(buffer.data());
	auto import_options = ImportOptionsFromTrain(options);
	if (options.task == BoostTask::MULTICLASS) {
		import_options.n_classes = n_classes;
		import_options.n_classes_set = true;
	}
	return ImportLightGBMText(dump, import_options);
}

#endif // LightGBM linked

} // namespace

bool NativeTrainerCompiled(BoostBackend backend) {
	switch (backend) {
	case BoostBackend::XGBOOST:
#if defined(DUCKBOOST_WITH_XGBOOST)
		return true;
#else
		return false;
#endif
	case BoostBackend::LIGHTGBM:
#if defined(DUCKBOOST_WITH_LIGHTGBM)
		return true;
#else
		return false;
#endif
	case BoostBackend::CATBOOST:
#if defined(DUCKBOOST_WITH_CATBOOST)
		return true;
#else
		return false;
#endif
	case BoostBackend::REFERENCE:
	default:
		return false;
	}
}

bool NativeTrainerLinked(BoostBackend backend) {
#if defined(DUCKBOOST_NATIVE_STUB)
	(void)backend;
	return false;
#else
	switch (backend) {
	case BoostBackend::XGBOOST:
#if defined(DUCKBOOST_WITH_XGBOOST)
		return true;
#else
		return false;
#endif
	case BoostBackend::LIGHTGBM:
#if defined(DUCKBOOST_WITH_LIGHTGBM)
		return true;
#else
		return false;
#endif
	case BoostBackend::CATBOOST:
		// CatBoost ships a model-application C API, not a public in-process training C API.
		return false;
	case BoostBackend::REFERENCE:
	default:
		return false;
	}
#endif
}

BoostModel TrainNative(const vector<double> &y, const vector<vector<double>> &x, const TrainOptions &options,
                       const vector<double> &weights, const vector<bool> &is_validation) {
	if (!options.categorical_features.empty() && options.backend != BoostBackend::LIGHTGBM) {
		throw NotImplementedException("duckboost: categorical_features requires backend='lightgbm' or 'reference'");
	}
	if (!NativeTrainerCompiled(options.backend)) {
		throw NotImplementedException("duckboost: native training for backend '%s' is not linked in this build. "
		                              "Configure with -DDUCKBOOST_WITH_%s=ON (and install the vendor library), "
		                              "or use backend='reference' / duckboost_import().",
		                              BackendToString(options.backend),
		                              StringUtil::Upper(BackendToString(options.backend)));
	}

#if defined(DUCKBOOST_NATIVE_STUB)
	throw NotImplementedException("duckboost: native trainer for backend '%s' is compiled as a stub "
	                              "(DUCKBOOST_NATIVE_STUB_ONLY). Rebuild with the vendor library linked, "
	                              "or use duckboost_import() / backend='reference'.",
	                              BackendToString(options.backend));
#endif

	switch (options.backend) {
	case BoostBackend::XGBOOST:
#if defined(DUCKBOOST_WITH_XGBOOST) && !defined(DUCKBOOST_NATIVE_STUB)
		return TrainWithXGBoost(y, x, options, weights, is_validation);
#else
		break;
#endif
	case BoostBackend::LIGHTGBM:
#if defined(DUCKBOOST_WITH_LIGHTGBM) && !defined(DUCKBOOST_NATIVE_STUB)
		return TrainWithLightGBM(y, x, options, weights, is_validation);
#else
		break;
#endif
	case BoostBackend::CATBOOST:
		throw NotImplementedException(
		    "duckboost: CatBoost has no public in-process training C API. "
		    "Train with the CatBoost CLI/Python API and load the JSON dump via duckboost_import('catboost', ...).");
	default:
		break;
	}

	throw NotImplementedException("duckboost: native trainer for backend '%s' is not available in this build. "
	                              "Use duckboost_import() or backend='reference'.",
	                              BackendToString(options.backend));
}

} // namespace duckboost
} // namespace duckdb
