#include "duckboost/native_train.hpp"
#include "duckboost/import.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/string_util.hpp"

#include <cmath>
#include <cstring>
#include <limits>
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
int XGBoosterDumpModelEx(BoosterHandle handle, const char *fmap, int with_stats, const char *format, bst_ulong *out_len,
                         const char ***out_dump_array);
int XGBoosterDumpModelExWithFeatures(BoosterHandle handle, int fnum, const char **fname, const char **ftype,
                                     int with_stats, const char *format, bst_ulong *out_len, const char ***out_models);
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

string ObjectiveForLightGBM(BoostTask task, idx_t n_classes) {
	switch (task) {
	case BoostTask::BINARY:
		return "binary";
	case BoostTask::MULTICLASS:
		if (n_classes < 2) {
			throw InvalidInputException("duckboost: lightgbm multiclass train requires n_classes >= 2");
		}
		return "multiclass";
	case BoostTask::REGRESSION:
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

BoostModel TrainWithXGBoost(const vector<double> &y, const vector<vector<double>> &x, const TrainOptions &options,
                            const vector<double> &weights) {
	EnsureRectangular(y, x);
	const idx_t nrow = y.size();
	const idx_t ncol = x[0].size();
	const idx_t n_classes = InferClassCount(y, options);

	vector<float> flat(nrow * ncol);
	vector<float> labels(nrow);
	for (idx_t i = 0; i < nrow; i++) {
		labels[i] = static_cast<float>(y[i]);
		for (idx_t j = 0; j < ncol; j++) {
			flat[i * ncol + j] = static_cast<float>(x[i][j]);
		}
	}

	DMatrixHandle dmat = nullptr;
	if (XGDMatrixCreateFromMat(flat.data(), static_cast<bst_ulong>(nrow), static_cast<bst_ulong>(ncol),
	                           std::numeric_limits<float>::quiet_NaN(), &dmat) != 0) {
		ThrowXGBoostError("XGDMatrixCreateFromMat");
	}
	if (XGDMatrixSetFloatInfo(dmat, "label", labels.data(), static_cast<bst_ulong>(nrow)) != 0) {
		XGDMatrixFree(dmat);
		ThrowXGBoostError("XGDMatrixSetFloatInfo(label)");
	}
	if (!weights.empty()) {
		if (weights.size() != nrow) {
			XGDMatrixFree(dmat);
			throw InvalidInputException("duckboost: sample weight count must match row count for xgboost train");
		}
		vector<float> weight_f(nrow);
		for (idx_t i = 0; i < nrow; i++) {
			weight_f[i] = static_cast<float>(weights[i]);
		}
		if (XGDMatrixSetFloatInfo(dmat, "weight", weight_f.data(), static_cast<bst_ulong>(nrow)) != 0) {
			XGDMatrixFree(dmat);
			ThrowXGBoostError("XGDMatrixSetFloatInfo(weight)");
		}
	}

	BoosterHandle booster = nullptr;
	const DMatrixHandle dmats[] = {dmat};
	if (XGBoosterCreate(dmats, 1, &booster) != 0) {
		XGDMatrixFree(dmat);
		ThrowXGBoostError("XGBoosterCreate");
	}

	auto set_param = [&](const char *name, const string &value) {
		if (XGBoosterSetParam(booster, name, value.c_str()) != 0) {
			XGBoosterFree(booster);
			XGDMatrixFree(dmat);
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
	if (options.task == BoostTask::MULTICLASS) {
		set_param("num_class", std::to_string(n_classes));
	}

	for (idx_t iter = 0; iter < options.n_estimators; iter++) {
		if (XGBoosterUpdateOneIter(booster, static_cast<int>(iter), dmat) != 0) {
			XGBoosterFree(booster);
			XGDMatrixFree(dmat);
			ThrowXGBoostError("XGBoosterUpdateOneIter");
		}
	}

	bst_ulong out_len = 0;
	const char **out_dump = nullptr;
	int dump_rc = 0;
	if (!options.feature_names.empty() && options.feature_names.size() == ncol) {
		vector<const char *> fnames(ncol);
		vector<const char *> ftypes(ncol, "q");
		for (idx_t i = 0; i < ncol; i++) {
			fnames[i] = options.feature_names[i].c_str();
		}
		dump_rc = XGBoosterDumpModelExWithFeatures(booster, static_cast<int>(ncol), fnames.data(), ftypes.data(), 0,
		                                           "json", &out_len, &out_dump);
	} else {
		dump_rc = XGBoosterDumpModelEx(booster, "", 0, "json", &out_len, &out_dump);
	}
	if (dump_rc != 0 || !out_dump) {
		XGBoosterFree(booster);
		XGDMatrixFree(dmat);
		ThrowXGBoostError("XGBoosterDumpModelEx");
	}

	string dump = "[";
	for (bst_ulong i = 0; i < out_len; i++) {
		if (i > 0) {
			dump += ",";
		}
		dump += out_dump[i] ? out_dump[i] : "{}";
	}
	dump += "]";

	XGBoosterFree(booster);
	XGDMatrixFree(dmat);

	auto import_options = ImportOptionsFromTrain(options);
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

BoostModel TrainWithLightGBM(const vector<double> &y, const vector<vector<double>> &x, const TrainOptions &options,
                             const vector<double> &weights) {
	EnsureRectangular(y, x);
	const auto nrow = NumericCast<int32_t>(y.size());
	const auto ncol = NumericCast<int32_t>(x[0].size());
	const idx_t n_classes = InferClassCount(y, options);

	vector<double> flat(static_cast<idx_t>(nrow) * static_cast<idx_t>(ncol));
	vector<float> labels(static_cast<idx_t>(nrow));
	for (int32_t i = 0; i < nrow; i++) {
		labels[static_cast<idx_t>(i)] = static_cast<float>(y[static_cast<idx_t>(i)]);
		for (int32_t j = 0; j < ncol; j++) {
			flat[static_cast<idx_t>(i) * static_cast<idx_t>(ncol) + static_cast<idx_t>(j)] =
			    x[static_cast<idx_t>(i)][static_cast<idx_t>(j)];
		}
	}

	string dataset_params = "max_bin=" + std::to_string(MaxValue<idx_t>(options.max_bins, 2));

	DatasetHandle dataset = nullptr;
	if (LGBM_DatasetCreateFromMat(flat.data(), C_API_DTYPE_FLOAT64, nrow, ncol, 1, dataset_params.c_str(), nullptr,
	                              &dataset) != 0) {
		ThrowLightGBMError("LGBM_DatasetCreateFromMat");
	}
	if (LGBM_DatasetSetField(dataset, "label", labels.data(), nrow, C_API_DTYPE_FLOAT32) != 0) {
		LGBM_DatasetFree(dataset);
		ThrowLightGBMError("LGBM_DatasetSetField(label)");
	}
	if (!weights.empty()) {
		if (weights.size() != static_cast<idx_t>(nrow)) {
			LGBM_DatasetFree(dataset);
			throw InvalidInputException("duckboost: sample weight count must match row count for lightgbm train");
		}
		vector<float> weight_f(static_cast<idx_t>(nrow));
		for (int32_t i = 0; i < nrow; i++) {
			weight_f[static_cast<idx_t>(i)] = static_cast<float>(weights[static_cast<idx_t>(i)]);
		}
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

	string params = StringUtil::Format(
	    "objective=%s learning_rate=%g num_leaves=%llu max_depth=%llu min_data_in_leaf=%llu "
	    "lambda_l2=%g lambda_l1=%g min_gain_to_split=%g bagging_fraction=%g feature_fraction=%g "
	    "verbosity=-1 force_col_wise=true",
	    ObjectiveForLightGBM(options.task, n_classes), options.learning_rate,
	    (unsigned long long)MaxValue<idx_t>(2, 1ULL << MinValue<idx_t>(options.max_depth, 10)),
	    (unsigned long long)options.max_depth, (unsigned long long)MaxValue<idx_t>(options.min_samples_leaf, 1),
	    options.reg_lambda, options.reg_alpha, options.min_split_gain, options.subsample, options.colsample_bytree);
	if (options.task == BoostTask::MULTICLASS) {
		params += " num_class=" + std::to_string(n_classes);
	}

	BoosterHandle booster = nullptr;
	if (LGBM_BoosterCreate(dataset, params.c_str(), &booster) != 0) {
		LGBM_DatasetFree(dataset);
		ThrowLightGBMError("LGBM_BoosterCreate");
	}

	for (idx_t iter = 0; iter < options.n_estimators; iter++) {
		int is_finished = 0;
		if (LGBM_BoosterUpdateOneIter(booster, &is_finished) != 0) {
			LGBM_BoosterFree(booster);
			LGBM_DatasetFree(dataset);
			ThrowLightGBMError("LGBM_BoosterUpdateOneIter");
		}
		if (is_finished) {
			break;
		}
	}

	int64_t out_len = 0;
	if (LGBM_BoosterSaveModelToString(booster, 0, -1, 0, 0, &out_len, nullptr) != 0) {
		LGBM_BoosterFree(booster);
		LGBM_DatasetFree(dataset);
		ThrowLightGBMError("LGBM_BoosterSaveModelToString(size)");
	}
	vector<char> buffer(static_cast<idx_t>(out_len) + 1);
	if (LGBM_BoosterSaveModelToString(booster, 0, -1, 0, out_len, &out_len, buffer.data()) != 0) {
		LGBM_BoosterFree(booster);
		LGBM_DatasetFree(dataset);
		ThrowLightGBMError("LGBM_BoosterSaveModelToString");
	}

	LGBM_BoosterFree(booster);
	LGBM_DatasetFree(dataset);

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
                       const vector<double> &weights) {
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
		return TrainWithXGBoost(y, x, options, weights);
#else
		break;
#endif
	case BoostBackend::LIGHTGBM:
#if defined(DUCKBOOST_WITH_LIGHTGBM) && !defined(DUCKBOOST_NATIVE_STUB)
		return TrainWithLightGBM(y, x, options, weights);
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
