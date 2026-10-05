#include "duckboost/functions.hpp"
#include "duckboost/import.hpp"
#include "duckboost/model.hpp"
#include "duckboost/native_train.hpp"

#include "duckdb/catalog/default/default_functions.hpp"
#include "duckdb/catalog/default/default_table_functions.hpp"
#include "duckdb/common/constants.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/vector/flat_vector.hpp"
#include "duckdb/common/vector/list_vector.hpp"
#include "duckdb/common/vector/map_vector.hpp"
#include "duckdb/common/vector/vector_writer.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace duckdb {
namespace duckboost {

namespace {

unordered_map<string, string> MapVectorToOptions(Vector &map_vector, idx_t row) {
	unordered_map<string, string> options;
	if (map_vector.GetType().id() != LogicalTypeId::MAP) {
		return options;
	}
	UnifiedVectorFormat map_format;
	map_vector.ToUnifiedFormat(map_format);
	auto map_idx = map_format.sel->get_index(row);
	if (!map_format.validity.RowIsValid(map_idx)) {
		return options;
	}
	auto list_data = UnifiedVectorFormat::GetData<list_entry_t>(map_format);
	auto entry = list_data[map_idx];
	auto &keys = MapVector::GetKeys(map_vector);
	auto &values = MapVector::GetValues(map_vector);
	UnifiedVectorFormat key_format;
	UnifiedVectorFormat value_format;
	keys.ToUnifiedFormat(key_format);
	values.ToUnifiedFormat(value_format);
	auto key_data = UnifiedVectorFormat::GetData<string_t>(key_format);
	auto value_data = UnifiedVectorFormat::GetData<string_t>(value_format);
	for (idx_t i = 0; i < entry.length; i++) {
		auto key_idx = key_format.sel->get_index(entry.offset + i);
		auto value_idx = value_format.sel->get_index(entry.offset + i);
		if (!key_format.validity.RowIsValid(key_idx) || !value_format.validity.RowIsValid(value_idx)) {
			continue;
		}
		options[key_data[key_idx].GetString()] = value_data[value_idx].GetString();
	}
	return options;
}

vector<double> ReadFeatureList(Vector &list_vector, idx_t row) {
	UnifiedVectorFormat list_format;
	list_vector.ToUnifiedFormat(list_format);
	auto list_idx = list_format.sel->get_index(row);
	if (!list_format.validity.RowIsValid(list_idx)) {
		throw InvalidInputException("duckboost: feature list cannot be NULL");
	}
	auto list_data = UnifiedVectorFormat::GetData<list_entry_t>(list_format);
	auto entry = list_data[list_idx];
	auto &child = ListVector::GetChild(list_vector);
	UnifiedVectorFormat child_format;
	child.ToUnifiedFormat(child_format);
	auto child_data = UnifiedVectorFormat::GetData<double>(child_format);
	vector<double> features;
	features.reserve(entry.length);
	for (idx_t i = 0; i < entry.length; i++) {
		auto child_idx = child_format.sel->get_index(entry.offset + i);
		if (!child_format.validity.RowIsValid(child_idx)) {
			// SQL NULL → NaN so the reference trainer can learn XGBoost-style default directions.
			features.push_back(std::numeric_limits<double>::quiet_NaN());
			continue;
		}
		features.push_back(child_data[child_idx]);
	}
	return features;
}

vector<string> ReadVarcharList(Vector &list_vector, idx_t row) {
	UnifiedVectorFormat list_format;
	list_vector.ToUnifiedFormat(list_format);
	auto list_idx = list_format.sel->get_index(row);
	if (!list_format.validity.RowIsValid(list_idx)) {
		return {};
	}
	auto list_data = UnifiedVectorFormat::GetData<list_entry_t>(list_format);
	auto entry = list_data[list_idx];
	auto &child = ListVector::GetChild(list_vector);
	UnifiedVectorFormat child_format;
	child.ToUnifiedFormat(child_format);
	auto child_data = UnifiedVectorFormat::GetData<string_t>(child_format);
	vector<string> values;
	values.reserve(entry.length);
	for (idx_t i = 0; i < entry.length; i++) {
		auto child_idx = child_format.sel->get_index(entry.offset + i);
		if (!child_format.validity.RowIsValid(child_idx)) {
			throw InvalidInputException("duckboost: feature column names cannot be NULL");
		}
		values.push_back(child_data[child_idx].GetString());
	}
	return values;
}

// Total order on doubles with every NaN equal to every other NaN and after all numbers, so std::sort stays well
// defined when a feature is NaN.
bool TotalLess(double a, double b) {
	if (std::isnan(a) || std::isnan(b)) {
		return !std::isnan(a) && std::isnan(b);
	}
	return a < b;
}

// Parallel aggregation hands rows to an aggregate in thread-dependent order, and both the trainer (tie-breaking,
// floating-point sums) and the metrics depend on row order. Sorting first makes the result depend only on which
// rows went in.
void SortRowsCanonically(vector<double> &y, vector<vector<double>> &x, vector<double> *weights,
                         vector<bool> *is_validation) {
	vector<idx_t> order(y.size());
	std::iota(order.begin(), order.end(), idx_t(0));
	std::sort(order.begin(), order.end(), [&](idx_t a, idx_t b) {
		if (TotalLess(y[a], y[b]) || TotalLess(y[b], y[a])) {
			return TotalLess(y[a], y[b]);
		}
		if (weights && (TotalLess((*weights)[a], (*weights)[b]) || TotalLess((*weights)[b], (*weights)[a]))) {
			return TotalLess((*weights)[a], (*weights)[b]);
		}
		if (is_validation && (*is_validation)[a] != (*is_validation)[b]) {
			return !(*is_validation)[a] && (*is_validation)[b];
		}
		return std::lexicographical_compare(x[a].begin(), x[a].end(), x[b].begin(), x[b].end(), TotalLess);
	});
	vector<double> sorted_y;
	vector<vector<double>> sorted_x;
	vector<double> sorted_w;
	vector<bool> sorted_v;
	sorted_y.reserve(y.size());
	sorted_x.reserve(x.size());
	if (weights) {
		sorted_w.reserve(weights->size());
	}
	if (is_validation) {
		sorted_v.reserve(is_validation->size());
	}
	for (auto index : order) {
		sorted_y.push_back(y[index]);
		sorted_x.push_back(std::move(x[index]));
		if (weights) {
			sorted_w.push_back((*weights)[index]);
		}
		if (is_validation) {
			sorted_v.push_back((*is_validation)[index]);
		}
	}
	y = std::move(sorted_y);
	x = std::move(sorted_x);
	if (weights) {
		*weights = std::move(sorted_w);
	}
	if (is_validation) {
		*is_validation = std::move(sorted_v);
	}
}

struct TrainDataset {
	vector<double> y;
	vector<vector<double>> x;
	vector<double> weights;
	bool has_weights = false;
	vector<bool> is_validation;
	bool has_is_validation = false;
	TrainOptions options;
	bool options_set = false;
};

struct TrainState {
	TrainDataset *data = nullptr;
};

struct TrainOperation {
	template <class STATE>
	static void Initialize(STATE &state) {
		state.data = nullptr;
	}

	template <class STATE>
	static void Destroy(STATE &state, AggregateInputData &) {
		if (state.data) {
			delete state.data;
			state.data = nullptr;
		}
	}

	static bool IgnoreNull() {
		return false;
	}
};

void EnsureTrainState(TrainState &state) {
	if (!state.data) {
		state.data = new TrainDataset();
	}
}

void TrainUpdate(Vector inputs[], AggregateInputData &, idx_t input_count, Vector &state_vector, idx_t count) {
	D_ASSERT(input_count >= 2);
	auto &y_vector = inputs[0];
	auto &x_vector = inputs[1];
	UnifiedVectorFormat y_format;
	y_vector.ToUnifiedFormat(y_format);
	auto y_data = UnifiedVectorFormat::GetData<double>(y_format);

	// Signatures (optional args after features, in order weight / is_validation / options):
	// (y, features)
	// (y, features, options MAP)
	// (y, features, weight DOUBLE)
	// (y, features, weight DOUBLE, options MAP)
	// (y, features, is_validation BOOLEAN)
	// (y, features, is_validation BOOLEAN, options MAP)
	// (y, features, weight DOUBLE, is_validation BOOLEAN)
	// (y, features, weight DOUBLE, is_validation BOOLEAN, options MAP)
	bool has_weight = false;
	bool has_is_validation = false;
	bool has_options = false;
	idx_t weight_arg = 0;
	idx_t is_validation_arg = 0;
	idx_t options_arg = 0;
	for (idx_t arg = 2; arg < input_count; arg++) {
		const auto type_id = inputs[arg].GetType().id();
		if (type_id == LogicalTypeId::MAP) {
			has_options = true;
			options_arg = arg;
		} else if (type_id == LogicalTypeId::BOOLEAN) {
			has_is_validation = true;
			is_validation_arg = arg;
		} else {
			has_weight = true;
			weight_arg = arg;
		}
	}

	UnifiedVectorFormat weight_format;
	const double *weight_data = nullptr;
	if (has_weight) {
		inputs[weight_arg].ToUnifiedFormat(weight_format);
		weight_data = UnifiedVectorFormat::GetData<double>(weight_format);
	}
	UnifiedVectorFormat is_validation_format;
	const bool *is_validation_data = nullptr;
	if (has_is_validation) {
		inputs[is_validation_arg].ToUnifiedFormat(is_validation_format);
		is_validation_data = UnifiedVectorFormat::GetData<bool>(is_validation_format);
	}

	UnifiedVectorFormat state_format;
	state_vector.ToUnifiedFormat(state_format);
	auto states = UnifiedVectorFormat::GetData<TrainState *>(state_format);

	for (idx_t i = 0; i < count; i++) {
		auto y_idx = y_format.sel->get_index(i);
		if (!y_format.validity.RowIsValid(y_idx)) {
			throw InvalidInputException("duckboost: target y cannot be NULL");
		}
		auto state_idx = state_format.sel->get_index(i);
		auto &state = *states[state_idx];
		EnsureTrainState(state);
		if (has_options && !state.data->options_set) {
			state.data->options = TrainOptions::FromMap(MapVectorToOptions(inputs[options_arg], i));
			state.data->options_set = true;
		}
		state.data->y.push_back(y_data[y_idx]);
		state.data->x.push_back(ReadFeatureList(x_vector, i));
		if (has_weight) {
			auto w_idx = weight_format.sel->get_index(i);
			if (!weight_format.validity.RowIsValid(w_idx)) {
				throw InvalidInputException("duckboost: sample weight cannot be NULL");
			}
			state.data->weights.push_back(weight_data[w_idx]);
			state.data->has_weights = true;
		}
		if (has_is_validation) {
			auto v_idx = is_validation_format.sel->get_index(i);
			// NULL means false so macros can default the argument and omit a mask.
			const bool flag = is_validation_format.validity.RowIsValid(v_idx) && is_validation_data[v_idx];
			state.data->is_validation.push_back(flag);
			state.data->has_is_validation = true;
		}
	}
}

void TrainCombine(Vector &source, Vector &target, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat source_format;
	UnifiedVectorFormat target_format;
	source.ToUnifiedFormat(source_format);
	target.ToUnifiedFormat(target_format);
	auto source_states = UnifiedVectorFormat::GetData<TrainState *>(source_format);
	auto target_states = UnifiedVectorFormat::GetData<TrainState *>(target_format);
	for (idx_t i = 0; i < count; i++) {
		auto &src = *source_states[source_format.sel->get_index(i)];
		auto &dst = *target_states[target_format.sel->get_index(i)];
		if (!src.data) {
			continue;
		}
		EnsureTrainState(dst);
		if (!dst.data->options_set && src.data->options_set) {
			dst.data->options = src.data->options;
			dst.data->options_set = true;
		}
		dst.data->y.insert(dst.data->y.end(), src.data->y.begin(), src.data->y.end());
		dst.data->x.insert(dst.data->x.end(), src.data->x.begin(), src.data->x.end());
		if (src.data->has_weights) {
			dst.data->weights.insert(dst.data->weights.end(), src.data->weights.begin(), src.data->weights.end());
			dst.data->has_weights = true;
		}
		if (src.data->has_is_validation) {
			dst.data->is_validation.insert(dst.data->is_validation.end(), src.data->is_validation.begin(),
			                               src.data->is_validation.end());
			dst.data->has_is_validation = true;
		}
	}
}

void TrainFinalize(Vector &state_vector, AggregateFinalizeInputData &, Vector &result, idx_t count, idx_t offset) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	UnifiedVectorFormat state_format;
	state_vector.ToUnifiedFormat(state_format);
	auto states = UnifiedVectorFormat::GetData<TrainState *>(state_format);
	auto writer = FlatVector::Writer<string_t>(result, count, offset);
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[state_format.sel->get_index(i)];
		if (!state.data || state.data->y.empty()) {
			writer.WriteNull();
			continue;
		}
		auto options = state.data->options_set ? state.data->options : TrainOptions();
		SortRowsCanonically(state.data->y, state.data->x, state.data->has_weights ? &state.data->weights : nullptr,
		                    state.data->has_is_validation ? &state.data->is_validation : nullptr);
		const vector<double> empty_weights;
		const vector<bool> empty_is_validation;
		auto model = TrainModel(state.data->y, state.data->x, options,
		                        state.data->has_weights ? state.data->weights : empty_weights,
		                        state.data->has_is_validation ? state.data->is_validation : empty_is_validation);
		writer.WriteValue(StringVector::AddString(result, model.ToJSON()));
	}
}

AggregateFunction GetTrainFunction(bool with_weight, bool with_is_validation, bool with_options) {
	auto feature_type = LogicalType::LIST(LogicalType::DOUBLE);
	auto options_type = LogicalType::MAP(LogicalType::VARCHAR, LogicalType::VARCHAR);
	vector<LogicalType> args = {LogicalType::DOUBLE, feature_type};
	if (with_weight) {
		args.push_back(LogicalType::DOUBLE);
	}
	if (with_is_validation) {
		args.push_back(LogicalType::BOOLEAN);
	}
	if (with_options) {
		args.push_back(options_type);
	}
	AggregateFunction fun(
	    args, LogicalType::VARCHAR, AggregateFunction::StateSize<TrainState>,
	    AggregateFunction::StateInitialize<TrainState, TrainOperation, AggregateDestructorType::LEGACY>, TrainUpdate,
	    TrainCombine, TrainFinalize, AggregateFunction::NoClusterUpdate(), AggregateFunction::NoBind(),
	    AggregateFunction::StateDestroy<TrainState, TrainOperation>);
	fun.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	fun.GetSignature().GetParameter(0).SetName("y");
	fun.GetSignature().GetParameter(1).SetName("features");
	idx_t next = 2;
	if (with_weight) {
		fun.GetSignature().GetParameter(next++).SetName("weight");
	}
	if (with_is_validation) {
		fun.GetSignature().GetParameter(next++).SetName("is_validation");
	}
	if (with_options) {
		fun.GetSignature().GetParameter(next).SetName("options");
	}
	return fun;
}

//! Reuse one parsed model when the JSON string repeats across a chunk (the common constant-model case).
struct CachedModelParse {
	string json;
	BoostModel model;
	bool valid = false;

	const BoostModel &Get(const string &model_json) {
		if (!valid || json != model_json) {
			json = model_json;
			model = BoostModel::FromJSON(model_json);
			valid = true;
		}
		return model;
	}
};

void PredictFunction(DataChunk &args, ExpressionState &, Vector &result) {
	auto count = args.size();
	UnifiedVectorFormat model_format;
	args.data[0].ToUnifiedFormat(model_format);
	auto model_data = UnifiedVectorFormat::GetData<string_t>(model_format);
	auto writer = FlatVector::Writer<double>(result, count);
	CachedModelParse cache;
	for (idx_t i = 0; i < count; i++) {
		auto model_idx = model_format.sel->get_index(i);
		if (!model_format.validity.RowIsValid(model_idx)) {
			writer.WriteNull();
			continue;
		}
		auto &model = cache.Get(model_data[model_idx].GetString());
		auto features = ReadFeatureList(args.data[1], i);
		writer.WriteValue(model.Predict(features));
	}
}

void PredictProbaFunction(DataChunk &args, ExpressionState &, Vector &result) {
	auto count = args.size();
	result.SetVectorType(VectorType::FLAT_VECTOR);
	UnifiedVectorFormat model_format;
	args.data[0].ToUnifiedFormat(model_format);
	auto model_data = UnifiedVectorFormat::GetData<string_t>(model_format);
	auto writer = FlatVector::Writer<VectorListType<double>>(result, count);
	CachedModelParse cache;
	for (idx_t i = 0; i < count; i++) {
		auto model_idx = model_format.sel->get_index(i);
		if (!model_format.validity.RowIsValid(model_idx)) {
			writer.WriteNull();
			continue;
		}
		auto &model = cache.Get(model_data[model_idx].GetString());
		auto features = ReadFeatureList(args.data[1], i);
		auto proba = model.PredictProba(features);
		idx_t class_idx = 0;
		for (auto &child : writer.WriteList(proba.size())) {
			child.WriteValue(proba[class_idx++]);
		}
	}
}

void EvaluateFunction(DataChunk &args, ExpressionState &, Vector &result) {
	auto count = args.size();
	UnifiedVectorFormat model_format;
	UnifiedVectorFormat y_format;
	args.data[0].ToUnifiedFormat(model_format);
	args.data[1].ToUnifiedFormat(y_format);
	auto model_data = UnifiedVectorFormat::GetData<string_t>(model_format);
	auto y_data = UnifiedVectorFormat::GetData<double>(y_format);
	auto writer = FlatVector::Writer<double>(result, count);
	CachedModelParse cache;

	// Row-wise evaluate is awkward; support a convenience aggregate-style path via lists? For MVP, compute
	// one-row metrics when users pass scalar model + scalar y + features, and document duckboost_evaluate_agg.
	for (idx_t i = 0; i < count; i++) {
		auto model_idx = model_format.sel->get_index(i);
		auto y_idx = y_format.sel->get_index(i);
		if (!model_format.validity.RowIsValid(model_idx) || !y_format.validity.RowIsValid(y_idx)) {
			writer.WriteNull();
			continue;
		}
		auto &model = cache.Get(model_data[model_idx].GetString());
		auto features = ReadFeatureList(args.data[2], i);
		EvalOptions options;
		options.metric = "auto";
		if (args.ColumnCount() >= 4) {
			options = EvalOptions::FromMap(MapVectorToOptions(args.data[3], i));
		}
		// Per-row absolute/squared error helpers for streaming metrics.
		auto pred = model.Predict(features);
		auto y = y_data[y_idx];
		auto metric = options.metric;
		if (metric == "brier") {
			metric = "brier_score";
		} else if (metric == "auc_ovr") {
			metric = "roc_auc_ovr";
		} else if (metric == "auc_ovo") {
			metric = "roc_auc_ovo";
		}
		if (metric.empty() || metric == "auto") {
			if (model.task == BoostTask::BINARY || model.task == BoostTask::MULTICLASS) {
				metric = "accuracy";
			} else if (model.loss == RegressionLoss::ABSOLUTE_ERROR) {
				metric = "mae";
			} else if (model.loss == RegressionLoss::QUANTILE) {
				metric = "pinball";
			} else if (model.loss == RegressionLoss::EXPECTILE) {
				metric = "expectile";
			} else {
				metric = "rmse";
			}
		}
		if (metric == "mae") {
			writer.WriteValue(std::fabs(pred - y));
		} else if (metric == "pinball" || metric == "quantile") {
			auto error = pred - y;
			writer.WriteValue(error >= 0 ? (1.0 - model.objective_alpha) * error : -model.objective_alpha * error);
		} else if (metric == "expectile") {
			auto error = pred - y;
			auto asymmetry = y >= pred ? model.objective_alpha : 1.0 - model.objective_alpha;
			writer.WriteValue(asymmetry * error * error);
		} else if (metric == "accuracy") {
			if (model.task == BoostTask::MULTICLASS) {
				writer.WriteValue(pred == y ? 1.0 : 0.0);
			} else {
				writer.WriteValue((pred >= 0.5 ? 1.0 : 0.0) == y ? 1.0 : 0.0);
			}
		} else if (metric == "logloss") {
			if (model.task == BoostTask::MULTICLASS) {
				auto proba = model.PredictProba(features);
				auto label = static_cast<idx_t>(y);
				if (label >= proba.size()) {
					throw InvalidInputException("duckboost: multiclass label out of range during logloss");
				}
				auto p = std::min(1.0 - 1e-15, std::max(1e-15, proba[label]));
				writer.WriteValue(-std::log(p));
			} else {
				auto p = std::min(1.0 - 1e-15, std::max(1e-15, pred));
				writer.WriteValue(-(y * std::log(p) + (1.0 - y) * std::log(1.0 - p)));
			}
		} else if (metric == "rmse") {
			// Squared error contribution; take the square root after averaging rows.
			auto err = pred - y;
			writer.WriteValue(err * err);
		} else if (metric == "brier_score") {
			if (model.task != BoostTask::BINARY && model.task != BoostTask::MULTICLASS) {
				throw InvalidInputException("duckboost: metric 'brier_score' requires a binary or multiclass model");
			}
			auto proba = model.PredictProba(features);
			if (model.task == BoostTask::BINARY) {
				if (!std::isfinite(y) || y < 0 || y != std::floor(y) || y > 1) {
					throw InvalidInputException("duckboost: binary brier_score expects labels 0 or 1");
				}
				auto p = proba.size() >= 2 ? proba[1] : pred;
				auto err = y - p;
				writer.WriteValue(err * err);
			} else {
				if (!std::isfinite(y) || y < 0 || y != std::floor(y)) {
					throw InvalidInputException("duckboost: brier_score expects non-negative integer class labels");
				}
				auto label = static_cast<idx_t>(y);
				if (label >= proba.size()) {
					throw InvalidInputException("duckboost: multiclass label out of range during brier_score");
				}
				double loss = 0;
				for (idx_t c = 0; c < proba.size(); c++) {
					auto target = c == label ? 1.0 : 0.0;
					auto err = target - proba[c];
					loss += err * err;
				}
				writer.WriteValue(loss);
			}
		} else if (metric == "roc_auc_ovr" || metric == "roc_auc_ovo") {
			throw InvalidInputException("duckboost: metric '%s' is dataset-level only; use duckboost_evaluate_agg",
			                            metric);
		} else {
			throw InvalidInputException("duckboost: unknown metric '%s'", options.metric);
		}
	}
}

struct EvaluateDataset {
	vector<double> y;
	vector<vector<double>> x;
	string model_json;
	EvalOptions options;
	bool options_set = false;
	bool model_set = false;
};

struct EvaluateAggState {
	EvaluateDataset *data = nullptr;
};

struct EvaluateAggOperation {
	template <class STATE>
	static void Initialize(STATE &state) {
		state.data = nullptr;
	}

	template <class STATE>
	static void Destroy(STATE &state, AggregateInputData &) {
		if (state.data) {
			delete state.data;
			state.data = nullptr;
		}
	}
};

void EvaluateAggUpdate(Vector inputs[], AggregateInputData &, idx_t input_count, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat state_format;
	state_vector.ToUnifiedFormat(state_format);
	auto states = UnifiedVectorFormat::GetData<EvaluateAggState *>(state_format);
	UnifiedVectorFormat model_format;
	UnifiedVectorFormat y_format;
	inputs[0].ToUnifiedFormat(model_format);
	inputs[1].ToUnifiedFormat(y_format);
	auto model_data = UnifiedVectorFormat::GetData<string_t>(model_format);
	auto y_data = UnifiedVectorFormat::GetData<double>(y_format);

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[state_format.sel->get_index(i)];
		if (!state.data) {
			state.data = new EvaluateDataset();
		}
		auto model_idx = model_format.sel->get_index(i);
		auto y_idx = y_format.sel->get_index(i);
		if (!model_format.validity.RowIsValid(model_idx) || !y_format.validity.RowIsValid(y_idx)) {
			throw InvalidInputException("duckboost: model and y cannot be NULL in evaluate_agg");
		}
		if (!state.data->model_set) {
			state.data->model_json = model_data[model_idx].GetString();
			state.data->model_set = true;
		}
		if (input_count >= 4 && !state.data->options_set) {
			state.data->options = EvalOptions::FromMap(MapVectorToOptions(inputs[3], i));
			state.data->options_set = true;
		}
		state.data->y.push_back(y_data[y_idx]);
		state.data->x.push_back(ReadFeatureList(inputs[2], i));
	}
}

void EvaluateAggCombine(Vector &source, Vector &target, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat source_format;
	UnifiedVectorFormat target_format;
	source.ToUnifiedFormat(source_format);
	target.ToUnifiedFormat(target_format);
	auto source_states = UnifiedVectorFormat::GetData<EvaluateAggState *>(source_format);
	auto target_states = UnifiedVectorFormat::GetData<EvaluateAggState *>(target_format);
	for (idx_t i = 0; i < count; i++) {
		auto &src = *source_states[source_format.sel->get_index(i)];
		auto &dst = *target_states[target_format.sel->get_index(i)];
		if (!src.data) {
			continue;
		}
		if (!dst.data) {
			dst.data = new EvaluateDataset();
		}
		if (!dst.data->model_set && src.data->model_set) {
			dst.data->model_json = src.data->model_json;
			dst.data->model_set = true;
		}
		if (!dst.data->options_set && src.data->options_set) {
			dst.data->options = src.data->options;
			dst.data->options_set = true;
		}
		dst.data->y.insert(dst.data->y.end(), src.data->y.begin(), src.data->y.end());
		dst.data->x.insert(dst.data->x.end(), src.data->x.begin(), src.data->x.end());
	}
}

void EvaluateAggFinalize(Vector &state_vector, AggregateFinalizeInputData &, Vector &result, idx_t count,
                         idx_t offset) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	UnifiedVectorFormat state_format;
	state_vector.ToUnifiedFormat(state_format);
	auto states = UnifiedVectorFormat::GetData<EvaluateAggState *>(state_format);
	auto writer = FlatVector::Writer<double>(result, count, offset);
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[state_format.sel->get_index(i)];
		if (!state.data || !state.data->model_set || state.data->y.empty()) {
			writer.WriteNull();
			continue;
		}
		auto model = BoostModel::FromJSON(state.data->model_json);
		auto options = state.data->options_set ? state.data->options : EvalOptions();
		if (options.metric.empty()) {
			options.metric = "auto";
		}
		SortRowsCanonically(state.data->y, state.data->x, nullptr, nullptr);
		writer.WriteValue(EvaluateModel(model, state.data->y, state.data->x, options));
	}
}

AggregateFunction GetEvaluateAggFunction(bool with_options) {
	auto feature_type = LogicalType::LIST(LogicalType::DOUBLE);
	auto options_type = LogicalType::MAP(LogicalType::VARCHAR, LogicalType::VARCHAR);
	vector<LogicalType> args = {LogicalType::VARCHAR, LogicalType::DOUBLE, feature_type};
	if (with_options) {
		args.push_back(options_type);
	}
	AggregateFunction fun(
	    args, LogicalType::DOUBLE, AggregateFunction::StateSize<EvaluateAggState>,
	    AggregateFunction::StateInitialize<EvaluateAggState, EvaluateAggOperation, AggregateDestructorType::LEGACY>,
	    EvaluateAggUpdate, EvaluateAggCombine, EvaluateAggFinalize, AggregateFunction::NoClusterUpdate(),
	    AggregateFunction::NoBind(), AggregateFunction::StateDestroy<EvaluateAggState, EvaluateAggOperation>);
	fun.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	fun.GetSignature().GetParameter(0).SetName("model");
	fun.GetSignature().GetParameter(1).SetName("y");
	fun.GetSignature().GetParameter(2).SetName("features");
	if (with_options) {
		fun.GetSignature().GetParameter(3).SetName("options");
	}
	return fun;
}

void ToSQLFunction(DataChunk &args, ExpressionState &, Vector &result) {
	auto count = args.size();
	UnifiedVectorFormat model_format;
	UnifiedVectorFormat table_format;
	args.data[0].ToUnifiedFormat(model_format);
	args.data[1].ToUnifiedFormat(table_format);
	auto model_data = UnifiedVectorFormat::GetData<string_t>(model_format);
	auto table_data = UnifiedVectorFormat::GetData<string_t>(table_format);
	auto writer = FlatVector::Writer<string_t>(result, count);
	for (idx_t i = 0; i < count; i++) {
		auto model_idx = model_format.sel->get_index(i);
		auto table_idx = table_format.sel->get_index(i);
		if (!model_format.validity.RowIsValid(model_idx) || !table_format.validity.RowIsValid(table_idx)) {
			writer.WriteNull();
			continue;
		}
		auto model = BoostModel::FromJSON(model_data[model_idx].GetString());
		auto feature_columns = ReadVarcharList(args.data[2], i);
		SqlExportOptions options;
		if (args.ColumnCount() >= 4) {
			options = SqlExportOptions::FromMap(MapVectorToOptions(args.data[3], i));
		}
		auto sql = ExportModelSQL(model, table_data[table_idx].GetString(), feature_columns, options);
		writer.WriteValue(StringVector::AddString(result, sql));
	}
}

void ImportFunction(DataChunk &args, ExpressionState &, Vector &result) {
	auto count = args.size();
	UnifiedVectorFormat backend_format;
	UnifiedVectorFormat model_format;
	args.data[0].ToUnifiedFormat(backend_format);
	args.data[1].ToUnifiedFormat(model_format);
	auto backend_data = UnifiedVectorFormat::GetData<string_t>(backend_format);
	auto model_data = UnifiedVectorFormat::GetData<string_t>(model_format);
	auto writer = FlatVector::Writer<string_t>(result, count);
	for (idx_t i = 0; i < count; i++) {
		auto backend_idx = backend_format.sel->get_index(i);
		auto model_idx = model_format.sel->get_index(i);
		if (!model_format.validity.RowIsValid(model_idx) || !backend_format.validity.RowIsValid(backend_idx)) {
			writer.WriteNull();
			continue;
		}
		auto backend = BackendFromString(backend_data[backend_idx].GetString());
		ImportOptions options;
		if (args.ColumnCount() >= 3) {
			options = ImportOptions::FromMap(MapVectorToOptions(args.data[2], i));
		}
		auto model = ImportModel(backend, model_data[model_idx].GetString(), options);
		writer.WriteValue(StringVector::AddString(result, model.ToJSON()));
	}
}

struct BackendsData : public GlobalTableFunctionState {
	idx_t offset = 0;
};

unique_ptr<FunctionData> BackendsBind(ClientContext &, TableFunctionBindInput &, vector<LogicalType> &return_types,
                                      vector<Identifier> &names) {
	names = {"backend", "training_supported", "notes"};
	return_types = {LogicalType::VARCHAR, LogicalType::BOOLEAN, LogicalType::VARCHAR};
	return nullptr;
}

unique_ptr<GlobalTableFunctionState> BackendsInit(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<BackendsData>();
}

void BackendsFunction(ClientContext &, TableFunctionInput &data, DataChunk &output) {
	auto &state = data.global_state->Cast<BackendsData>();
	static const BoostBackend backends[] = {BoostBackend::REFERENCE, BoostBackend::XGBOOST, BoostBackend::LIGHTGBM,
	                                        BoostBackend::CATBOOST};
	static constexpr idx_t backend_count = sizeof(backends) / sizeof(backends[0]);
	if (state.offset >= backend_count) {
		return;
	}
	const idx_t remaining = backend_count - state.offset;
	const idx_t count = MinValue<idx_t>(remaining, STANDARD_VECTOR_SIZE);
	auto backend_writer = FlatVector::Writer<string_t>(output.data[0], count);
	auto supported_writer = FlatVector::Writer<bool>(output.data[1], count);
	auto notes_writer = FlatVector::Writer<string_t>(output.data[2], count);
	for (idx_t i = 0; i < count; i++) {
		auto backend = backends[state.offset + i];
		backend_writer.WriteValue(StringVector::AddString(output.data[0], BackendToString(backend)));
		supported_writer.WriteValue(BackendTrainingSupported(backend));
		notes_writer.WriteValue(StringVector::AddString(output.data[2], BackendCapabilityNote(backend)));
	}
	state.offset += count;
}

struct BuildInfoRow {
	const char *name;
	bool enabled;
	const char *notes;
};

struct BuildInfoData : public GlobalTableFunctionState {
	idx_t offset = 0;
};

unique_ptr<FunctionData> BuildInfoBind(ClientContext &, TableFunctionBindInput &, vector<LogicalType> &return_types,
                                       vector<Identifier> &names) {
	names = {"name", "enabled", "notes"};
	return_types = {LogicalType::VARCHAR, LogicalType::BOOLEAN, LogicalType::VARCHAR};
	return nullptr;
}

unique_ptr<GlobalTableFunctionState> BuildInfoInit(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<BuildInfoData>();
}

void BuildInfoFunction(ClientContext &, TableFunctionInput &data, DataChunk &output) {
	auto &state = data.global_state->Cast<BuildInfoData>();
	static const BuildInfoRow props[] = {
	    {"DUCKBOOST_WITH_XGBOOST",
#if defined(DUCKBOOST_WITH_XGBOOST)
	     true,
#else
	     false,
#endif
	     "XGBoost native trainer compile flag"},
	    {"DUCKBOOST_WITH_LIGHTGBM",
#if defined(DUCKBOOST_WITH_LIGHTGBM)
	     true,
#else
	     false,
#endif
	     "LightGBM native trainer compile flag"},
	    {"DUCKBOOST_WITH_CATBOOST",
#if defined(DUCKBOOST_WITH_CATBOOST)
	     true,
#else
	     false,
#endif
	     "CatBoost native trainer compile flag"},
	    {"DUCKBOOST_NATIVE_STUB",
#if defined(DUCKBOOST_NATIVE_STUB)
	     true,
#else
	     false,
#endif
	     "Native trainers compiled as stubs (no vendor link)"},
	    {"native_xgboost_compiled", NativeTrainerCompiled(BoostBackend::XGBOOST), "NativeTrainerCompiled(xgboost)"},
	    {"native_lightgbm_compiled", NativeTrainerCompiled(BoostBackend::LIGHTGBM), "NativeTrainerCompiled(lightgbm)"},
	    {"native_catboost_compiled", NativeTrainerCompiled(BoostBackend::CATBOOST), "NativeTrainerCompiled(catboost)"},
	    {"native_xgboost_linked", NativeTrainerLinked(BoostBackend::XGBOOST),
	     "NativeTrainerLinked(xgboost) — C API train bridge"},
	    {"native_lightgbm_linked", NativeTrainerLinked(BoostBackend::LIGHTGBM),
	     "NativeTrainerLinked(lightgbm) — C API train bridge"},
	    {"native_catboost_linked", NativeTrainerLinked(BoostBackend::CATBOOST),
	     "Always false: CatBoost has no public train C API"},
	};
	static constexpr idx_t prop_count = sizeof(props) / sizeof(props[0]);
	if (state.offset >= prop_count) {
		return;
	}
	const idx_t remaining = prop_count - state.offset;
	const idx_t count = MinValue<idx_t>(remaining, STANDARD_VECTOR_SIZE);
	auto name_writer = FlatVector::Writer<string_t>(output.data[0], count);
	auto enabled_writer = FlatVector::Writer<bool>(output.data[1], count);
	auto notes_writer = FlatVector::Writer<string_t>(output.data[2], count);
	for (idx_t i = 0; i < count; i++) {
		auto &prop = props[state.offset + i];
		name_writer.WriteValue(StringVector::AddString(output.data[0], prop.name));
		enabled_writer.WriteValue(prop.enabled);
		notes_writer.WriteValue(StringVector::AddString(output.data[2], prop.notes));
	}
	state.offset += count;
}

} // namespace

// clang-format off
static const DefaultTableMacro duckboost_table_macros[] = {
	{DEFAULT_SCHEMA, "duckboost_fit", {"source", "y", "features", nullptr}, {{"weight", "NULL"}, {"is_validation", "NULL"}, {"options", "MAP {}"}, {nullptr, nullptr}}, R"(
SELECT duckboost_train(y, features, coalesce(weight, 1.0), coalesce(is_validation, false), options) AS model
FROM query_table(source::VARCHAR)
)"},
	{DEFAULT_SCHEMA, "duckboost_score", {"model", "source", "features", nullptr}, {{nullptr, nullptr}}, R"(
SELECT *, duckboost_predict(model, features) AS prediction
FROM query_table(source::VARCHAR)
)"},
	// The split macros rank rows by a hash of the whole row mixed with the seed, so the assignment depends only on
	// row contents and seed, not on physical row order. hash(row, seed) mixes the seed too weakly, hence the xor.
	{DEFAULT_SCHEMA, "duckboost_initial_split", {"source", nullptr}, {{"prop", "0.75"}, {"strata", "NULL"}, {"seed", "42"}, {nullptr, nullptr}}, R"(
WITH __duckboost_hashed AS (
	SELECT __duckboost_row.*,
		hash(xor(hash(__duckboost_row), hash(seed::BIGINT))) AS __duckboost_hash,
		strata AS __duckboost_stratum
	FROM query_table(source::VARCHAR) __duckboost_row
), __duckboost_ranked AS (
	SELECT *,
		row_number() OVER (PARTITION BY __duckboost_stratum ORDER BY __duckboost_hash) AS __duckboost_rank,
		count(*) OVER (PARTITION BY __duckboost_stratum) AS __duckboost_size,
		count(DISTINCT __duckboost_stratum) OVER () AS __duckboost_strata,
		count(*) OVER () AS __duckboost_rows
	FROM __duckboost_hashed
)
SELECT * EXCLUDE (__duckboost_hash, __duckboost_stratum, __duckboost_rank, __duckboost_size, __duckboost_strata, __duckboost_rows),
	CASE
		WHEN prop IS NULL OR prop <= 0 OR prop >= 1
			THEN error('duckboost_initial_split: prop must be strictly between 0 and 1')
		WHEN __duckboost_strata > 1 AND __duckboost_strata * 5 > __duckboost_rows
			THEN error('duckboost: strata has fewer than 5 rows per value on average; bin a numeric column first, e.g. strata := ntile(4) OVER (ORDER BY price)')
		WHEN __duckboost_rank <= round(prop * __duckboost_size) THEN 'train'
		ELSE 'test'
	END AS split
FROM __duckboost_ranked
)"},
	{DEFAULT_SCHEMA, "duckboost_initial_validation_split", {"source", nullptr}, {{"prop", "[0.6, 0.2]"}, {"strata", "NULL"}, {"seed", "42"}, {nullptr, nullptr}}, R"(
WITH __duckboost_hashed AS (
	SELECT __duckboost_row.*,
		hash(xor(hash(__duckboost_row), hash(seed::BIGINT))) AS __duckboost_hash,
		strata AS __duckboost_stratum
	FROM query_table(source::VARCHAR) __duckboost_row
), __duckboost_ranked AS (
	SELECT *,
		row_number() OVER (PARTITION BY __duckboost_stratum ORDER BY __duckboost_hash) AS __duckboost_rank,
		count(*) OVER (PARTITION BY __duckboost_stratum) AS __duckboost_size,
		count(DISTINCT __duckboost_stratum) OVER () AS __duckboost_strata,
		count(*) OVER () AS __duckboost_rows
	FROM __duckboost_hashed
)
SELECT * EXCLUDE (__duckboost_hash, __duckboost_stratum, __duckboost_rank, __duckboost_size, __duckboost_strata, __duckboost_rows),
	CASE
		WHEN len(prop) IS DISTINCT FROM 2 OR prop[1] <= 0 OR prop[2] <= 0 OR prop[1] + prop[2] >= 1
			THEN error('duckboost_initial_validation_split: prop must be [train, validation], both > 0 and summing to less than 1; the rest is test')
		WHEN __duckboost_strata > 1 AND __duckboost_strata * 5 > __duckboost_rows
			THEN error('duckboost: strata has fewer than 5 rows per value on average; bin a numeric column first, e.g. strata := ntile(4) OVER (ORDER BY price)')
		WHEN __duckboost_rank <= round(prop[1] * __duckboost_size) THEN 'train'
		WHEN __duckboost_rank <= round((prop[1] + prop[2]) * __duckboost_size) THEN 'validation'
		ELSE 'test'
	END AS split
FROM __duckboost_ranked
)"},
	// Rows are dealt round-robin in (stratum, hash) order, so folds differ in size by at most one row and every
	// stratum is spread evenly across folds.
	{DEFAULT_SCHEMA, "duckboost_vfold", {"source", nullptr}, {{"v", "5"}, {"strata", "NULL"}, {"seed", "42"}, {nullptr, nullptr}}, R"(
WITH __duckboost_hashed AS (
	SELECT __duckboost_row.*,
		hash(xor(hash(__duckboost_row), hash(seed::BIGINT))) AS __duckboost_hash,
		strata AS __duckboost_stratum
	FROM query_table(source::VARCHAR) __duckboost_row
), __duckboost_ranked AS (
	SELECT *,
		row_number() OVER (ORDER BY __duckboost_stratum, __duckboost_hash) AS __duckboost_rank,
		count(DISTINCT __duckboost_stratum) OVER () AS __duckboost_strata,
		count(*) OVER () AS __duckboost_rows
	FROM __duckboost_hashed
)
SELECT * EXCLUDE (__duckboost_hash, __duckboost_stratum, __duckboost_rank, __duckboost_strata, __duckboost_rows),
	CASE
		WHEN v IS NULL OR v < 2 OR v > __duckboost_rows
			THEN error('duckboost_vfold: v must be at least 2 and at most the number of rows')
		WHEN __duckboost_strata > 1 AND __duckboost_strata * 5 > __duckboost_rows
			THEN error('duckboost: strata has fewer than 5 rows per value on average; bin a numeric column first, e.g. strata := ntile(4) OVER (ORDER BY price)')
		ELSE ((__duckboost_rank - 1) % v + 1)::INTEGER
	END AS fold
FROM __duckboost_ranked
)"},
	{nullptr, nullptr, {nullptr}, {{nullptr, nullptr}}, nullptr}
};

static const DefaultMacro duckboost_scalar_macros[] = {
	{DEFAULT_SCHEMA, "duckboost_levels", R"((x, min_count := 1) AS
	list_sort([__duckboost_level.key FOR __duckboost_level IN map_entries(histogram(x)) IF __duckboost_level.value >= min_count]))"},
	{DEFAULT_SCHEMA, "duckboost_other", R"((x, levels, other := 'other') AS
	CASE WHEN x IS NULL THEN NULL WHEN list_contains(levels, x) THEN x ELSE other END)"},
	// Dummy encoding drops the first level as the reference; one_hot keeps every level.
	{DEFAULT_SCHEMA, "duckboost_dummy", R"((x, levels, one_hot := false) AS
	[CASE WHEN x = __duckboost_level THEN 1.0::DOUBLE ELSE 0.0::DOUBLE END
	 FOR __duckboost_level IN levels[CASE WHEN one_hot THEN 1 ELSE 2 END:]])"},
	{DEFAULT_SCHEMA, "duckboost_dummy_names", R"((prefix, levels, one_hot := false) AS
	[prefix || '_' || regexp_replace(__duckboost_level::VARCHAR, '[^A-Za-z0-9_]', '_', 'g')
	 FOR __duckboost_level IN levels[CASE WHEN one_hot THEN 1 ELSE 2 END:]])"},
	{DEFAULT_SCHEMA, "duckboost_integer", R"((x, levels) AS
	coalesce(list_position(levels, x), 0)::DOUBLE)"},
	{nullptr, nullptr, nullptr}
};
// clang-format on

void RegisterDuckBoostMacros(ExtensionLoader &loader) {
	for (idx_t index = 0; duckboost_table_macros[index].name != nullptr; index++) {
		auto info = DefaultTableFunctionGenerator::CreateTableMacroInfo(duckboost_table_macros[index]);
		loader.RegisterFunction(*info);
	}
	for (idx_t index = 0; duckboost_scalar_macros[index].name != nullptr; index++) {
		auto info = DefaultFunctionGenerator::CreateInternalMacroInfo(duckboost_scalar_macros[index]);
		loader.RegisterFunction(*info);
	}
}

unordered_map<string, string> ValueMapToOptions(const Value &map_value) {
	unordered_map<string, string> options;
	if (map_value.IsNull()) {
		return options;
	}
	for (auto &entry : MapValue::GetChildren(map_value)) {
		auto &kv = StructValue::GetChildren(entry);
		if (kv[0].IsNull() || kv[1].IsNull()) {
			continue;
		}
		options[StringValue::Get(kv[0])] = StringValue::Get(kv[1]);
	}
	return options;
}

struct ImportanceBindData : public TableFunctionData {
	vector<FeatureImportance> rows;
};

struct ImportanceData : public GlobalTableFunctionState {
	idx_t offset = 0;
};

unique_ptr<FunctionData> ImportanceBind(ClientContext &, TableFunctionBindInput &input,
                                        vector<LogicalType> &return_types, vector<Identifier> &names) {
	names = {"variable", "feature_index", "gain", "cover", "frequency", "importance"};
	return_types = {LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::DOUBLE,
	                LogicalType::DOUBLE,  LogicalType::BIGINT, LogicalType::DOUBLE};
	if (input.inputs.empty() || input.inputs[0].IsNull()) {
		throw InvalidInputException("duckboost_importance: model must not be NULL");
	}
	unordered_map<string, string> option_map;
	if (input.inputs.size() >= 2 && !input.inputs[1].IsNull()) {
		option_map = ValueMapToOptions(input.inputs[1]);
	}
	for (auto &np : input.named_parameters) {
		auto key = StringUtil::Lower(np.first.GetIdentifierName());
		if (key == "options" && !np.second.IsNull()) {
			auto named = ValueMapToOptions(np.second);
			option_map.insert(named.begin(), named.end());
		}
	}
	auto options = ImportanceOptions::FromMap(option_map);
	auto model = BoostModel::FromJSON(StringValue::Get(input.inputs[0]));
	auto result = make_uniq<ImportanceBindData>();
	result->rows = ComputeFeatureImportance(model, options);
	return std::move(result);
}

unique_ptr<GlobalTableFunctionState> ImportanceInit(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<ImportanceData>();
}

void ImportanceFunction(ClientContext &, TableFunctionInput &data, DataChunk &output) {
	auto &bind = data.bind_data->Cast<ImportanceBindData>();
	auto &state = data.global_state->Cast<ImportanceData>();
	if (state.offset >= bind.rows.size()) {
		return;
	}
	const idx_t remaining = bind.rows.size() - state.offset;
	const idx_t count = MinValue<idx_t>(remaining, STANDARD_VECTOR_SIZE);
	auto variable_writer = FlatVector::Writer<string_t>(output.data[0], count);
	auto index_writer = FlatVector::Writer<int64_t>(output.data[1], count);
	auto gain_writer = FlatVector::Writer<double>(output.data[2], count);
	auto cover_writer = FlatVector::Writer<double>(output.data[3], count);
	auto frequency_writer = FlatVector::Writer<int64_t>(output.data[4], count);
	auto importance_writer = FlatVector::Writer<double>(output.data[5], count);
	for (idx_t i = 0; i < count; i++) {
		auto &row = bind.rows[state.offset + i];
		variable_writer.WriteValue(StringVector::AddString(output.data[0], row.variable));
		index_writer.WriteValue(NumericCast<int64_t>(row.feature_index));
		gain_writer.WriteValue(row.gain);
		cover_writer.WriteValue(row.cover);
		frequency_writer.WriteValue(NumericCast<int64_t>(row.frequency));
		importance_writer.WriteValue(row.importance);
	}
	state.offset += count;
}

void RegisterDuckBoostFunctions(ExtensionLoader &loader) {
	AggregateFunctionSet train_set("duckboost_train");
	train_set.AddFunction(GetTrainFunction(false, false, false));
	train_set.AddFunction(GetTrainFunction(false, false, true));
	train_set.AddFunction(GetTrainFunction(false, true, false));
	train_set.AddFunction(GetTrainFunction(false, true, true));
	train_set.AddFunction(GetTrainFunction(true, false, false));
	train_set.AddFunction(GetTrainFunction(true, false, true));
	train_set.AddFunction(GetTrainFunction(true, true, false));
	train_set.AddFunction(GetTrainFunction(true, true, true));
	loader.RegisterFunction(train_set);

	ScalarFunctionSet predict_set("duckboost_predict");
	ScalarFunction predict_fun({}, LogicalType::DOUBLE, PredictFunction);
	predict_fun.SetFallible();
	predict_fun.GetSignature()
	    .AddParameter("model", LogicalType::VARCHAR)
	    .AddParameter("features", LogicalType::LIST(LogicalType::DOUBLE));
	predict_set.AddFunction(predict_fun);
	loader.RegisterFunction(predict_set);

	ScalarFunctionSet predict_proba_set("duckboost_predict_proba");
	ScalarFunction predict_proba_fun({}, LogicalType::LIST(LogicalType::DOUBLE), PredictProbaFunction);
	predict_proba_fun.SetFallible();
	predict_proba_fun.GetSignature()
	    .AddParameter("model", LogicalType::VARCHAR)
	    .AddParameter("features", LogicalType::LIST(LogicalType::DOUBLE));
	predict_proba_set.AddFunction(predict_proba_fun);
	loader.RegisterFunction(predict_proba_set);

	ScalarFunctionSet evaluate_set("duckboost_evaluate");
	ScalarFunction evaluate_fun({}, LogicalType::DOUBLE, EvaluateFunction);
	evaluate_fun.SetFallible();
	evaluate_fun.GetSignature()
	    .AddParameter("model", LogicalType::VARCHAR)
	    .AddParameter("y", LogicalType::DOUBLE)
	    .AddParameter("features", LogicalType::LIST(LogicalType::DOUBLE));
	evaluate_set.AddFunction(evaluate_fun);
	ScalarFunction evaluate_opts({}, LogicalType::DOUBLE, EvaluateFunction);
	evaluate_opts.SetFallible();
	evaluate_opts.GetSignature()
	    .AddParameter("model", LogicalType::VARCHAR)
	    .AddParameter("y", LogicalType::DOUBLE)
	    .AddParameter("features", LogicalType::LIST(LogicalType::DOUBLE))
	    .AddParameter("options", LogicalType::MAP(LogicalType::VARCHAR, LogicalType::VARCHAR));
	evaluate_set.AddFunction(evaluate_opts);
	loader.RegisterFunction(evaluate_set);

	AggregateFunctionSet evaluate_agg_set("duckboost_evaluate_agg");
	evaluate_agg_set.AddFunction(GetEvaluateAggFunction(false));
	evaluate_agg_set.AddFunction(GetEvaluateAggFunction(true));
	loader.RegisterFunction(evaluate_agg_set);

	ScalarFunctionSet to_sql_set("duckboost_to_sql");
	ScalarFunction to_sql_fun({}, LogicalType::VARCHAR, ToSQLFunction);
	to_sql_fun.SetFallible();
	to_sql_fun.GetSignature()
	    .AddParameter("model", LogicalType::VARCHAR)
	    .AddParameter("table_name", LogicalType::VARCHAR)
	    .AddParameter("feature_columns", LogicalType::LIST(LogicalType::VARCHAR));
	to_sql_set.AddFunction(to_sql_fun);
	ScalarFunction to_sql_opts({}, LogicalType::VARCHAR, ToSQLFunction);
	to_sql_opts.SetFallible();
	to_sql_opts.GetSignature()
	    .AddParameter("model", LogicalType::VARCHAR)
	    .AddParameter("table_name", LogicalType::VARCHAR)
	    .AddParameter("feature_columns", LogicalType::LIST(LogicalType::VARCHAR))
	    .AddParameter("options", LogicalType::MAP(LogicalType::VARCHAR, LogicalType::VARCHAR));
	to_sql_set.AddFunction(to_sql_opts);
	loader.RegisterFunction(to_sql_set);

	ScalarFunctionSet import_set("duckboost_import");
	ScalarFunction import_fun({}, LogicalType::VARCHAR, ImportFunction);
	import_fun.SetFallible();
	import_fun.GetSignature().AddParameter("backend", LogicalType::VARCHAR).AddParameter("dump", LogicalType::VARCHAR);
	import_set.AddFunction(import_fun);
	ScalarFunction import_opts({}, LogicalType::VARCHAR, ImportFunction);
	import_opts.SetFallible();
	import_opts.GetSignature()
	    .AddParameter("backend", LogicalType::VARCHAR)
	    .AddParameter("dump", LogicalType::VARCHAR)
	    .AddParameter("options", LogicalType::MAP(LogicalType::VARCHAR, LogicalType::VARCHAR));
	import_set.AddFunction(import_opts);
	loader.RegisterFunction(import_set);

	TableFunction backends_fun("duckboost_backends", {}, BackendsFunction, BackendsBind, BackendsInit);
	loader.RegisterFunction(backends_fun);

	TableFunction build_info_fun("duckboost_build_info", {}, BuildInfoFunction, BuildInfoBind, BuildInfoInit);
	loader.RegisterFunction(build_info_fun);

	TableFunctionSet importance_set("duckboost_importance");
	{
		FunctionSignature sig;
		sig.AddParameter("model", LogicalType::VARCHAR);
		TableFunction importance_fun(std::move(sig), ImportanceFunction, ImportanceBind, ImportanceInit);
		importance_set.AddFunction(std::move(importance_fun));
	}
	{
		FunctionSignature sig;
		sig.AddParameter("model", LogicalType::VARCHAR);
		sig.AddParameter("options", LogicalType::MAP(LogicalType::VARCHAR, LogicalType::VARCHAR));
		TableFunction importance_opts(std::move(sig), ImportanceFunction, ImportanceBind, ImportanceInit);
		importance_set.AddFunction(std::move(importance_opts));
	}
	loader.RegisterFunction(importance_set);

	RegisterDuckBoostMacros(loader);
}

} // namespace duckboost
} // namespace duckdb
