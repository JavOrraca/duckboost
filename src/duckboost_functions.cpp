#include "duckboost/functions.hpp"
#include "duckboost/import.hpp"
#include "duckboost/model.hpp"
#include "duckboost/native_train.hpp"

#include "duckdb/catalog/default/default_table_functions.hpp"
#include "duckdb/common/constants.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/vector/flat_vector.hpp"
#include "duckdb/common/vector/list_vector.hpp"
#include "duckdb/common/vector/map_vector.hpp"
#include "duckdb/common/vector/vector_writer.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"

#include <cmath>

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
			throw InvalidInputException("duckboost: feature values cannot be NULL");
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

struct TrainDataset {
	vector<double> y;
	vector<vector<double>> x;
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
		if (input_count >= 3 && !state.data->options_set) {
			state.data->options = TrainOptions::FromMap(MapVectorToOptions(inputs[2], i));
			state.data->options_set = true;
		}
		state.data->y.push_back(y_data[y_idx]);
		state.data->x.push_back(ReadFeatureList(x_vector, i));
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
		auto model = TrainModel(state.data->y, state.data->x, options);
		writer.WriteValue(StringVector::AddString(result, model.ToJSON()));
	}
}

AggregateFunction GetTrainFunction(bool with_options) {
	auto feature_type = LogicalType::LIST(LogicalType::DOUBLE);
	auto options_type = LogicalType::MAP(LogicalType::VARCHAR, LogicalType::VARCHAR);
	vector<LogicalType> args = {LogicalType::DOUBLE, feature_type};
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
	if (with_options) {
		fun.GetSignature().GetParameter(2).SetName("options");
	}
	return fun;
}

void PredictFunction(DataChunk &args, ExpressionState &, Vector &result) {
	auto count = args.size();
	UnifiedVectorFormat model_format;
	args.data[0].ToUnifiedFormat(model_format);
	auto model_data = UnifiedVectorFormat::GetData<string_t>(model_format);
	auto writer = FlatVector::Writer<double>(result, count);
	for (idx_t i = 0; i < count; i++) {
		auto model_idx = model_format.sel->get_index(i);
		if (!model_format.validity.RowIsValid(model_idx)) {
			writer.WriteNull();
			continue;
		}
		auto model = BoostModel::FromJSON(model_data[model_idx].GetString());
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
	for (idx_t i = 0; i < count; i++) {
		auto model_idx = model_format.sel->get_index(i);
		if (!model_format.validity.RowIsValid(model_idx)) {
			writer.WriteNull();
			continue;
		}
		auto model = BoostModel::FromJSON(model_data[model_idx].GetString());
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

	// Row-wise evaluate is awkward; support a convenience aggregate-style path via lists? For MVP, compute
	// one-row metrics when users pass scalar model + scalar y + features, and document duckboost_evaluate_agg.
	for (idx_t i = 0; i < count; i++) {
		auto model_idx = model_format.sel->get_index(i);
		auto y_idx = y_format.sel->get_index(i);
		if (!model_format.validity.RowIsValid(model_idx) || !y_format.validity.RowIsValid(y_idx)) {
			writer.WriteNull();
			continue;
		}
		auto model = BoostModel::FromJSON(model_data[model_idx].GetString());
		auto features = ReadFeatureList(args.data[2], i);
		EvalOptions options;
		options.metric = "auto";
		if (args.ColumnCount() >= 4) {
			options = EvalOptions::FromMap(MapVectorToOptions(args.data[3], i));
		}
		// Per-row absolute/squared error helpers for streaming metrics.
		auto pred = model.Predict(features);
		auto y = y_data[y_idx];
		if (options.metric == "mae" || (options.metric == "auto" && model.task == BoostTask::REGRESSION && false)) {
			writer.WriteValue(std::fabs(pred - y));
		} else if (options.metric == "accuracy" ||
		           (options.metric == "auto" &&
		            (model.task == BoostTask::BINARY || model.task == BoostTask::MULTICLASS))) {
			if (model.task == BoostTask::MULTICLASS) {
				writer.WriteValue(pred == y ? 1.0 : 0.0);
			} else {
				writer.WriteValue((pred >= 0.5 ? 1.0 : 0.0) == y ? 1.0 : 0.0);
			}
		} else if (options.metric == "logloss") {
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
		} else {
			// default: squared error contribution (mean externally for RMSE)
			auto err = pred - y;
			writer.WriteValue(err * err);
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
	{DEFAULT_SCHEMA, "duckboost_fit", {"source", "y", "features", nullptr}, {{"options", "MAP {}"}, {nullptr, nullptr}}, R"(
SELECT duckboost_train(y, features, options) AS model
FROM query_table(source::VARCHAR)
)"},
	{DEFAULT_SCHEMA, "duckboost_score", {"model", "source", "features", nullptr}, {{nullptr, nullptr}}, R"(
SELECT *, duckboost_predict(model, features) AS prediction
FROM query_table(source::VARCHAR)
)"},
	{nullptr, nullptr, {nullptr}, {{nullptr, nullptr}}, nullptr}
};
// clang-format on

void RegisterDuckBoostMacros(ExtensionLoader &loader) {
	ParserOptions parser_options;
	for (idx_t index = 0; duckboost_table_macros[index].name != nullptr; index++) {
		auto info = DefaultTableFunctionGenerator::CreateTableMacroInfo(duckboost_table_macros[index], parser_options);
		loader.RegisterFunction(*info);
	}
}

void RegisterDuckBoostFunctions(ExtensionLoader &loader) {
	AggregateFunctionSet train_set("duckboost_train");
	train_set.AddFunction(GetTrainFunction(false));
	train_set.AddFunction(GetTrainFunction(true));
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
	evaluate_fun.GetSignature()
	    .AddParameter("model", LogicalType::VARCHAR)
	    .AddParameter("y", LogicalType::DOUBLE)
	    .AddParameter("features", LogicalType::LIST(LogicalType::DOUBLE));
	evaluate_set.AddFunction(evaluate_fun);
	ScalarFunction evaluate_opts({}, LogicalType::DOUBLE, EvaluateFunction);
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

	RegisterDuckBoostMacros(loader);
}

} // namespace duckboost
} // namespace duckdb
