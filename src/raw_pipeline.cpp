#include "duckboost/raw_pipeline.hpp"
#include "duckboost/functions.hpp"
#include "duckboost/json_util.hpp"
#include "duckboost/model.hpp"

#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/common/sql_identifier.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/vector/flat_vector.hpp"
#include "duckdb/common/vector/vector_writer.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "yyjson.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>

namespace duckdb {
namespace duckboost {

namespace {
using namespace duckdb_yyjson;

const double MISSING = std::numeric_limits<double>::quiet_NaN();

struct RawFeature {
	string name;
	string type;
	vector<string> levels;
	double median = MISSING;

	bool IsCategory() const {
		return type == "VARCHAR";
	}

	double Encode(const Value &value) const {
		if (value.IsNull()) {
			return median;
		}
		if (IsCategory()) {
			auto &category = StringValue::Get(value);
			auto entry = std::lower_bound(levels.begin(), levels.end(), category);
			return entry == levels.end() || *entry != category ? MISSING : static_cast<double>(entry - levels.begin());
		}
		double numeric = value.GetValue<double>();
		return std::isfinite(numeric) ? numeric : median;
	}

	string ExpressionSQL(const string &column) const {
		if (!IsCategory()) {
			auto numeric = "CAST(" + column + " AS DOUBLE)";
			return "CASE WHEN isfinite(" + numeric + ") THEN " + numeric + " ELSE " +
			       (std::isnan(median) ? string("NULL::DOUBLE") : FormatDoubleSQL(median)) + " END";
		}
		if (levels.empty()) {
			return "NULL::DOUBLE";
		}
		// Compare UTF-8 bytes, also when the raw column or session has a collation.
		string sql = "CASE encode(CAST(" + column + " AS VARCHAR))";
		for (idx_t i = 0; i < levels.size(); i++) {
			sql += " WHEN " + Value::BLOB_RAW(levels[i]).ToSQLString() + " THEN " + std::to_string(i) + "::DOUBLE";
		}
		return sql + " ELSE NULL::DOUBLE END";
	}
};

void RequireNamedStruct(const LogicalType &type) {
	if (type.id() != LogicalTypeId::STRUCT) {
		throw InvalidInputException("duckboost: raw features require a named STRUCT, received %s", type.ToString());
	}
}

struct RawPipeline {
	vector<RawFeature> features;
	bool impute = false;
	BoostModel model;

	vector<idx_t> MatchColumns(const LogicalType &type) const {
		RequireNamedStruct(type);
		case_insensitive_map_t<idx_t> names;
		auto &children = StructType::GetChildTypes(type);
		for (idx_t i = 0; i < children.size(); i++) {
			names.emplace(children[i].first.GetIdentifierName(), i);
		}
		vector<idx_t> indices;
		for (auto &f : features) {
			auto match = names.find(f.name);
			if (match == names.end()) {
				throw InvalidInputException("duckboost: missing required raw column '%s'", f.name);
			}
			auto &actual = children[match->second].second;
			if (actual.ToString() != f.type) {
				throw InvalidInputException("duckboost: raw column '%s' expected %s, received %s", f.name, f.type,
				                            actual.ToString());
			}
			indices.push_back(match->second);
		}
		return indices;
	}

	vector<double> Encode(const Value &row, const vector<idx_t> &indices) const {
		auto &children = StructValue::GetChildren(row);
		vector<double> encoded;
		encoded.reserve(features.size());
		for (idx_t i = 0; i < features.size(); i++) {
			encoded.push_back(features[i].Encode(children[indices[i]]));
		}
		return encoded;
	}

	string ToJSON() const;
	static RawPipeline FromJSON(const string &json);
	string ExportSQL(const string &table, const SqlExportOptions &options) const;
};

string JSONText(yyjson_val *value) {
	if (!yyjson_is_str(value)) {
		throw InvalidInputException("duckboost: missing or malformed text in raw pipeline");
	}
	return string(yyjson_get_str(value), yyjson_get_len(value));
}

string TakeJSON(char *buffer) {
	std::unique_ptr<char, decltype(&free)> owned(buffer, free);
	if (!owned) {
		throw InvalidInputException("duckboost: could not serialize raw pipeline");
	}
	return string(owned.get());
}

string RawPipeline::ToJSON() const {
	std::unique_ptr<yyjson_mut_doc, decltype(&yyjson_mut_doc_free)> owner(yyjson_mut_doc_new(nullptr),
	                                                                      yyjson_mut_doc_free);
	if (!owner) {
		throw OutOfMemoryException("duckboost: allocating raw pipeline JSON");
	}
	auto doc = owner.get();
	auto root = yyjson_mut_obj(doc);
	yyjson_mut_doc_set_root(doc, root);
	yyjson_mut_obj_add_uint(doc, root, "duckboost_version", 3);
	yyjson_mut_obj_add_str(doc, root, "artifact_type", "raw_pipeline");
	yyjson_mut_obj_add_str(doc, root, "numeric_imputation", impute ? "median" : "none");
	yyjson_mut_obj_add_str(doc, root, "numeric_missing", "null_or_nonfinite");
	yyjson_mut_obj_add_str(doc, root, "category_missing", "tree_missing");
	yyjson_mut_obj_add_str(doc, root, "category_unseen", "tree_missing");
	auto schema = yyjson_mut_obj_add_arr(doc, root, "features");
	for (auto &f : features) {
		auto field = yyjson_mut_arr_add_obj(doc, schema);
		yyjson_mut_obj_add_strn(doc, field, "name", f.name.data(), f.name.size());
		yyjson_mut_obj_add_strn(doc, field, "type", f.type.data(), f.type.size());
		if (f.IsCategory()) {
			auto levels = yyjson_mut_obj_add_arr(doc, field, "levels");
			for (auto &level : f.levels) {
				yyjson_mut_arr_add_strn(doc, levels, level.data(), level.size());
			}
		} else if (std::isfinite(f.median)) {
			yyjson_mut_obj_add_real(doc, field, "median", f.median);
		} else {
			yyjson_mut_obj_add_null(doc, field, "median");
		}
	}
	auto ensemble = model.ToJSON();
	yyjson_mut_obj_add_val(doc, root, "model", yyjson_mut_rawn(doc, ensemble.data(), ensemble.size()));
	return TakeJSON(yyjson_mut_write(doc, 0, nullptr));
}

RawPipeline RawPipeline::FromJSON(const string &json) {
	std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)> doc(yyjson_read(json.data(), json.size(), 0),
	                                                            yyjson_doc_free);
	auto root = doc ? yyjson_doc_get_root(doc.get()) : nullptr;
	if (yyjson_get_uint(yyjson_obj_get(root, "duckboost_version")) != 3 ||
	    JSONText(yyjson_obj_get(root, "artifact_type")) != "raw_pipeline") {
		throw InvalidInputException("duckboost: expected a v3 raw pipeline from duckboost_train_raw");
	}
	RawPipeline result;
	auto mode = JSONText(yyjson_obj_get(root, "numeric_imputation"));
	result.impute = mode == "median";
	if ((!result.impute && mode != "none") ||
	    JSONText(yyjson_obj_get(root, "numeric_missing")) != "null_or_nonfinite" ||
	    JSONText(yyjson_obj_get(root, "category_missing")) != "tree_missing" ||
	    JSONText(yyjson_obj_get(root, "category_unseen")) != "tree_missing") {
		throw InvalidInputException("duckboost: unsupported raw pipeline missing-value policy");
	}
	auto schema = yyjson_obj_get(root, "features");
	case_insensitive_set_t names;
	size_t i, n;
	yyjson_val *field;
	yyjson_arr_foreach(schema, i, n, field) {
		RawFeature f;
		f.name = JSONText(yyjson_obj_get(field, "name"));
		f.type = JSONText(yyjson_obj_get(field, "type"));
		if (f.name.empty() || !names.insert(f.name).second) {
			throw InvalidInputException("duckboost: invalid or duplicate raw feature name");
		}
		if (f.IsCategory()) {
			auto levels = yyjson_obj_get(field, "levels");
			if (!yyjson_is_arr(levels)) {
				throw InvalidInputException("duckboost: missing category dictionary for '%s'", f.name);
			}
			size_t j, count;
			yyjson_val *level;
			yyjson_arr_foreach(levels, j, count, level) {
				auto text = JSONText(level);
				if (!f.levels.empty() && f.levels.back() >= text) {
					throw InvalidInputException("duckboost: unsorted category dictionary for '%s'", f.name);
				}
				f.levels.push_back(std::move(text));
			}
		} else {
			auto median = yyjson_obj_get(field, "median");
			if (!yyjson_is_null(median)) {
				if (!result.impute || !yyjson_is_num(median) || !std::isfinite(yyjson_get_num(median))) {
					throw InvalidInputException("duckboost: invalid median for '%s'", f.name);
				}
				f.median = yyjson_get_num(median);
			}
		}
		result.features.push_back(std::move(f));
	}
	auto ensemble = yyjson_obj_get(root, "model");
	if (result.features.empty() || !yyjson_is_obj(ensemble)) {
		throw InvalidInputException("duckboost: raw pipeline requires named features and a model");
	}
	result.model = BoostModel::FromJSON(TakeJSON(yyjson_val_write(ensemble, 0, nullptr)));
	if (result.model.n_raw_features != result.features.size()) {
		throw InvalidInputException("duckboost: raw schema width differs from trained model");
	}
	return result;
}

string RawPipeline::ExportSQL(const string &table, const SqlExportOptions &options) const {
	if (table.empty()) {
		throw InvalidInputException("duckboost: table_name must not be empty");
	}
	auto inner_options = options;
	inner_options.keep_columns = "input_row";
	inner_options.prediction_alias = "prediction";
	vector<string> columns;
	string select = "SELECT struct_pack(*COLUMNS(r.*)) AS input_row";
	string check;
	for (idx_t i = 0; i < features.size(); i++) {
		auto &f = features[i];
		auto column = "r." + SQLQuotedIdentifier::ToString(f.name);
		auto name = "f" + std::to_string(i);
		columns.push_back(name);
		select += ", " + f.ExpressionSQL(column) + " AS " + name;
		// Check every required column even in constant or shallow models.
		if (!check.empty()) {
			check += " AND ";
		}
		check += "CASE WHEN typeof(" + column + ") = " + SQLString::ToString(f.type) + " THEN TRUE ELSE error(" +
		         SQLString::ToString("duckboost: raw column '" + f.name + "' expected " + f.type) + ") END";
	}
	auto cte = StringUtil::CIEquals(table, "encoded") ? "encoded_features" : "encoded";
	auto scores = ExportModelSQL(model, cte, columns, inner_options);
	string projection;
	case_insensitive_set_t output_names = {options.prediction_alias};
	auto keep = options.keep_columns;
	StringUtil::Trim(keep);
	if (keep == "*") {
		projection = "unnest(s.input_row), ";
	} else if (!keep.empty()) {
		for (auto name : StringUtil::Split(keep, ',')) {
			StringUtil::Trim(name);
			if (name.empty() || !output_names.insert(name).second) {
				throw InvalidInputException("duckboost: invalid or duplicate keep_columns name '%s'", name);
			}
			auto id = SQLQuotedIdentifier::ToString(name);
			projection += "s.input_row." + id + " AS " + id + ", ";
		}
	}
	projection += "s.prediction AS " + SQLQuotedIdentifier::ToString(options.prediction_alias);
	if (options.proba) {
		idx_t width = model.task == BoostTask::BINARY ? 2 : model.n_classes;
		for (idx_t i = 0; i < width; i++) {
			auto name = "proba_" + std::to_string(i);
			if (!output_names.insert(name).second) {
				throw InvalidInputException("duckboost: output column collides with %s", name);
			}
			projection += ", s." + name;
		}
	}
	return "WITH " + string(cte) + " AS (" + select + " FROM " + SQLQuotedIdentifier::ToString(table) + " AS r WHERE " +
	       check + ") SELECT " + projection + " FROM (" + scores + ") AS s";
}

struct RawRow {
	double target;
	Value raw;
	bool validation;
};

struct RawFit {
	vector<RawRow> rows;
	unordered_map<string, string> options;

	RawPipeline Fit();
};

RawPipeline RawFit::Fit() {
	RawPipeline pipeline;
	auto train_options = options;
	for (auto it = train_options.begin(); it != train_options.end(); ++it) {
		if (StringUtil::CIEquals(it->first, "numeric_imputation")) {
			auto mode = StringUtil::Lower(it->second);
			if (mode != "none" && mode != "median") {
				throw InvalidInputException("duckboost: numeric_imputation must be none or median");
			}
			pipeline.impute = mode == "median";
			train_options.erase(it);
			break;
		}
	}
	auto training = TrainOptions::FromMap(train_options);
	if (!training.feature_names.empty() || !training.categorical_features.empty()) {
		throw InvalidInputException("duckboost: raw feature names and categorical features come from the named STRUCT");
	}
	// Canonical order makes the implicit seed-based holdout independent of aggregate combine order.
	std::sort(rows.begin(), rows.end(), [](const RawRow &a, const RawRow &b) {
		if (a.target != b.target) {
			return a.target < b.target;
		}
		if (a.validation != b.validation) {
			return a.validation < b.validation;
		}
		return a.raw < b.raw;
	});
	vector<double> y;
	vector<bool> mask;
	y.reserve(rows.size());
	mask.reserve(rows.size());
	for (auto &r : rows) {
		y.push_back(r.target);
		mask.push_back(r.validation);
	}
	TrainingRows split;
	ResolveTrainValidRows(rows.size(), training, mask, split.train_rows, split.valid_rows);
	auto &schema = StructType::GetChildTypes(rows.front().raw.type());
	for (idx_t col = 0; col < schema.size(); col++) {
		RawFeature f;
		f.name = schema[col].first.GetIdentifierName();
		f.type = schema[col].second.ToString();
		if (!f.IsCategory() && !schema[col].second.IsNumeric()) {
			throw InvalidInputException("duckboost: raw column '%s' must be numeric or VARCHAR, received %s", f.name,
			                            f.type);
		}
		training.feature_names.push_back("f" + std::to_string(col));
		if (f.IsCategory()) {
			training.categorical_features.push_back(std::to_string(col));
			for (auto row : split.train_rows) {
				auto &value = StructValue::GetChildren(rows[row].raw)[col];
				if (!value.IsNull()) {
					f.levels.push_back(StringValue::Get(value));
				}
			}
			std::sort(f.levels.begin(), f.levels.end());
			f.levels.erase(std::unique(f.levels.begin(), f.levels.end()), f.levels.end());
		} else if (pipeline.impute) {
			vector<double> values;
			for (auto row : split.train_rows) {
				auto &value = StructValue::GetChildren(rows[row].raw)[col];
				if (!value.IsNull()) {
					auto number = value.GetValue<double>();
					if (std::isfinite(number)) {
						values.push_back(number);
					}
				}
			}
			if (!values.empty()) {
				std::sort(values.begin(), values.end());
				auto a = values[(values.size() - 1) / 2];
				auto b = values[values.size() / 2];
				f.median = (a < 0 && b > 0) ? (a + b) / 2 : a + (b - a) / 2;
			}
		}
		pipeline.features.push_back(std::move(f));
	}
	auto indices = pipeline.MatchColumns(rows.front().raw.type());
	vector<vector<double>> x;
	x.reserve(rows.size());
	for (auto &row : rows) {
		x.push_back(pipeline.Encode(row.raw, indices));
	}
	pipeline.model = TrainModel(y, x, training, {}, split);
	return pipeline;
}

struct RawState {
	RawFit *data;
};
struct RawOperation {
	template <class STATE>
	static void Initialize(STATE &state) {
		state.data = nullptr;
	}
	template <class STATE>
	static void Destroy(STATE &state, AggregateInputData &) {
		delete state.data;
	}
};

void RawUpdate(Vector inputs[], AggregateInputData &, idx_t n_inputs, Vector &state_vector, idx_t count) {
	RequireNamedStruct(inputs[1].GetType());
	auto has_mask = n_inputs > 2 && inputs[2].GetType().id() == LogicalTypeId::BOOLEAN;
	auto has_options = inputs[n_inputs - 1].GetType().id() == LogicalTypeId::MAP;
	UnifiedVectorFormat format;
	state_vector.ToUnifiedFormat(format);
	auto states = UnifiedVectorFormat::GetData<RawState *>(format);
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[format.sel->get_index(i)];
		if (!state.data) {
			state.data = new RawFit();
			if (has_options) {
				state.data->options = MapVectorToOptions(inputs[n_inputs - 1], i);
			}
		} else if (has_options && (i == 0 || inputs[n_inputs - 1].GetVectorType() != VectorType::CONSTANT_VECTOR) &&
		           state.data->options != MapVectorToOptions(inputs[n_inputs - 1], i)) {
			throw InvalidInputException("duckboost: raw training options must be constant per fit");
		}
		auto target = inputs[0].GetValue(i);
		auto raw = inputs[1].GetValue(i);
		if (target.IsNull() || !std::isfinite(target.GetValue<double>()) || raw.IsNull()) {
			throw InvalidInputException("duckboost: raw training requires a finite target and a non-NULL STRUCT");
		}
		bool validation = false;
		if (has_mask) {
			auto flag = inputs[2].GetValue(i);
			validation = !flag.IsNull() && flag.GetValue<bool>();
		}
		state.data->rows.push_back({target.GetValue<double>(), std::move(raw), validation});
	}
}

void RawCombine(Vector &source, Vector &target, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat src, dst;
	source.ToUnifiedFormat(src);
	target.ToUnifiedFormat(dst);
	auto sources = UnifiedVectorFormat::GetData<RawState *>(src);
	auto targets = UnifiedVectorFormat::GetData<RawState *>(dst);
	for (idx_t i = 0; i < count; i++) {
		auto &from = *sources[src.sel->get_index(i)];
		auto &to = *targets[dst.sel->get_index(i)];
		if (!from.data) {
			continue;
		}
		if (!to.data) {
			to.data = new RawFit(*from.data);
		} else {
			if (to.data->options != from.data->options) {
				throw InvalidInputException("duckboost: raw training options must be constant per fit");
			}
			to.data->rows.insert(to.data->rows.end(), from.data->rows.begin(), from.data->rows.end());
		}
	}
}

void RawFinalize(Vector &state_vector, AggregateFinalizeInputData &, Vector &result, idx_t count, idx_t offset) {
	UnifiedVectorFormat format;
	state_vector.ToUnifiedFormat(format);
	auto states = UnifiedVectorFormat::GetData<RawState *>(format);
	auto writer = FlatVector::Writer<string_t>(result, count, offset);
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[format.sel->get_index(i)];
		if (!state.data || state.data->rows.empty()) {
			writer.WriteNull();
			continue;
		}
		auto model = state.data->Fit();
		writer.WriteValue(StringVector::AddString(result, model.ToJSON()));
	}
}

void RawPredict(DataChunk &args, ExpressionState &, Vector &result) {
	RawPipeline model;
	string cached_json;
	vector<idx_t> indices;
	bool valid = false;
	auto writer = FlatVector::Writer<double>(result, args.size());
	for (idx_t i = 0; i < args.size(); i++) {
		auto json = args.data[0].GetValue(i);
		auto row = args.data[1].GetValue(i);
		if (json.IsNull() || row.IsNull()) {
			writer.WriteNull();
			continue;
		}
		auto &text = StringValue::Get(json);
		if (!valid || cached_json != text) {
			model = RawPipeline::FromJSON(text);
			indices = model.MatchColumns(row.type());
			cached_json = text;
			valid = true;
		}
		writer.WriteValue(model.model.Predict(model.Encode(row, indices)));
	}
}

void RawToSQL(DataChunk &args, ExpressionState &, Vector &result) {
	auto writer = FlatVector::Writer<string_t>(result, args.size());
	for (idx_t i = 0; i < args.size(); i++) {
		auto json = args.data[0].GetValue(i);
		auto table = args.data[1].GetValue(i);
		if (json.IsNull() || table.IsNull()) {
			writer.WriteNull();
			continue;
		}
		auto model = RawPipeline::FromJSON(StringValue::Get(json));
		auto options = args.ColumnCount() > 2 ? SqlExportOptions::FromMap(MapVectorToOptions(args.data[2], i))
		                                      : SqlExportOptions();
		writer.WriteValue(StringVector::AddString(result, model.ExportSQL(StringValue::Get(table), options)));
	}
}

} // namespace

void RegisterRawFunctions(ExtensionLoader &loader) {
	auto opts = LogicalType::MAP(LogicalType::VARCHAR, LogicalType::VARCHAR);
	AggregateFunctionSet train("duckboost_train_raw");
	for (bool mask : {false, true}) {
		for (bool options : {false, true}) {
			vector<LogicalType> args = {LogicalType::DOUBLE, LogicalType::ANY};
			if (mask) {
				args.push_back(LogicalType::BOOLEAN);
			}
			if (options) {
				args.push_back(opts);
			}
			AggregateFunction fn(
			    args, LogicalType::VARCHAR, AggregateFunction::StateSize<RawState>,
			    AggregateFunction::StateInitialize<RawState, RawOperation, AggregateDestructorType::LEGACY>, RawUpdate,
			    RawCombine, RawFinalize, AggregateFunction::NoClusterUpdate(), AggregateFunction::NoBind(),
			    AggregateFunction::StateDestroy<RawState, RawOperation>);
			fn.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
			fn.GetSignature().GetParameter(0).SetName("y");
			fn.GetSignature().GetParameter(1).SetName("raw");
			if (mask) {
				fn.GetSignature().GetParameter(2).SetName("is_validation");
			}
			if (options) {
				fn.GetSignature().GetParameter(args.size() - 1).SetName("options");
			}
			train.AddFunction(fn);
		}
	}
	loader.RegisterFunction(train);
	ScalarFunction predict("duckboost_predict_raw", {LogicalType::VARCHAR, LogicalType::ANY}, LogicalType::DOUBLE,
	                       RawPredict);
	predict.SetFallible();
	predict.GetSignature().GetParameter(0).SetName("model");
	predict.GetSignature().GetParameter(1).SetName("raw");
	loader.RegisterFunction(predict);
	ScalarFunctionSet exporter("duckboost_to_sql_raw");
	for (bool options : {false, true}) {
		vector<LogicalType> args = {LogicalType::VARCHAR, LogicalType::VARCHAR};
		if (options) {
			args.push_back(opts);
		}
		ScalarFunction fn(args, LogicalType::VARCHAR, RawToSQL);
		fn.SetFallible();
		fn.GetSignature().GetParameter(0).SetName("model");
		fn.GetSignature().GetParameter(1).SetName("table_name");
		if (options) {
			fn.GetSignature().GetParameter(2).SetName("options");
		}
		exporter.AddFunction(fn);
	}
	loader.RegisterFunction(exporter);
}

} // namespace duckboost
} // namespace duckdb
