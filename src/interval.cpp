#include "duckboost/functions.hpp"
#include "duckboost/json_util.hpp"
#include "duckboost/model.hpp"

#include "duckdb/common/string_util.hpp"
#include "duckdb/common/vector/flat_vector.hpp"
#include "duckdb/common/vector/struct_vector.hpp"
#include "duckdb/common/vector/vector_writer.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/common/sql_identifier.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace duckdb {
namespace duckboost {
namespace {

using QuantileFits = std::array<BoostModel, 3>;
using QuantileJSON = std::array<string, 3>;

void ValidateFits(const QuantileFits &fits) {
	for (auto &fit : fits) {
		if (fit.task != BoostTask::REGRESSION || fit.loss != RegressionLoss::QUANTILE || !fit.ctr_features.empty()) {
			throw InvalidInputException("duckboost: expected three quantile regression fits over the input features");
		}
		if (fit.n_features != fits[0].n_features || fit.feature_names != fits[0].feature_names) {
			throw InvalidInputException("duckboost: interval fits must have the same feature names and order");
		}
	}
	if (!(fits[0].objective_alpha > 0 && fits[0].objective_alpha < 0.5 && fits[1].objective_alpha == 0.5 &&
	      fits[2].objective_alpha > 0.5 && fits[2].objective_alpha < 1)) {
		throw InvalidInputException("duckboost: expected lower, median (0.5) and upper quantile fits in that order");
	}
}

void ValidateAlpha(double alpha) {
	if (!(alpha > 0 && alpha < 1)) {
		throw InvalidInputException("duckboost: interval alpha must be strictly between 0 and 1");
	}
}

idx_t CalibrationRank(idx_t n, double alpha) {
	const double rank = std::ceil((static_cast<double>(n) + 1) * (1 - alpha));
	if (rank > static_cast<double>(n) || n == 0) {
		throw InvalidInputException("duckboost: insufficient calibration rows for the requested coverage");
	}
	return static_cast<idx_t>(rank);
}

std::array<double, 3> OrderedQuantiles(const QuantileFits &fits, const vector<double> &features) {
	std::array<double, 3> predictions;
	for (idx_t i = 0; i < fits.size(); ++i) {
		predictions[i] = fits[i].Predict(features);
		if (!std::isfinite(predictions[i])) {
			throw InvalidInputException("duckboost: interval quantile predictions must be finite");
		}
	}
	std::sort(predictions.begin(), predictions.end());
	return predictions;
}

struct IntervalModel {
	QuantileFits fits;
	double alpha = std::numeric_limits<double>::quiet_NaN();
	double correction = std::numeric_limits<double>::quiet_NaN();
	idx_t n_calibration = 0;

	string ToJSON() const {
		return "{\"duckboost_version\":3,\"artifact_type\":\"calibrated_interval\",\"crossing\":\"rearrange\","
		       "\"alpha\":" +
		       FormatDouble(alpha) + ",\"n_calibration\":" + std::to_string(n_calibration) +
		       ",\"correction\":" + FormatDouble(correction) + ",\"quantiles\":[" + fits[0].ToJSON() + "," +
		       fits[1].ToJSON() + "," + fits[2].ToJSON() + "]}";
	}

	static IntervalModel FromJSON(const string &json) {
		IntervalModel model;
		JsonParser parser(json);
		parser.Expect('{');
		double version = 0;
		string kind, crossing;
		do {
			auto field = parser.ParseString();
			parser.Expect(':');
			if (field == "duckboost_version") {
				version = parser.ParseNumber();
			} else if (field == "artifact_type") {
				kind = parser.ParseString();
			} else if (field == "crossing") {
				crossing = parser.ParseString();
			} else if (field == "alpha") {
				model.alpha = parser.ParseNumber();
			} else if (field == "correction") {
				model.correction = parser.ParseNumber();
			} else if (field == "n_calibration") {
				double n = parser.ParseNumber();
				if (!(n >= 1 && n < static_cast<double>(std::numeric_limits<idx_t>::max()) && n == std::floor(n))) {
					throw InvalidInputException("duckboost: invalid interval calibration count");
				}
				model.n_calibration = static_cast<idx_t>(n);
			} else if (field == "quantiles") {
				parser.Expect('[');
				for (idx_t i = 0; i < model.fits.size(); ++i) {
					if (i) {
						parser.Expect(',');
					}
					const auto start = parser.pos;
					parser.SkipValue();
					model.fits[i] = BoostModel::FromJSON(json.substr(start, parser.pos - start));
				}
				parser.Expect(']');
			} else {
				throw InvalidInputException("duckboost: unknown interval field '%s'", field);
			}
		} while (parser.TryConsume(','));
		parser.Expect('}');
		if (!parser.AtEnd() || kind != "calibrated_interval" || version != 3 || crossing != "rearrange" ||
		    !std::isfinite(model.correction)) {
			throw InvalidInputException("duckboost: expected a format 3 calibrated interval model");
		}
		ValidateAlpha(model.alpha);
		CalibrationRank(model.n_calibration, model.alpha);
		ValidateFits(model.fits);
		return model;
	}
};

struct CalibrationData {
	QuantileJSON json;
	IntervalModel model;
	vector<double> scores;

	CalibrationData(const std::array<string_t, 3> &input, double alpha) {
		ValidateAlpha(alpha);
		model.alpha = alpha;
		for (idx_t i = 0; i < input.size(); ++i) {
			json[i] = input[i].GetString();
			model.fits[i] = BoostModel::FromJSON(json[i]);
		}
		ValidateFits(model.fits);
	}
};

struct CalibrationState {
	CalibrationData *data;
};

struct CalibrationOperation {
	template <class STATE>
	static void Initialize(STATE &state) {
		state.data = nullptr;
	}
	template <class STATE>
	static void Destroy(STATE &state, AggregateInputData &) {
		delete state.data;
		state.data = nullptr;
	}
};

void CalibrateUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat state_format;
	state_vector.ToUnifiedFormat(state_format);
	auto states = UnifiedVectorFormat::GetData<CalibrationState *>(state_format);
	std::array<UnifiedVectorFormat, 5> formats;
	for (idx_t j = 0; j < formats.size(); ++j) {
		inputs[j].ToUnifiedFormat(formats[j]);
	}
	for (idx_t i = 0; i < count; ++i) {
		auto &state = *states[state_format.sel->get_index(i)];
		std::array<idx_t, 5> indices;
		for (idx_t j = 0; j < formats.size(); ++j) {
			indices[j] = formats[j].sel->get_index(i);
			if (!formats[j].validity.RowIsValid(indices[j])) {
				throw InvalidInputException("duckboost: interval fits, alpha and calibration target must not be NULL");
			}
		}
		std::array<string_t, 3> json;
		for (idx_t j = 0; j < json.size(); ++j) {
			json[j] = UnifiedVectorFormat::GetData<string_t>(formats[j])[indices[j]];
		}
		const double alpha = UnifiedVectorFormat::GetData<double>(formats[3])[indices[3]];
		if (!state.data) {
			state.data = new CalibrationData(json, alpha);
		} else if (alpha != state.data->model.alpha || json[0] != string_t(state.data->json[0]) ||
		           json[1] != string_t(state.data->json[1]) || json[2] != string_t(state.data->json[2])) {
			throw InvalidInputException(
			    "duckboost: quantile fits and alpha must be constant within a calibration group");
		}
		const double y = UnifiedVectorFormat::GetData<double>(formats[4])[indices[4]];
		auto quantiles = OrderedQuantiles(state.data->model.fits, ReadFeatureList(inputs[5], i));
		double score = std::max(quantiles[0] - y, y - quantiles[2]);
		if (!std::isfinite(score)) {
			throw InvalidInputException("duckboost: calibration targets and scores must be finite");
		}
		state.data->scores.push_back(score);
	}
}

void CalibrateCombine(Vector &source, Vector &target, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat src_format, dst_format;
	source.ToUnifiedFormat(src_format);
	target.ToUnifiedFormat(dst_format);
	auto sources = UnifiedVectorFormat::GetData<CalibrationState *>(src_format);
	auto targets = UnifiedVectorFormat::GetData<CalibrationState *>(dst_format);
	for (idx_t i = 0; i < count; ++i) {
		auto src = sources[src_format.sel->get_index(i)]->data;
		auto &dst = targets[dst_format.sel->get_index(i)]->data;
		if (!src) {
			continue;
		}
		if (!dst) {
			dst = new CalibrationData(*src);
			continue;
		}
		if (src->json != dst->json || src->model.alpha != dst->model.alpha) {
			throw InvalidInputException(
			    "duckboost: quantile fits and alpha must be constant within a calibration group");
		}
		dst->scores.insert(dst->scores.end(), src->scores.begin(), src->scores.end());
	}
}

void CalibrateFinalize(Vector &state_vector, AggregateFinalizeInputData &, Vector &result, idx_t count, idx_t offset) {
	UnifiedVectorFormat format;
	state_vector.ToUnifiedFormat(format);
	auto states = UnifiedVectorFormat::GetData<CalibrationState *>(format);
	auto writer = FlatVector::Writer<string_t>(result, count, offset);
	for (idx_t i = 0; i < count; ++i) {
		auto data = states[format.sel->get_index(i)]->data;
		if (!data) {
			throw InvalidInputException("duckboost: insufficient calibration rows for the requested coverage");
		}
		auto &scores = data->scores;
		const auto rank = CalibrationRank(scores.size(), data->model.alpha);
		auto selected = scores.begin() + (rank - 1);
		std::nth_element(scores.begin(), selected, scores.end());
		data->model.correction = *selected;
		data->model.n_calibration = scores.size();
		writer.WriteValue(StringVector::AddString(result, data->model.ToJSON()));
	}
}

void PredictInterval(DataChunk &args, ExpressionState &, Vector &result) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto &bounds = StructVector::GetEntries(result);
	UnifiedVectorFormat format;
	args.data[0].ToUnifiedFormat(format);
	auto input = UnifiedVectorFormat::GetData<string_t>(format);
	IntervalModel model;
	string_t previous;
	bool parsed = false;
	for (idx_t i = 0; i < args.size(); ++i) {
		const auto index = format.sel->get_index(i);
		if (!format.validity.RowIsValid(index)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		if (!parsed || input[index] != previous) {
			model = IntervalModel::FromJSON(input[index].GetString());
			previous = input[index];
			parsed = true;
		}
		auto ordered = OrderedQuantiles(model.fits, ReadFeatureList(args.data[1], i));
		ordered[0] -= model.correction;
		ordered[2] += model.correction;
		for (idx_t j = 0; j < bounds.size(); ++j) {
			FlatVector::GetDataMutable<double>(bounds[j])[i] = ordered[j];
		}
	}
}

string IntervalSQL(const IntervalModel &model, const string &table, const vector<string> &features,
                   const vector<string> &keep) {
	if (table.empty() || features.size() != model.fits[0].n_features) {
		throw InvalidInputException("duckboost: interval export requires a table and one column name per feature");
	}
	string prefix;
	unordered_set<string> reserved {"lower", "median", "upper", "__duckboost_quantiles"};
	for (auto &name : keep) {
		if (name.empty() || !reserved.insert(StringUtil::Lower(name)).second) {
			throw InvalidInputException("duckboost: empty, duplicate or reserved interval output column '%s'", name);
		}
		prefix += SQLQuotedIdentifier(name) + ", ";
	}
	// The inner projection evaluates all fits on the same row and preserves caller-supplied keys.
	string inner = "SELECT " + prefix + "list_sort([";
	for (idx_t i = 0; i < model.fits.size(); ++i) {
		if (i) {
			inner += ", ";
		}
		inner += "(" + ExportScoreExpression(model.fits[i], features) + ")";
	}
	inner += "]) AS __duckboost_quantiles FROM " + SQLQuotedIdentifier(table);
	const auto correction = FormatDoubleSQL(model.correction);
	return "SELECT " + prefix + "(__duckboost_quantiles[1] - (" + correction +
	       ")) AS lower, __duckboost_quantiles[2] AS median, (__duckboost_quantiles[3] + (" + correction +
	       ")) AS upper FROM (" + inner + ") AS bands";
}

void IntervalToSQL(DataChunk &args, ExpressionState &, Vector &result) {
	auto writer = FlatVector::Writer<string_t>(result, args.size());
	for (idx_t i = 0; i < args.size(); ++i) {
		auto json = args.data[0].GetValue(i);
		auto table = args.data[1].GetValue(i);
		if (json.IsNull() || table.IsNull()) {
			writer.WriteNull();
			continue;
		}
		auto model = IntervalModel::FromJSON(StringValue::Get(json));
		auto features = ReadVarcharList(args.data[2], i);
		auto keep = args.ColumnCount() == 4 ? ReadVarcharList(args.data[3], i) : vector<string>();
		writer.WriteValue(StringVector::AddString(result, IntervalSQL(model, StringValue::Get(table), features, keep)));
	}
}

} // namespace

void RegisterIntervalFunctions(ExtensionLoader &loader) {
	auto numeric_list = LogicalType::LIST(LogicalType::DOUBLE);
	auto names = LogicalType::LIST(LogicalType::VARCHAR);
	auto interval = LogicalType::STRUCT(
	    {{"lower", LogicalType::DOUBLE}, {"median", LogicalType::DOUBLE}, {"upper", LogicalType::DOUBLE}});
	AggregateFunction calibrate(
	    "duckboost_calibrate_interval",
	    {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::DOUBLE, LogicalType::DOUBLE,
		 numeric_list},
	    LogicalType::VARCHAR, AggregateFunction::StateSize<CalibrationState>,
	    AggregateFunction::StateInitialize<CalibrationState, CalibrationOperation, AggregateDestructorType::LEGACY>,
	    CalibrateUpdate, CalibrateCombine, CalibrateFinalize, AggregateFunction::NoClusterUpdate(),
	    AggregateFunction::NoBind(), AggregateFunction::StateDestroy<CalibrationState, CalibrationOperation>);
	const char *parameters[] = {"lower_fit", "median_fit", "upper_fit", "alpha", "y", "features"};
	for (idx_t i = 0; i < 6; ++i) {
		calibrate.GetSignature().GetParameter(i).SetName(parameters[i]);
	}
	calibrate.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	loader.RegisterFunction(calibrate);
	ScalarFunction predict("duckboost_predict_interval", {}, interval, PredictInterval);
	predict.GetSignature().AddParameter("model", LogicalType::VARCHAR).AddParameter("features", numeric_list);
	predict.SetFallible();
	loader.RegisterFunction(predict);

	ScalarFunctionSet export_set("duckboost_interval_to_sql");
	ScalarFunction export_sql({}, LogicalType::VARCHAR, IntervalToSQL);
	export_sql.GetSignature()
	    .AddParameter("model", LogicalType::VARCHAR)
	    .AddParameter("table_name", LogicalType::VARCHAR)
	    .AddParameter("feature_columns", names);
	export_sql.SetFallible();
	export_set.AddFunction(export_sql);
	export_sql.GetSignature().AddParameter("keep_columns", names);
	export_set.AddFunction(export_sql);
	loader.RegisterFunction(export_set);
}

} // namespace duckboost
} // namespace duckdb
