#include "duckboost/model.hpp"
#include "duckboost/native_train.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/numeric_utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace duckdb {
namespace duckboost {

namespace {

struct SplitCandidate {
	idx_t feature = 0;
	double threshold = 0;
	double gain = -std::numeric_limits<double>::infinity();
	double left_value = 0;
	double right_value = 0;
};

double Mean(const vector<double> &values) {
	if (values.empty()) {
		return 0;
	}
	double sum = 0;
	for (auto v : values) {
		sum += v;
	}
	return sum / static_cast<double>(values.size());
}

vector<double> UniqueSorted(vector<double> values) {
	std::sort(values.begin(), values.end());
	values.erase(std::unique(values.begin(), values.end()), values.end());
	return values;
}

vector<double> CandidateThresholds(const vector<double> &column, idx_t max_bins) {
	auto unique = UniqueSorted(column);
	if (unique.size() <= 1) {
		return {};
	}
	if (unique.size() <= max_bins + 1) {
		vector<double> thresholds;
		for (idx_t i = 0; i + 1 < unique.size(); i++) {
			thresholds.push_back(0.5 * (unique[i] + unique[i + 1]));
		}
		return thresholds;
	}
	vector<double> thresholds;
	for (idx_t b = 1; b <= max_bins; b++) {
		double q = static_cast<double>(b) / static_cast<double>(max_bins + 1);
		idx_t idx = static_cast<idx_t>(q * static_cast<double>(unique.size() - 1));
		if (idx + 1 < unique.size()) {
			thresholds.push_back(0.5 * (unique[idx] + unique[idx + 1]));
		}
	}
	return UniqueSorted(std::move(thresholds));
}

double LeafValueRegression(const vector<double> &residuals, const vector<idx_t> &rows) {
	if (rows.empty()) {
		return 0;
	}
	double sum = 0;
	for (auto row : rows) {
		sum += residuals[row];
	}
	return sum / static_cast<double>(rows.size());
}

double LeafValueBinary(const vector<double> &gradients, const vector<double> &hessians, const vector<idx_t> &rows) {
	double g = 0;
	double h = 0;
	for (auto row : rows) {
		g += gradients[row];
		h += hessians[row];
	}
	if (h <= 1e-12) {
		return 0;
	}
	return -g / h;
}

double SplitGainRegression(const vector<double> &residuals, const vector<idx_t> &left, const vector<idx_t> &right) {
	vector<idx_t> all = left;
	all.insert(all.end(), right.begin(), right.end());
	auto left_mean = LeafValueRegression(residuals, left);
	auto right_mean = LeafValueRegression(residuals, right);
	auto parent = LeafValueRegression(residuals, all);
	double left_sse = 0;
	double right_sse = 0;
	double parent_sse = 0;
	for (auto row : left) {
		auto left_err = residuals[row] - left_mean;
		auto parent_err = residuals[row] - parent;
		left_sse += left_err * left_err;
		parent_sse += parent_err * parent_err;
	}
	for (auto row : right) {
		auto right_err = residuals[row] - right_mean;
		auto parent_err = residuals[row] - parent;
		right_sse += right_err * right_err;
		parent_sse += parent_err * parent_err;
	}
	return parent_sse - (left_sse + right_sse);
}

double SplitGainBinary(const vector<double> &gradients, const vector<double> &hessians, const vector<idx_t> &left,
                       const vector<idx_t> &right) {
	auto score = [&](const vector<idx_t> &rows) {
		double g = 0;
		double h = 0;
		for (auto row : rows) {
			g += gradients[row];
			h += hessians[row];
		}
		if (h <= 1e-12) {
			return 0.0;
		}
		return (g * g) / h;
	};
	vector<idx_t> all = left;
	all.insert(all.end(), right.begin(), right.end());
	return score(left) + score(right) - score(all);
}

SplitCandidate FindBestSplitRegression(const vector<vector<double>> &x, const vector<double> &residuals,
                                       const vector<idx_t> &rows, const TrainOptions &options) {
	SplitCandidate best;
	if (rows.size() < 2 * options.min_samples_leaf) {
		return best;
	}
	idx_t n_features = x.empty() ? 0 : x[0].size();
	for (idx_t f = 0; f < n_features; f++) {
		vector<double> column;
		column.reserve(rows.size());
		for (auto row : rows) {
			column.push_back(x[row][f]);
		}
		auto thresholds = CandidateThresholds(column, options.max_bins);
		for (auto threshold : thresholds) {
			vector<idx_t> left;
			vector<idx_t> right;
			left.reserve(rows.size());
			right.reserve(rows.size());
			for (auto row : rows) {
				if (x[row][f] < threshold) {
					left.push_back(row);
				} else {
					right.push_back(row);
				}
			}
			if (left.size() < options.min_samples_leaf || right.size() < options.min_samples_leaf) {
				continue;
			}
			auto gain = SplitGainRegression(residuals, left, right);
			if (gain > best.gain) {
				best.gain = gain;
				best.feature = f;
				best.threshold = threshold;
				best.left_value = LeafValueRegression(residuals, left);
				best.right_value = LeafValueRegression(residuals, right);
			}
		}
	}
	return best;
}

SplitCandidate FindBestSplitBinary(const vector<vector<double>> &x, const vector<double> &gradients,
                                   const vector<double> &hessians, const vector<idx_t> &rows,
                                   const TrainOptions &options) {
	SplitCandidate best;
	if (rows.size() < 2 * options.min_samples_leaf) {
		return best;
	}
	idx_t n_features = x.empty() ? 0 : x[0].size();
	for (idx_t f = 0; f < n_features; f++) {
		vector<double> column;
		column.reserve(rows.size());
		for (auto row : rows) {
			column.push_back(x[row][f]);
		}
		auto thresholds = CandidateThresholds(column, options.max_bins);
		for (auto threshold : thresholds) {
			vector<idx_t> left;
			vector<idx_t> right;
			left.reserve(rows.size());
			right.reserve(rows.size());
			for (auto row : rows) {
				if (x[row][f] < threshold) {
					left.push_back(row);
				} else {
					right.push_back(row);
				}
			}
			if (left.size() < options.min_samples_leaf || right.size() < options.min_samples_leaf) {
				continue;
			}
			auto gain = SplitGainBinary(gradients, hessians, left, right);
			if (gain > best.gain) {
				best.gain = gain;
				best.feature = f;
				best.threshold = threshold;
				best.left_value = LeafValueBinary(gradients, hessians, left);
				best.right_value = LeafValueBinary(gradients, hessians, right);
			}
		}
	}
	return best;
}

idx_t BuildLeaf(BoostTree &tree, double value) {
	TreeNode node;
	node.is_leaf = true;
	node.value = value;
	tree.nodes.push_back(node);
	return tree.nodes.size() - 1;
}

idx_t BuildTreeRegression(BoostTree &tree, const vector<vector<double>> &x, const vector<double> &residuals,
                          const vector<idx_t> &rows, idx_t depth, const TrainOptions &options) {
	if (depth >= options.max_depth || rows.size() < 2 * options.min_samples_leaf) {
		return BuildLeaf(tree, LeafValueRegression(residuals, rows));
	}
	auto split = FindBestSplitRegression(x, residuals, rows, options);
	if (!std::isfinite(split.gain) || split.gain <= 1e-12) {
		return BuildLeaf(tree, LeafValueRegression(residuals, rows));
	}
	vector<idx_t> left_rows;
	vector<idx_t> right_rows;
	for (auto row : rows) {
		if (x[row][split.feature] < split.threshold) {
			left_rows.push_back(row);
		} else {
			right_rows.push_back(row);
		}
	}
	TreeNode node;
	node.is_leaf = false;
	node.feature = split.feature;
	node.threshold = split.threshold;
	auto node_idx = tree.nodes.size();
	tree.nodes.push_back(node);
	tree.nodes[node_idx].left = BuildTreeRegression(tree, x, residuals, left_rows, depth + 1, options);
	tree.nodes[node_idx].right = BuildTreeRegression(tree, x, residuals, right_rows, depth + 1, options);
	return node_idx;
}

idx_t BuildTreeBinary(BoostTree &tree, const vector<vector<double>> &x, const vector<double> &gradients,
                      const vector<double> &hessians, const vector<idx_t> &rows, idx_t depth,
                      const TrainOptions &options) {
	if (depth >= options.max_depth || rows.size() < 2 * options.min_samples_leaf) {
		return BuildLeaf(tree, LeafValueBinary(gradients, hessians, rows));
	}
	auto split = FindBestSplitBinary(x, gradients, hessians, rows, options);
	if (!std::isfinite(split.gain) || split.gain <= 1e-12) {
		return BuildLeaf(tree, LeafValueBinary(gradients, hessians, rows));
	}
	vector<idx_t> left_rows;
	vector<idx_t> right_rows;
	for (auto row : rows) {
		if (x[row][split.feature] < split.threshold) {
			left_rows.push_back(row);
		} else {
			right_rows.push_back(row);
		}
	}
	TreeNode node;
	node.is_leaf = false;
	node.feature = split.feature;
	node.threshold = split.threshold;
	auto node_idx = tree.nodes.size();
	tree.nodes.push_back(node);
	tree.nodes[node_idx].left = BuildTreeBinary(tree, x, gradients, hessians, left_rows, depth + 1, options);
	tree.nodes[node_idx].right = BuildTreeBinary(tree, x, gradients, hessians, right_rows, depth + 1, options);
	return node_idx;
}

BoostModel TrainReference(const vector<double> &y, const vector<vector<double>> &x, const TrainOptions &options) {
	if (y.size() != x.size()) {
		throw InvalidInputException("duckboost: y/x row count mismatch");
	}
	if (y.empty()) {
		throw InvalidInputException("duckboost: cannot train on empty dataset");
	}
	idx_t n_features = x[0].size();
	for (auto &row : x) {
		if (row.size() != n_features) {
			throw InvalidInputException("duckboost: jagged feature rows are not supported");
		}
	}

	BoostModel model;
	model.backend = BoostBackend::REFERENCE;
	model.task = options.task;
	model.learning_rate = options.learning_rate;
	model.n_features = n_features;
	model.feature_names = options.feature_names;
	if (model.feature_names.empty()) {
		for (idx_t i = 0; i < n_features; i++) {
			model.feature_names.push_back("f" + std::to_string(i));
		}
	} else if (model.feature_names.size() != n_features) {
		throw InvalidInputException("duckboost: feature_names count (%llu) must match feature width (%llu)",
		                            (unsigned long long)model.feature_names.size(), (unsigned long long)n_features);
	}

	vector<idx_t> all_rows(y.size());
	std::iota(all_rows.begin(), all_rows.end(), 0);

	if (options.task == BoostTask::REGRESSION) {
		model.base_score = Mean(y);
		vector<double> prediction(y.size(), model.base_score);
		for (idx_t round = 0; round < options.n_estimators; round++) {
			vector<double> residuals(y.size());
			for (idx_t i = 0; i < y.size(); i++) {
				residuals[i] = y[i] - prediction[i];
			}
			BoostTree tree;
			BuildTreeRegression(tree, x, residuals, all_rows, 0, options);
			for (idx_t i = 0; i < y.size(); i++) {
				prediction[i] += options.learning_rate * [&]() {
					idx_t node_idx = 0;
					while (true) {
						auto &node = tree.nodes[node_idx];
						if (node.is_leaf) {
							return node.value;
						}
						node_idx = x[i][node.feature] < node.threshold ? node.left : node.right;
					}
				}();
			}
			model.trees.push_back(std::move(tree));
		}
		return model;
	}

	// binary logistic boosting
	for (auto label : y) {
		if (!(label == 0.0 || label == 1.0)) {
			throw InvalidInputException("duckboost: binary task requires labels in {0, 1}");
		}
	}
	double pos = Mean(y);
	pos = std::min(1.0 - 1e-6, std::max(1e-6, pos));
	model.base_score = std::log(pos / (1.0 - pos));
	vector<double> raw(y.size(), model.base_score);
	for (idx_t round = 0; round < options.n_estimators; round++) {
		vector<double> gradients(y.size());
		vector<double> hessians(y.size());
		for (idx_t i = 0; i < y.size(); i++) {
			double p = 1.0 / (1.0 + std::exp(-raw[i]));
			gradients[i] = p - y[i];
			hessians[i] = std::max(p * (1.0 - p), 1e-6);
		}
		BoostTree tree;
		BuildTreeBinary(tree, x, gradients, hessians, all_rows, 0, options);
		for (idx_t i = 0; i < y.size(); i++) {
			idx_t node_idx = 0;
			while (true) {
				auto &node = tree.nodes[node_idx];
				if (node.is_leaf) {
					raw[i] += options.learning_rate * node.value;
					break;
				}
				node_idx = x[i][node.feature] < node.threshold ? node.left : node.right;
			}
		}
		model.trees.push_back(std::move(tree));
	}
	return model;
}

} // namespace

BoostModel TrainModel(const vector<double> &y, const vector<vector<double>> &x, const TrainOptions &options) {
	if (options.backend == BoostBackend::REFERENCE) {
		return TrainReference(y, x, options);
	}
	if (NativeTrainerCompiled(options.backend)) {
		return TrainNative(y, x, options);
	}
	throw NotImplementedException(
	    "duckboost: native training for backend '%s' is not linked in this build. "
	    "Use backend='reference' to train in-process, or duckboost_import() with a vendor dump. "
	    "Optional CMake flags: DUCKBOOST_WITH_XGBOOST / DUCKBOOST_WITH_LIGHTGBM / DUCKBOOST_WITH_CATBOOST "
	    "(add DUCKBOOST_NATIVE_STUB_ONLY=ON to compile stubs without vendor libs).",
	    BackendToString(options.backend));
}

double EvaluateModel(const BoostModel &model, const vector<double> &y, const vector<vector<double>> &x,
                     const EvalOptions &options) {
	if (y.size() != x.size()) {
		throw InvalidInputException("duckboost: y/x row count mismatch during evaluate");
	}
	if (y.empty()) {
		throw InvalidInputException("duckboost: cannot evaluate on empty dataset");
	}
	auto metric = options.metric;
	if (metric.empty() || metric == "auto") {
		if (model.task == BoostTask::BINARY || model.task == BoostTask::MULTICLASS) {
			metric = "accuracy";
		} else {
			metric = "rmse";
		}
	}

	if (metric == "rmse") {
		double sse = 0;
		for (idx_t i = 0; i < y.size(); i++) {
			auto err = model.Predict(x[i]) - y[i];
			sse += err * err;
		}
		return std::sqrt(sse / static_cast<double>(y.size()));
	}
	if (metric == "mae") {
		double sae = 0;
		for (idx_t i = 0; i < y.size(); i++) {
			sae += std::fabs(model.Predict(x[i]) - y[i]);
		}
		return sae / static_cast<double>(y.size());
	}
	if (metric == "accuracy") {
		idx_t correct = 0;
		for (idx_t i = 0; i < y.size(); i++) {
			double pred;
			if (model.task == BoostTask::MULTICLASS) {
				pred = model.Predict(x[i]);
			} else {
				pred = model.Predict(x[i]) >= 0.5 ? 1.0 : 0.0;
			}
			if (pred == y[i]) {
				correct++;
			}
		}
		return static_cast<double>(correct) / static_cast<double>(y.size());
	}
	if (metric == "logloss") {
		if (model.task == BoostTask::MULTICLASS) {
			double loss = 0;
			for (idx_t i = 0; i < y.size(); i++) {
				auto proba = model.PredictProba(x[i]);
				auto label = static_cast<idx_t>(y[i]);
				if (label >= proba.size()) {
					throw InvalidInputException("duckboost: multiclass label out of range during logloss");
				}
				auto p = std::min(1.0 - 1e-15, std::max(1e-15, proba[label]));
				loss += -std::log(p);
			}
			return loss / static_cast<double>(y.size());
		}
		double loss = 0;
		for (idx_t i = 0; i < y.size(); i++) {
			auto p = std::min(1.0 - 1e-15, std::max(1e-15, model.Predict(x[i])));
			loss += -(y[i] * std::log(p) + (1.0 - y[i]) * std::log(1.0 - p));
		}
		return loss / static_cast<double>(y.size());
	}
	throw InvalidInputException("duckboost: unknown metric '%s' (expected auto, rmse, mae, accuracy, logloss)",
	                            options.metric);
}

} // namespace duckboost
} // namespace duckdb
