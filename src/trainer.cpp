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

//! Midpoints between distinct values of an ascending column, thinned to at most max_bins quantiles.
vector<double> CandidateThresholds(const vector<double> &sorted_column, idx_t max_bins) {
	vector<double> unique;
	for (auto v : sorted_column) {
		if (unique.empty() || v != unique.back()) {
			unique.push_back(v);
		}
	}
	if (unique.size() <= 1) {
		return {};
	}
	vector<double> thresholds;
	if (unique.size() <= max_bins + 1) {
		for (idx_t i = 0; i + 1 < unique.size(); i++) {
			thresholds.push_back(0.5 * (unique[i] + unique[i + 1]));
		}
		return thresholds;
	}
	for (idx_t b = 1; b <= max_bins; b++) {
		double q = static_cast<double>(b) / static_cast<double>(max_bins + 1);
		idx_t idx = static_cast<idx_t>(q * static_cast<double>(unique.size() - 1));
		if (idx + 1 < unique.size()) {
			auto threshold = 0.5 * (unique[idx] + unique[idx + 1]);
			if (thresholds.empty() || threshold != thresholds.back()) {
				thresholds.push_back(threshold);
			}
		}
	}
	return thresholds;
}

double NewtonLeaf(double g, double h) {
	if (h <= 1e-12) {
		return 0;
	}
	return -g / h;
}

double NewtonScore(double g, double h) {
	if (h <= 1e-12) {
		return 0;
	}
	return (g * g) / h;
}

//! Second-order split search. Squared error is the special case g = prediction - y, h = 1.
SplitCandidate FindBestSplit(const vector<vector<double>> &x, const vector<double> &gradients,
                             const vector<double> &hessians, const vector<idx_t> &rows, const TrainOptions &options) {
	SplitCandidate best;
	const idx_t n = rows.size();
	if (n < 2 * options.min_samples_leaf) {
		return best;
	}
	double g_total = 0;
	double h_total = 0;
	for (auto row : rows) {
		g_total += gradients[row];
		h_total += hessians[row];
	}
	const double parent_score = NewtonScore(g_total, h_total);

	idx_t n_features = x.empty() ? 0 : x[0].size();
	vector<std::pair<double, idx_t>> sorted(n);
	vector<idx_t> order(n);
	vector<double> column(n);
	for (idx_t f = 0; f < n_features; f++) {
		for (idx_t i = 0; i < n; i++) {
			sorted[i] = {x[rows[i]][f], rows[i]};
		}
		std::sort(sorted.begin(), sorted.end());
		for (idx_t i = 0; i < n; i++) {
			column[i] = sorted[i].first;
			order[i] = sorted[i].second;
		}
		auto thresholds = CandidateThresholds(column, options.max_bins);
		// Sweep: rows with value < threshold form a growing prefix of the sorted order.
		idx_t pos = 0;
		double g_left = 0;
		double h_left = 0;
		for (auto threshold : thresholds) {
			while (pos < n && column[pos] < threshold) {
				g_left += gradients[order[pos]];
				h_left += hessians[order[pos]];
				pos++;
			}
			if (pos < options.min_samples_leaf || n - pos < options.min_samples_leaf) {
				continue;
			}
			auto gain = NewtonScore(g_left, h_left) + NewtonScore(g_total - g_left, h_total - h_left) - parent_score;
			if (gain > best.gain) {
				best.gain = gain;
				best.feature = f;
				best.threshold = threshold;
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

idx_t BuildTree(BoostTree &tree, const vector<vector<double>> &x, const vector<double> &gradients,
                const vector<double> &hessians, const vector<idx_t> &rows, idx_t depth, const TrainOptions &options) {
	auto leaf_value = [&]() {
		double g = 0;
		double h = 0;
		for (auto row : rows) {
			g += gradients[row];
			h += hessians[row];
		}
		return NewtonLeaf(g, h);
	};
	if (depth >= options.max_depth || rows.size() < 2 * options.min_samples_leaf) {
		return BuildLeaf(tree, leaf_value());
	}
	auto split = FindBestSplit(x, gradients, hessians, rows, options);
	if (!std::isfinite(split.gain) || split.gain <= 1e-12) {
		return BuildLeaf(tree, leaf_value());
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
	tree.nodes[node_idx].left = BuildTree(tree, x, gradients, hessians, left_rows, depth + 1, options);
	tree.nodes[node_idx].right = BuildTree(tree, x, gradients, hessians, right_rows, depth + 1, options);
	return node_idx;
}

double EvalTreeRow(const BoostTree &tree, const vector<double> &row) {
	idx_t node_idx = 0;
	while (true) {
		auto &node = tree.nodes[node_idx];
		if (node.is_leaf) {
			return node.value;
		}
		node_idx = row[node.feature] < node.threshold ? node.left : node.right;
	}
}

//! Softmax cross-entropy boosting: each round fits one Newton tree per class (layout [round][class]).
void TrainMulticlass(BoostModel &model, const vector<double> &y, const vector<vector<double>> &x,
                     const vector<idx_t> &all_rows, const TrainOptions &options) {
	const idx_t n_classes = ResolveClassCount(y, options);
	const idx_t n_rows = y.size();
	model.n_classes = n_classes;
	model.base_score = 0;

	vector<idx_t> labels(n_rows);
	vector<double> counts(n_classes, 0);
	for (idx_t i = 0; i < n_rows; i++) {
		labels[i] = static_cast<idx_t>(y[i]);
		counts[labels[i]] += 1;
	}
	model.base_scores.resize(n_classes);
	for (idx_t c = 0; c < n_classes; c++) {
		auto prior = std::max(counts[c] / static_cast<double>(n_rows), 1e-6);
		model.base_scores[c] = std::log(prior);
	}

	// raw[i * n_classes + c] is the running score of class c for row i
	vector<double> raw(n_rows * n_classes);
	for (idx_t i = 0; i < n_rows; i++) {
		for (idx_t c = 0; c < n_classes; c++) {
			raw[i * n_classes + c] = model.base_scores[c];
		}
	}

	vector<double> proba(n_rows * n_classes);
	vector<double> gradients(n_rows);
	vector<double> hessians(n_rows);
	for (idx_t round = 0; round < options.n_estimators; round++) {
		for (idx_t i = 0; i < n_rows; i++) {
			auto *scores = &raw[i * n_classes];
			double max_score = *std::max_element(scores, scores + n_classes);
			double sum = 0;
			for (idx_t c = 0; c < n_classes; c++) {
				proba[i * n_classes + c] = std::exp(scores[c] - max_score);
				sum += proba[i * n_classes + c];
			}
			for (idx_t c = 0; c < n_classes; c++) {
				proba[i * n_classes + c] /= sum;
			}
		}
		// All trees of a round use the probabilities from the start of the round.
		vector<BoostTree> round_trees(n_classes);
		for (idx_t c = 0; c < n_classes; c++) {
			for (idx_t i = 0; i < n_rows; i++) {
				double p = proba[i * n_classes + c];
				gradients[i] = p - (labels[i] == c ? 1.0 : 0.0);
				hessians[i] = std::max(p * (1.0 - p), 1e-6);
			}
			BuildTree(round_trees[c], x, gradients, hessians, all_rows, 0, options);
		}
		for (idx_t c = 0; c < n_classes; c++) {
			for (idx_t i = 0; i < n_rows; i++) {
				raw[i * n_classes + c] += options.learning_rate * EvalTreeRow(round_trees[c], x[i]);
			}
			model.trees.push_back(std::move(round_trees[c]));
		}
	}
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

	if (options.task == BoostTask::MULTICLASS) {
		TrainMulticlass(model, y, x, all_rows, options);
		return model;
	}

	if (options.task == BoostTask::BINARY) {
		for (auto label : y) {
			if (!(label == 0.0 || label == 1.0)) {
				throw InvalidInputException("duckboost: binary task requires labels in {0, 1}");
			}
		}
		double pos = Mean(y);
		pos = std::min(1.0 - 1e-6, std::max(1e-6, pos));
		model.base_score = std::log(pos / (1.0 - pos));
	} else {
		model.base_score = Mean(y);
	}

	vector<double> raw(y.size(), model.base_score);
	vector<double> gradients(y.size());
	vector<double> hessians(y.size(), 1.0);
	for (idx_t round = 0; round < options.n_estimators; round++) {
		for (idx_t i = 0; i < y.size(); i++) {
			if (options.task == BoostTask::BINARY) {
				double p = 1.0 / (1.0 + std::exp(-raw[i]));
				gradients[i] = p - y[i];
				hessians[i] = std::max(p * (1.0 - p), 1e-6);
			} else {
				gradients[i] = raw[i] - y[i];
			}
		}
		BoostTree tree;
		BuildTree(tree, x, gradients, hessians, all_rows, 0, options);
		for (idx_t i = 0; i < y.size(); i++) {
			raw[i] += options.learning_rate * EvalTreeRow(tree, x[i]);
		}
		model.trees.push_back(std::move(tree));
	}
	return model;
}

} // namespace

idx_t ResolveClassCount(const vector<double> &y, const TrainOptions &options) {
	double max_label = -1;
	for (auto v : y) {
		if (!std::isfinite(v) || v < 0 || v != std::floor(v)) {
			throw InvalidInputException(
			    "duckboost: multiclass labels must be integer class indices 0, 1, ..., n_classes - 1 (got %s)",
			    std::to_string(v));
		}
		max_label = std::max(max_label, v);
	}
	auto n_classes = static_cast<idx_t>(max_label) + 1;
	if (options.n_classes > 0) {
		if (n_classes > options.n_classes) {
			throw InvalidInputException("duckboost: multiclass label %llu is out of range for n_classes = %llu",
			                            (unsigned long long)(n_classes - 1), (unsigned long long)options.n_classes);
		}
		n_classes = options.n_classes;
	}
	if (n_classes < 2) {
		throw InvalidInputException("duckboost: multiclass train requires at least 2 classes");
	}
	return n_classes;
}

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
