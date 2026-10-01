#include "duckboost/model.hpp"
#include "duckboost/native_train.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/string_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <utility>

namespace duckdb {
namespace duckboost {

namespace {

struct GradStat {
	double g = 0;
	double h = 0;

	void Add(double gg, double hh) {
		g += gg;
		h += hh;
	}

	void Add(const GradStat &other) {
		g += other.g;
		h += other.h;
	}

	GradStat Without(const GradStat &other) const {
		return {g - other.g, h - other.h};
	}
};

struct SplitCandidate {
	idx_t feature = 0;
	double threshold = 0;
	double gain = -std::numeric_limits<double>::infinity();
	SplitCompare compare = SplitCompare::LESS;
	bool default_left = true;
	//! Membership set for IN splits (sorted unique); empty for LESS/EQUAL.
	vector<double> categories;
};

struct FeatureBinning {
	//! Upper bounds of bins 0 .. n-2; bin n-1 is everything above the last bound.
	vector<double> thresholds;
	bool categorical = false;
};

static constexpr uint32_t MISSING_BIN = std::numeric_limits<uint32_t>::max();

struct SimpleRng {
	uint64_t state;

	explicit SimpleRng(uint64_t seed) : state(seed ? seed : 0x9e3779b97f4a7c15ULL) {
	}

	uint64_t Next() {
		state = state * 6364136223846793005ULL + 1;
		return state;
	}

	idx_t Bounded(idx_t n) {
		if (n <= 1) {
			return 0;
		}
		return static_cast<idx_t>(Next() % n);
	}
};

bool IsMissing(double value) {
	return std::isnan(value);
}

double SoftThreshold(double g, double alpha) {
	if (g > alpha) {
		return g - alpha;
	}
	if (g < -alpha) {
		return g + alpha;
	}
	return 0;
}

double LeafWeight(const GradStat &stat, const TrainOptions &options) {
	if (stat.h <= 0) {
		return 0;
	}
	return -SoftThreshold(stat.g, options.reg_alpha) / (stat.h + options.reg_lambda);
}

double LeafScore(const GradStat &stat, const TrainOptions &options) {
	auto g = SoftThreshold(stat.g, options.reg_alpha);
	return (g * g) / (stat.h + options.reg_lambda);
}

double SplitGain(const GradStat &left, const GradStat &right, const GradStat &parent, const TrainOptions &options) {
	return 0.5 * (LeafScore(left, options) + LeafScore(right, options) - LeafScore(parent, options)) -
	       options.min_split_gain;
}

GradStat SumStats(const vector<double> &gradients, const vector<double> &hessians, const vector<idx_t> &rows) {
	GradStat sum;
	for (auto row : rows) {
		sum.Add(gradients[row], hessians[row]);
	}
	return sum;
}

//! Midpoints between distinct present values, thinned to at most max_bins quantiles.
vector<double> CandidateThresholds(const vector<double> &sorted_present_values, idx_t max_bins) {
	if (sorted_present_values.size() <= 1) {
		return {};
	}
	vector<double> unique;
	unique.reserve(sorted_present_values.size());
	for (auto value : sorted_present_values) {
		if (unique.empty() || value != unique.back()) {
			unique.push_back(value);
		}
	}
	if (unique.size() <= 1) {
		return {};
	}
	vector<double> thresholds;
	if (unique.size() <= max_bins + 1) {
		thresholds.reserve(unique.size() - 1);
		for (idx_t i = 0; i + 1 < unique.size(); i++) {
			thresholds.push_back(0.5 * (unique[i] + unique[i + 1]));
		}
		return thresholds;
	}
	for (idx_t b = 1; b <= max_bins; b++) {
		double q = static_cast<double>(b) / static_cast<double>(max_bins + 1);
		idx_t idx = static_cast<idx_t>(q * static_cast<double>(unique.size() - 1));
		if (idx + 1 < unique.size()) {
			thresholds.push_back(0.5 * (unique[idx] + unique[idx + 1]));
		}
	}
	std::sort(thresholds.begin(), thresholds.end());
	thresholds.erase(std::unique(thresholds.begin(), thresholds.end()), thresholds.end());
	return thresholds;
}

vector<FeatureBinning> BuildFeatureBinnings(const vector<vector<double>> &x, const vector<idx_t> &rows,
                                            const vector<bool> &categorical_features, idx_t max_bins) {
	const idx_t n_features = x.empty() ? 0 : x[0].size();
	vector<FeatureBinning> binnings(n_features);
	for (idx_t f = 0; f < n_features; f++) {
		if (categorical_features[f]) {
			binnings[f].categorical = true;
			continue;
		}
		vector<double> values;
		values.reserve(rows.size());
		for (auto row : rows) {
			auto value = x[row][f];
			if (!IsMissing(value)) {
				values.push_back(value);
			}
		}
		std::sort(values.begin(), values.end());
		binnings[f].thresholds = CandidateThresholds(values, max_bins);
	}
	return binnings;
}

vector<vector<uint32_t>> AssignFeatureBins(const vector<vector<double>> &x, const vector<FeatureBinning> &binnings) {
	vector<vector<uint32_t>> bins(x.size(), vector<uint32_t>(binnings.size(), MISSING_BIN));
	for (idx_t row = 0; row < x.size(); row++) {
		for (idx_t f = 0; f < binnings.size(); f++) {
			if (binnings[f].categorical) {
				continue;
			}
			auto value = x[row][f];
			if (IsMissing(value) || binnings[f].thresholds.empty()) {
				bins[row][f] = MISSING_BIN;
				continue;
			}
			bins[row][f] = static_cast<uint32_t>(
			    std::upper_bound(binnings[f].thresholds.begin(), binnings[f].thresholds.end(), value) -
			    binnings[f].thresholds.begin());
		}
	}
	return bins;
}

bool ChildFeasible(const GradStat &stat, idx_t row_count, const TrainOptions &options) {
	return row_count >= options.min_samples_leaf && stat.h >= options.min_child_weight;
}

void ConsiderSplit(SplitCandidate &best, idx_t feature, double threshold, SplitCompare compare, bool default_left,
                   const GradStat &left, const GradStat &right, const GradStat &parent, idx_t left_rows,
                   idx_t right_rows, const TrainOptions &options, const vector<double> &categories = {}) {
	if (!ChildFeasible(left, left_rows, options) || !ChildFeasible(right, right_rows, options)) {
		return;
	}
	auto gain = SplitGain(left, right, parent, options);
	if (std::isfinite(gain) && gain > best.gain) {
		best.gain = gain;
		best.feature = feature;
		best.threshold = threshold;
		best.compare = compare;
		best.default_left = default_left;
		best.categories = categories;
	}
}

void ConsiderCategoricalPartition(SplitCandidate &best, idx_t feature, const vector<double> &right_categories,
                                  const GradStat &matching, idx_t matching_count, const GradStat &missing_stat,
                                  idx_t missing_count, idx_t present_count, const GradStat &parent,
                                  const TrainOptions &options) {
	if (right_categories.empty() || matching_count == 0 || matching_count >= present_count) {
		return;
	}
	auto rest = parent.Without(matching).Without(missing_stat);
	auto rest_count = present_count - matching_count;
	SplitCompare compare = right_categories.size() == 1 ? SplitCompare::EQUAL : SplitCompare::IN;
	double threshold = right_categories.size() == 1 ? right_categories[0] : 0;
	vector<double> categories = right_categories.size() == 1 ? vector<double> {} : right_categories;

	GradStat left_m = rest;
	left_m.Add(missing_stat);
	ConsiderSplit(best, feature, threshold, compare, true, left_m, matching, parent, rest_count + missing_count,
	              matching_count, options, categories);
	GradStat right_m = matching;
	right_m.Add(missing_stat);
	ConsiderSplit(best, feature, threshold, compare, false, rest, right_m, parent, rest_count,
	              matching_count + missing_count, options, categories);
}

//! Order categories by local target statistic (g/h) and scan contiguous partitions — CatBoost/LightGBM style.
void FindBestCategoricalSplit(SplitCandidate &best, idx_t feature, const vector<vector<double>> &x,
                              const vector<double> &gradients, const vector<double> &hessians,
                              const vector<idx_t> &rows, const GradStat &parent, const TrainOptions &options) {
	struct CatStat {
		double category = 0;
		GradStat stat;
		idx_t count = 0;
	};
	unordered_map<double, CatStat> by_category;
	GradStat missing_stat;
	idx_t missing_count = 0;
	idx_t present_count = 0;
	for (auto row : rows) {
		auto value = x[row][feature];
		if (IsMissing(value)) {
			missing_stat.Add(gradients[row], hessians[row]);
			missing_count++;
			continue;
		}
		auto &entry = by_category[value];
		entry.category = value;
		entry.stat.Add(gradients[row], hessians[row]);
		entry.count++;
		present_count++;
	}
	if (by_category.size() < 2) {
		return;
	}
	vector<CatStat> ordered;
	ordered.reserve(by_category.size());
	for (auto &entry : by_category) {
		ordered.push_back(entry.second);
	}
	std::sort(ordered.begin(), ordered.end(), [](const CatStat &a, const CatStat &b) {
		double ta = a.stat.g / MaxValue(a.stat.h, 1e-12);
		double tb = b.stat.g / MaxValue(b.stat.h, 1e-12);
		if (ta != tb) {
			return ta < tb;
		}
		return a.category < b.category;
	});

	GradStat prefix;
	idx_t prefix_count = 0;
	vector<double> prefix_categories;
	prefix_categories.reserve(ordered.size());
	for (idx_t i = 0; i + 1 < ordered.size(); i++) {
		prefix.Add(ordered[i].stat);
		prefix_count += ordered[i].count;
		prefix_categories.push_back(ordered[i].category);
		auto suffix_count = present_count - prefix_count;
		// Prefer emitting the smaller membership set on the right (match → right).
		if (prefix_count <= suffix_count) {
			auto cats = prefix_categories;
			std::sort(cats.begin(), cats.end());
			ConsiderCategoricalPartition(best, feature, cats, prefix, prefix_count, missing_stat, missing_count,
			                             present_count, parent, options);
		} else {
			vector<double> suffix_categories;
			suffix_categories.reserve(ordered.size() - i - 1);
			GradStat suffix;
			for (idx_t j = i + 1; j < ordered.size(); j++) {
				suffix_categories.push_back(ordered[j].category);
				suffix.Add(ordered[j].stat);
			}
			std::sort(suffix_categories.begin(), suffix_categories.end());
			ConsiderCategoricalPartition(best, feature, suffix_categories, suffix, suffix_count, missing_stat,
			                             missing_count, present_count, parent, options);
		}
	}
}

void FindBestHistogramSplit(SplitCandidate &best, idx_t feature, const vector<vector<uint32_t>> &bins,
                            const vector<double> &thresholds, const vector<double> &gradients,
                            const vector<double> &hessians, const vector<idx_t> &rows, const GradStat &parent,
                            const TrainOptions &options) {
	if (thresholds.empty()) {
		return;
	}
	const idx_t n_bins = thresholds.size() + 1;
	vector<GradStat> hist(n_bins);
	vector<idx_t> hist_count(n_bins, 0);
	GradStat missing_stat;
	idx_t missing_count = 0;
	idx_t present_count = 0;
	for (auto row : rows) {
		auto bin = bins[row][feature];
		if (bin == MISSING_BIN) {
			missing_stat.Add(gradients[row], hessians[row]);
			missing_count++;
			continue;
		}
		if (bin >= n_bins) {
			continue;
		}
		hist[bin].Add(gradients[row], hessians[row]);
		hist_count[bin]++;
		present_count++;
	}
	if (present_count < 2) {
		return;
	}
	GradStat left_present;
	idx_t left_count = 0;
	for (idx_t b = 0; b + 1 < n_bins; b++) {
		left_present.Add(hist[b]);
		left_count += hist_count[b];
		auto right_count = present_count - left_count;
		if (left_count == 0 || right_count == 0) {
			continue;
		}
		auto right_present = parent.Without(left_present).Without(missing_stat);
		auto threshold = thresholds[b];
		GradStat left_m = left_present;
		left_m.Add(missing_stat);
		ConsiderSplit(best, feature, threshold, SplitCompare::LESS, true, left_m, right_present, parent,
		              left_count + missing_count, right_count, options);
		GradStat right_m = right_present;
		right_m.Add(missing_stat);
		ConsiderSplit(best, feature, threshold, SplitCompare::LESS, false, left_present, right_m, parent, left_count,
		              right_count + missing_count, options);
	}
}

SplitCandidate FindBestSplit(const vector<vector<double>> &x, const vector<vector<uint32_t>> &bins,
                             const vector<FeatureBinning> &binnings, const vector<double> &gradients,
                             const vector<double> &hessians, const vector<idx_t> &rows,
                             const vector<idx_t> &feature_subset, const TrainOptions &options) {
	SplitCandidate best;
	if (rows.size() < 2 * options.min_samples_leaf) {
		return best;
	}
	auto parent = SumStats(gradients, hessians, rows);
	for (auto f : feature_subset) {
		if (binnings[f].categorical) {
			FindBestCategoricalSplit(best, f, x, gradients, hessians, rows, parent, options);
		} else {
			FindBestHistogramSplit(best, f, bins, binnings[f].thresholds, gradients, hessians, rows, parent, options);
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

TreeNode MakeSplitNode(const SplitCandidate &split, const GradStat &parent) {
	TreeNode node;
	node.is_leaf = false;
	node.feature = split.feature;
	node.threshold = split.threshold;
	node.compare = split.compare;
	node.categories = split.categories;
	node.default_left = split.default_left;
	node.gain = split.gain;
	node.cover = parent.h;
	return node;
}

void PartitionRows(const vector<vector<double>> &x, const vector<idx_t> &rows, const SplitCandidate &split,
                   vector<idx_t> &left_rows, vector<idx_t> &right_rows) {
	left_rows.clear();
	right_rows.clear();
	left_rows.reserve(rows.size());
	right_rows.reserve(rows.size());
	TreeNode node;
	node.threshold = split.threshold;
	node.compare = split.compare;
	node.categories = split.categories;
	node.default_left = split.default_left;
	for (auto row : rows) {
		auto value = x[row][split.feature];
		if (NodeGoesLeft(node, value)) {
			left_rows.push_back(row);
		} else {
			right_rows.push_back(row);
		}
	}
}

idx_t GrowDepthwise(BoostTree &tree, const vector<vector<double>> &x, const vector<vector<uint32_t>> &bins,
                    const vector<FeatureBinning> &binnings, const vector<double> &gradients,
                    const vector<double> &hessians, const vector<idx_t> &rows, const vector<idx_t> &feature_subset,
                    idx_t depth, const TrainOptions &options) {
	auto parent = SumStats(gradients, hessians, rows);
	if (depth >= options.max_depth || rows.size() < 2 * options.min_samples_leaf) {
		return BuildLeaf(tree, LeafWeight(parent, options));
	}
	auto split = FindBestSplit(x, bins, binnings, gradients, hessians, rows, feature_subset, options);
	if (!std::isfinite(split.gain) || split.gain <= 0) {
		return BuildLeaf(tree, LeafWeight(parent, options));
	}
	vector<idx_t> left_rows;
	vector<idx_t> right_rows;
	PartitionRows(x, rows, split, left_rows, right_rows);
	if (left_rows.empty() || right_rows.empty()) {
		return BuildLeaf(tree, LeafWeight(parent, options));
	}

	auto node = MakeSplitNode(split, parent);
	auto node_idx = tree.nodes.size();
	tree.nodes.push_back(node);
	tree.nodes[node_idx].left =
	    GrowDepthwise(tree, x, bins, binnings, gradients, hessians, left_rows, feature_subset, depth + 1, options);
	tree.nodes[node_idx].right =
	    GrowDepthwise(tree, x, bins, binnings, gradients, hessians, right_rows, feature_subset, depth + 1, options);
	return node_idx;
}

struct LossguideCandidate {
	idx_t node_idx;
	idx_t depth;
	SplitCandidate split;
	vector<idx_t> rows;
};

struct LossguideCandidateCompare {
	bool operator()(const LossguideCandidate &a, const LossguideCandidate &b) const {
		if (a.split.gain != b.split.gain) {
			return a.split.gain < b.split.gain;
		}
		return a.node_idx > b.node_idx;
	}
};

using LossguideQueue = std::priority_queue<LossguideCandidate, vector<LossguideCandidate>, LossguideCandidateCompare>;

void EnqueueLossguideCandidate(LossguideQueue &candidates, const vector<vector<double>> &x,
                               const vector<vector<uint32_t>> &bins, const vector<FeatureBinning> &binnings,
                               const vector<double> &gradients, const vector<double> &hessians, vector<idx_t> rows,
                               const vector<idx_t> &feature_subset, idx_t node_idx, idx_t depth,
                               const TrainOptions &options) {
	if ((options.max_depth > 0 && depth >= options.max_depth) || rows.size() < 2 * options.min_samples_leaf) {
		return;
	}
	auto split = FindBestSplit(x, bins, binnings, gradients, hessians, rows, feature_subset, options);
	if (!std::isfinite(split.gain) || split.gain <= 0) {
		return;
	}
	candidates.push({node_idx, depth, split, std::move(rows)});
}

void GrowLossguide(BoostTree &tree, const vector<vector<double>> &x, const vector<vector<uint32_t>> &bins,
                   const vector<FeatureBinning> &binnings, const vector<double> &gradients,
                   const vector<double> &hessians, const vector<idx_t> &rows, const vector<idx_t> &feature_subset,
                   const TrainOptions &options) {
	auto root_stat = SumStats(gradients, hessians, rows);
	BuildLeaf(tree, LeafWeight(root_stat, options));

	LossguideQueue candidates;
	EnqueueLossguideCandidate(candidates, x, bins, binnings, gradients, hessians, rows, feature_subset, 0, 0, options);
	idx_t leaf_count = 1;
	while (leaf_count < options.max_leaves && !candidates.empty()) {
		auto candidate = candidates.top();
		candidates.pop();

		vector<idx_t> left_rows;
		vector<idx_t> right_rows;
		PartitionRows(x, candidate.rows, candidate.split, left_rows, right_rows);
		if (left_rows.empty() || right_rows.empty()) {
			continue;
		}

		auto parent = SumStats(gradients, hessians, candidate.rows);
		auto left_stat = SumStats(gradients, hessians, left_rows);
		auto right_stat = SumStats(gradients, hessians, right_rows);
		auto left_idx = BuildLeaf(tree, LeafWeight(left_stat, options));
		auto right_idx = BuildLeaf(tree, LeafWeight(right_stat, options));

		auto node = MakeSplitNode(candidate.split, parent);
		node.left = left_idx;
		node.right = right_idx;
		tree.nodes[candidate.node_idx] = node;
		leaf_count++;

		EnqueueLossguideCandidate(candidates, x, bins, binnings, gradients, hessians, std::move(left_rows),
		                          feature_subset, left_idx, candidate.depth + 1, options);
		EnqueueLossguideCandidate(candidates, x, bins, binnings, gradients, hessians, std::move(right_rows),
		                          feature_subset, right_idx, candidate.depth + 1, options);
	}
}

double LeafSplitGain(const vector<vector<double>> &x, const vector<double> &gradients, const vector<double> &hessians,
                     const vector<idx_t> &rows, const SplitCandidate &split, const TrainOptions &options) {
	if (rows.size() < 2 * options.min_samples_leaf) {
		return 0;
	}
	vector<idx_t> left_rows;
	vector<idx_t> right_rows;
	PartitionRows(x, rows, split, left_rows, right_rows);
	if (left_rows.empty() || right_rows.empty()) {
		return 0;
	}
	auto parent = SumStats(gradients, hessians, rows);
	auto left = SumStats(gradients, hessians, left_rows);
	auto right = SumStats(gradients, hessians, right_rows);
	if (!ChildFeasible(left, left_rows.size(), options) || !ChildFeasible(right, right_rows.size(), options)) {
		return 0;
	}
	auto gain = SplitGain(left, right, parent, options);
	return std::isfinite(gain) && gain > 0 ? gain : 0;
}

//! CatBoost-style symmetric trees: one shared (feature, threshold) per depth level.
void GrowOblivious(BoostTree &tree, const vector<vector<double>> &x, const vector<FeatureBinning> &binnings,
                   const vector<double> &gradients, const vector<double> &hessians, const vector<idx_t> &rows,
                   const vector<idx_t> &feature_subset, const TrainOptions &options) {
	auto root_stat = SumStats(gradients, hessians, rows);
	BuildLeaf(tree, LeafWeight(root_stat, options));

	vector<vector<idx_t>> level_rows;
	level_rows.push_back(rows);
	vector<idx_t> level_nodes = {0};

	for (idx_t depth = 0; depth < options.max_depth; depth++) {
		SplitCandidate best;
		double best_total = -std::numeric_limits<double>::infinity();

		// Evaluate shared continuous candidates from global histogram boundaries.
		for (auto f : feature_subset) {
			if (binnings[f].categorical) {
				continue;
			}
			auto &thresholds = binnings[f].thresholds;
			for (idx_t b = 0; b < thresholds.size(); b++) {
				for (bool default_left : {true, false}) {
					SplitCandidate candidate;
					candidate.feature = f;
					candidate.threshold = thresholds[b];
					candidate.compare = SplitCompare::LESS;
					candidate.default_left = default_left;
					double total = 0;
					bool any = false;
					for (auto &leaf_rows : level_rows) {
						auto gain = LeafSplitGain(x, gradients, hessians, leaf_rows, candidate, options);
						if (gain > 0) {
							any = true;
							total += gain;
						}
					}
					if (any && total > best_total) {
						best_total = total;
						best = candidate;
						best.gain = total;
					}
				}
			}
		}

		// Shared categorical partitions: pool target statistics across the current level, then score.
		for (auto f : feature_subset) {
			if (!binnings[f].categorical) {
				continue;
			}
			vector<idx_t> pooled;
			for (auto &leaf_rows : level_rows) {
				pooled.insert(pooled.end(), leaf_rows.begin(), leaf_rows.end());
			}
			if (pooled.size() < 2 * options.min_samples_leaf) {
				continue;
			}
			auto parent = SumStats(gradients, hessians, pooled);
			SplitCandidate local;
			FindBestCategoricalSplit(local, f, x, gradients, hessians, pooled, parent, options);
			if (!std::isfinite(local.gain) || local.gain <= 0) {
				continue;
			}
			double total = 0;
			bool any = false;
			for (auto &leaf_rows : level_rows) {
				auto gain = LeafSplitGain(x, gradients, hessians, leaf_rows, local, options);
				if (gain > 0) {
					any = true;
					total += gain;
				}
			}
			if (any && total > best_total) {
				best_total = total;
				best = local;
				best.gain = total;
			}
		}

		if (!std::isfinite(best_total) || best_total <= 0) {
			break;
		}

		vector<vector<idx_t>> next_rows;
		vector<idx_t> next_nodes;
		next_rows.reserve(level_rows.size() * 2);
		next_nodes.reserve(level_nodes.size() * 2);
		for (idx_t i = 0; i < level_nodes.size(); i++) {
			vector<idx_t> left_rows;
			vector<idx_t> right_rows;
			PartitionRows(x, level_rows[i], best, left_rows, right_rows);
			auto parent = SumStats(gradients, hessians, level_rows[i]);
			auto left_stat = left_rows.empty() ? parent : SumStats(gradients, hessians, left_rows);
			auto right_stat = right_rows.empty() ? parent : SumStats(gradients, hessians, right_rows);
			if (left_rows.empty()) {
				left_stat = {};
			}
			if (right_rows.empty()) {
				right_stat = {};
			}
			auto left_idx = BuildLeaf(tree, LeafWeight(left_rows.empty() ? parent : left_stat, options));
			auto right_idx = BuildLeaf(tree, LeafWeight(right_rows.empty() ? parent : right_stat, options));
			auto node = MakeSplitNode(best, parent);
			node.gain = LeafSplitGain(x, gradients, hessians, level_rows[i], best, options);
			node.left = left_idx;
			node.right = right_idx;
			tree.nodes[level_nodes[i]] = node;
			next_rows.push_back(std::move(left_rows));
			next_rows.push_back(std::move(right_rows));
			next_nodes.push_back(left_idx);
			next_nodes.push_back(right_idx);
		}
		level_rows = std::move(next_rows);
		level_nodes = std::move(next_nodes);
	}
}

void GrowTree(BoostTree &tree, const vector<vector<double>> &x, const vector<vector<uint32_t>> &bins,
              const vector<FeatureBinning> &binnings, const vector<double> &gradients, const vector<double> &hessians,
              const vector<idx_t> &rows, const vector<idx_t> &feature_subset, const TrainOptions &options) {
	if (options.growth_policy == GrowthPolicy::LOSSGUIDE) {
		GrowLossguide(tree, x, bins, binnings, gradients, hessians, rows, feature_subset, options);
		return;
	}
	if (options.growth_policy == GrowthPolicy::OBLIVIOUS) {
		GrowOblivious(tree, x, binnings, gradients, hessians, rows, feature_subset, options);
		return;
	}
	GrowDepthwise(tree, x, bins, binnings, gradients, hessians, rows, feature_subset, 0, options);
}

vector<idx_t> SampleRows(const vector<idx_t> &pool, double subsample, SimpleRng &rng) {
	if (subsample >= 1.0 || pool.empty()) {
		return pool;
	}
	auto all = pool;
	idx_t keep = MaxValue<idx_t>(1, static_cast<idx_t>(std::ceil(subsample * static_cast<double>(all.size()))));
	keep = MinValue<idx_t>(keep, all.size());
	for (idx_t i = 0; i < keep; i++) {
		idx_t j = i + rng.Bounded(all.size() - i);
		std::swap(all[i], all[j]);
	}
	all.resize(keep);
	std::sort(all.begin(), all.end());
	return all;
}

vector<idx_t> SampleFeatures(idx_t n_features, double colsample, SimpleRng &rng) {
	vector<idx_t> all(n_features);
	std::iota(all.begin(), all.end(), 0);
	if (colsample >= 1.0 || n_features == 0) {
		return all;
	}
	idx_t keep = MaxValue<idx_t>(1, static_cast<idx_t>(std::ceil(colsample * static_cast<double>(n_features))));
	keep = MinValue<idx_t>(keep, n_features);
	for (idx_t i = 0; i < keep; i++) {
		idx_t j = i + rng.Bounded(n_features - i);
		std::swap(all[i], all[j]);
	}
	all.resize(keep);
	std::sort(all.begin(), all.end());
	return all;
}

double ApplyTree(const BoostTree &tree, const vector<double> &features) {
	idx_t node_idx = 0;
	while (true) {
		auto &node = tree.nodes[node_idx];
		if (node.is_leaf) {
			return node.value;
		}
		node_idx = NodeGoesLeft(node, features[node.feature]) ? node.left : node.right;
	}
}

vector<double> NormalizeWeights(const vector<double> &y, const vector<double> &weights_in) {
	vector<double> weights(y.size(), 1.0);
	if (!weights_in.empty()) {
		if (weights_in.size() != y.size()) {
			throw InvalidInputException("duckboost: sample weight count (%llu) must match row count (%llu)",
			                            (unsigned long long)weights_in.size(), (unsigned long long)y.size());
		}
		for (idx_t i = 0; i < y.size(); i++) {
			if (!std::isfinite(weights_in[i]) || weights_in[i] < 0) {
				throw InvalidInputException("duckboost: sample weights must be finite and >= 0");
			}
			weights[i] = weights_in[i];
		}
	}
	return weights;
}

void ApplyClassWeights(vector<double> &weights, const vector<double> &y, const TrainOptions &options, idx_t n_classes) {
	if (options.class_weight.empty()) {
		return;
	}
	vector<double> multipliers(n_classes, 1.0);
	auto lower = StringUtil::Lower(options.class_weight);
	if (lower == "balanced") {
		vector<double> counts(n_classes, 0);
		double total = 0;
		for (idx_t i = 0; i < y.size(); i++) {
			auto label = static_cast<idx_t>(y[i]);
			if (label >= n_classes) {
				continue;
			}
			counts[label] += weights[i];
			total += weights[i];
		}
		for (idx_t c = 0; c < n_classes; c++) {
			if (counts[c] <= 0) {
				multipliers[c] = 0;
			} else {
				multipliers[c] = total / (static_cast<double>(n_classes) * counts[c]);
			}
		}
	} else {
		auto parts = StringUtil::Split(options.class_weight, ',');
		if (parts.size() != n_classes) {
			throw InvalidInputException("duckboost: class_weight list length (%llu) must match n_classes (%llu)",
			                            (unsigned long long)parts.size(), (unsigned long long)n_classes);
		}
		for (idx_t c = 0; c < n_classes; c++) {
			StringUtil::Trim(parts[c]);
			multipliers[c] = std::stod(parts[c]);
			if (!(multipliers[c] >= 0) || !std::isfinite(multipliers[c])) {
				throw InvalidInputException("duckboost: class_weight values must be finite and >= 0");
			}
		}
	}
	for (idx_t i = 0; i < y.size(); i++) {
		auto label = static_cast<idx_t>(y[i]);
		if (label < n_classes) {
			weights[i] *= multipliers[label];
		}
	}
}

double WeightedMean(const vector<double> &y, const vector<double> &weights, const vector<idx_t> &rows) {
	double sum = 0;
	double wsum = 0;
	for (auto row : rows) {
		sum += weights[row] * y[row];
		wsum += weights[row];
	}
	return wsum > 0 ? sum / wsum : 0;
}

double WeightedQuantile(const vector<double> &values, const vector<double> &weights, const vector<idx_t> &rows,
                        double alpha) {
	vector<std::pair<double, double>> ordered;
	ordered.reserve(rows.size());
	double total_weight = 0;
	for (auto row : rows) {
		if (weights[row] <= 0) {
			continue;
		}
		ordered.emplace_back(values[row], weights[row]);
		total_weight += weights[row];
	}
	if (ordered.empty() || total_weight <= 0) {
		return 0;
	}
	std::sort(ordered.begin(), ordered.end(),
	          [](const std::pair<double, double> &a, const std::pair<double, double> &b) { return a.first < b.first; });
	double cumulative = 0;
	const double target = alpha * total_weight;
	for (auto &entry : ordered) {
		cumulative += entry.second;
		if (cumulative >= target) {
			return entry.first;
		}
	}
	return ordered.back().first;
}

double WeightedExpectile(const vector<double> &values, const vector<double> &weights, const vector<idx_t> &rows,
                         double tau) {
	if (tau == 0.5) {
		return WeightedMean(values, weights, rows);
	}
	bool found = false;
	double lower = 0;
	double upper = 0;
	double total_weight = 0;
	for (auto row : rows) {
		if (weights[row] <= 0) {
			continue;
		}
		if (!found) {
			lower = values[row];
			upper = values[row];
			found = true;
		} else {
			lower = MinValue(lower, values[row]);
			upper = MaxValue(upper, values[row]);
		}
		total_weight += weights[row];
	}
	if (!found || total_weight <= 0) {
		return 0;
	}
	for (idx_t iteration = 0; iteration < 100; iteration++) {
		auto candidate = 0.5 * (lower + upper);
		double gradient = 0;
		for (auto row : rows) {
			auto asymmetry = values[row] >= candidate ? tau : 1.0 - tau;
			gradient += weights[row] * asymmetry * (candidate - values[row]);
		}
		if (gradient > 0) {
			upper = candidate;
		} else {
			lower = candidate;
		}
	}
	return 0.5 * (lower + upper);
}

void RenewQuantileLeaves(BoostTree &tree, const vector<vector<double>> &x, const vector<double> &y,
                         const vector<double> &prediction, const vector<double> &weights, const vector<idx_t> &rows,
                         double alpha) {
	vector<vector<idx_t>> leaf_rows(tree.nodes.size());
	vector<double> residuals(y.size());
	for (auto row : rows) {
		idx_t node_idx = 0;
		while (!tree.nodes[node_idx].is_leaf) {
			auto &node = tree.nodes[node_idx];
			node_idx = NodeGoesLeft(node, x[row][node.feature]) ? node.left : node.right;
		}
		leaf_rows[node_idx].push_back(row);
		residuals[row] = y[row] - prediction[row];
	}
	for (idx_t node_idx = 0; node_idx < tree.nodes.size(); node_idx++) {
		if (tree.nodes[node_idx].is_leaf) {
			tree.nodes[node_idx].value = WeightedQuantile(residuals, weights, leaf_rows[node_idx], alpha);
		}
	}
}

double RegressionLossValue(RegressionLoss loss, double alpha, double y, double prediction) {
	auto error = prediction - y;
	switch (loss) {
	case RegressionLoss::ABSOLUTE_ERROR:
		return std::fabs(error);
	case RegressionLoss::QUANTILE:
		return error >= 0 ? (1.0 - alpha) * error : -alpha * error;
	case RegressionLoss::EXPECTILE: {
		auto asymmetry = y >= prediction ? alpha : 1.0 - alpha;
		return asymmetry * error * error;
	}
	case RegressionLoss::SQUARED_ERROR:
	default:
		return error * error;
	}
}

double EvalValidRegression(const vector<double> &y, const vector<double> &prediction, const vector<idx_t> &rows,
                           const vector<double> &weights, RegressionLoss loss, double alpha) {
	double total_loss = 0;
	double wsum = 0;
	for (auto row : rows) {
		total_loss += weights[row] * RegressionLossValue(loss, alpha, y[row], prediction[row]);
		wsum += weights[row];
	}
	return wsum > 0 ? total_loss / wsum : std::numeric_limits<double>::infinity();
}

double EvalValidBinaryLogloss(const vector<double> &y, const vector<double> &prediction, const vector<idx_t> &rows,
                              const vector<double> &weights) {
	double loss = 0;
	double wsum = 0;
	for (auto row : rows) {
		double p = 1.0 / (1.0 + std::exp(-prediction[row]));
		p = std::min(1.0 - 1e-15, std::max(1e-15, p));
		loss += weights[row] * -(y[row] * std::log(p) + (1.0 - y[row]) * std::log(1.0 - p));
		wsum += weights[row];
	}
	return wsum > 0 ? loss / wsum : std::numeric_limits<double>::infinity();
}

double EvalValidMulticlassLogloss(const vector<double> &y, const vector<vector<double>> &prediction, idx_t n_classes,
                                  const vector<idx_t> &rows, const vector<double> &weights) {
	double loss = 0;
	double wsum = 0;
	for (auto row : rows) {
		double max_score = prediction[row][0];
		for (idx_t c = 1; c < n_classes; c++) {
			max_score = MaxValue(max_score, prediction[row][c]);
		}
		double sum_exp = 0;
		for (idx_t c = 0; c < n_classes; c++) {
			sum_exp += std::exp(prediction[row][c] - max_score);
		}
		auto label = static_cast<idx_t>(y[row]);
		double p = std::exp(prediction[row][label] - max_score) / sum_exp;
		p = std::min(1.0 - 1e-15, std::max(1e-15, p));
		loss += weights[row] * -std::log(p);
		wsum += weights[row];
	}
	return wsum > 0 ? loss / wsum : std::numeric_limits<double>::infinity();
}

vector<bool> ResolveCategoricalFeatures(const vector<string> &tokens, const vector<string> &feature_names) {
	vector<bool> categorical(feature_names.size(), false);
	for (auto &token : tokens) {
		bool is_index = !token.empty();
		for (auto ch : token) {
			if (ch < '0' || ch > '9') {
				is_index = false;
				break;
			}
		}
		if (is_index) {
			idx_t feature_idx = 0;
			for (auto ch : token) {
				auto digit = static_cast<idx_t>(ch - '0');
				if (feature_idx > (std::numeric_limits<idx_t>::max() - digit) / 10) {
					throw InvalidInputException("duckboost: categorical feature index '%s' is out of range", token);
				}
				feature_idx = feature_idx * 10 + digit;
			}
			if (feature_idx >= feature_names.size()) {
				throw InvalidInputException("duckboost: categorical feature index '%s' is out of range", token);
			}
			categorical[feature_idx] = true;
			continue;
		}

		auto match = std::find(feature_names.begin(), feature_names.end(), token);
		if (match == feature_names.end()) {
			throw InvalidInputException("duckboost: unknown categorical feature '%s'", token);
		}
		categorical[static_cast<idx_t>(match - feature_names.begin())] = true;
	}
	return categorical;
}

BoostModel TrainReference(const vector<double> &y, const vector<vector<double>> &x, const TrainOptions &options,
                          const vector<double> &weights_in) {
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
	model.loss = options.loss;
	model.objective_alpha = options.objective_alpha;
	model.learning_rate = options.learning_rate;
	model.n_features = n_features;
	model.n_raw_features = n_features;
	model.feature_names = options.feature_names;
	if (model.feature_names.empty()) {
		for (idx_t i = 0; i < n_features; i++) {
			model.feature_names.push_back("f" + std::to_string(i));
		}
	} else if (model.feature_names.size() != n_features) {
		throw InvalidInputException("duckboost: feature_names count (%llu) must match feature width (%llu)",
		                            (unsigned long long)model.feature_names.size(), (unsigned long long)n_features);
	}
	auto categorical_features = ResolveCategoricalFeatures(options.categorical_features, model.feature_names);

	auto weights = NormalizeWeights(y, weights_in);
	SimpleRng rng(options.seed);

	vector<idx_t> all_rows(y.size());
	std::iota(all_rows.begin(), all_rows.end(), 0);
	vector<idx_t> train_rows = all_rows;
	vector<idx_t> valid_rows;
	if (options.validation_fraction > 0 && options.early_stopping_rounds > 0 && y.size() >= 4) {
		auto shuffled = all_rows;
		for (idx_t i = 0; i < shuffled.size(); i++) {
			idx_t j = i + rng.Bounded(shuffled.size() - i);
			std::swap(shuffled[i], shuffled[j]);
		}
		idx_t valid_n = MaxValue<idx_t>(
		    1, static_cast<idx_t>(std::floor(options.validation_fraction * static_cast<double>(y.size()))));
		valid_n = MinValue<idx_t>(valid_n, y.size() - 1);
		valid_rows.assign(shuffled.begin(), shuffled.begin() + valid_n);
		train_rows.assign(shuffled.begin() + valid_n, shuffled.end());
		std::sort(train_rows.begin(), train_rows.end());
		std::sort(valid_rows.begin(), valid_rows.end());
	}

	// Global quantile histogram boundaries (and per-row bin ids) are fixed before boosting.
	auto binnings = BuildFeatureBinnings(x, train_rows, categorical_features, options.max_bins);
	auto bins = AssignFeatureBins(x, binnings);

	double best_valid = std::numeric_limits<double>::infinity();
	idx_t best_rounds = 0;
	idx_t rounds_since_improve = 0;

	if (options.task == BoostTask::MULTICLASS) {
		const idx_t n_classes = ResolveClassCount(y, options);
		for (auto label : y) {
			if (static_cast<idx_t>(label) >= n_classes) {
				throw InvalidInputException("duckboost: multiclass label %g >= n_classes %llu", label,
				                            (unsigned long long)n_classes);
			}
		}
		model.n_classes = n_classes;
		ApplyClassWeights(weights, y, options, n_classes);

		vector<double> class_weight_sum(n_classes, 0);
		double total_w = 0;
		for (auto row : train_rows) {
			auto label = static_cast<idx_t>(y[row]);
			class_weight_sum[label] += weights[row];
			total_w += weights[row];
		}
		model.base_scores.assign(n_classes, 0);
		for (idx_t c = 0; c < n_classes; c++) {
			auto prior = class_weight_sum[c] / MaxValue(total_w, 1e-12);
			prior = std::min(1.0 - 1e-6, std::max(1e-6, prior));
			model.base_scores[c] = std::log(prior);
		}
		model.base_score = model.base_scores[0];

		vector<vector<double>> prediction(y.size(), model.base_scores);
		for (idx_t round = 0; round < options.n_estimators; round++) {
			auto grow_rows = SampleRows(train_rows, options.subsample, rng);
			auto feature_subset = SampleFeatures(n_features, options.colsample_bytree, rng);
			for (idx_t c = 0; c < n_classes; c++) {
				vector<double> gradients(y.size());
				vector<double> hessians(y.size());
				for (idx_t i = 0; i < y.size(); i++) {
					double max_score = prediction[i][0];
					for (idx_t k = 1; k < n_classes; k++) {
						max_score = MaxValue(max_score, prediction[i][k]);
					}
					double sum_exp = 0;
					for (idx_t k = 0; k < n_classes; k++) {
						sum_exp += std::exp(prediction[i][k] - max_score);
					}
					double p = std::exp(prediction[i][c] - max_score) / sum_exp;
					double target = (static_cast<idx_t>(y[i]) == c) ? 1.0 : 0.0;
					gradients[i] = weights[i] * (p - target);
					hessians[i] = weights[i] * std::max(p * (1.0 - p), 1e-6);
				}
				BoostTree tree;
				GrowTree(tree, x, bins, binnings, gradients, hessians, grow_rows, feature_subset, options);
				for (idx_t i = 0; i < y.size(); i++) {
					prediction[i][c] += options.learning_rate * ApplyTree(tree, x[i]);
				}
				model.trees.push_back(std::move(tree));
			}
			if (!valid_rows.empty()) {
				auto metric = EvalValidMulticlassLogloss(y, prediction, n_classes, valid_rows, weights);
				if (metric < best_valid - 1e-12) {
					best_valid = metric;
					best_rounds = round + 1;
					rounds_since_improve = 0;
				} else {
					rounds_since_improve++;
					if (rounds_since_improve >= options.early_stopping_rounds) {
						break;
					}
				}
			}
		}
		if (!valid_rows.empty() && best_rounds > 0 && best_rounds * n_classes < model.trees.size()) {
			model.trees.resize(best_rounds * n_classes);
		}
		return model;
	}

	if (options.task == BoostTask::BINARY) {
		ApplyClassWeights(weights, y, options, 2);
		for (auto label : y) {
			if (!(label == 0.0 || label == 1.0)) {
				throw InvalidInputException("duckboost: binary task requires labels in {0, 1}");
			}
		}
		double pos = WeightedMean(y, weights, train_rows);
		pos = std::min(1.0 - 1e-6, std::max(1e-6, pos));
		model.base_score = std::log(pos / (1.0 - pos));
	} else {
		if (!options.class_weight.empty()) {
			throw InvalidInputException("duckboost: class_weight is only supported for binary and multiclass tasks");
		}
		switch (options.loss) {
		case RegressionLoss::ABSOLUTE_ERROR:
			model.base_score = WeightedQuantile(y, weights, train_rows, 0.5);
			break;
		case RegressionLoss::QUANTILE:
			model.base_score = WeightedQuantile(y, weights, train_rows, options.objective_alpha);
			break;
		case RegressionLoss::EXPECTILE:
			model.base_score = WeightedExpectile(y, weights, train_rows, options.objective_alpha);
			break;
		case RegressionLoss::SQUARED_ERROR:
		default:
			model.base_score = WeightedMean(y, weights, train_rows);
			break;
		}
	}
	model.n_classes = 1;

	vector<double> prediction(y.size(), model.base_score);
	for (idx_t round = 0; round < options.n_estimators; round++) {
		vector<double> gradients(y.size());
		vector<double> hessians(y.size(), 1.0);
		for (idx_t i = 0; i < y.size(); i++) {
			if (options.task == BoostTask::BINARY) {
				double p = 1.0 / (1.0 + std::exp(-prediction[i]));
				gradients[i] = weights[i] * (p - y[i]);
				hessians[i] = weights[i] * std::max(p * (1.0 - p), 1e-6);
			} else {
				switch (options.loss) {
				case RegressionLoss::ABSOLUTE_ERROR:
					gradients[i] = weights[i] * ((prediction[i] > y[i]) - (prediction[i] < y[i]));
					hessians[i] = weights[i];
					break;
				case RegressionLoss::QUANTILE:
					gradients[i] =
					    weights[i] * (prediction[i] >= y[i] ? 1.0 - options.objective_alpha : -options.objective_alpha);
					hessians[i] = weights[i];
					break;
				case RegressionLoss::EXPECTILE: {
					auto asymmetry =
					    y[i] >= prediction[i] ? 2.0 * options.objective_alpha : 2.0 * (1.0 - options.objective_alpha);
					gradients[i] = weights[i] * asymmetry * (prediction[i] - y[i]);
					hessians[i] = weights[i] * asymmetry;
					break;
				}
				case RegressionLoss::SQUARED_ERROR:
				default:
					gradients[i] = weights[i] * (prediction[i] - y[i]);
					hessians[i] = weights[i];
					break;
				}
			}
		}
		auto grow_rows = SampleRows(train_rows, options.subsample, rng);
		auto feature_subset = SampleFeatures(n_features, options.colsample_bytree, rng);
		BoostTree tree;
		GrowTree(tree, x, bins, binnings, gradients, hessians, grow_rows, feature_subset, options);
		if (options.task == BoostTask::REGRESSION &&
		    (options.loss == RegressionLoss::ABSOLUTE_ERROR || options.loss == RegressionLoss::QUANTILE)) {
			auto alpha = options.loss == RegressionLoss::ABSOLUTE_ERROR ? 0.5 : options.objective_alpha;
			RenewQuantileLeaves(tree, x, y, prediction, weights, grow_rows, alpha);
		}
		for (idx_t i = 0; i < y.size(); i++) {
			prediction[i] += options.learning_rate * ApplyTree(tree, x[i]);
		}
		model.trees.push_back(std::move(tree));

		if (!valid_rows.empty()) {
			double metric =
			    options.task == BoostTask::BINARY
			        ? EvalValidBinaryLogloss(y, prediction, valid_rows, weights)
			        : EvalValidRegression(y, prediction, valid_rows, weights, options.loss, options.objective_alpha);
			if (metric < best_valid - 1e-12) {
				best_valid = metric;
				best_rounds = model.trees.size();
				rounds_since_improve = 0;
			} else {
				rounds_since_improve++;
				if (rounds_since_improve >= options.early_stopping_rounds) {
					break;
				}
			}
		}
	}

	if (!valid_rows.empty() && best_rounds > 0 && best_rounds < model.trees.size()) {
		model.trees.resize(best_rounds);
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

BoostModel TrainModel(const vector<double> &y, const vector<vector<double>> &x, const TrainOptions &options,
                      const vector<double> &weights) {
	if (options.backend == BoostBackend::REFERENCE) {
		return TrainReference(y, x, options, weights);
	}
	if (options.growth_policy == GrowthPolicy::OBLIVIOUS && options.backend != BoostBackend::REFERENCE) {
		throw NotImplementedException("duckboost: growth_policy='oblivious' requires backend='reference'");
	}
	if (options.backend == BoostBackend::XGBOOST || options.backend == BoostBackend::CATBOOST) {
		if (options.growth_policy_set || options.max_leaves_set || options.growth_policy != GrowthPolicy::DEPTHWISE) {
			throw NotImplementedException(
			    "duckboost: growth_policy and max_leaves require backend='lightgbm' or 'reference'");
		}
		if (options.loss != RegressionLoss::SQUARED_ERROR || options.objective_alpha_set) {
			throw NotImplementedException(
			    "duckboost: objective '%s' requires backend='lightgbm' or 'reference' for native training",
			    RegressionLossToString(options.loss));
		}
		if (!options.categorical_features.empty()) {
			throw NotImplementedException("duckboost: categorical_features requires backend='lightgbm' or 'reference'");
		}
	}
	if (options.backend == BoostBackend::LIGHTGBM && options.loss == RegressionLoss::EXPECTILE) {
		throw NotImplementedException("duckboost: expectile is supported by the reference backend only");
	}
	if (NativeTrainerCompiled(options.backend)) {
		if (options.class_weight.empty()) {
			return TrainNative(y, x, options, weights);
		}
		if (options.task == BoostTask::REGRESSION) {
			throw InvalidInputException("duckboost: class_weight is only supported for binary and multiclass tasks");
		}
		auto native_weights = NormalizeWeights(y, weights);
		ApplyClassWeights(native_weights, y, options,
		                  options.task == BoostTask::MULTICLASS ? ResolveClassCount(y, options) : 2);
		return TrainNative(y, x, options, native_weights);
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
			switch (model.loss) {
			case RegressionLoss::ABSOLUTE_ERROR:
				metric = "mae";
				break;
			case RegressionLoss::QUANTILE:
				metric = "pinball";
				break;
			case RegressionLoss::EXPECTILE:
				metric = "expectile";
				break;
			case RegressionLoss::SQUARED_ERROR:
			default:
				metric = "rmse";
				break;
			}
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
	if (metric == "pinball" || metric == "quantile") {
		double loss = 0;
		for (idx_t i = 0; i < y.size(); i++) {
			loss += RegressionLossValue(RegressionLoss::QUANTILE, model.objective_alpha, y[i], model.Predict(x[i]));
		}
		return loss / static_cast<double>(y.size());
	}
	if (metric == "expectile") {
		double loss = 0;
		for (idx_t i = 0; i < y.size(); i++) {
			loss += RegressionLossValue(RegressionLoss::EXPECTILE, model.objective_alpha, y[i], model.Predict(x[i]));
		}
		return loss / static_cast<double>(y.size());
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
	throw InvalidInputException(
	    "duckboost: unknown metric '%s' (expected auto, rmse, mae, pinball, quantile, expectile, accuracy, logloss)",
	    options.metric);
}

} // namespace duckboost
} // namespace duckdb
