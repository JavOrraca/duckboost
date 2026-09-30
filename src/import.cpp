#include "duckboost/import.hpp"
#include "duckboost/json_util.hpp"

#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/unordered_map.hpp"

#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <sstream>

namespace duckdb {
namespace duckboost {

namespace {

double ThresholdForLessEqual(double border) {
	// duckboost uses feature < threshold; convert vendor "<=" / CatBoost False paths.
	if (!std::isfinite(border)) {
		return border;
	}
	auto next = std::nextafter(border, std::numeric_limits<double>::infinity());
	// Avoid subnormals: they round-trip poorly through text/JSON numeric parsers.
	if (std::fpclassify(next) == FP_SUBNORMAL || next == 0.0) {
		return std::numeric_limits<double>::min();
	}
	return next;
}

idx_t ParseFeatureIndex(const string &name, unordered_map<string, idx_t> &name_to_idx, idx_t &n_features) {
	auto it = name_to_idx.find(name);
	if (it != name_to_idx.end()) {
		return it->second;
	}
	if (!name.empty() && (name[0] == 'f' || name[0] == 'F')) {
		bool all_digits = true;
		for (idx_t i = 1; i < name.size(); i++) {
			if (!std::isdigit(static_cast<unsigned char>(name[i]))) {
				all_digits = false;
				break;
			}
		}
		if (all_digits && name.size() > 1) {
			auto idx = static_cast<idx_t>(std::stoull(name.substr(1)));
			n_features = MaxValue<idx_t>(n_features, idx + 1);
			name_to_idx[name] = idx;
			return idx;
		}
	}
	auto idx = n_features++;
	name_to_idx[name] = idx;
	return idx;
}

void ApplyImportOptions(BoostModel &model, const ImportOptions &options) {
	if (options.task_set) {
		model.task = options.task;
	}
	if (options.base_score_set) {
		model.base_score = options.base_score;
		if (model.task != BoostTask::MULTICLASS) {
			model.base_scores.clear();
		}
	}
	if (options.learning_rate_set) {
		model.learning_rate = options.learning_rate;
	}
	if (options.n_classes_set) {
		model.n_classes = options.n_classes;
		if (model.n_classes >= 2) {
			model.task = BoostTask::MULTICLASS;
		}
	}
	if (!options.feature_names.empty()) {
		model.feature_names = options.feature_names;
		model.n_features = MaxValue<idx_t>(model.n_features, model.feature_names.size());
	}
	if (model.feature_names.size() < model.n_features) {
		model.feature_names.resize(model.n_features);
	}
	for (idx_t i = 0; i < model.feature_names.size(); i++) {
		if (model.feature_names[i].empty()) {
			model.feature_names[i] = "f" + std::to_string(i);
		}
	}
	if (model.task == BoostTask::MULTICLASS && model.n_classes < 2) {
		throw InvalidInputException("duckboost: multiclass models require n_classes >= 2");
	}
	if (model.task != BoostTask::MULTICLASS) {
		model.n_classes = 1;
	}
}

struct XGBNode {
	bool is_leaf = false;
	idx_t nodeid = 0;
	idx_t yes = 0;
	idx_t no = 0;
	idx_t missing = 0;
	bool has_missing = false;
	string split;
	double split_condition = 0;
	double leaf = 0;
	vector<XGBNode> children;
};

XGBNode ParseXGBNode(JsonParser &p) {
	p.Expect('{');
	XGBNode node;
	bool first = true;
	while (!p.TryConsume('}')) {
		if (!first) {
			p.Expect(',');
		}
		first = false;
		auto key = p.ParseString();
		p.Expect(':');
		if (key == "nodeid") {
			node.nodeid = static_cast<idx_t>(p.ParseNumber());
		} else if (key == "leaf") {
			node.is_leaf = true;
			node.leaf = p.ParseNumber();
		} else if (key == "split") {
			node.split = p.ParseString();
		} else if (key == "split_condition") {
			node.split_condition = p.ParseNumber();
		} else if (key == "yes") {
			node.yes = static_cast<idx_t>(p.ParseNumber());
		} else if (key == "no") {
			node.no = static_cast<idx_t>(p.ParseNumber());
		} else if (key == "missing") {
			node.missing = static_cast<idx_t>(p.ParseNumber());
			node.has_missing = true;
		} else if (key == "children") {
			p.Expect('[');
			bool first_child = true;
			while (!p.TryConsume(']')) {
				if (!first_child) {
					p.Expect(',');
				}
				first_child = false;
				node.children.push_back(ParseXGBNode(p));
			}
		} else {
			p.SkipValue();
		}
	}
	return node;
}

idx_t ConvertXGBNode(const XGBNode &node, BoostTree &tree, unordered_map<string, idx_t> &name_to_idx,
                     idx_t &n_features) {
	if (node.is_leaf || node.children.empty()) {
		TreeNode leaf;
		leaf.is_leaf = true;
		leaf.value = node.leaf;
		tree.nodes.push_back(leaf);
		return tree.nodes.size() - 1;
	}
	TreeNode out;
	out.is_leaf = false;
	out.feature = ParseFeatureIndex(node.split, name_to_idx, n_features);
	out.threshold = node.split_condition;
	auto idx = tree.nodes.size();
	tree.nodes.push_back(out);

	const XGBNode *yes_child = nullptr;
	const XGBNode *no_child = nullptr;
	for (auto &child : node.children) {
		if (child.nodeid == node.yes) {
			yes_child = &child;
		} else if (child.nodeid == node.no) {
			no_child = &child;
		}
	}
	if (!yes_child || !no_child) {
		if (node.children.size() != 2) {
			throw InvalidInputException("duckboost: xgboost node %llu missing yes/no children",
			                            (unsigned long long)node.nodeid);
		}
		yes_child = &node.children[0];
		no_child = &node.children[1];
	}
	// XGBoost: yes = feature < threshold (left), no = else (right). missing points at yes or no.
	tree.nodes[idx].default_left = !node.has_missing || node.missing == node.yes;
	tree.nodes[idx].left = ConvertXGBNode(*yes_child, tree, name_to_idx, n_features);
	tree.nodes[idx].right = ConvertXGBNode(*no_child, tree, name_to_idx, n_features);
	return idx;
}

vector<string> SplitWhitespace(const string &line) {
	vector<string> parts;
	std::stringstream ss(line);
	string part;
	while (ss >> part) {
		parts.push_back(part);
	}
	return parts;
}

vector<double> ParseDoubleList(const string &value) {
	vector<double> out;
	if (value.empty()) {
		return out;
	}
	auto parts = StringUtil::Split(value, ' ');
	for (auto &part : parts) {
		StringUtil::Trim(part);
		if (part.empty()) {
			continue;
		}
		out.push_back(std::stod(part));
	}
	return out;
}

vector<int64_t> ParseIntList(const string &value) {
	vector<int64_t> out;
	if (value.empty()) {
		return out;
	}
	auto parts = StringUtil::Split(value, ' ');
	for (auto &part : parts) {
		StringUtil::Trim(part);
		if (part.empty()) {
			continue;
		}
		out.push_back(std::stoll(part));
	}
	return out;
}

idx_t ConvertLGBNode(idx_t internal_idx, const vector<int64_t> &split_feature, const vector<double> &threshold,
                     const vector<int64_t> &left_child, const vector<int64_t> &right_child,
                     const vector<int64_t> &decision_type, const vector<idx_t> &cat_boundaries,
                     const vector<uint32_t> &cat_threshold, const vector<double> &leaf_value, double shrinkage,
                     BoostTree &tree) {
	if (internal_idx >= split_feature.size()) {
		throw InvalidInputException("duckboost: lightgbm internal node index out of range");
	}
	TreeNode node;
	node.is_leaf = false;
	node.feature = static_cast<idx_t>(split_feature[internal_idx]);
	auto decision = decision_type.empty() ? int64_t(0) : decision_type[internal_idx];
	if (decision < 0 || decision > std::numeric_limits<uint8_t>::max()) {
		throw InvalidInputException("duckboost: invalid lightgbm decision_type");
	}
	auto categorical = (decision & 1) != 0;
	if (categorical) {
		auto raw_cat_idx = threshold[internal_idx];
		if (!std::isfinite(raw_cat_idx) || raw_cat_idx < 0 || std::floor(raw_cat_idx) != raw_cat_idx ||
		    raw_cat_idx >= static_cast<double>(cat_boundaries.size() - 1)) {
			throw InvalidInputException("duckboost: lightgbm categorical threshold index out of range");
		}
		auto cat_idx = static_cast<idx_t>(raw_cat_idx);
		auto begin = cat_boundaries[cat_idx];
		auto end = cat_boundaries[cat_idx + 1];
		for (idx_t word_idx = begin; word_idx < end; word_idx++) {
			auto word = cat_threshold[word_idx];
			for (idx_t bit = 0; bit < 32; bit++) {
				if ((word & (uint32_t(1) << bit)) != 0) {
					node.categories.push_back(static_cast<double>(32 * (word_idx - begin) + bit));
				}
			}
		}
		if (node.categories.empty()) {
			throw InvalidInputException("duckboost: lightgbm categorical split has no categories");
		}
		node.default_left = true;
		if (node.categories.size() == 1) {
			node.compare = SplitCompare::EQUAL;
			node.threshold = node.categories[0];
			node.categories.clear();
		} else {
			node.compare = SplitCompare::IN;
		}
	} else {
		node.threshold = ThresholdForLessEqual(threshold[internal_idx]);
		node.default_left = (decision & 2) != 0;
		switch ((decision >> 2) & 3) {
		case 0:
			node.missing_type = SplitMissingType::NONE;
			break;
		case 1:
			node.missing_type = SplitMissingType::ZERO;
			break;
		case 2:
			node.missing_type = SplitMissingType::NAN_VALUE;
			break;
		default:
			throw InvalidInputException("duckboost: invalid lightgbm missing_type");
		}
	}
	auto idx = tree.nodes.size();
	tree.nodes.push_back(node);

	auto attach = [&](int64_t child) -> idx_t {
		if (child < 0) {
			TreeNode leaf;
			leaf.is_leaf = true;
			auto leaf_idx = static_cast<idx_t>(~child);
			if (leaf_idx >= leaf_value.size()) {
				throw InvalidInputException("duckboost: lightgbm leaf index out of range");
			}
			leaf.value = leaf_value[leaf_idx] * shrinkage;
			tree.nodes.push_back(leaf);
			return tree.nodes.size() - 1;
		}
		return ConvertLGBNode(static_cast<idx_t>(child), split_feature, threshold, left_child, right_child,
		                      decision_type, cat_boundaries, cat_threshold, leaf_value, shrinkage, tree);
	};

	// LightGBM categorical matches go left. Duckboost EQUAL/IN matches go right.
	tree.nodes[idx].left = attach(categorical ? right_child[internal_idx] : left_child[internal_idx]);
	tree.nodes[idx].right = attach(categorical ? left_child[internal_idx] : right_child[internal_idx]);
	return idx;
}

idx_t ExpandCatBoost(idx_t depth, idx_t path_bits, const vector<idx_t> &features, const vector<double> &thresholds,
                     const vector<SplitCompare> &compares, const vector<double> &leaf_values, BoostTree &tree) {
	if (depth == features.size()) {
		if (path_bits >= leaf_values.size()) {
			throw InvalidInputException("duckboost: catboost leaf index out of range");
		}
		TreeNode leaf;
		leaf.is_leaf = true;
		leaf.value = leaf_values[path_bits];
		tree.nodes.push_back(leaf);
		return tree.nodes.size() - 1;
	}
	TreeNode node;
	node.is_leaf = false;
	node.feature = features[depth];
	node.threshold = thresholds[depth];
	node.compare = compares[depth];
	auto idx = tree.nodes.size();
	tree.nodes.push_back(node);
	// False → left / bit 0; True → right / bit 1
	tree.nodes[idx].left = ExpandCatBoost(depth + 1, path_bits << 1, features, thresholds, compares, leaf_values, tree);
	tree.nodes[idx].right =
	    ExpandCatBoost(depth + 1, (path_bits << 1) | 1, features, thresholds, compares, leaf_values, tree);
	return idx;
}

enum class CatSplitKind : uint8_t { FLOAT = 0, ONE_HOT = 1, CTR = 2 };

struct GlobalCatSplit {
	CatSplitKind kind = CatSplitKind::FLOAT;
	idx_t feature = 0;
	double border_or_value = 0;
	idx_t ctr_index = 0; // index into features_info.ctrs / model.ctr_features
};

} // namespace

ImportOptions ImportOptions::FromMap(const unordered_map<string, string> &options) {
	ImportOptions result;
	for (auto &entry : options) {
		auto key = StringUtil::Lower(entry.first);
		auto &value = entry.second;
		if (key == "task" || key == "objective") {
			result.task = TaskFromString(value);
			result.task_set = true;
		} else if (key == "base_score" || key == "bias") {
			result.base_score = std::stod(value);
			result.base_score_set = true;
		} else if (key == "learning_rate" || key == "eta" || key == "lr" || key == "scale") {
			result.learning_rate = std::stod(value);
			result.learning_rate_set = true;
		} else if (key == "feature_names") {
			result.feature_names = StringUtil::Split(value, ',');
			for (auto &name : result.feature_names) {
				StringUtil::Trim(name);
			}
		} else if (key == "n_classes" || key == "classes_count" || key == "num_class" || key == "num_classes") {
			result.n_classes = static_cast<idx_t>(std::stoull(value));
			result.n_classes_set = true;
		} else {
			throw InvalidInputException("duckboost: unknown import option '%s'", entry.first);
		}
	}
	return result;
}

bool LooksLikeDuckBoostJSON(const string &dump) {
	auto trimmed = dump;
	StringUtil::Trim(trimmed);
	return trimmed.find("\"duckboost_version\"") != string::npos;
}

BoostModel ImportXGBoostJSON(const string &dump, const ImportOptions &options) {
	JsonParser p(dump);
	p.Expect('[');
	BoostModel model;
	model.backend = BoostBackend::XGBOOST;
	model.learning_rate = 1.0;
	model.task = BoostTask::REGRESSION;
	unordered_map<string, idx_t> name_to_idx;
	bool first = true;
	while (!p.TryConsume(']')) {
		if (!first) {
			p.Expect(',');
		}
		first = false;
		auto root = ParseXGBNode(p);
		BoostTree tree;
		ConvertXGBNode(root, tree, name_to_idx, model.n_features);
		if (tree.nodes.empty()) {
			throw InvalidInputException("duckboost: empty xgboost tree");
		}
		model.trees.push_back(std::move(tree));
	}
	for (auto &entry : name_to_idx) {
		if (model.feature_names.size() < model.n_features) {
			model.feature_names.resize(model.n_features);
		}
		model.feature_names[entry.second] = entry.first;
	}
	ApplyImportOptions(model, options);
	return model;
}

BoostModel ImportLightGBMText(const string &dump, const ImportOptions &options) {
	BoostModel model;
	model.backend = BoostBackend::LIGHTGBM;
	model.learning_rate = 1.0;
	model.task = BoostTask::REGRESSION;

	auto lines = StringUtil::Split(dump, '\n');
	std::map<string, string> tree_fields;
	bool in_tree = false;
	auto flush_tree = [&]() {
		if (!in_tree) {
			return;
		}
		auto num_leaves = static_cast<idx_t>(std::stoull(tree_fields["num_leaves"]));
		if (num_leaves == 0) {
			throw InvalidInputException("duckboost: lightgbm tree has zero leaves");
		}
		auto split_feature = ParseIntList(tree_fields["split_feature"]);
		auto threshold = ParseDoubleList(tree_fields["threshold"]);
		auto left_child = ParseIntList(tree_fields["left_child"]);
		auto right_child = ParseIntList(tree_fields["right_child"]);
		auto leaf_value = ParseDoubleList(tree_fields["leaf_value"]);
		auto decision_type = ParseIntList(tree_fields.count("decision_type") ? tree_fields["decision_type"] : "");
		auto num_cat = tree_fields.count("num_cat") ? static_cast<idx_t>(std::stoull(tree_fields["num_cat"])) : 0;
		auto raw_cat_boundaries =
		    ParseIntList(tree_fields.count("cat_boundaries") ? tree_fields["cat_boundaries"] : "");
		auto raw_cat_threshold = ParseIntList(tree_fields.count("cat_threshold") ? tree_fields["cat_threshold"] : "");
		vector<idx_t> cat_boundaries;
		vector<uint32_t> cat_threshold;
		if (num_cat == 0) {
			if ((!raw_cat_boundaries.empty() &&
			     (raw_cat_boundaries.size() != 1 || raw_cat_boundaries[0] != 0)) ||
			    !raw_cat_threshold.empty()) {
				throw InvalidInputException("duckboost: lightgbm categorical arrays present with num_cat=0");
			}
			cat_boundaries.push_back(0);
		} else {
			if (raw_cat_boundaries.size() != num_cat + 1) {
				throw InvalidInputException("duckboost: lightgbm cat_boundaries size does not match num_cat");
			}
			for (idx_t i = 0; i < raw_cat_boundaries.size(); i++) {
				auto boundary = raw_cat_boundaries[i];
				if (boundary < 0 || (i == 0 && boundary != 0) ||
				    (i > 0 && boundary <= raw_cat_boundaries[i - 1]) ||
				    static_cast<uint64_t>(boundary) > raw_cat_threshold.size()) {
					throw InvalidInputException("duckboost: malformed lightgbm cat_boundaries");
				}
				cat_boundaries.push_back(static_cast<idx_t>(boundary));
			}
			if (cat_boundaries.back() != raw_cat_threshold.size()) {
				throw InvalidInputException("duckboost: lightgbm cat_boundaries do not cover cat_threshold");
			}
			for (auto word : raw_cat_threshold) {
				if (word < 0 || static_cast<uint64_t>(word) > std::numeric_limits<uint32_t>::max()) {
					throw InvalidInputException("duckboost: invalid lightgbm cat_threshold word");
				}
				cat_threshold.push_back(static_cast<uint32_t>(word));
			}
		}
		double shrinkage = 1.0;
		if (tree_fields.count("shrinkage")) {
			shrinkage = std::stod(tree_fields["shrinkage"]);
		}
		if (leaf_value.size() != num_leaves) {
			throw InvalidInputException("duckboost: lightgbm leaf_value size mismatch");
		}
		if (num_cat > num_leaves - 1) {
			throw InvalidInputException("duckboost: lightgbm num_cat exceeds split count");
		}
		BoostTree tree;
		if (num_leaves == 1) {
			TreeNode leaf;
			leaf.is_leaf = true;
			leaf.value = leaf_value[0] * shrinkage;
			tree.nodes.push_back(leaf);
		} else {
			if (split_feature.size() != num_leaves - 1 || threshold.size() != num_leaves - 1 ||
			    left_child.size() != num_leaves - 1 || right_child.size() != num_leaves - 1) {
				throw InvalidInputException("duckboost: lightgbm split array size mismatch");
			}
			if (!decision_type.empty() && decision_type.size() != num_leaves - 1) {
				throw InvalidInputException("duckboost: lightgbm decision_type size mismatch");
			}
			idx_t categorical_count = 0;
			for (auto dt : decision_type) {
				if (dt & 1) {
					categorical_count++;
				}
			}
			if (categorical_count != num_cat) {
				throw InvalidInputException("duckboost: lightgbm num_cat does not match categorical splits");
			}
			for (auto f : split_feature) {
				if (f < 0) {
					throw InvalidInputException("duckboost: lightgbm split_feature must be non-negative");
				}
				model.n_features = MaxValue<idx_t>(model.n_features, static_cast<idx_t>(f) + 1);
			}
			ConvertLGBNode(0, split_feature, threshold, left_child, right_child, decision_type, cat_boundaries,
			               cat_threshold, leaf_value, shrinkage, tree);
		}
		model.trees.push_back(std::move(tree));
		tree_fields.clear();
		in_tree = false;
	};

	for (auto &raw_line : lines) {
		auto line = raw_line;
		StringUtil::Trim(line);
		if (line.empty()) {
			continue;
		}
		if (line.rfind("Tree=", 0) == 0) {
			flush_tree();
			in_tree = true;
			continue;
		}
		if (line == "end of trees" || line.rfind("tree_sizes=", 0) == 0) {
			flush_tree();
			continue;
		}
		auto eq = line.find('=');
		if (eq == string::npos) {
			continue;
		}
		auto key = line.substr(0, eq);
		auto value = line.substr(eq + 1);
		StringUtil::Trim(key);
		StringUtil::Trim(value);
		if (in_tree) {
			tree_fields[key] = value;
			continue;
		}
		if (key == "objective") {
			auto lower = StringUtil::Lower(value);
			if (lower.find("binary") != string::npos || lower.find("logistic") != string::npos) {
				model.task = BoostTask::BINARY;
			} else {
				model.task = BoostTask::REGRESSION;
			}
		} else if (key == "feature_names") {
			model.feature_names = SplitWhitespace(value);
			model.n_features = MaxValue<idx_t>(model.n_features, model.feature_names.size());
		} else if (key == "max_feature_idx") {
			model.n_features = MaxValue<idx_t>(model.n_features, static_cast<idx_t>(std::stoull(value)) + 1);
		}
	}
	flush_tree();
	if (model.trees.empty()) {
		throw InvalidInputException("duckboost: no trees found in lightgbm dump");
	}
	ApplyImportOptions(model, options);
	return model;
}

BoostModel ImportCatBoostJSON(const string &dump, const ImportOptions &options) {
	JsonParser p(dump);
	p.Expect('{');
	BoostModel model;
	model.backend = BoostBackend::CATBOOST;
	model.learning_rate = 1.0;
	model.task = BoostTask::REGRESSION;

	struct FloatFeatureInfo {
		idx_t feature_index = 0;
		idx_t flat_feature_index = 0;
		vector<double> borders;
	};
	struct CatFeatureInfo {
		idx_t feature_index = 0;
		idx_t flat_feature_index = 0;
		vector<double> one_hot_values;
	};
	struct CtrElementInfo {
		string element_type = "cat_feature_value";
		idx_t feature_index = 0; // cat_feature_index or float_feature_index before remap
		double border_or_value = 0;
	};
	struct CtrInfo {
		string ctr_type = "Counter";
		string identifier;
		double prior_numerator = 0;
		double prior_denominator = 1;
		double scale = 1;
		double shift = 0;
		vector<double> borders;
		vector<CtrElementInfo> elements;
	};

	vector<FloatFeatureInfo> float_features;
	vector<CatFeatureInfo> cat_features;
	vector<CtrInfo> ctr_infos;
	unordered_map<string, CtrFeatureSpec> ctr_data_by_id;
	double scale = 1.0;
	vector<double> biases = {0.0};
	bool has_trees = false;
	idx_t detected_classes = 0;

	struct TreeSplit {
		idx_t feature = 0;
		double border = 0;
		SplitCompare compare = SplitCompare::LESS;
		CatSplitKind kind = CatSplitKind::FLOAT;
		idx_t ctr_index = 0;
		bool resolved = false;
		idx_t split_index = 0;
		string split_type = "FloatFeature";
	};
	struct PendingTree {
		vector<double> leaf_values;
		vector<TreeSplit> splits;
	};
	vector<PendingTree> pending_trees;

	bool first = true;
	while (!p.TryConsume('}')) {
		if (!first) {
			p.Expect(',');
		}
		first = false;
		auto key = p.ParseString();
		p.Expect(':');
		if (key == "features_info") {
			p.Expect('{');
			bool first_fi = true;
			while (!p.TryConsume('}')) {
				if (!first_fi) {
					p.Expect(',');
				}
				first_fi = false;
				auto fi_key = p.ParseString();
				p.Expect(':');
				if (fi_key == "float_features") {
					p.Expect('[');
					bool first_feat = true;
					while (!p.TryConsume(']')) {
						if (!first_feat) {
							p.Expect(',');
						}
						first_feat = false;
						p.Expect('{');
						FloatFeatureInfo feat;
						bool has_feature_index = false;
						bool has_flat = false;
						bool first_field = true;
						while (!p.TryConsume('}')) {
							if (!first_field) {
								p.Expect(',');
							}
							first_field = false;
							auto fkey = p.ParseString();
							p.Expect(':');
							if (fkey == "feature_index") {
								feat.feature_index = static_cast<idx_t>(p.ParseNumber());
								has_feature_index = true;
							} else if (fkey == "flat_feature_index") {
								feat.flat_feature_index = static_cast<idx_t>(p.ParseNumber());
								has_flat = true;
							} else if (fkey == "borders") {
								p.Expect('[');
								bool first_border = true;
								while (!p.TryConsume(']')) {
									if (!first_border) {
										p.Expect(',');
									}
									first_border = false;
									feat.borders.push_back(p.ParseNumber());
								}
							} else {
								p.SkipValue();
							}
						}
						if (!has_flat && has_feature_index) {
							feat.flat_feature_index = feat.feature_index;
						}
						if (!has_feature_index && has_flat) {
							feat.feature_index = feat.flat_feature_index;
						}
						float_features.push_back(std::move(feat));
					}
				} else if (fi_key == "categorical_features") {
					p.Expect('[');
					bool first_feat = true;
					while (!p.TryConsume(']')) {
						if (!first_feat) {
							p.Expect(',');
						}
						first_feat = false;
						p.Expect('{');
						CatFeatureInfo feat;
						bool has_feature_index = false;
						bool has_flat = false;
						bool first_field = true;
						while (!p.TryConsume('}')) {
							if (!first_field) {
								p.Expect(',');
							}
							first_field = false;
							auto fkey = p.ParseString();
							p.Expect(':');
							if (fkey == "feature_index") {
								feat.feature_index = static_cast<idx_t>(p.ParseNumber());
								has_feature_index = true;
							} else if (fkey == "flat_feature_index") {
								feat.flat_feature_index = static_cast<idx_t>(p.ParseNumber());
								has_flat = true;
							} else if (fkey == "values") {
								p.Expect('[');
								bool first_val = true;
								while (!p.TryConsume(']')) {
									if (!first_val) {
										p.Expect(',');
									}
									first_val = false;
									feat.one_hot_values.push_back(p.ParseNumber());
								}
							} else {
								p.SkipValue();
							}
						}
						if (!has_flat && has_feature_index) {
							feat.flat_feature_index = feat.feature_index;
						}
						if (!has_feature_index && has_flat) {
							feat.feature_index = feat.flat_feature_index;
						}
						cat_features.push_back(std::move(feat));
					}
				} else if (fi_key == "ctrs") {
					p.Expect('[');
					bool first_ctr = true;
					while (!p.TryConsume(']')) {
						if (!first_ctr) {
							p.Expect(',');
						}
						first_ctr = false;
						p.Expect('{');
						CtrInfo ctr;
						bool first_field = true;
						while (!p.TryConsume('}')) {
							if (!first_field) {
								p.Expect(',');
							}
							first_field = false;
							auto ckey = p.ParseString();
							p.Expect(':');
							if (ckey == "ctr_type" || ckey == "type") {
								ctr.ctr_type = p.ParseString();
							} else if (ckey == "identifier") {
								ctr.identifier = p.ParseString();
							} else if (ckey == "prior_numerator") {
								ctr.prior_numerator = p.ParseNumber();
							} else if (ckey == "prior_denomerator" || // typos:ignore
							           ckey == "prior_denominator") {
								ctr.prior_denominator = p.ParseNumber();
							} else if (ckey == "scale") {
								ctr.scale = p.ParseNumber();
							} else if (ckey == "shift") {
								ctr.shift = p.ParseNumber();
							} else if (ckey == "borders") {
								p.Expect('[');
								bool first_border = true;
								while (!p.TryConsume(']')) {
									if (!first_border) {
										p.Expect(',');
									}
									first_border = false;
									ctr.borders.push_back(p.ParseNumber());
								}
							} else if (ckey == "elements") {
								p.Expect('[');
								bool first_el = true;
								while (!p.TryConsume(']')) {
									if (!first_el) {
										p.Expect(',');
									}
									first_el = false;
									p.Expect('{');
									CtrElementInfo element;
									bool first_efield = true;
									while (!p.TryConsume('}')) {
										if (!first_efield) {
											p.Expect(',');
										}
										first_efield = false;
										auto ekey = p.ParseString();
										p.Expect(':');
										if (ekey == "combination_element") {
											element.element_type = p.ParseString();
										} else if (ekey == "cat_feature_index" || ekey == "float_feature_index" ||
										           ekey == "flat_feature_index") {
											element.feature_index = static_cast<idx_t>(p.ParseNumber());
										} else if (ekey == "border" || ekey == "value") {
											element.border_or_value = p.ParseNumber();
										} else {
											p.SkipValue();
										}
									}
									ctr.elements.push_back(std::move(element));
								}
							} else {
								p.SkipValue();
							}
						}
						ctr_infos.push_back(std::move(ctr));
					}
				} else {
					p.SkipValue();
				}
			}
		} else if (key == "ctr_data") {
			p.Expect('{');
			bool first_cd = true;
			while (!p.TryConsume('}')) {
				if (!first_cd) {
					p.Expect(',');
				}
				first_cd = false;
				auto identifier = p.ParseString();
				p.Expect(':');
				p.Expect('{');
				CtrFeatureSpec spec;
				spec.ctr_type = "Counter";
				bool first_field = true;
				while (!p.TryConsume('}')) {
					if (!first_field) {
						p.Expect(',');
					}
					first_field = false;
					auto dkey = p.ParseString();
					p.Expect(':');
					if (dkey == "counter_denominator") {
						spec.counter_denominator = static_cast<int64_t>(p.ParseNumber());
					} else if (dkey == "hash_stride") {
						p.ParseNumber();
					} else if (dkey == "hash_map") {
						p.Expect('[');
						// Detect stride by checking identifier later; parse as flat list of numbers/strings.
						vector<string> tokens;
						bool first_tok = true;
						while (!p.TryConsume(']')) {
							if (!first_tok) {
								p.Expect(',');
							}
							first_tok = false;
							if (p.Peek() == '"') {
								tokens.push_back(p.ParseString());
							} else {
								tokens.push_back(std::to_string(static_cast<int64_t>(p.ParseNumber())));
							}
						}
						// Store raw tokens temporarily in hash_keys as 0 and values as sentinel via string side
						// channel: We'll reinterpret after ctr_type is known; keep tokens in hash_values size as packed
						// later. For now assume Counter stride=2 (key, count) unless Borders (key, fail, success).
						// Parse as Counter by default; Borders rewrite when matched to ctr_infos.
						for (idx_t i = 0; i + 1 < tokens.size();) {
							spec.hash_keys.push_back(std::stoull(tokens[i]));
							spec.hash_values.push_back(std::stoll(tokens[i + 1]));
							if (i + 2 < tokens.size()) {
								// peek: if next looks like another key (large) vs success count — deferred
							}
							i += 2;
							// Keep alt empty for Counter; Borders fixed up below when identifier type known.
							(void)i;
						}
						// Save full token list encoded: put stride hint in hash_values_alt[0] as token count.
						spec.hash_values_alt.clear();
						for (auto &tok : tokens) {
							// reuse: store nothing; re-parse from keys/values only for Counter.
							(void)tok;
						}
						// Re-parse with flexible stride based on identifier substring.
						spec.hash_keys.clear();
						spec.hash_values.clear();
						spec.hash_values_alt.clear();
						idx_t stride = 2;
						auto id_lower = StringUtil::Lower(identifier);
						if (id_lower.find("borders") != string::npos || id_lower.find("buckets") != string::npos) {
							stride = 3;
							spec.ctr_type = "Borders";
						}
						for (idx_t i = 0; i + stride - 1 < tokens.size(); i += stride) {
							spec.hash_keys.push_back(std::stoull(tokens[i]));
							spec.hash_values.push_back(std::stoll(tokens[i + 1]));
							if (stride == 3) {
								spec.hash_values_alt.push_back(std::stoll(tokens[i + 2]));
							}
						}
					} else {
						p.SkipValue();
					}
				}
				ctr_data_by_id[identifier] = std::move(spec);
			}
		} else if (key == "oblivious_trees") {
			has_trees = true;
			p.Expect('[');
			bool first_tree = true;
			while (!p.TryConsume(']')) {
				if (!first_tree) {
					p.Expect(',');
				}
				first_tree = false;
				p.Expect('{');
				PendingTree tree;
				bool first_field = true;
				while (!p.TryConsume('}')) {
					if (!first_field) {
						p.Expect(',');
					}
					first_field = false;
					auto tkey = p.ParseString();
					p.Expect(':');
					if (tkey == "leaf_values") {
						p.Expect('[');
						bool first_leaf = true;
						while (!p.TryConsume(']')) {
							if (!first_leaf) {
								p.Expect(',');
							}
							first_leaf = false;
							tree.leaf_values.push_back(p.ParseNumber());
						}
					} else if (tkey == "splits") {
						p.Expect('[');
						bool first_split = true;
						while (!p.TryConsume(']')) {
							if (!first_split) {
								p.Expect(',');
							}
							first_split = false;
							p.Expect('{');
							TreeSplit split;
							bool has_border = false;
							bool has_feature = false;
							bool has_value = false;
							bool first_sfield = true;
							while (!p.TryConsume('}')) {
								if (!first_sfield) {
									p.Expect(',');
								}
								first_sfield = false;
								auto skey = p.ParseString();
								p.Expect(':');
								if (skey == "split_index") {
									split.split_index = static_cast<idx_t>(p.ParseNumber());
								} else if (skey == "float_feature_index" || skey == "flat_feature_index") {
									split.feature = static_cast<idx_t>(p.ParseNumber());
									has_feature = true;
								} else if (skey == "cat_feature_index") {
									split.feature = static_cast<idx_t>(p.ParseNumber());
									has_feature = true;
								} else if (skey == "border") {
									split.border = p.ParseNumber();
									has_border = true;
								} else if (skey == "value") {
									split.border = p.ParseNumber();
									has_value = true;
								} else if (skey == "split_type") {
									split.split_type = p.ParseString();
								} else {
									p.SkipValue();
								}
							}
							auto st = split.split_type;
							if (st == "FloatFeature") {
								split.kind = CatSplitKind::FLOAT;
								split.compare = SplitCompare::LESS;
								split.resolved = has_border && has_feature;
							} else if (st == "OneHotFeature") {
								split.kind = CatSplitKind::ONE_HOT;
								split.compare = SplitCompare::EQUAL;
								split.resolved = has_value && has_feature;
							} else if (st == "OnlineCtr") {
								split.kind = CatSplitKind::CTR;
								split.compare = SplitCompare::LESS;
								split.resolved = false; // always resolve via split_index / ctr tables
							} else {
								throw NotImplementedException("duckboost: catboost split_type '%s' is not supported",
								                              st);
							}
							tree.splits.push_back(split);
						}
					} else {
						p.SkipValue();
					}
				}
				pending_trees.push_back(std::move(tree));
			}
		} else if (key == "scale_and_bias") {
			p.Expect('[');
			scale = p.ParseNumber();
			biases.clear();
			if (p.TryConsume(',')) {
				if (p.Peek() == '[') {
					p.Expect('[');
					bool first_bias = true;
					while (!p.TryConsume(']')) {
						if (!first_bias) {
							p.Expect(',');
						}
						first_bias = false;
						biases.push_back(p.ParseNumber());
					}
				} else {
					biases.push_back(p.ParseNumber());
				}
			}
			if (biases.empty()) {
				biases.push_back(0.0);
			}
			p.Expect(']');
		} else if (key == "model_info") {
			p.Expect('{');
			bool first_mi = true;
			while (!p.TryConsume('}')) {
				if (!first_mi) {
					p.Expect(',');
				}
				first_mi = false;
				auto mi_key = p.ParseString();
				p.Expect(':');
				if (mi_key == "params" && p.Peek() == '{') {
					p.Expect('{');
					bool first_param = true;
					while (!p.TryConsume('}')) {
						if (!first_param) {
							p.Expect(',');
						}
						first_param = false;
						auto pk = p.ParseString();
						p.Expect(':');
						if ((pk == "loss_function" || pk == "objective") && p.Peek() == '"') {
							auto loss = StringUtil::Lower(p.ParseString());
							if (loss.find("multiclass") != string::npos || loss.find("multi_class") != string::npos) {
								model.task = BoostTask::MULTICLASS;
							} else if (loss.find("logloss") != string::npos ||
							           loss.find("crossentropy") != string::npos) {
								model.task = BoostTask::BINARY;
							}
						} else if ((pk == "classes_count" || pk == "class_count") && p.Peek() != '"') {
							detected_classes = static_cast<idx_t>(p.ParseNumber());
						} else {
							p.SkipValue();
						}
					}
				} else {
					p.SkipValue();
				}
			}
		} else {
			p.SkipValue();
		}
	}

	if (!has_trees) {
		throw InvalidInputException("duckboost: catboost JSON missing oblivious_trees");
	}

	auto remap_cat_index = [&](idx_t cat_idx) -> idx_t {
		if (cat_idx >= cat_features.size()) {
			throw InvalidInputException("duckboost: catboost CTR cat_feature_index out of range");
		}
		return cat_features[cat_idx].flat_feature_index;
	};
	auto remap_float_index = [&](idx_t float_idx) -> idx_t {
		if (float_idx >= float_features.size()) {
			// Some dumps already store flat indices under float_feature_index.
			return float_idx;
		}
		return float_features[float_idx].flat_feature_index;
	};

	vector<GlobalCatSplit> global_splits;
	for (auto &feat : float_features) {
		model.n_features = MaxValue<idx_t>(model.n_features, feat.flat_feature_index + 1);
		for (auto border : feat.borders) {
			GlobalCatSplit split;
			split.kind = CatSplitKind::FLOAT;
			split.feature = feat.flat_feature_index;
			split.border_or_value = border;
			global_splits.push_back(split);
		}
	}
	for (auto &feat : cat_features) {
		model.n_features = MaxValue<idx_t>(model.n_features, feat.flat_feature_index + 1);
		for (auto value : feat.one_hot_values) {
			GlobalCatSplit split;
			split.kind = CatSplitKind::ONE_HOT;
			split.feature = feat.flat_feature_index;
			split.border_or_value = value;
			global_splits.push_back(split);
		}
	}

	model.n_raw_features = model.n_features;
	// Allocate synthetic CTR feature slots and append CTR borders to global split index.
	for (idx_t ci = 0; ci < ctr_infos.size(); ci++) {
		auto &ctr = ctr_infos[ci];
		auto synth = model.n_features;
		model.n_features = synth + 1;
		CtrFeatureSpec spec;
		spec.feature_index = synth;
		spec.ctr_type = ctr.ctr_type;
		spec.prior_numerator = ctr.prior_numerator;
		spec.prior_denominator = ctr.prior_denominator;
		spec.scale = ctr.scale;
		spec.shift = ctr.shift;
		for (auto &el : ctr.elements) {
			CtrCombineElement out;
			auto lower = StringUtil::Lower(el.element_type);
			if (lower == "cat_feature_value") {
				out.kind = CtrElementKind::CAT_FEATURE_VALUE;
				out.feature_index = remap_cat_index(el.feature_index);
				spec.cat_feature_indices.push_back(out.feature_index);
			} else if (lower == "float_feature") {
				out.kind = CtrElementKind::FLOAT_FEATURE;
				out.feature_index = remap_float_index(el.feature_index);
				out.border_or_value = el.border_or_value;
			} else if (lower == "cat_feature_exact_value") {
				out.kind = CtrElementKind::CAT_FEATURE_EXACT_VALUE;
				out.feature_index = remap_cat_index(el.feature_index);
				out.border_or_value = el.border_or_value;
			} else {
				throw NotImplementedException("duckboost: unsupported CTR combination_element '%s'", el.element_type);
			}
			spec.elements.push_back(out);
		}
		if (!ctr.identifier.empty() && ctr_data_by_id.count(ctr.identifier)) {
			auto &data = ctr_data_by_id[ctr.identifier];
			spec.hash_keys = data.hash_keys;
			spec.hash_values = data.hash_values;
			spec.hash_values_alt = data.hash_values_alt;
			spec.counter_denominator = data.counter_denominator;
			if (!data.ctr_type.empty()) {
				spec.ctr_type = data.ctr_type;
			}
		} else if (!ctr_data_by_id.empty()) {
			// Try fuzzy match on ctr_type in identifier keys.
			for (auto &entry : ctr_data_by_id) {
				if (entry.first.find(ctr.ctr_type) != string::npos ||
				    StringUtil::Lower(entry.first).find(StringUtil::Lower(ctr.ctr_type)) != string::npos) {
					// Prefer exact identifier; skip fuzzy if identifier set.
					if (ctr.identifier.empty()) {
						spec.hash_keys = entry.second.hash_keys;
						spec.hash_values = entry.second.hash_values;
						spec.hash_values_alt = entry.second.hash_values_alt;
						spec.counter_denominator = entry.second.counter_denominator;
					}
				}
			}
		}
		model.ctr_features.push_back(spec);
		for (auto border : ctr.borders) {
			GlobalCatSplit split;
			split.kind = CatSplitKind::CTR;
			split.feature = synth;
			split.border_or_value = border;
			split.ctr_index = ci;
			global_splits.push_back(split);
		}
	}

	for (auto &pending : pending_trees) {
		vector<idx_t> features;
		vector<double> thresholds;
		vector<SplitCompare> compares;
		features.reserve(pending.splits.size());
		thresholds.reserve(pending.splits.size());
		compares.reserve(pending.splits.size());
		for (auto &split : pending.splits) {
			idx_t feature = split.feature;
			double border = split.border;
			SplitCompare compare = split.compare;
			if (split.kind == CatSplitKind::ONE_HOT && split.resolved) {
				// Inline OneHot uses cat_feature_index; map to flat when possible.
				if (feature < cat_features.size() && cat_features[feature].feature_index == feature) {
					feature = cat_features[feature].flat_feature_index;
				} else {
					for (auto &cf : cat_features) {
						if (cf.feature_index == split.feature) {
							feature = cf.flat_feature_index;
							break;
						}
					}
				}
			} else if (!split.resolved || split.kind == CatSplitKind::CTR) {
				if (split.split_index >= global_splits.size()) {
					throw InvalidInputException("duckboost: catboost split_index %llu out of range",
					                            (unsigned long long)split.split_index);
				}
				auto &gs = global_splits[split.split_index];
				feature = gs.feature;
				border = gs.border_or_value;
				if (gs.kind == CatSplitKind::ONE_HOT) {
					compare = SplitCompare::EQUAL;
				} else if (gs.kind == CatSplitKind::CTR) {
					compare = SplitCompare::LESS;
					if (gs.ctr_index >= model.ctr_features.size() ||
					    model.ctr_features[gs.ctr_index].hash_keys.empty()) {
						throw NotImplementedException(
						    "duckboost: CatBoost OnlineCtr split requires ctr_data in the JSON dump "
						    "(save_model with pool=...)");
					}
				} else {
					compare = SplitCompare::LESS;
				}
			}
			model.n_features = MaxValue<idx_t>(model.n_features, feature + 1);
			features.push_back(feature);
			if (compare == SplitCompare::EQUAL) {
				thresholds.push_back(border);
			} else {
				// CatBoost True when feature > border; duckboost left uses feature < threshold.
				thresholds.push_back(ThresholdForLessEqual(border));
			}
			compares.push_back(compare);
		}
		idx_t leaves_per_class = idx_t(1) << features.size();
		if (leaves_per_class == 0 || pending.leaf_values.size() % leaves_per_class != 0) {
			throw InvalidInputException(
			    "duckboost: catboost tree leaf_values size %llu is not a positive multiple of 2^depth (%llu)",
			    (unsigned long long)pending.leaf_values.size(), (unsigned long long)leaves_per_class);
		}
		idx_t tree_classes = pending.leaf_values.size() / leaves_per_class;
		if (tree_classes == 0) {
			throw InvalidInputException("duckboost: catboost tree has zero classes");
		}
		if (detected_classes == 0) {
			detected_classes = tree_classes;
		} else if (detected_classes != tree_classes) {
			throw InvalidInputException("duckboost: catboost inconsistent class count across trees (%llu vs %llu)",
			                            (unsigned long long)detected_classes, (unsigned long long)tree_classes);
		}
		for (idx_t c = 0; c < tree_classes; c++) {
			vector<double> class_leaves(leaves_per_class);
			for (idx_t leaf = 0; leaf < leaves_per_class; leaf++) {
				class_leaves[leaf] = pending.leaf_values[c * leaves_per_class + leaf] * scale;
			}
			BoostTree tree;
			ExpandCatBoost(0, 0, features, thresholds, compares, class_leaves, tree);
			model.trees.push_back(std::move(tree));
		}
	}

	if (detected_classes >= 2) {
		model.task = BoostTask::MULTICLASS;
		model.n_classes = detected_classes;
		model.base_scores = biases;
		if (model.base_scores.size() < model.n_classes) {
			model.base_scores.resize(model.n_classes, model.base_scores.empty() ? 0.0 : model.base_scores.back());
		} else if (model.base_scores.size() > model.n_classes) {
			model.base_scores.resize(model.n_classes);
		}
		model.base_score = model.base_scores.empty() ? 0.0 : model.base_scores[0];
	} else {
		model.n_classes = 1;
		model.base_score = biases.empty() ? 0.0 : biases[0];
		model.base_scores.clear();
	}
	ApplyImportOptions(model, options);
	return model;
}

BoostModel ImportModel(BoostBackend backend, const string &dump, const ImportOptions &options) {
	if (dump.empty()) {
		throw InvalidInputException("duckboost: empty model dump");
	}
	if (LooksLikeDuckBoostJSON(dump) || backend == BoostBackend::REFERENCE) {
		auto model = BoostModel::FromJSON(dump);
		if (backend != BoostBackend::REFERENCE) {
			model.backend = backend;
		}
		ApplyImportOptions(model, options);
		return model;
	}
	switch (backend) {
	case BoostBackend::XGBOOST:
		return ImportXGBoostJSON(dump, options);
	case BoostBackend::LIGHTGBM:
		return ImportLightGBMText(dump, options);
	case BoostBackend::CATBOOST:
		return ImportCatBoostJSON(dump, options);
	case BoostBackend::REFERENCE:
	default:
		throw InvalidInputException("duckboost: cannot import vendor dump with backend 'reference'");
	}
}

} // namespace duckboost
} // namespace duckdb
