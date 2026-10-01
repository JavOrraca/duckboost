//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckboost/native_train.hpp
//
// Optional native XGBoost / LightGBM / CatBoost trainers (DUCKBOOST_WITH_*).
//===----------------------------------------------------------------------===//

#pragma once

#include "duckboost/model.hpp"

namespace duckdb {
namespace duckboost {

//! True when this build compiled with DUCKBOOST_WITH_<BACKEND> (stub or linked).
bool NativeTrainerCompiled(BoostBackend backend);

//! True when the vendor library is linked (not DUCKBOOST_NATIVE_STUB) and a train bridge exists.
bool NativeTrainerLinked(BoostBackend backend);

//! Train using a vendor library when linked; stub / unlinked builds throw NotImplementedException.
//! weights empty ⇒ unit weights (forwarded to the vendor library when linked).
//! is_validation empty ⇒ optional random hold-out via options; otherwise per-row external validation.
BoostModel TrainNative(const vector<double> &y, const vector<vector<double>> &x, const TrainOptions &options,
                       const vector<double> &weights = {}, const vector<bool> &is_validation = {});

} // namespace duckboost
} // namespace duckdb
