//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckboost/json_util.hpp
//
// Minimal JSON helpers shared by model serialization and dump importers.
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string.hpp"
#include "duckdb/common/typedefs.hpp"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <sstream>

namespace duckdb {
namespace duckboost {

inline string EscapeJSON(const string &input) {
	string out;
	out.reserve(input.size() + 8);
	for (auto c : input) {
		switch (c) {
		case '"':
			out += "\\\"";
			break;
		case '\\':
			out += "\\\\";
			break;
		case '\n':
			out += "\\n";
			break;
		case '\r':
			out += "\\r";
			break;
		case '\t':
			out += "\\t";
			break;
		default:
			out.push_back(c);
			break;
		}
	}
	return out;
}

inline string FormatDouble(double value) {
	char buf[64];
	std::snprintf(buf, sizeof(buf), "%.17g", value);
	return string(buf);
}

struct JsonParser {
	explicit JsonParser(const string &input_p) : input(input_p), pos(0) {
	}

	const string &input;
	idx_t pos;

	void SkipWs() {
		while (pos < input.size() && std::isspace(static_cast<unsigned char>(input[pos]))) {
			pos++;
		}
	}

	bool AtEnd() {
		SkipWs();
		return pos >= input.size();
	}

	char Peek() {
		SkipWs();
		if (pos >= input.size()) {
			throw InvalidInputException("duckboost: unexpected end of JSON");
		}
		return input[pos];
	}

	char Consume() {
		auto c = Peek();
		pos++;
		return c;
	}

	void Expect(char c) {
		auto got = Consume();
		if (got != c) {
			throw InvalidInputException("duckboost: expected '%c' in JSON, got '%c'", c, got);
		}
	}

	bool TryConsume(char c) {
		SkipWs();
		if (pos < input.size() && input[pos] == c) {
			pos++;
			return true;
		}
		return false;
	}

	string ParseString() {
		Expect('"');
		string out;
		while (pos < input.size()) {
			auto c = input[pos++];
			if (c == '"') {
				return out;
			}
			if (c == '\\') {
				if (pos >= input.size()) {
					throw InvalidInputException("duckboost: truncated escape in JSON string");
				}
				auto e = input[pos++];
				switch (e) {
				case '"':
				case '\\':
				case '/':
					out.push_back(e);
					break;
				case 'n':
					out.push_back('\n');
					break;
				case 'r':
					out.push_back('\r');
					break;
				case 't':
					out.push_back('\t');
					break;
				default:
					out.push_back(e);
					break;
				}
			} else {
				out.push_back(c);
			}
		}
		throw InvalidInputException("duckboost: unterminated JSON string");
	}

	double ParseNumber() {
		SkipWs();
		if (pos >= input.size()) {
			throw InvalidInputException("duckboost: expected number in JSON");
		}
		char *end = nullptr;
		// strtod (unlike stod) tolerates subnormals / underflow without throwing.
		auto value = std::strtod(input.c_str() + pos, &end);
		if (end == input.c_str() + pos) {
			throw InvalidInputException("duckboost: expected number in JSON");
		}
		pos = static_cast<idx_t>(end - input.c_str());
		return value;
	}

	bool ParseBool() {
		SkipWs();
		if (input.compare(pos, 4, "true") == 0) {
			pos += 4;
			return true;
		}
		if (input.compare(pos, 5, "false") == 0) {
			pos += 5;
			return false;
		}
		throw InvalidInputException("duckboost: expected boolean in JSON");
	}

	void SkipValue() {
		SkipWs();
		auto c = Peek();
		if (c == '"') {
			ParseString();
		} else if (c == '{') {
			Expect('{');
			if (!TryConsume('}')) {
				do {
					ParseString();
					Expect(':');
					SkipValue();
				} while (TryConsume(','));
				Expect('}');
			}
		} else if (c == '[') {
			Expect('[');
			if (!TryConsume(']')) {
				do {
					SkipValue();
				} while (TryConsume(','));
				Expect(']');
			}
		} else if (c == 't' || c == 'f') {
			ParseBool();
		} else if (c == 'n') {
			if (input.compare(pos, 4, "null") != 0) {
				throw InvalidInputException("duckboost: expected null in JSON");
			}
			pos += 4;
		} else {
			ParseNumber();
		}
	}
};

} // namespace duckboost
} // namespace duckdb
