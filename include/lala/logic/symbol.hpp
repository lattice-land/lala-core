// Copyright 2026 Pierre Talbot

#ifndef LALA_CORE_SYMBOL_HPP
#define LALA_CORE_SYMBOL_HPP

#include <cstdio>
#include <string>
#include <vector>
#include <optional>
#include <unordered_map>
#include "battery/utility.hpp"
#include "battery/string.hpp"

namespace lala {

/** The table of the names of the logical variables, shared by the whole process.
 * Each name is stored once: the keys of `ids` are the only copies of the names, and `names` points to them.
 * The nodes of a `std::unordered_map` are never moved (not even when the table is rehashed), hence these pointers remain valid.
 * Both directions are in O(1): name -> identifier by hashing, identifier -> name by indexing.
 * The table is only used on the host (not on the GPU), and it is not thread-safe.
 * The names are never removed. */
class SymbolTable {
  std::unordered_map<std::string, int> ids;
  std::vector<const std::string*> names;

  SymbolTable() = default;

public:
  SymbolTable(const SymbolTable&) = delete;
  SymbolTable& operator=(const SymbolTable&) = delete;

  static SymbolTable& instance() {
    static SymbolTable table;
    return table;
  }

  /** \return The identifier of `name`, which is added to the table if it is not already in it. */
  int intern(const std::string& name) {
    auto [it, inserted] = ids.try_emplace(name, static_cast<int>(names.size()));
    if(inserted) {
      names.push_back(&it->first);
    }
    return it->second;
  }

  /** \return The identifier of `name` if it is in the table (without adding it). */
  std::optional<int> find(const std::string& name) const {
    auto it = ids.find(name);
    if(it != ids.end()) {
      return it->second;
    }
    return {};
  }

  /** \pre `0 <= id < size()`. */
  const std::string& name(int id) const {
    return *names[id];
  }

  size_t size() const {
    return names.size();
  }
};

/** A symbol is the name of a logical variable, represented by its identifier in the `SymbolTable`.
 * Comparing and copying symbols is constant time, and they can be used on the GPU.
 * The constructors from strings and `data()` are only available on the host, as they use the `SymbolTable`. */
class SymID {
  int id_;

public:
  /** An invalid symbol. */
  CUDA constexpr SymID(): id_(-1) {}

  /** Intern `name` in the `SymbolTable`. */
  SymID(const char* name): id_(SymbolTable::instance().intern(std::string(name))) {}
  SymID(const std::string& name): id_(SymbolTable::instance().intern(name)) {}
  template <class Alloc>
  SymID(const battery::string<Alloc>& name): SymID(name.data()) {}

  CUDA constexpr SymID(const SymID&) = default;
  CUDA constexpr SymID& operator=(const SymID&) = default;

  /** \return The symbol of `name` if `name` is in the `SymbolTable` (without adding it). */
  static std::optional<SymID> find(const char* name) {
    auto id = SymbolTable::instance().find(std::string(name));
    if(id.has_value()) {
      return SymID::from_id(*id);
    }
    return {};
  }

  CUDA static constexpr SymID from_id(int id) {
    SymID s;
    s.id_ = id;
    return s;
  }

  CUDA constexpr int id() const { return id_; }
  CUDA constexpr bool is_valid() const { return id_ >= 0; }

  /** The name of the symbol. */
  const char* data() const {
    return SymbolTable::instance().name(id_).c_str();
  }

  const std::string& str() const {
    return SymbolTable::instance().name(id_);
  }

  CUDA void print() const {
    #ifdef __CUDA_ARCH__
      printf("#%d", id_);
    #else
      if(is_valid()) {
        printf("%s", data());
      }
      else {
        printf("<invalid symbol>");
      }
    #endif
  }
};

CUDA constexpr inline bool operator==(const SymID& a, const SymID& b) { return a.id() == b.id(); }
CUDA constexpr inline bool operator!=(const SymID& a, const SymID& b) { return a.id() != b.id(); }
CUDA constexpr inline bool operator<(const SymID& a, const SymID& b) { return a.id() < b.id(); }

} // namespace lala

#endif // LALA_CORE_SYMBOL_HPP
