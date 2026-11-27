// Copyright 2021 Pierre Talbot

/** The environment mapping logical variable names to abstract variables (`AVar`).
 * It holds no logical formula: interpreting a formula into an environment is done by the
 * interpretation layer (`turbo/include/interpretation.hpp`), which is why this class has no
 * `interpret` member and no dependency on `IDiagnostics`. */

#ifndef LALA_CORE_ENV_HPP
#define LALA_CORE_ENV_HPP

#include "battery/utility.hpp"
#include "battery/vector.hpp"
#include "battery/string.hpp"
#include "battery/tuple.hpp"
#include "battery/variant.hpp"
#include "ast.hpp"
#include "algorithm.hpp"

#include <functional>

namespace lala {

template<class Allocator>
struct Variable {
  template<class T>
  using bvector = battery::vector<T, Allocator>;

  SymID name;
  Sort<Allocator> sort;
  bvector<AVar> avars;

  Variable(Variable<Allocator>&&) = default;
  Variable(const Variable<Allocator>&) = default;

  CUDA NI Variable(SymID name, const Sort<Allocator>& sort, AVar av, const Allocator& allocator = Allocator{})
    : name(name), sort(sort, allocator), avars(allocator)
  {
    avars.push_back(av);
  }

  template <class Alloc2>
  CUDA NI Variable(const Variable<Alloc2>& other, const Allocator& allocator = Allocator{})
    : name(other.name)
    , sort(other.sort, allocator)
    , avars(other.avars, allocator)
  {}

  CUDA NI std::optional<AVar> avar_of(AType aty) const {
    for(int i = 0; i < avars.size(); ++i) {
      if(avars[i].aty() == aty) {
        return avars[i];
      }
    }
    return {};
  }
};

/** A `VarEnv` is a variable environment mapping between logical variables and abstract variables.
 * The logical variables are found by their symbol in constant time, on the CPU as well as on the GPU. */
template <class Allocator>
class VarEnv {
public:
  using allocator_type = Allocator;
  using this_type = VarEnv<Allocator>;

  constexpr static const char* name = "VarEnv";

  template<class T>
  using bvector = battery::vector<T, Allocator>;

  using variable_type = Variable<Allocator>;

  template <class Alloc2>
  friend class VarEnv;

private:
  bvector<variable_type> lvars;
  bvector<bvector<size_t>> avar2lvar;
  /** `symbol2lvar[s]` is the index in `lvars` of the variable whose symbol has the identifier `s`, or -1 if there is none. */
  bvector<int> symbol2lvar;

  CUDA NI std::optional<size_t> lvar_index_of(SymID lv) const {
    if(lv.id() >= 0 && lv.id() < symbol2lvar.size() && symbol2lvar[lv.id()] != -1) {
      return static_cast<size_t>(symbol2lvar[lv.id()]);
    }
    return {};
  }

  CUDA NI void set_lvar_index(SymID lv, int index) {
    assert(lv.is_valid());
    while(symbol2lvar.size() <= lv.id()) {
      symbol2lvar.push_back(-1);
    }
    symbol2lvar[lv.id()] = index;
  }

public:
  CUDA NI AType extends_abstract_dom() {
    avar2lvar.push_back(bvector<size_t>(get_allocator()));
    return static_cast<AType>(avar2lvar.size()) - 1;
  }

  CUDA NI void extends_abstract_doms(AType aty) {
    assert(aty != UNTYPED);
    while(aty >= avar2lvar.size()) {
      extends_abstract_dom();
    }
  }

  template <class Alloc2>
  CUDA NI AVar extends_vars(AType aty, SymID name, const Sort<Alloc2>& sort) {
    extends_abstract_doms(aty);
    AVar avar(aty, static_cast<int>(avar2lvar[aty].size()));
    // We first verify the variable doesn't already exist.
    auto lvar_idx = lvar_index_of(name);
    if(lvar_idx.has_value()) {
      auto avar_opt = lvars[*lvar_idx].avar_of(aty);
      if(avar_opt.has_value()) {
        return *avar_opt;
      }
      else {
        lvars[*lvar_idx].avars.push_back(avar);
      }
    }
    else {
      lvar_idx = {lvars.size()};
      set_lvar_index(name, static_cast<int>(lvars.size()));
      lvars.push_back(Variable<allocator_type>{name, Sort<allocator_type>(sort, get_allocator()), avar, get_allocator()});
    }
    avar2lvar[aty].push_back(*lvar_idx);
    return avar;
  }

public:
  CUDA VarEnv(const Allocator& allocator): lvars(allocator), avar2lvar(allocator), symbol2lvar(allocator) {}
  CUDA VarEnv(this_type&& other)
    : lvars(std::move(other.lvars)), avar2lvar(std::move(other.avar2lvar)), symbol2lvar(std::move(other.symbol2lvar)) {}
  CUDA VarEnv(): VarEnv(Allocator{}) {}
  CUDA VarEnv(const this_type& other)
    : lvars(other.lvars), avar2lvar(other.avar2lvar), symbol2lvar(other.symbol2lvar) {}

  template <class Alloc2>
  CUDA VarEnv(const VarEnv<Alloc2>& other, const Allocator& allocator = Allocator{})
    : lvars(other.lvars, allocator)
    , avar2lvar(other.avar2lvar, allocator)
    , symbol2lvar(other.symbol2lvar, allocator)
  {}

  CUDA this_type& operator=(this_type&& other) {
    lvars = std::move(other.lvars);
    avar2lvar = std::move(other.avar2lvar);
    symbol2lvar = std::move(other.symbol2lvar);
    return *this;
  }

  CUDA this_type& operator=(const this_type& other) {
    lvars = other.lvars;
    avar2lvar = other.avar2lvar;
    symbol2lvar = other.symbol2lvar;
    return *this;
  }

  template <class Alloc2>
  CUDA this_type& operator=(const VarEnv<Alloc2>& other) {
    lvars = other.lvars;
    avar2lvar = other.avar2lvar;
    symbol2lvar = other.symbol2lvar;
    return *this;
  }

  CUDA allocator_type get_allocator() const {
    return lvars.get_allocator();
  }

  CUDA size_t num_abstract_doms() const {
    return avar2lvar.size();
  }

  CUDA size_t num_vars() const {
    return lvars.size();
  }

  CUDA size_t num_vars_in(AType aty) const {
    if(aty >= avar2lvar.size()) {
      return 0;
    }
    else {
      return avar2lvar[aty].size();
    }
  }

  CUDA void print() const {
    printf("Environment (%lu variables): \n", num_vars());
    printf("index\t name\t sort\tavars\n");
    for(int i = 0; i < num_vars(); ++i) {
      printf("%d\t", i);
      const auto& var = lvars[i];
      var.name.print();
      printf("\t");
      var.sort.print(); printf("\t\t");
      for(int j = 0; j < var.avars.size(); ++j) {
        printf("(%d,%d)", var.avars[j].aty(), var.avars[j].vid());
        if(j != var.avars.size() - 1) {
          printf(",");
        }
      }
      printf("\n");
    }
  }

  CUDA NI std::optional<std::reference_wrapper<const variable_type>> variable_of(SymID lv) const {
    auto r = lvar_index_of(lv);
    if(r.has_value()) {
      return std::cref(lvars[*r]);
    }
    else {
      return {};
    }
  }

  /** Find a variable by its name (only on the host, through the `SymbolTable`). */
  std::optional<std::reference_wrapper<const variable_type>> variable_of(const char* lv) const {
    auto s = SymID::find(lv);
    if(s.has_value()) {
      return variable_of(*s);
    }
    return {};
  }

  template <class Alloc2>
  std::optional<std::reference_wrapper<const variable_type>> variable_of(const battery::string<Alloc2>& lv) const {
    return variable_of(lv.data());
  }

  CUDA bool contains(SymID lv) const {
    return lvar_index_of(lv).has_value();
  }

  bool contains(const char* lv) const {
    return variable_of(lv).has_value();
  }

  template <class Alloc2>
  bool contains(const battery::string<Alloc2>& lv) const {
    return contains(lv.data());
  }

  CUDA bool contains(AVar av) const {
    if(!av.is_untyped()) {
      return avar2lvar.size() > av.aty() && avar2lvar[av.aty()].size() > av.vid();
    }
    return false;
  }

  CUDA const variable_type& operator[](int i) const {
    return lvars[i];
  }

  CUDA const variable_type& operator[](AVar av) const {
    return lvars[avar2lvar[av.aty()][av.vid()]];
  }

  CUDA SymID name_of(AVar av) const {
    return (*this)[av].name;
  }

  CUDA const Sort<Allocator>& sort_of(AVar av) const {
    return (*this)[av].sort;
  }

  struct snapshot_type {
    bvector<size_t> lvars_snap;
    bvector<size_t> avar2lvar_snap;
  };

  /** Save the state of the environment. */
  CUDA NI snapshot_type snapshot() const {
    snapshot_type snap;
    for(int i = 0; i < lvars.size(); ++i) {
      snap.lvars_snap.push_back(lvars[i].avars.size());
    }
    for(int i = 0; i < avar2lvar.size(); ++i) {
      snap.avar2lvar_snap.push_back(avar2lvar[i].size());
    }
    return std::move(snap);
  }

  /** Restore the environment to its previous state `snap`. */
  CUDA NI void restore(const snapshot_type& snap) {
    assert(lvars.size() >= snap.lvars_snap.size());
    assert(avar2lvar.size() >= snap.avar2lvar_snap.size());
    while(lvars.size() > snap.lvars_snap.size()) {
      symbol2lvar[lvars.back().name.id()] = -1;
      lvars.pop_back();
    }
    for(int i = 0; i < lvars.size(); ++i) {
      lvars[i].avars.resize(snap.lvars_snap[i]);
    }
    while(avar2lvar.size() > snap.avar2lvar_snap.size()) {
      avar2lvar.pop_back();
    }
    for(int i = 0; i < avar2lvar.size(); ++i) {
      avar2lvar[i].resize(snap.avar2lvar_snap[i]);
    }
  }
};

/** Given a formula `f` and an environment, return the first variable occurring in `f` or `{}` if `f` has no variable in `env`. */
template <class F, class Env>
CUDA NI std::optional<std::reference_wrapper<const typename Env::variable_type>> var_in(const F& f, const Env& env) {
  const auto& g = var_in(f);
  switch(g.index()) {
    case F::E:
      return env.variable_of(battery::get<0>(g.exists()));
    case F::LV:
      return env.variable_of(g.lv());
  }
  return {};
}
}

#endif
