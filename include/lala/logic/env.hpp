// Copyright 2021 Pierre Talbot

#ifndef LALA_CORE_ENV_HPP
#define LALA_CORE_ENV_HPP

#include "battery/utility.hpp"
#include "battery/vector.hpp"
#include "battery/string.hpp"
#include "battery/string.hpp"
#include "battery/tuple.hpp"
#include "battery/variant.hpp"
#include "ast.hpp"
#include "iresult.hpp"

namespace lala {

template<class Allocator>
struct Variable {
  template<class T>
  using bvector = battery::vector<T, Allocator>;
  using bstring = battery::string<Allocator>;

  bstring name;
  Sort<Allocator> sort;
  bvector<AVar> avars;

  CUDA NI Variable(const bstring& name, const Sort<Allocator>& sort, AVar av, const Allocator& allocator = Allocator())
    : name(name, allocator), sort(sort, allocator), avars(allocator)
  {
    avars.push_back(av);
  }

  template <class Alloc2>
  CUDA NI Variable(const Variable<Alloc2>& other, const Allocator& allocator = Allocator())
    : name(other.name, allocator)
    , sort(other.sort, allocator)
    , avars(other.avars, allocator)
  {}

  CUDA NI thrust::optional<AVar> avar_of(AType aty) const {
    for(int i = 0; i < avars.size(); ++i) {
      if(avars[i].aty() == aty) {
        return avars[i];
      }
    }
    return {};
  }
};

/** A `VarEnv` is a variable environment mapping between logical variables and abstract variables. */
template <class Allocator>
class VarEnv {
  template <class F> using fstring = battery::string<typename F::allocator_type>;
public:
  using allocator_type = Allocator;
  using this_type = VarEnv<Allocator>;

  template<class F>
  using iresult = IResult<AVar, F>;

  constexpr static const char* name = "VarEnv";

  template<class T>
  using bvector = battery::vector<T, Allocator>;
  using bstring = battery::string<Allocator>;

  using variable_type = Variable<Allocator>;

  template <class Alloc2>
  friend class VarEnv;

private:
  bvector<variable_type> lvars;
  bvector<bvector<int>> avar2lvar;

public:
  CUDA NI AType extends_abstract_dom() {
    avar2lvar.push_back(bvector<int>(get_allocator()));
    return avar2lvar.size() - 1;
  }

private:

  CUDA NI void extends_abstract_doms(AType aty) {
    assert(aty != UNTYPED);
    while(aty >= avar2lvar.size()) {
      extends_abstract_dom();
    }
  }

  template <class Alloc2, class Alloc3>
  CUDA NI AVar extends_vars(AType aty, const battery::string<Alloc2>& name, const Sort<Alloc3>& sort) {
    extends_abstract_doms(aty);
    AVar avar(aty, avar2lvar[aty].size());
    // We first verify the variable doesn't already exist.
    auto lvar_idx = lvar_index_of(name.data());
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
      lvars.push_back(Variable(name, sort, avar, get_allocator()));
    }
    avar2lvar[aty].push_back(*lvar_idx);
    return avar;
  }

  // Variable redeclaration does not lead to an error, instead the abstract type of the variable is added to the abstract variables list (`avars`) of the variable.
  template <bool diagnose = false, class F>
  CUDA NI bool interpret_existential(const F& f, AVar& avar, IDiagnostics<F>& diagnostics) {
    const auto& vname = battery::get<0>(f.exists());
    if(f.type() == UNTYPED) {
      RETURN_INTERPRETATION_ERROR("Untyped abstract type: variable `" + vname + "` has no abstract type.");
    }
    auto var = variable_of(vname);
    if(var.has_value()) {
      if(var->sort != battery::get<1>(f.exists())) {
        RETURN_INTERPRETATION_ERROR("Invalid redeclaration with different sort: variable `" + vname + "` has already been declared and the sort does not coincide.");
      }
    }
    avar = extends_vars(f.type(), vname, battery::get<1>(f.exists()));
    return true;
  }

  template <bool diagnose = false, class F>
  CUDA NI bool interpret_lv(const F& f, AVar& avar, IDiagnostics<F>& diagnostics) {
    const auto& vname = f.lv();
    auto var = variable_of(vname);
    if(var.has_value()) {
      if(f.type() != UNTYPED) {
        auto avar = var->avar_of(f.type());
        if(avar.has_value()) {
          avar = AVar(*avar);
          return true;
        }
        else {
          RETURN_INTERPRETATION_ERROR("Variable `" + vname + "` has not been declared in the abstract domain `" + fstring<F>::from_int(f.type()) + "`.");
        }
      }
      else {
        if(var->avars.size() == 1) {
          avar = AVar(var->avars[0]);
          return true;
        }
        else {
          RETURN_INTERPRETATION_ERROR("Variable occurrence `" + vname + "` is untyped, but exists in multiple abstract domains.");
        }
      }
    }
    else {
      RETURN_INTERPRETATION_ERROR("Undeclared variable `" + vname + "`.");
    }
  }

  CUDA NI thrust::optional<int> lvar_index_of(const char* lv) const {
    for(int i = 0; i < lvars.size(); ++i) {
      if(lvars[i].name == lv) {
        return i;
      }
    }
    return {};
  }

public:
  CUDA VarEnv(const Allocator& allocator): lvars(allocator), avar2lvar(allocator) {}
  CUDA VarEnv(this_type&& other): lvars(std::move(other.lvars)), avar2lvar(std::move(other.avar2lvar)) {}
  CUDA VarEnv(): VarEnv(Allocator()) {}

  template <class Alloc2>
  CUDA VarEnv(const VarEnv<Alloc2>& other, const Allocator& allocator = Allocator())
    : lvars(other.lvars, allocator)
    , avar2lvar(other.avar2lvar, allocator)
  {}

  CUDA this_type& operator=(this_type&& other) {
    lvars = std::move(other.lvars);
    avar2lvar = std::move(other.avar2lvar);
    return *this;
  }

  CUDA this_type& operator=(const this_type& other) {
    lvars = other.lvars;
    avar2lvar = other.avar2lvar;
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

  /** A variable environment can interpret formulas of two forms:
   *    - Existential formula with a valid abstract type (`f.type() != UNTYPED`).
   *    - Variable occurrence.
   * It returns an abstract variable (`AVar`) corresponding to the variable created (existential) or already presents (occurrence). */
  template <bool diagnose = false, class F>
  CUDA NI bool interpret(const F& f, AVar& avar, IDiagnostics<F>& diagnostics) {
    if(f.is(F::E)) {
      return interpret_existential<diagnose>(f, avar, diagnostics);
    }
    else if(f.is(F::LV)) {
      return interpret_lv<diagnose>(f, avar, diagnostics);
    }
    else if(f.is(F::V)) {
      if(contains(f.v())) {
        avar = f.v();
        return true;
      }
      else {
        RETURN_INTERPRETATION_ERROR("Undeclared abstract variable `" + fstring<F>::from_int(f.v().aty()) + ", " + fstring<F>::from_int(f.v().vid()) + "`.");
      }
    }
    else {
      RETURN_INTERPRETATION_ERROR("Unsupported formula: `VarEnv` can only interpret quantifiers and occurrences of variables.");
    }
  }

  CUDA NI thrust::optional<const variable_type&> variable_of(const char* lv) const {
    auto r = lvar_index_of(lv);
    if(r.has_value()) {
      return lvars[*r];
    }
    else {
      return {};
    }
  }

  template <class Alloc2>
  CUDA thrust::optional<const variable_type&> variable_of(const battery::string<Alloc2>& lv) const {
    return variable_of(lv.data());
  }

  template <class Alloc2>
  CUDA bool contains(const battery::string<Alloc2>& lv) const {
    return contains(lv.data());
  }

  CUDA bool contains(const char* lv) const {
    return variable_of(lv).has_value();
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

  CUDA const bstring& name_of(AVar av) const {
    return (*this)[av].name;
  }

  CUDA const Sort<Allocator>& sort_of(AVar av) const {
    return (*this)[av].sort;
  }

  struct snapshot_type {
    bvector<int> lvars_snap;
    bvector<int> avar2lvar_snap;
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
CUDA NI thrust::optional<const typename Env::variable_type&> var_in(const F& f, const Env& env) {
  const auto& g = var_in(f);
  switch(g.index()) {
    case F::V:
      if(g.v().is_untyped()) { return {}; }
      else { return env[g.v()]; }
    case F::E:
      return env.variable_of(battery::get<0>(g.exists()));
    case F::LV:
      return env.variable_of(g.lv());
  }
  return {};
}
}

#endif
