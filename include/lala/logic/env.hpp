// Copyright 2021 Pierre Talbot

#ifndef LALA_CORE_ENV_HPP
#define LALA_CORE_ENV_HPP

#include "battery/utility.hpp"
#include "battery/vector.hpp"
#include "battery/string.hpp"
#include "battery/tuple.hpp"
#include "battery/variant.hpp"
#include "ast.hpp"
#include "diagnostics.hpp"

#include <string>
#include <unordered_map>

namespace lala {

template<class Allocator>
struct Variable {
  template<class T>
  using bvector = battery::vector<T, Allocator>;
  using bstring = battery::string<Allocator>;

  bstring name;
  Sort<Allocator> sort;
  bvector<AVar> avars;

  Variable(Variable<Allocator>&&) = default;
  Variable(const Variable<Allocator>&) = default;

  CUDA NI Variable(const bstring& name, const Sort<Allocator>& sort, AVar av, const Allocator& allocator = Allocator{})
    : name(name, allocator), sort(sort, allocator), avars(allocator)
  {
    avars.push_back(av);
  }

  template <class Alloc2>
  CUDA NI Variable(const Variable<Alloc2>& other, const Allocator& allocator = Allocator{})
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

template <class Allocator>
struct ListVarIndex {
  using allocator_type = Allocator;
  using this_type = ListVarIndex<Allocator>;
  using variable_type = Variable<Allocator>;

  template<class T>
  using bvector = battery::vector<T, Allocator>;
  using bstring = battery::string<Allocator>;

  bvector<variable_type>* lvars;

  CUDA ListVarIndex(bvector<variable_type>* lvars): lvars(lvars) {}
  CUDA ListVarIndex(this_type&&, bvector<variable_type>* lvars): lvars(lvars) {}
  CUDA ListVarIndex(const this_type&, bvector<variable_type>* lvars): lvars(lvars) {}

  template <class Alloc2>
  CUDA ListVarIndex(const ListVarIndex<Alloc2>&, bvector<variable_type>* lvars)
    : lvars(lvars)
  {}

  // For this operator=, we suppose `lvars` is updated before.
  CUDA this_type& operator=(this_type&& other) {
    return *this;
  }

  CUDA this_type& operator=(const this_type& other) {
    return *this;
  }

  CUDA thrust::optional<int> lvar_index_of(const char* lv) const {
    for(int i = 0; i < lvars->size(); ++i) {
      if((*lvars)[i].name == lv) {
        return i;
      }
    }
    return {};
  }

  CUDA void push_back(variable_type&& var) {
    lvars->push_back(std::move(var));
  }

  CUDA void erase(const char* lv) {}

  CUDA void set_lvars(bvector<variable_type>* lvars) {
    this->lvars = lvars;
  }
};

template <class Allocator>
struct HashMapVarIndex {
  using allocator_type = Allocator;
  using this_type = ListVarIndex<Allocator>;
  using variable_type = Variable<Allocator>;

  template<class T>
  using bvector = battery::vector<T, Allocator>;
  using bstring = battery::string<Allocator>;

  bvector<variable_type>* lvars;
  std::unordered_map<std::string, int> lvar_index;

  HashMapVarIndex(bvector<variable_type>* lvars): lvars(lvars) {
    for(int i = 0; i < lvars->size(); ++i) {
      lvar_index[std::string((*lvars)[i].name.data())] = i;
    }
  }

  HashMapVarIndex(this_type&& other, bvector<variable_type>* lvars)
   : lvars(lvars), lvar_index(std::move(other.lvar_index)) {}

  HashMapVarIndex(const this_type& other, bvector<variable_type>* lvars)
   : lvars(lvars), lvar_index(other.lvar_index) {}

  template <class Alloc2>
  HashMapVarIndex(const HashMapVarIndex<Alloc2>& other, bvector<variable_type>* lvars)
    : lvars(lvars), lvar_index(other.lvar_index)
  {}

  // For this operator=, we suppose `lvars` is updated before.
  this_type& operator=(this_type&& other) {
    lvar_index = std::move(other.lvar_index);
    return *this;
  }
  this_type& operator=(const this_type& other) {
    lvar_index = other.lvar_index;
    return *this;
  }

  thrust::optional<int> lvar_index_of(const char* lv) const {
    auto it = lvar_index.find(std::string(lv));
    if(it != lvar_index.end()) {
      return {it->second};
    }
    return {};
  }

  void push_back(variable_type&& var) {
    lvar_index[std::string(var.name.data())] = lvars->size();
    lvars->push_back(std::move(var));
  }

  void erase(const char* lv) {
    lvar_index.erase(std::string(lv));
  }

  void set_lvars(bvector<variable_type>* lvars) {
    this->lvars = lvars;
  }
};

template <class Allocator>
struct DispatchIndex {
  using allocator_type = Allocator;
  using this_type = ListVarIndex<Allocator>;
  using variable_type = Variable<Allocator>;

  template<class T>
  using bvector = battery::vector<T, Allocator>;
  using bstring = battery::string<Allocator>;

  battery::unique_ptr<HashMapVarIndex<allocator_type>, allocator_type> cpu_index;
  battery::unique_ptr<ListVarIndex<allocator_type>, allocator_type> gpu_index;

  CUDA DispatchIndex(bvector<variable_type>* lvars): cpu_index(nullptr), gpu_index(nullptr) {
    gpu_index = std::move(battery::allocate_unique<ListVarIndex<allocator_type>>(lvars->get_allocator(), lvars));
    #ifndef __CUDA_ARCH__
      cpu_index = std::move(battery::allocate_unique<HashMapVarIndex<allocator_type>>(lvars->get_allocator(), lvars));
    #endif
  }

  CUDA DispatchIndex(this_type&& other, bvector<variable_type>* lvars)
   : gpu_index(std::move(other.gpu_index))
  {
    #ifndef __CUDA_ARCH__
      cpu_index = std::move(other.cpu_index);
    #endif
  }

  CUDA DispatchIndex(const this_type& other, bvector<variable_type>* lvars)
  {
    gpu_index = std::move(battery::allocate_unique<ListVarIndex<allocator_type>>(lvars->get_allocator(), *other.gpu_index, lvars));
    #ifndef __CUDA_ARCH__
      cpu_index = std::move(battery::allocate_unique<HashMapVarIndex<allocator_type>>(lvars->get_allocator(), *other.cpu_index, lvars));
    #endif
  }

  template <class Alloc2>
  CUDA DispatchIndex(const DispatchIndex<Alloc2>& other, bvector<variable_type>* lvars)
  {
    gpu_index = std::move(battery::allocate_unique<ListVarIndex<allocator_type>>(lvars->get_allocator(), *other.gpu_index, lvars));
    #ifndef __CUDA_ARCH__
      cpu_index = std::move(battery::allocate_unique<HashMapVarIndex<allocator_type>>(lvars->get_allocator(), *other.cpu_index, lvars));
    #endif
  }

  // For this operator=, we suppose `lvars` is updated before.
  CUDA this_type& operator=(this_type&& other) {
    gpu_index = std::move(other.gpu_index);
    #ifndef __CUDA_ARCH__
      cpu_index = std::move(other.cpu_index);
    #endif
    return *this;
  }

  CUDA this_type& operator=(const this_type& other) {
    *gpu_index = *other.gpu_index;
    #ifndef __CUDA_ARCH__
      *cpu_index = *other.cpu_index;
    #endif
    return *this;
  }

  CUDA thrust::optional<int> lvar_index_of(const char* lv) const {
    #ifdef __CUDA_ARCH__
      return gpu_index->lvar_index_of(lv);
    #else
      return cpu_index->lvar_index_of(lv);
    #endif
  }

  CUDA void push_back(variable_type&& var) {
    #ifdef __CUDA_ARCH__
      gpu_index->push_back(std::move(var));
    #else
      cpu_index->push_back(std::move(var));
    #endif
  }

  CUDA void erase(const char* lv) {
    #ifdef __CUDA_ARCH__
      gpu_index->erase(lv);
    #else
      cpu_index->erase(lv);
    #endif
  }

  CUDA void set_lvars(bvector<variable_type>* lvars) {
    gpu_index->set_lvars(lvars);
    #ifndef __CUDA_ARCH__
      cpu_index->set_lvars(lvars);
    #endif
  }
};

/** A `VarEnv` is a variable environment mapping between logical variables and abstract variables. */
template <class Allocator>
class VarEnv {
  template <class F> using fstring = battery::string<typename F::allocator_type>;
public:
  using allocator_type = Allocator;
  using this_type = VarEnv<Allocator>;

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
  DispatchIndex<allocator_type> var_index; // Note that this must always be declared *after* `lvars` because it stores a reference to it.

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
    auto lvar_idx = var_index.lvar_index_of(name.data());
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
      var_index.push_back(Variable<allocator_type>{name, sort, avar, get_allocator()});
    }
    avar2lvar[aty].push_back(*lvar_idx);
    return avar;
  }

  // Variable redeclaration does not lead to an error, instead the abstract type of the variable is added to the abstract variables list (`avars`) of the variable.
  template <bool diagnose = false, class F>
  CUDA NI bool interpret_existential(const F& f, AVar& avar, IDiagnostics& diagnostics) {
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
  CUDA NI bool interpret_lv(const F& f, AVar& avar, IDiagnostics& diagnostics) {
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
        // We take the first abstract variable as a representative. Need more thought on this, but currently we need it for the simplifier, because each variable is typed in both PC and Simplifier, and this interpretation fails.

        // if(var->avars.size() == 1) {
          avar = AVar(var->avars[0]);
          return true;
        // }
        // else {
        //   RETURN_INTERPRETATION_ERROR("Variable occurrence `" + vname + "` is untyped, but exists in multiple abstract domains.");
        // }
      }
    }
    else {
      RETURN_INTERPRETATION_ERROR("Undeclared variable `" + vname + "`.");
    }
  }

public:
  CUDA VarEnv(const Allocator& allocator): lvars(allocator), avar2lvar(allocator), var_index(&lvars) {}
  CUDA VarEnv(this_type&& other): lvars(std::move(other.lvars)), avar2lvar(std::move(other.avar2lvar)), var_index(std::move(other.var_index), &lvars) {}
  CUDA VarEnv(): VarEnv(Allocator{}) {}
  CUDA VarEnv(const this_type& other): lvars(other.lvars), avar2lvar(other.avar2lvar), var_index(other.var_index, &lvars) {}

  template <class Alloc2>
  CUDA VarEnv(const VarEnv<Alloc2>& other, const Allocator& allocator = Allocator{})
    : lvars(other.lvars, allocator)
    , avar2lvar(other.avar2lvar, allocator)
    , var_index(other.var_index, &lvars)
  {}

  CUDA this_type& operator=(this_type&& other) {
    lvars = std::move(other.lvars);
    avar2lvar = std::move(other.avar2lvar);
    var_index = std::move(other.var_index);
    var_index.set_lvars(&lvars);
    return *this;
  }

  CUDA this_type& operator=(const this_type& other) {
    lvars = other.lvars;
    avar2lvar = other.avar2lvar;
    var_index = DispatchIndex<allocator_type>(other.var_index, &lvars);
    var_index.set_lvars(&lvars);
    return *this;
  }

  template <class Alloc2>
  CUDA this_type& operator=(const VarEnv<Alloc2>& other) {
    lvars = other.lvars;
    avar2lvar = other.avar2lvar;
    var_index = DispatchIndex<allocator_type>(other.var_index, &lvars);
    var_index.set_lvars(&lvars);
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
  CUDA NI bool interpret(const F& f, AVar& avar, IDiagnostics& diagnostics) {
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
    auto r = var_index.lvar_index_of(lv);
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
      var_index.erase(lvars.back().name.data());
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
