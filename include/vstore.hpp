// Copyright 2021 Pierre Talbot

#ifndef VSTORE_HPP
#define VSTORE_HPP

#include "ast.hpp"
#include "z.hpp"
#include "arithmetic.hpp"

namespace lala {

/** The variable store abstract domain is a _domain transformer_ built on top of an abstract universe `U`.
Concretization function: \f$ \gamma(\rho) \sqcap_{x \in \pi(\rho)} \gamma_{U_x}(\rho(x)) \f$.
The top element is smashed and the equality between two stores is represented by the following equivalence relation, for two stores \f$ S \f$ and \f$ T \f$:
\f$ S \equiv T \Leftrightarrow \forall{x \in \mathit{Vars}},~S(x) = T(x) \lor \exists{x \in \mathit{dom}(S)},\exists{y \in \mathit{dom}(T)},~S(x) = \top \land T(x) = \top \f$.
Intuitively, it means that either all elements are equal or both stores have a top element, in which case they "collapse" to the top element, and are considered equal.

Limitation:
  - The capacity of the variable store is fixed at initialization and cannot be modified afterwards.

Template parameters:
  - `U` is the type of the abstract universe.
  - `EnvAllocator` is the allocator of the mapping between names (LVar) and indices in the store (AVar).
  - `DataAllocator` is the allocator of the underlying array of universes.
On some architecture, such as GPU, it is better to store the data in the shared memory whenever possible, while the environment can stay in the global memory because it is usually not queried during solving (but before or after). */
template<class U, class EnvAllocator, class DataAllocator = EnvAllocator>
class VStore {
public:
  using universe_type = U;
  using allocator_type = EnvAllocator;
  using fast_allocator_type = DataAllocator;
  using this_type = VStore<U, DataAllocator, EnvAllocator>;

  struct var_dom {
    AVar avar;
    universe_type dom;
    var_dom(AVar avar, const universe_type& dom): avar(avar), dom(dom) {}
  };
  using tell_type = battery::vector<var_dom, allocator_type>;
  using env_type = VarEnv<allocator_type>;

  using snapshot_type = battery::vector<universe_type, allocator_type>;

  template<class F>
  using iresult = IResult<tell_type, F>;

  constexpr static const char* name = "VStore";

private:
  using store_type = battery::vector<universe_type, fast_allocator_type>;
  using memory_type = typename universe_type::memory_type;

  store_type data;
  env_type env;
  BInc<memory_type> is_at_top;

public:
  /** Initialize an empty store of size `capacity`. The capacity cannot be changed later on. */
  CUDA VStore(AType uid, size_t capacity, const fast_allocator_type& falloc, const allocator_type& alloc = allocator_type())
   : data(falloc), env(uid, capacity), is_at_top(false)
  {
    // We want the size of `data` to be 0 initially.
    // New variables are added using push_back in interpret.
    data.reserve(capacity);
  }

  template<class R>
  CUDA VStore(const VStore<R, allocator_type, fast_allocator_type>& other)
    : data(other.data), env(other), is_at_top(other.is_at_top)
  {}

  template<class R, class EA, class DA>
  CUDA VStore(const VStore<R, EA, DA>& other, const fast_allocator_type& falloc, const allocator_type& alloc = allocator_type())
    : data(other.data, falloc), env(other, alloc), is_at_top(other.is_at_top)
  {}

  /** Completely copy the vstore `other` in the current element.
   *  `deps` can be empty and is not used (since this abstract domain does not have dependencies). */
  template<class Alloc2>
  CUDA VStore(const VStore<U, Alloc2>& other, const AbstractDeps<Allocator, FAllocator>& deps)
   : VStore(other, deps.get_fast_allocator(), deps.get_allocator()) {}

  CUDA allocator_type get_allocator() const {
    return env.get_allocator();
  }

  CUDA fast_allocator_type get_fast_allocator() const {
    return data.get_allocator();
  }

  CUDA AType uid() const {
    return env.uid();
  }

  /** Returns the number of variables currently represented by this abstract element. */
  CUDA local::ZInc vars() const {
    return data.size();
  }

  CUDA static this_type bot(AType uid = UNTYPED,
    size_t capacity = 0,
    const fast_allocator_type& falloc = fast_allocator_type(),
    const allocator_type& alloc = allocator_type())
  {
    return VStore(uid, capacity, falloc, alloc);
  }

  /** A special symbolic element representing top. */
  CUDA static this_type top(AType uid = UNTYPED,
    const fast_allocator_type& falloc = fast_allocator_type(),
    const allocator_type& alloc = allocator_type())
  {
    return VStore(uid, 0, falloc, alloc).tell_top();
  }

  CUDA this_type& tell_top() {
    is_at_top.tell_top();
    return *this;
  }

  /** `true` if at least one element is equal to top in the store, `false` otherwise. */
  CUDA local::BInc is_top() const {
    return is_at_top;
  }

  /** The bottom element of a store of `n` variables, is when all variables are at bottom, or the store is empty.
   * We do not expect to use this operation a lot, so its complexity is linear in the number of variables. */
  CUDA local::BDec is_bot() const {
    if(is_at_top) { return false; }
    for(int i = 0; i < data.size(); ++i) {
      if(!data[i].is_bot()) {
        return false;
      }
    }
    return true;
  }

  /** Take a snapshot of the current variable store.
   * Precondition: `!is_top()`. */
  CUDA snapshot_type snapshot() const {
    assert(!is_top());
    return snapshot_type(data);
  }

  CUDA this_type& restore(const snapshot_type& snap) {
    assert(snap.size() == data.size());
    is_at_top.tell_bot();
    for(int i = 0; i < snap.size(); ++i) {
      data[i].dtell(snap[i]);
    }
    while(data.size() > snap.size()) {
      data.pop_back();
      env.pop_back();
    }
    return *this;
  }

private:
  template <class F>
  CUDA iresult<F> interpret_existential(const F& f) {
    assert(f.is(F::E));
    auto u = universe_type::interpret(f);
    if(u.has_value()) {
      const auto& vname = battery::get<0>(f.exists());
      if(env.contains(vname)) {
        return iresult<F>(IError<F>(true, name, "Invalid redeclaration: variable `" + vname + "` has already been declared.", f)).join_warnings(std::move(u));
      }
      else {
        env.add(vname);
        data.push_back(u.value());
        return iresult<F>(tell_type()).join_warnings(std::move(u));
      }
    }
    else {
      return iresult<F>(IError<F>(true, name, "Invalid redeclaration: variable `" + vname + "` has already been declared.", f)).join_errors(std::move(u));
    }
  }

  /** Interpret a predicate without variables. */
  template <class F>
  CUDA iresult<F> interpret_zero_predicate(const F& f) const {
    auto u = universe_type::interpret(f);
    if(u.is_ok()) {
      if(!u.value().is_bot()) {
        return iresult<F>(IError<F>(true, name, "Only `true` can be interpreted in the store without being named.", f)).join_warnings(std::move(u));
      }
      else {
        return iresult<F>(tell_type()).join_warnings(std::move(u));
      }
    }
    else {
      return iresult<F>(IError<F>(true, name, "Could not interpret a predicate without variable in the underlying abstract universe.", f)).join_errors(std::move(u));
    }
  }

  /** Interpret a predicate with a single variable occurrence. */
  template <class F>
  CUDA iresult<F> interpret_unary_predicate(const F& f) const {
    auto u = universe_type::interpret(f);
    if(u.is_ok()) {
      tell_type res;
      if(!u.value().is_bot()) {
        auto fv = var_in(f, env);
        AVar v;
        if(fv.is(F::V)) {
          v = fv.v();
          if(AID(v) != uid() && AID(v) != UNTYPED) {
            return iresult<F>(IError<F>(true, name, "An abstract variable was encountered with a wrong abstract type for the current VStore.", f)).join_warnings(std::move(u));
          }
        }
        else if(fv.is(F::LV)) {
          auto av = env.to_avar(fv.lv());
          if(!av.has_value()) {
            return iresult<F>(IError<F>(true, name, "The variable `" + vname + "` is undeclared.", f)).join_warnings(std::move(u));
          }
          avar = *av;
        }
        else {
          return iresult<F>(IError<F>(true, name, "Existential quantifier must be interpreted in VStore and not in the underlying universe.", f)).join_warnings(std::move(u));
        }
        res.push_back(var_dom(v, u.value()));
      }
      return iresult<F>(std::move(res)).join_warnings(std::move(u));
    }
    else {
      return iresult<F>(IError<F>(true, name, "Could not interpret a unary predicate in the underlying abstract universe.", f)).join_errors(std::move(u));
    }
  }

  template <class F>
  CUDA iresult<F> interpret_predicate(const F& f) const {
    if(f.is(F::E)) {
      return interpret_existential(f);
    }
    else {
      switch(num_vars(f)) {
        case 0: return interpret_zero_predicate(f);
        case 1: return interpret_unary_predicate(f);
        default: return iresult<F>(IError<F>(true, name, "Interpretation of n-ary predicate is not supported in VStore.", f));
      }
    }
  }

public:
  /** The store of variables lattice expects a conjunctive formula \f$ c_1 \land \ldots \land c_n \f$ in which all components \f$ c_i \f$ are formulas with a single variable (including existential quantifiers) that can be handled by the abstract universe `U`.
   *
    I. Approximation
    ================

    Exact and under-approximation of \f$ \land \f$ are treated in the same way (nothing special is performed for under-approximation).
    Over-approximation of \f$ \land \f$ allows the abstract domain to ignore components of the formula that cannot be interpreted in the underlying abstract universe.

    II. Existential quantifier
    ==========================

    Variables must be existentially quantified before a formula containing variables can be interpreted.
    Variables are immediately added to `VStore` and initialized to \f$ \bot_U \f$, hence existential quantifiers never need to be joined in this abstract domain.
    Shadowing/redeclaration of variables with existential quantifier is not supported.
    Variables are added to the current abstract element only if `interpret(f).is_ok()`.
  */
  template <class F>
  CUDA iresult<F> interpret(const F& f) {
    if((f.type() == UNTYPED || f.type() == uid())
     && f.is(F::Seq) && f.sig() == AND)
    {
      int current_vars = vars();
      const typename F::Sequence& seq = f.seq();
      auto res = iresult<F>(tell_type());
      for(int i = 0; i < seq.size(); ++i) {
        auto r = interpret_predicate(seq[i]);
        if(r.has_value()) {
          for(int j = 0; j < r.value().size(); ++j) {
            res.value().push_back(r.value()[j]);
          }
          res.join_warnings(std::move(r));
        }
        else if(f.approx() == OVER) {
          auto warning = IError<F>(false, name, "A component of the conjunction was ignored (allowed by over-approximation)", f);
          warning.add_suberror(std::move(r.error()));
          res.push_warning(std::move(warning));
        }
        else {
          res = iresult<F>(IError<F>(true, name, "Could not interpret a component of the conjunction", f))
            .join_errors(std::move(r))
            .join_warnings(std::move(res));
        }
      }
      return res;
    }
    else {
      return interpret_predicate(f);
    }
  }

  CUDA const universe_type& project(AVar x) const {
    assert(AID(x) == uid());
    assert(VID(x) < data.size());
    return data[VID(x)];
  }

  template <class Mem>
  CUDA this_type& tell(AVar x, const universe_type& dom, BInc<Mem>& has_changed) {
    data[VID(x)].tell(dom, has_changed);
    is_at_top.tell(data[VID(x)].is_top());
    return *this;
  }

  template <class Mem>
  CUDA this_type& tell(const tell_type& t, BInc<Mem>& has_changed) {
    for(int i = 0; i < t.size(); ++i) {
      tell(t[i].avar, t[i].dom, has_changed);
    }
    return *this;
  }

  /** Whenever `this` is different from `top`, we extract its data into `ua`.
   * For now, we suppose VStore is only used to store under-approximation, I'm not sure yet how we would interact with over-approximation.
   * The variable environment of `ua` is supposed to be already initialized. */
  template<class Alloc2>
  CUDA bool extract(VStore<U, Alloc2>& ua) const {
    if(is_top()) {
      return false;
    }
    if(&ua != this) {
      if(ua.env.size() != env.size()) {
        ua.env = env;
      }
      ua.data = data;
      ua.is_at_top = false;
    }
    return true;
  }

  CUDA const Env& environment() const { return env; }
};

} // namespace lala

#endif
