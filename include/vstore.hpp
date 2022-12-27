// Copyright 2021 Pierre Talbot

#ifndef VSTORE_HPP
#define VSTORE_HPP

#include "logic/logic.hpp"
#include "universes/upset_universe.hpp"
#include "copy_dag_helper.hpp"

namespace lala {

/** The variable store abstract domain is a _domain transformer_ built on top of an abstract universe `U`.
Concretization function: \f$ \gamma(\rho) \sqcap_{x \in \pi(\rho)} \gamma_{U_x}(\rho(x)) \f$.
The top element is smashed and the equality between two stores is represented by the following equivalence relation, for two stores \f$ S \f$ and \f$ T \f$:
\f$ S \equiv T \Leftrightarrow \forall{x \in \mathit{Vars}},~S(x) = T(x) \lor \exists{x \in \mathit{dom}(S)},\exists{y \in \mathit{dom}(T)},~S(x) = \top \land T(x) = \top \f$.
Intuitively, it means that either all elements are equal or both stores have a top element, in which case they "collapse" to the top element, and are considered equal.

The bottom element is the element \f$ \langle \bot, \ldots \rangle \f$, that is an infinite number of variables initialized to bottom.
In practice, we cannot represent infinite collections, so we represent bottom either as the empty collection or with a finite number of bottom elements.
Any finite store \f$ \langle x_1, \ldots, x_n \rangle \f$ should be seen as the concrete store \f$ \langle x_1, \ldots, x_n, \bot, \ldots \rangle \f$.

This semantics has implication when joining or merging two elements.
For instance, \f$ \langle 1 \rangle.\mathit{dtell}(\langle \bot, 4 \rangle) \f$ will be equal to bottom, in that case represented by \f$ \langle \bot \rangle \f$.

Limitation:
  - The size of the variable store is fixed at initialization and cannot be modified afterwards.
    The reason is that resizing an array does not currently fit the PCCP model of computation.
  - You can only tell store that are smaller or equal in size (unless the extra elements are set to bottom).

Template parameters:
  - `U` is the type of the abstract universe.
  - `Allocator` is the allocator of the underlying array of universes.
On some architecture, such as GPU, it is better to store the data in the shared memory whenever possible. */
template<class U, class Allocator>
class VStore {
public:
  using universe_type = U;
  using allocator_type = Allocator;
  using this_type = VStore<universe_type, allocator_type>;

  struct var_dom {
    int idx;
    universe_type dom;
    var_dom() = default;
    var_dom(int idx, const universe_type& dom): idx(idx), dom(dom) {}
  };

  template <class Alloc>
  using tell_type = battery::vector<var_dom, Alloc>;

  template <class Alloc = allocator_type>
  using snapshot_type = battery::vector<universe_type, Alloc>;

  template<class Env, class F>
  using iresult = IResult<tell_type<typename Env::allocator_type>, F>;

  constexpr static const char* name = "VStore";

  template<class U2, class Alloc2>
  friend class VStore;

private:
  using store_type = battery::vector<universe_type, allocator_type>;
  using memory_type = typename universe_type::memory_type;

  AType atype;
  store_type data;
  BInc<memory_type> is_at_top;

public:
  CUDA VStore(const this_type& other)
    : atype(other.atype), data(other.data), is_at_top(other.is_at_top)
  {}

  /** Initialize a store with `size` bottom elements. The size cannot be changed later on. */
  CUDA VStore(AType atype, size_t size, const allocator_type& alloc = allocator_type())
   : atype(atype), data(size, alloc), is_at_top(false)
  {}

  template<class R>
  CUDA VStore(const VStore<R, allocator_type>& other)
    : atype(other.atype), data(other.data), is_at_top(other.is_at_top)
  {}

  template<class R, class Alloc2>
  CUDA VStore(const VStore<R, Alloc2>& other, const allocator_type& alloc = allocator_type())
    : atype(other.atype), data(other.data, alloc), is_at_top(other.is_at_top)
  {}

  /** Copy the vstore `other` in the current element.
   *  `deps` can be empty and is not used (since this abstract domain does not have dependencies). */
  template<class Alloc2, class Alloc3>
  CUDA VStore(const VStore<U, Alloc2>& other, const AbstractDeps<Alloc3, allocator_type>& deps)
   : VStore(other, deps.get_fast_allocator()) {}

  CUDA VStore(this_type&& other):
    atype(other.atype), data(std::move(other.data)), is_at_top(other.is_at_top) {}

  CUDA allocator_type get_allocator() const {
    return data.get_allocator();
  }

  CUDA AType aty() const {
    return atype;
  }

  /** Returns the number of variables currently represented by this abstract element.
   * This value does not change. */
  CUDA size_t vars() const {
    return data.size();
  }

  CUDA static this_type bot(AType atype = UNTYPED,
    const allocator_type& alloc = allocator_type())
  {
    return VStore(atype, 0, alloc);
  }

  /** A special symbolic element representing top. */
  CUDA static this_type top(AType atype = UNTYPED,
    const allocator_type& alloc = allocator_type())
  {
    return std::move(VStore(atype, 0, alloc).tell_top());
  }

  /** `true` if at least one element is equal to top in the store, `false` otherwise. */
  CUDA local::BInc is_top() const {
    return is_at_top;
  }

  /** The bottom element of a store of `n` variables, is when all variables are at bottom, or the store is empty.
   * We do not expect to use this operation a lot, so its complexity is linear in the number of variables. */
  CUDA local::BDec is_bot() const {
    if(is_at_top) { return false; }
    for(int i = 0; i < vars(); ++i) {
      if(!data[i].is_bot()) {
        return false;
      }
    }
    return true;
  }

  /** Take a snapshot of the current variable store. */
  template <class Alloc = allocator_type>
  CUDA snapshot_type<Alloc> snapshot(const Alloc& alloc = Alloc()) const {
    return snapshot_type(data, alloc);
  }

  template <class Alloc>
  CUDA this_type& restore(const snapshot_type<Alloc>& snap) {
    assert(snap.size() == data.size());
    is_at_top.dtell_bot();
    for(int i = 0; i < snap.size(); ++i) {
      data[i].dtell(snap[i]);
      is_at_top.tell(data[i].is_top());
    }
    return *this;
  }

private:
  template <class F, class Env>
  CUDA iresult<Env, F> interpret_existential(const F& f, Env& env) const {
    using TellType = tell_type<typename Env::allocator_type>;
    assert(f.is(F::E));
    auto u = universe_type::interpret(f, env);
    if(u.has_value()) {
      if(env.num_vars_in(atype) >= vars()) {
        return iresult<Env, F>(IError<F>(true, name, "The variable could not be interpreted because the store is full.", f));
      }
      auto avar = env.interpret(f.map_atype(atype));
      if(avar.has_value()) {
        if(u.value().is_bot()) {
          return std::move(iresult<Env, F>(TellType(env.get_allocator())).join_warnings(std::move(u)));
        }
        else {
          TellType res(env.get_allocator());
          res.push_back(var_dom(avar.value().vid(), u.value()));
          return std::move(iresult<Env, F>(std::move(res)).join_warnings(std::move(u)));
        }
      }
      else {
        return std::move(avar).template map_error<TellType>();
      }
    }
    else {
      return std::move(u).template map_error<TellType>();
    }
  }

  /** Interpret a predicate without variables. */
  template <class F, class Env>
  CUDA iresult<Env, F> interpret_zero_predicate(const F& f, const Env& env) const {
    if(f.is_true()) {
      return std::move(iresult<Env, F>(tell_type(env.get_allocator())));
    }
    else if(f.is_false()) {
      tell_type res(env.get_allocator());
      res.push_back(var_dom(-1, U::top()));
      return std::move(iresult<Env, F>(std::move(res)));
    }
    else {
      return std::move(iresult<Env, F>(IError<F>(true, name, "Only `true` and `false` can be interpreted in the store without being named.", f)));
    }
  }

  /** Interpret a predicate with a single variable occurrence. */
  template <class F, class Env>
  CUDA iresult<Env, F> interpret_unary_predicate(const F& f, const Env& env) const {
    auto u = universe_type::interpret(f, env);
    if(u.has_value()) {
      tell_type res(env.get_allocator());
      auto var = var_in(f, env);
      if(!var.has_value()) {
        return std::move(iresult<Env, F>(IError<F>(true, name, "Undeclared variable.", f)).join_warnings(std::move(u)));
      }
      if(!u.value().is_bot()) {
        auto avar = var->avar_of(atype);
        if(!avar.has_value()) {
          return std::move(iresult<Env, F>(IError<F>(true, name,
              "The variable was not declared in the current abstract element (but exists in other abstract elements).", f))
            .join_warnings(std::move(u)));
        }
        res.push_back(var_dom(avar->vid(), u.value()));
      }
      return std::move(iresult<Env, F>(std::move(res)).join_warnings(std::move(u)));
    }
    else {
      return std::move(iresult<Env, F>(IError<F>(true, name, "Could not interpret a unary predicate in the underlying abstract universe.", f)).join_errors(std::move(u)));
    }
  }

  template <class F, class Env>
  CUDA iresult<Env, F> interpret_predicate(const F& f, Env& env) const {
    if(f.is(F::E)) {
      return interpret_existential(f, env);
    }
    else {
      switch(num_vars(f)) {
        case 0: return interpret_zero_predicate(f, env);
        case 1: return interpret_unary_predicate(f, env);
        default: return iresult<Env, F>(IError<F>(true, name, "Interpretation of n-ary predicate is not supported in VStore.", f));
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
    Variables are immediately assigned to an index of `VStore` and initialized to \f$ \bot_U \f$, if the universe's interpretation is different from bottom, the result of the VStore interpretation need to be `tell` later on in the store.
    Shadowing/redeclaration of variables with existential quantifier is not supported.
    Variables are added to the current abstract element only if `interpret(f).has_value()`.

    As a quirk, different stores might be produced if quantifiers do not appear in the same order.
    This is because we attribute the first available index to variables when interpreting the quantifier.
    The store will only be equivalent when considering the `env` structure.
  */
  template <class F, class Env>
  CUDA iresult<Env, F> interpret_in(const F& f, Env& env) const {
    if((f.type() == UNTYPED || f.type() == aty())
     && f.is(F::Seq) && f.sig() == AND)
    {
      const typename F::Sequence& seq = f.seq();
      auto res = iresult<Env, F>(tell_type(env.get_allocator()));
      for(int i = 0; i < seq.size(); ++i) {
        auto r = interpret_predicate(seq[i], env);
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
          res = std::move(iresult<Env, F>(IError<F>(true, name, "Could not interpret a component of the conjunction", f))
            .join_errors(std::move(r))
            .join_warnings(std::move(res)));
        }
      }
      return res;
    }
    else {
      return interpret_predicate(f, env);
    }
  }

  /** The static version of interpret creates a store with exactly the number of existentially quantified variables occurring in `f`.
   * All the existentially quantified variables must be untyped.
   * The UID of the store will be an UID that is free in `env`. */
  template <class F, class Env>
  CUDA static IResult<this_type, F> interpret(const F& f, Env& env, allocator_type alloc = allocator_type()) {
    this_type store(env.num_abstract_doms(), num_quantified_untyped_vars(f), alloc);
    auto r = store.interpret_in(f, env);
    if(r.has_value()) {
      store.tell(r.value());
      return std::move(r).map(std::move(store));
    }
    else {
      return std::move(r).template map_error<this_type>();
    }
  }

  /** The projection must stay const, otherwise the user might tell new information in the universe, but we need to know in case we reach `top`. */
  CUDA const universe_type& project(AVar x) const {
    assert(x.aty() == aty());
    assert(x.vid() < data.size());
    return data[x.vid()];
  }

  /** See note on projection. */
  CUDA const universe_type& operator[](int x) const {
    return data[x];
  }

  CUDA this_type& tell_top() {
    is_at_top.tell_top();
    return *this;
  }

  /** Given an abstract variable `v`, `tell(VID(v), dom)` will update the domain of this variable with the new information `dom`. */
  CUDA this_type& tell(int x, const universe_type& dom) {
    data[x].tell(dom);
    is_at_top.tell(data[x].is_top());
    return *this;
  }

  template <class Mem>
  CUDA this_type& tell(int x, const universe_type& dom, BInc<Mem>& has_changed) {
    data[x].tell(dom, has_changed);
    is_at_top.tell(data[x].is_top());
    return *this;
  }

  CUDA this_type& tell(AVar x, const universe_type& dom) {
    assert(x.aty() == aty());
    return tell(x.vid(), dom);
  }

  template <class Mem>
  CUDA this_type& tell(AVar x, const universe_type& dom, BInc<Mem>& has_changed) {
    assert(x.aty() == aty());
    return tell(x.vid(), dom, has_changed);
  }

  template <class Alloc2>
  CUDA this_type& tell(const tell_type<Alloc2>& t) {
    if(t.size() > 0 && t[0].idx == -1) {
      return tell_top();
    }
    for(int i = 0; i < t.size(); ++i) {
      tell(t[i].idx, t[i].dom);
    }
    return *this;
  }

  template <class Alloc2, class Mem>
  CUDA this_type& tell(const tell_type<Alloc2>& t, BInc<Mem>& has_changed) {
    if(t.size() > 0 && t[0].idx == -1) {
      is_at_top.tell(true, has_changed);
      return *this;
    }
    for(int i = 0; i < t.size(); ++i) {
      tell(t[i].idx, t[i].dom, has_changed);
    }
    return *this;
  }

  /** Precondition: `other` must be smaller or equal in size than the current store. */
  template <class U2, class Alloc2, class Mem>
  CUDA this_type& tell(const VStore<U2, Alloc2>& other, BInc<Mem>& has_changed) {
    is_at_top.tell(other.is_at_top, has_changed);
    int min_size = battery::min(vars(), other.vars());
    for(int i = 0; i < min_size; ++i) {
      data[i].tell(other[i], has_changed);
    }
    for(int i = min_size; i < other.vars(); ++i) {
      assert(other[i].is_bot()); // the size of the current store cannot be modified.
    }
    return *this;
  }

  /** Precondition: `other` must be smaller or equal in size than the current store. */
  template <class U2, class Alloc2>
  CUDA this_type& tell(const VStore<U2, Alloc2>& other) {
    local::BInc has_changed;
    return tell(other, has_changed);
  }

  CUDA this_type& dtell_bot() {
    is_at_top.dtell_bot();
    for(int i = 0; i < data.size(); ++i) {
      data[i].dtell_bot();
    }
    return *this;
  }

  /** Precondition: `other` must be smaller or equal in size than the current store. */
  template <class U2, class Alloc2, class Mem>
  CUDA this_type& dtell(const VStore<U2, Alloc2>& other, BInc<Mem>& has_changed)  {
    if(other.is_top()) {
      return *this;
    }
    int min_size = battery::min(vars(), other.vars());
    is_at_top.dtell(other.is_at_top, has_changed);
    for(int i = 0; i < min_size; ++i) {
      data[i].dtell(other[i], has_changed);
    }
    for(int i = min_size; i < vars(); ++i) {
      data[i].dtell(U::bot(), has_changed);
    }
    for(int i = min_size; i < other.vars(); ++i) {
      assert(other[i].is_bot());
    }
    return *this;
  }

  /** Precondition: `other` must be smaller or equal in size than the current store. */
  template <class U2, class Alloc2, class Mem>
  CUDA this_type& dtell(const VStore<U2, Alloc2>& other)  {
    local::BInc has_changed;
    return dtell(other, has_changed);
  }

  /** \return `true` when we can deduce the content of `t` from the current domain.
   * For instance, if we have in the store `x = [0..10]`, we can deduce `x = [-1..11]` but we cannot deduce `x = [5..8]`. */
  template <class Alloc2>
  CUDA local::BInc ask(const tell_type<Alloc2>& t) const {
    for(int i = 0; i < t.size(); ++i) {
      if(data[t[i].idx] < t[i].dom) {
        return false;
      }
    }
    return true;
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
      ua.data = data;
      ua.is_at_top.dtell_bot();
    }
    return true;
  }
};

// Lattice operations.
// Note that we do not consider the logical names.
// These operations are only considering the indices of the elements.

template<class L, class K, class Alloc>
CUDA auto join(const VStore<L, Alloc>& a, const VStore<K, Alloc>& b)
{
  using U = decltype(join(a[0], b[0]));
  if(a.is_top() || b.is_top()) {
    return VStore<U, Alloc>::top(UNTYPED, a.get_allocator());
  }
  int max_size = battery::max(a.vars(), b.vars());
  int min_size = battery::min(a.vars(), b.vars());
  VStore<U, Alloc> res(UNTYPED, max_size, a.get_allocator());
  for(int i = 0; i < min_size; ++i) {
    res.tell(i, join(a[i], b[i]));
  }
  for(int i = min_size; i < a.vars(); ++i) {
    res.tell(i, a[i]);
  }
  for(int i = min_size; i < b.vars(); ++i) {
    res.tell(i, b[i]);
  }
  return res;
}

template<class L, class K, class Alloc>
CUDA auto meet(const VStore<L, Alloc>& a, const VStore<K, Alloc>& b)
{
  using U = decltype(meet(a[0], b[0]));
  if(a.is_top()) {
    if(b.is_top()) {
      return VStore<U, Alloc>::top(UNTYPED, a.get_allocator());
    }
    else {
      return VStore<U, Alloc>(b);
    }
  }
  else if(b.is_top()) {
    return VStore<U, Alloc>(a);
  }
  else {
    int min_size = battery::min(a.vars(), b.vars());
    VStore<U, Alloc> res(UNTYPED, min_size, a.get_allocator());
    for(int i = 0; i < min_size; ++i) {
      res.tell(i, meet(a[i], b[i]));
    }
    return res;
  }
}

template<class L, class K, class Alloc>
CUDA bool operator<=(const VStore<L, Alloc>& a, const VStore<K, Alloc>& b)
{
  if(b.is_top()) {
    return true;
  }
  else {
    int min_size = battery::min(a.vars(), b.vars());
    for(int i = 0; i < min_size; ++i) {
      if(a[i] > b[i]) {
        return false;
      }
    }
    for(int i = min_size; i < a.vars(); ++i) {
      if(!a[i].is_bot()) {
        return false;
      }
    }
  }
  return true;
}

template<class L, class K, class Alloc>
CUDA bool operator<(const VStore<L, Alloc>& a, const VStore<K, Alloc>& b)
{
  if(b.is_top()) {
    return !a.is_top();
  }
  else {
    int min_size = battery::min(a.vars(), b.vars());
    bool strict = false;
    for(int i = 0; i < b.vars(); ++i) {
      if(i < a.vars()) {
        if(a[i] < b[i]) {
          strict = true;
        }
        else if(a[i] > b[i]) {
          return false;
        }
      }
      else if(!b[i].is_bot()) {
        strict = true;
      }
    }
    for(int i = min_size; i < a.vars(); ++i) {
      if(!a[i].is_bot()) {
        return false;
      }
    }
    return strict;
  }
}

template<class L, class K, class Alloc>
CUDA bool operator>=(const VStore<L, Alloc>& a, const VStore<K, Alloc>& b)
{
  return b <= a;
}

template<class L, class K, class Alloc>
CUDA bool operator>(const VStore<L, Alloc>& a, const VStore<K, Alloc>& b)
{
  return b < a;
}

template<class L, class K, class Alloc>
CUDA bool operator==(const VStore<L, Alloc>& a, const VStore<K, Alloc>& b)
{
  if(a.is_top()) {
    return b.is_top();
  }
  else if(b.is_top()) {
    return false;
  }
  else {
    int min_size = battery::min(a.vars(), b.vars());
    for(int i = 0; i < min_size; ++i) {
      if(a[i] != b[i]) {
        return false;
      }
    }
    for(int i = min_size; i < a.vars(); ++i) {
      if(!a[i].is_bot()) {
        return false;
      }
    }
    for(int i = min_size; i < b.vars(); ++i) {
      if(!b[i].is_bot()) {
        return false;
      }
    }
  }
  return true;
}

template<class L, class K, class Alloc>
CUDA bool operator!=(const VStore<L, Alloc>& a, const VStore<K, Alloc>& b)
{
  return !(a == b);
}

template<class L, class Alloc>
std::ostream& operator<<(std::ostream &s, const VStore<L, Alloc> &vstore) {
  if(vstore.is_top()) {
    s << "\u22A4";
  }
  else {
    s << "<";
    for(int i = 0; i < vstore.vars(); ++i) {
      s << vstore[i] << (i+1 == vstore.vars() ? "" : ", ");
    }
    s << ">";
  }
  return s;
}

} // namespace lala

#endif
