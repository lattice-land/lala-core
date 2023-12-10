// Copyright 2021 Pierre Talbot

#ifndef LALA_CORE_VSTORE_HPP
#define LALA_CORE_VSTORE_HPP

#include "logic/logic.hpp"
#include "universes/primitive_upset.hpp"
#include "abstract_deps.hpp"
#include <optional>

namespace lala {

  struct NonAtomicExtraction {
    static constexpr bool atoms = false;
  };

  struct AtomicExtraction {
    static constexpr bool atoms = true;
  };

/** The variable store abstract domain is a _domain transformer_ built on top of an abstract universe `U`.
Concretization function: \f$ \gamma(\rho) := \bigcap_{x \in \pi(\rho)} \gamma_{U_x}(\rho(x)) \f$.
The top element is smashed and the equality between two stores is represented by the following equivalence relation, for two stores \f$ S \f$ and \f$ T \f$:
\f$ S \equiv T \Leftrightarrow \forall{x \in \mathit{Vars}},~S(x) = T(x) \lor \exists{x \in \mathit{dom}(S)},\exists{y \in \mathit{dom}(T)},~S(x) = \top \land T(y) = \top \f$.
Intuitively, it means that either all elements are equal or both stores have a top element, in which case they "collapse" to the top element, and are considered equal.

The bottom element is the element \f$ \langle \bot, \ldots \rangle \f$, that is an infinite number of variables initialized to bottom.
In practice, we cannot represent infinite collections, so we represent bottom either as the empty collection or with a finite number of bottom elements.
Any finite store \f$ \langle x_1, \ldots, x_n \rangle \f$ should be seen as the concrete store \f$ \langle x_1, \ldots, x_n, \bot, \ldots \rangle \f$.

This semantics has implication when joining or merging two elements.
For instance, \f$ \langle 1 \rangle.\mathit{dtell}(\langle \bot, 4 \rangle) \f$ will be equal to bottom, in that case represented by \f$ \langle \bot \rangle \f$.

Template parameters:
  - `U` is the type of the abstract universe.
  - `Allocator` is the allocator of the underlying array of universes. */
template<class U, class Allocator>
class VStore {
public:
  using universe_type = U;
  using local_universe = typename universe_type::local_type;
  using allocator_type = Allocator;
  using this_type = VStore<universe_type, allocator_type>;

  struct var_dom {
    AVar avar;
    local_universe dom;
    var_dom() = default;
    var_dom(const var_dom&) = default;
    CUDA var_dom(AVar avar, const local_universe& dom): avar(avar), dom(dom) {}
  };

  template <class Alloc>
  using tell_type = battery::vector<var_dom, Alloc>;

  template <class Alloc>
  using ask_type = tell_type<Alloc>;

  template <class Alloc = allocator_type>
  using snapshot_type = battery::vector<local_universe, Alloc>;

  constexpr static const bool is_abstract_universe = false;
  constexpr static const bool sequential = universe_type::sequential;
  constexpr static const bool is_totally_ordered = false;
  constexpr static const bool preserve_bot = true;
  constexpr static const bool preserve_top = true;
  constexpr static const bool preserve_join = universe_type::preserve_join;
  constexpr static const bool preserve_meet = universe_type::preserve_meet;
  constexpr static const bool injective_concretization = universe_type::injective_concretization;
  constexpr static const bool preserve_concrete_covers = universe_type::preserve_concrete_covers;
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

  /** Initialize an empty store. */
  CUDA VStore(AType atype, const allocator_type& alloc = allocator_type())
   : atype(atype), data(alloc), is_at_top(false)
  {}

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
   *  `deps` can be empty and is not used besides to get the allocator (since this abstract domain does not have dependencies). */
  template<class R, class Alloc2, class... Allocators>
  CUDA VStore(const VStore<R, Alloc2>& other, const AbstractDeps<Allocators...>& deps)
   : VStore(other, deps.template get_allocator<allocator_type>()) {}

  CUDA VStore(this_type&& other):
    atype(other.atype), data(std::move(other.data)), is_at_top(other.is_at_top) {}

  CUDA allocator_type get_allocator() const {
    return data.get_allocator();
  }

  CUDA AType aty() const {
    return atype;
  }

  /** Returns the number of variables currently represented by this abstract element. */
  CUDA size_t vars() const {
    return data.size();
  }

  CUDA static this_type bot(AType atype = UNTYPED,
    const allocator_type& alloc = allocator_type{})
  {
    return VStore(atype, alloc);
  }

  /** A special symbolic element representing top. */
  CUDA static this_type top(AType atype = UNTYPED,
    const allocator_type& alloc = allocator_type{})
  {
    return std::move(VStore(atype, alloc).tell_top());
  }

  /** `true` if at least one element is equal to top in the store, `false` otherwise. */
  CUDA local::BInc is_top() const {
    return is_at_top;
  }

  /** The bottom element of a store of `n` variables is when all variables are at bottom, or the store is empty.
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
    return snapshot_type<Alloc>(data, alloc);
  }

  template <class Alloc>
  CUDA this_type& restore(const snapshot_type<Alloc>& snap) {
    while(snap.size() < data.size()) {
      data.pop_back();
    }
    is_at_top.dtell_bot();
    for(int i = 0; i < snap.size(); ++i) {
      data[i].dtell(snap[i]);
      is_at_top.tell(data[i].is_top());
    }
    return *this;
  }

private:
  template <bool diagnose, class F, class Env, class Alloc2>
  CUDA NI bool interpret_existential(const F& f, Env& env, tell_type<Alloc2>& tell, IDiagnostics<F>& diagnostics) const {
    assert(f.is(F::E));
    var_dom k;
    if(local_universe::template interpret_tell<diagnose>(f, env, k.dom, diagnostics)) {
      if(env.interpret(f.map_atype(atype), k.avar, diagnostics)) {
        tell.push_back(k);
        return true;
      }
    }
    return false;
  }

  /** Interpret a predicate without variables. */
  template <bool diagnose, class F, class Env, class Alloc2>
  CUDA NI bool interpret_zero_predicate(const F& f, const Env& env, tell_type<Alloc2>& tell, IDiagnostics<F>& diagnostics) const {
    if(f.is_true()) {
      return true;
    }
    else if(f.is_false()) {
      tell.push_back(var_dom(AVar{}, U::top()));
      return true;
    }
    else {
      RETURN_INTERPRETATION_ERROR("Only `true` and `false` can be interpreted in the store without being named.");
    }
  }

  /** Interpret a predicate with a single variable occurrence. */
  template <IKind kind, bool diagnose, class F, class Env, class Alloc2>
  CUDA NI bool interpret_unary_predicate(const F& f, const Env& env, tell_type<Alloc2>& tell, IDiagnostics<F>& diagnostics) const {
    local_universe u;
    bool res = local_universe::template interpret<kind, diagnose>(f, env, u, diagnostics);
    if(res) {
      const auto& varf = var_in(f);
      // When it is not necessary, we try to avoid using the environment.
      // This is for instance useful when refinement operators add new constraints but do not have access to the environment (e.g., split()), and to avoid passing the environment around everywhere.
      if(varf.is(F::V)) {
        tell.push_back(var_dom(varf.v(), u));
      }
      else {
        auto var = var_in(f, env);
        if(!var.has_value()) {
          RETURN_INTERPRETATION_ERROR("Undeclared variable.");
        }
        auto avar = var->avar_of(atype);
        if(!avar.has_value()) {
          RETURN_INTERPRETATION_ERROR("The variable was not declared in the current abstract element (but exists in other abstract elements).");
        }
        tell.push_back(var_dom(*avar, u));
      }
      return true;
    }
    else {
      RETURN_INTERPRETATION_ERROR("Could not interpret a unary predicate in the underlying abstract universe.");
    }
  }

  template <IKind kind, bool diagnose, class F, class Env, class Alloc2>
  CUDA NI bool interpret_predicate(const F& f, Env& env, tell_type<Alloc2>& tell, IDiagnostics<F>& diagnostics) const {
    if(f.type() != UNTYPED && f.type() != aty()) {
      RETURN_INTERPRETATION_ERROR("The abstract type of this predicate does not match the one of the current abstract element.");
    }
    if constexpr(kind == IKind::TELL) {
      if(f.is(F::E)) {
        return interpret_existential<diagnose>(f, env, tell, diagnostics);
      }
    }
    switch(num_vars(f)) {
      case 0: return interpret_zero_predicate<diagnose>(f, env, tell, diagnostics);
      case 1: return interpret_unary_predicate<kind, diagnose>(f, env, tell, diagnostics);
      default: RETURN_INTERPRETATION_ERROR("Interpretation of n-ary predicate is not supported in VStore.");
    }
  }

public:
  template <IKind kind, bool diagnose = false, class F, class Env, class Alloc2>
  CUDA NI bool interpret(const F& f, Env& env, tell_type<Alloc2>& tell, IDiagnostics<F>& diagnostics) const {
    if(f.is_untyped() || f.type() == aty()) {
      return interpret_predicate<kind, diagnose>(f, env, tell, diagnostics);
    }
    else {
      RETURN_INTERPRETATION_ERROR("Interpretation of a formula with a different type.");
    }
  }

  /** The store of variables lattice expects a formula with a single variable (including existential quantifiers) that can be handled by the abstract universe `U`.
   *
   * Variables must be existentially quantified before a formula containing variables can be interpreted.
   * Variables are immediately assigned to an index of `VStore` and initialized to \f$ \bot_U \f$.
   * Shadowing/redeclaration of variables with existential quantifier is not supported.
   * The variable mapping is added to the environment only if the interpretation succeeds.

   * There is a small quirk: different stores might be produced if quantifiers do not appear in the same order.
   * This is because we attribute the first available index to variables when interpreting the quantifier.
   * In that case, the store will only be equivalent modulo the `env` structure.
  */
  template <bool diagnose = false, class F, class Env, class Alloc2>
  CUDA NI bool interpret_tell(const F& f, Env& env, tell_type<Alloc2>& tell, IDiagnostics<F>& diagnostics) const {
    return interpret<IKind::TELL, diagnose>(f, env, tell, diagnostics);
  }

  /** Similar to `interpret_tell` but do not support existential quantifier and therefore leaves `env` unchanged. */
  template <bool diagnose = false, class F, class Env, class Alloc2>
  CUDA NI bool interpret_ask(const F& f, const Env& env, ask_type<Alloc2>& ask, IDiagnostics<F>& diagnostics) const {
    return const_cast<this_type*>(this)->interpret<IKind::ASK, diagnose>(f, const_cast<Env&>(env), ask, diagnostics);
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

  /** Given an abstract variable `v`, `tell(VID(v), dom)` will update the domain of this variable with the new information `dom`.
   * This `tell` method follows PCCP's model, but the variable `x` must already be initialized in the store.
  */
  CUDA this_type& tell(int x, const universe_type& dom) {
    assert(x < data.size());
    data[x].tell(dom);
    is_at_top.tell(data[x].is_top());
    return *this;
  }

  /** This `tell` method follows PCCP's model, but the variable `x` must already be initialized in the store. */
  template <class Mem>
  CUDA this_type& tell(int x, const universe_type& dom, BInc<Mem>& has_changed) {
    assert(x < data.size());
    data[x].tell(dom, has_changed);
    is_at_top.tell(data[x].is_top());
    return *this;
  }

  /** This `tell` method follows PCCP's model, but the variable `x` must already be initialized in the store. */
  CUDA this_type& tell(AVar x, const universe_type& dom) {
    assert(x.aty() == aty());
    return tell(x.vid(), dom);
  }

  /** This `tell` method follows PCCP's model, but the variable `x` must already be initialized in the store. */
  template <class Mem>
  CUDA this_type& tell(AVar x, const universe_type& dom, BInc<Mem>& has_changed) {
    assert(x.aty() == aty());
    return tell(x.vid(), dom, has_changed);
  }

  /** This tell method can grow the store if required, and therefore do not satisfy the PCCP model.
   * /!\ It should not be used in parallel.
  */
  template <class Alloc2, class Mem>
  CUDA this_type& tell(const tell_type<Alloc2>& t, BInc<Mem>& has_changed) {
    if(t.size() == 0) {
      return *this;
    }
    if(t[0].avar == AVar{}) {
      is_at_top.tell(local::BInc(true), has_changed);
      return *this;
    }
    if(t.back().avar.vid() >= data.size()) {
      data.resize(t.back().avar.vid()+1);
    }
    for(int i = 0; i < t.size(); ++i) {
      tell(t[i].avar, t[i].dom, has_changed);
    }
    return *this;
  }

  /** This tell method can grow the store if required, and therefore do not satisfy the PCCP model.
   * /!\ It should not be used in parallel.
  */
  template <class Alloc2>
  CUDA this_type& tell(const tell_type<Alloc2>& t) {
    local::BInc has_changed;
    return tell(t, has_changed);
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
  CUDA local::BInc ask(const ask_type<Alloc2>& t) const {
    for(int i = 0; i < t.size(); ++i) {
      if(!(data[t[i].avar.vid()] >= t[i].dom)) {
        return false;
      }
    }
    return true;
  }

  /**  An abstract element is extractable when it is not equal to top.
   * If the strategy is `atoms`, we check the domains are singleton.
   */
  template<class ExtractionStrategy = NonAtomicExtraction>
  CUDA bool is_extractable(const ExtractionStrategy& strategy = ExtractionStrategy()) const {
    if(is_top()) {
      return false;
    }
    if constexpr(ExtractionStrategy::atoms) {
      for(int i = 0; i < data.size(); ++i) {
        if(data[i].lb() < dual<typename universe_type::LB>(data[i].ub())) {
          return false;
        }
      }
    }
    return true;
  }

  /** Whenever `this` is different from `top`, we extract its data into `ua`.
   * \pre `is_extractable()` must be `true`.
   * For now, we suppose VStore is only used to store under-approximation, I'm not sure yet how we would interact with over-approximation. */
  template<class U2, class Alloc2>
  CUDA void extract(VStore<U2, Alloc2>& ua) const {
    if((void*)&ua != (void*)this) {
      ua.data = data;
      ua.is_at_top.dtell_bot();
    }
  }

  template<class Env>
  CUDA NI TFormula<typename Env::allocator_type> deinterpret(const Env& env) const {
    using F = TFormula<typename Env::allocator_type>;
    typename F::Sequence seq{env.get_allocator()};
    for(int i = 0; i < data.size(); ++i) {
      AVar v(aty(), i);
      seq.push_back(F::make_exists(aty(), env.name_of(v), env.sort_of(v)));
      auto f = data[i].deinterpret(v, env);
      f.type_as(aty());
      map_avar_to_lvar(f, env);
      seq.push_back(std::move(f));
    }
    return F::make_nary(AND, std::move(seq), aty());
  }

  CUDA void print() const {
    for(int i = 0; i < vars(); ++i) {
      data[i].print();
      printf("%s", (i+1 == vars() ? "\n" : ", "));
    }
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

template<class L, class K, class Alloc1, class Alloc2>
CUDA bool operator<=(const VStore<L, Alloc1>& a, const VStore<K, Alloc2>& b)
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

template<class L, class K, class Alloc1, class Alloc2>
CUDA bool operator<(const VStore<L, Alloc1>& a, const VStore<K, Alloc2>& b)
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

template<class L, class K, class Alloc1, class Alloc2>
CUDA bool operator>=(const VStore<L, Alloc1>& a, const VStore<K, Alloc2>& b)
{
  return b <= a;
}

template<class L, class K, class Alloc1, class Alloc2>
CUDA bool operator>(const VStore<L, Alloc1>& a, const VStore<K, Alloc2>& b)
{
  return b < a;
}

template<class L, class K, class Alloc1, class Alloc2>
CUDA bool operator==(const VStore<L, Alloc1>& a, const VStore<K, Alloc2>& b)
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

template<class L, class K, class Alloc1, class Alloc2>
CUDA bool operator!=(const VStore<L, Alloc1>& a, const VStore<K, Alloc2>& b)
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
