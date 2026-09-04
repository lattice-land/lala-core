// Copyright 2021 Pierre Talbot

#ifndef LALA_CORE_VSTORE_HPP
#define LALA_CORE_VSTORE_HPP

#include "logic/logic.hpp"
#include "universes/arith_bound.hpp"
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
The bot element is smashed and the equality between two stores is represented by the following equivalence relation, for two stores \f$ S \f$ and \f$ T \f$:
\f$ S \equiv T \Leftrightarrow \forall{x \in \mathit{Vars}},~S(x) = T(x) \lor \exists{x \in \mathit{dom}(S)},\exists{y \in \mathit{dom}(T)},~S(x) = \bot \land T(y) = \bot \f$.
Intuitively, it means that either all elements are equal or both stores have a bot element, in which case they "collapse" to the bot element, and are considered equal.

The top element is the element \f$ \langle \top, \ldots \rangle \f$, that is an infinite number of variables initialized to top.
In practice, we cannot represent infinite collections, so we represent top either as the empty collection or with a finite number of top elements.
Any finite store \f$ \langle x_1, \ldots, x_n \rangle \f$ should be seen as the concrete store \f$ \langle x_1, \ldots, x_n, \top, \ldots \rangle \f$.

This semantics has implication when joining or merging two elements.
For instance, \f$ \langle 1 \rangle.\mathit{meet}(\langle \bot, 4 \rangle) \f$ will be equal to bottom, in that case represented by \f$ \langle \bot \rangle \f$.

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

  template <class Alloc>
  struct var_dom {
    AVar avar;
    local_universe dom;
    var_dom() = default;
    var_dom(const var_dom<Alloc>&) = default;
    CUDA explicit var_dom(const Alloc&) {}
    CUDA var_dom(AVar avar, const local_universe& dom): avar(avar), dom(dom) {}
    template <class VarDom>
    CUDA var_dom(const VarDom& other): avar(other.avar), dom(other.dom) {}
  };

  template <class Alloc>
  using tell_type = battery::vector<var_dom<Alloc>, Alloc>;

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
  B<memory_type> is_at_bot;

public:
  CUDA VStore(const this_type& other)
    : atype(other.atype), data(other.data), is_at_bot(other.is_at_bot)
  {}

  /** Initialize an empty store. */
  CUDA VStore(AType atype, const allocator_type& alloc = allocator_type())
   : atype(atype), data(alloc), is_at_bot(false)
  {}

  CUDA VStore(AType atype, int size, const allocator_type& alloc = allocator_type())
   : atype(atype), data(size, alloc), is_at_bot(false)
  {}

  template<class R>
  CUDA VStore(const VStore<R, allocator_type>& other)
    : atype(other.atype), data(other.data), is_at_bot(other.is_at_bot)
  {}

  template<class R, class Alloc2>
  CUDA VStore(const VStore<R, Alloc2>& other, const allocator_type& alloc = allocator_type())
    : atype(other.atype), data(other.data, alloc), is_at_bot(other.is_at_bot)
  {}

  /** Copy the vstore `other` in the current element.
   *  `deps` can be empty and is not used besides to get the allocator (since this abstract domain does not have dependencies). */
  template<class R, class Alloc2, class... Allocators>
  CUDA VStore(const VStore<R, Alloc2>& other, const AbstractDeps<Allocators...>& deps)
   : VStore(other, deps.template get_allocator<allocator_type>()) {}

  CUDA VStore(this_type&& other):
    atype(other.atype), data(std::move(other.data)), is_at_bot(other.is_at_bot) {}

  CUDA allocator_type get_allocator() const {
    return data.get_allocator();
  }

  CUDA AType aty() const {
    return atype;
  }

  /** Returns the number of variables currently represented by this abstract element. */
  CUDA int vars() const {
    return data.size();
  }

  CUDA static this_type top(AType atype = UNTYPED,
    const allocator_type& alloc = allocator_type{})
  {
    return VStore(atype, alloc);
  }

  /** A special symbolic element representing top. */
  CUDA static this_type bot(AType atype = UNTYPED,
    const allocator_type& alloc = allocator_type{})
  {
    auto s = VStore{atype, alloc};
    s.meet_bot();
    return std::move(s);
  }

  template <class Env>
  CUDA static this_type bot(Env& env,
    const allocator_type& alloc = allocator_type{})
  {
    return bot(env.extends_abstract_dom(), alloc);
  }

  template <class Env>
  CUDA static this_type top(Env& env,
    const allocator_type& alloc = allocator_type{})
  {
    return top(env.extends_abstract_dom(), alloc);
  }

  /** \return `true` if at least one element is equal to bot in the store, `false` otherwise.
   * @parallel @order-preserving @increasing
  */
  CUDA local::B is_bot() const {
    return is_at_bot;
  }

  /** The bottom element of a store of `n` variables is when all variables are at bottom, or the store is empty.
   * We do not expect to use this operation a lot, so its complexity is linear in the number of variables.
   * @parallel @order-preserving @decreasing */
  CUDA local::B is_top() const {
    if(is_at_bot) { return false; }
    for(int i = 0; i < vars(); ++i) {
      if(!data[i].is_top()) {
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
    is_at_bot = false;
    for(int i = 0; i < snap.size(); ++i) {
      data[i].join(snap[i]);
      is_at_bot.join(data[i].is_bot());
    }
    return *this;
  }

public:
  template <class Group, class Store>
  CUDA void copy_to(Group& group, Store& store) const {
    assert(vars() == store.vars());
    if(group.thread_rank() == 0) {
      store.is_at_bot = is_at_bot;
    }
    if(is_at_bot) {
      return;
    }
    for (int i = group.thread_rank(); i < store.vars(); i += group.num_threads()) {
      store.data[i] = data[i];
    }
  }

  template <class Store>
  CUDA void copy_to(Store& store) const {
    assert(vars() == store.vars());
    store.is_at_bot = is_at_bot;
    if(is_at_bot) {
      return;
    }
    for (int i = 0; i < store.vars(); ++i) {
      store.data[i] = data[i];
    }
  }

#ifdef __CUDACC__
  void prefetch(int dstDevice) const {
    // for compatibility in v1.2.8, useless anyway.
  }
#endif

  /** Change the allocator of the underlying data, and reallocate the memory without copying the old data. */
  CUDA void reset_data(allocator_type alloc) {
    data = store_type(data.size(), alloc);
  }

  template <class Univ>
  CUDA void project(AVar x, Univ& u) const {
    u.meet(project(x));
  }

  CUDA const universe_type& project(AVar x) const {
    assert(x.aty() == aty());
    assert(x.vid() < data.size());
    return data[x.vid()];
  }

  /** This projection must stay const, otherwise the user might tell new information in the universe, but we need to know in case we reach `top`. */
  CUDA const universe_type& operator[](int x) const {
    return data[x];
  }

  CUDA const universe_type& operator[](AVar x) const {
    return project(x);
  }

  CUDA void meet_bot() {
    is_at_bot.join_top();
  }

  /** Given an abstract variable `v`, `embed(VID(v), dom)` will update the domain of this variable with the new information `dom`.
   * This `embed` method follows PCCP's model, but the variable `x` must already be initialized in the store.
   * @parallel @order-preserving @increasing
  */
  CUDA bool embed(int x, const universe_type& dom) {
    assert(x < data.size());
    bool has_changed = data[x].meet(dom);
    if(has_changed && data[x].is_bot()) {
      is_at_bot.join_top();
    }
    return has_changed;
  }

  /** This `embed` method follows PCCP's model, but the variable `x` must already be initialized in the store. */
  CUDA bool embed(AVar x, const universe_type& dom) {
    assert(x.aty() == aty());
    return embed(x.vid(), dom);
  }

  /** This deduce method can grow the store if required, and therefore do not satisfy the PCCP model.
   * @sequential @order-preserving @increasing
  */
  template <class Alloc2>
  CUDA bool deduce(const tell_type<Alloc2>& t) {
    if(t.size() == 0) {
      return false;
    }
    int largest_vid = 0;
    for(int i = 0; i < t.size(); ++i) {
      if(t[i].avar == AVar{}) {
        return is_at_bot.join(local::B(true));
      }
      largest_vid = battery::max(largest_vid, t[i].avar.vid());
    }
    if(largest_vid >= data.size()) {
      data.resize(largest_vid+1);
    }
    bool has_changed = false;
    for(int i = 0; i < t.size(); ++i) {
      has_changed |= embed(t[i].avar, t[i].dom);
    }
    return has_changed;
  }

  /** Precondition: `other` must be smaller or equal in size than the current store. */
  template <class U2, class Alloc2>
  CUDA bool meet(const VStore<U2, Alloc2>& other) {
    bool has_changed = is_at_bot.join(other.is_at_bot);
    int min_size = battery::min(vars(), other.vars());
    for(int i = 0; i < min_size; ++i) {
      has_changed |= data[i].meet(other[i]);
    }
    for(int i = min_size; i < other.vars(); ++i) {
      assert(other[i].is_top()); // the size of the current store cannot be modified.
    }
    return has_changed;
  }

  CUDA void join_top() {
    is_at_bot.meet_bot();
    for(int i = 0; i < data.size(); ++i) {
      data[i].join_top();
    }
  }

  /** Precondition: `other` must be smaller or equal in size than the current store. */
  template <class U2, class Alloc2>
  CUDA bool join(const VStore<U2, Alloc2>& other)  {
    if(other.is_bot()) {
      return false;
    }
    int min_size = battery::min(vars(), other.vars());
    bool has_changed = is_at_bot.meet(other.is_at_bot);
    for(int i = 0; i < min_size; ++i) {
      has_changed |= data[i].join(other[i]);
    }
    for(int i = min_size; i < vars(); ++i) {
      has_changed |= data[i].join(U::top());
    }
    for(int i = min_size; i < other.vars(); ++i) {
      assert(other[i].is_top());
    }
    return has_changed;
  }

  /** \return `true` when we can deduce the content of `t` from the current domain.
   * For instance, if we have in the store `x = [0..10]`, we can deduce `x = [-1..11]` but we cannot deduce `x = [5..8]`.
   * @parallel @order-preserving @decreasing */
  template <class Alloc2>
  CUDA local::B ask(const ask_type<Alloc2>& t) const {
    for(int i = 0; i < t.size(); ++i) {
      if(!(data[t[i].avar.vid()] <= t[i].dom)) {
        return false;
      }
    }
    return true;
  }

  CUDA int num_deductions() const { return 0; }
  CUDA local::B deduce(int) const { assert(false); return false; }

  /**  An abstract element is extractable when it is not equal to bot.
   * If the strategy is `atoms`, we check the domains are singleton.
   */
  template<class ExtractionStrategy = NonAtomicExtraction>
  CUDA bool is_extractable(const ExtractionStrategy& strategy = ExtractionStrategy()) const {
    if(is_bot()) {
      return false;
    }
    if constexpr(ExtractionStrategy::atoms) {
      for(int i = 0; i < data.size(); ++i) {
        if(data[i].lb().value() != data[i].ub().value()) {
          return false;
        }
      }
    }
    return true;
  }

#ifdef __CUDACC__
  template<class ExtractionStrategy = NonAtomicExtraction>
  __device__ bool is_extractable(auto& group, const ExtractionStrategy& strategy = ExtractionStrategy()) const {
    if(is_bot()) {
      return false;
    }
    if constexpr(ExtractionStrategy::atoms) {
      __shared__ bool res;
      if(group.thread_rank() == 0) {
        res = true;
      }
      group.sync();
      for(int i = group.thread_rank(); i < data.size(); i += group.num_threads()) {
        if(data[i].lb().value() != data[i].ub().value()) {
          res = false;
        }
      }
      group.sync();
      return res;
    }
    else {
      return true;
    }
  }
#endif

  /** Whenever `this` is different from `bot`, we extract its data into `ua`.
   * \pre `is_extractable()` must be `true`.
   * For now, we suppose VStore is only used to store under-approximation, I'm not sure yet how we would interact with over-approximation. */
  template<class U2, class Alloc2>
  CUDA void extract(VStore<U2, Alloc2>& ua) const {
    if((void*)&ua != (void*)this) {
      ua.data = data;
      ua.is_at_bot.meet_bot();
    }
  }

private:

public:


  CUDA void print() const {
    if(is_bot()) {
      printf("\u22A5 | ");
    }
    printf("<");
    for(int i = 0; i < vars(); ++i) {
      data[i].print();
      printf("%s", (i+1 == vars() ? "" : ", "));
    }
    printf(">\n");
  }
};

// Lattice operations.
// Note that we do not consider the logical names.
// These operations are only considering the indices of the elements.

template<class L, class K, class Alloc>
CUDA auto fmeet(const VStore<L, Alloc>& a, const VStore<K, Alloc>& b)
{
  using U = decltype(fmeet(a[0], b[0]));
  if(a.is_bot() || b.is_bot()) {
    return VStore<U, Alloc>::bot(UNTYPED, a.get_allocator());
  }
  int max_size = battery::max(a.vars(), b.vars());
  int min_size = battery::min(a.vars(), b.vars());
  VStore<U, Alloc> res(UNTYPED, max_size, a.get_allocator());
  for(int i = 0; i < min_size; ++i) {
    res.embed(i, fmeet(a[i], b[i]));
  }
  for(int i = min_size; i < a.vars(); ++i) {
    res.embed(i, a[i]);
  }
  for(int i = min_size; i < b.vars(); ++i) {
    res.embed(i, b[i]);
  }
  return res;
}

template<class L, class K, class Alloc>
CUDA auto fjoin(const VStore<L, Alloc>& a, const VStore<K, Alloc>& b)
{
  using U = decltype(fjoin(a[0], b[0]));
  if(a.is_bot()) {
    if(b.is_bot()) {
      return VStore<U, Alloc>::bot(UNTYPED, a.get_allocator());
    }
    else {
      return VStore<U, Alloc>(b);
    }
  }
  else if(b.is_bot()) {
    return VStore<U, Alloc>(a);
  }
  else {
    int min_size = battery::min(a.vars(), b.vars());
    VStore<U, Alloc> res(UNTYPED, min_size, a.get_allocator());
    for(int i = 0; i < min_size; ++i) {
      res.embed(i, fjoin(a[i], b[i]));
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
      if(!(a[i] <= b[i])) {
        return false;
      }
    }
    for(int i = min_size; i < b.vars(); ++i) {
      if(!b[i].is_top()) {
        return false;
      }
    }
  }
  return true;
}

template<class L, class K, class Alloc1, class Alloc2>
CUDA bool operator<(const VStore<L, Alloc1>& a, const VStore<K, Alloc2>& b)
{
  if(a.is_bot()) {
    return !b.is_bot();
  }
  else {
    int min_size = battery::min(a.vars(), b.vars());
    bool strict = false;
    for(int i = 0; i < a.vars(); ++i) {
      if(i < b.vars()) {
        if(a[i] < b[i]) {
          strict = true;
        }
        else if(!(a[i] <= b[i])) {
          return false;
        }
      }
      else if(!a[i].is_top()) {
        strict = true;
        break;
      }
    }
    for(int i = min_size; i < b.vars(); ++i) {
      if(!b[i].is_top()) {
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
  if(a.is_bot()) {
    return b.is_bot();
  }
  else if(b.is_bot()) {
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
      if(!a[i].is_top()) {
        return false;
      }
    }
    for(int i = min_size; i < b.vars(); ++i) {
      if(!b[i].is_top()) {
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
  if(vstore.is_bot()) {
    s << "\u22A5: ";
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
