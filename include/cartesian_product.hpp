// Copyright 2021 Pierre Talbot

#ifndef CARTESIAN_PRODUCT_HPP
#define CARTESIAN_PRODUCT_HPP

#include "thrust/optional.h"
#include "utility.hpp"
#include "darray.hpp"
#include "ast.hpp"
#include "tuple.hpp"
#include "variant.hpp"
#include "z.hpp"

namespace lala {

/** The Cartesian product abstract domain is a _domain transformer_ combining several abstract domains.
Concretization function: \f$ \gamma((a_1, \ldots, a_n)) \sqcap_{i \leq n} \gamma_i(a_i) \f$. */
template<class...As>
class CartesianProduct {
public:
  template<size_t i>
  using TypeOf = typename battery::tuple_element<i, battery::tuple<As...>>::type;
  constexpr static size_t n = battery::tuple_size<battery::tuple<As...>>{};
  using this_type = CartesianProduct<As...>;
  using dual_type = CartesianProduct<typename As::dual_type...>;

  template<class...Bs> friend class CartesianProduct;

  using ValueType = battery::tuple<typename As::ValueType...>;
private:
  battery::tuple<As...> val;

public:
  CUDA CartesianProduct(const As&... as): val(battery::make_tuple(as...)) {}
  CUDA CartesianProduct(As&&... as): val(battery::make_tuple(std::forward<As>(as)...)) {}
  CUDA CartesianProduct(typename As::ValueType... vs): val(battery::make_tuple(As(vs)...)) {}
  CUDA CartesianProduct(const this_type& other): val(other.val) {}
  CUDA this_type& operator=(this_type&& other) {
    battery::tuple<As...> old = std::move(val);
    val = std::move(other.val);
    other.val = std::move(old);
    return *this;
  }

  /** Similar to \f$[\![\mathit{true}]\!]\f$. */
  CUDA static this_type bot() {
    return CartesianProduct(As::bot()...);
  }

  /** Similar to \f$[\![\mathit{false}]\!]\f$. */
  CUDA static this_type top() {
    return CartesianProduct(As::top()...);
  }

  CUDA dual_type dual() const {
    return dual_type(typename As::dual_type(get<As>(val).dual())...);
  }

  template<size_t i, typename Formula>
  CUDA static thrust::optional<this_type> interpret_one(const Formula& f) {
    auto one = TypeOf<i>::interpret(f);
    if(one.has_value()) {
      auto res = bot();
      get<i>(res.val) = std::move(one).value();
      return res;
    }
    return {};
  }

private:
  template<size_t i = 0, typename Formula>
  CUDA static thrust::optional<this_type> interpret_all(const Formula& f, this_type res, bool empty) {
    if constexpr(i == n) {
      if(empty) {
        return {};
      }
      else {
        return thrust::optional(std::move(res));
      }
    }
    else {
      auto one = TypeOf<i>::interpret(f);
      if(one.has_value()) {
        empty = false;
        get<i>(res.val) = std::move(one).value();
      }
      return interpret_all<i+1>(f, std::move(res), empty);
    }
  }

public:
  /** Interpret the formula `f` in all sub-universes in which `f` is interpretable. */
  template<typename Formula>
  CUDA static thrust::optional<this_type> interpret(const Formula& f) {
    return interpret_all(f, bot(), true);
  }

private:
  // The non-const version must stay private, otherwise it violates the PCCP model since the caller might not check if the updated value is strictly greater w.r.t. lattice order.
  template<size_t i>
  CUDA TypeOf<i>& project() {
    return get<i>(val);
  }

  template<size_t... I>
  CUDA ValueType value_(std::index_sequence<I...>) const {
    return ValueType(project<I>().value()...);
  }

  template<class O, class... Ls, class... Ks, size_t... I>
  CUDA typename leq_t<O, CartesianProduct<Ls...>, CartesianProduct<Ks...>>::type leq_(
    const CartesianProduct<Ls...>& a, const CartesianProduct<Ks...>& b,
    std::index_sequence<I...>)
  {
    return land(leq<typename O::TypeOf<I>>(project<I>(a), project<I>(b))...);
  }

  template<size_t... I>
  CUDA BInc is_top_(std::index_sequence<I...>) const {
    return lor(project<I>().is_top()...);
  }

  template<size_t... I>
  CUDA BDec is_bot_(std::index_sequence<I...>) const {
    return land(project<I>().is_bot()...);
  }
public:
  template<size_t i>
  CUDA const TypeOf<i>& project() const {
    return get<i>(val);
  }

  CUDA ValueType value() const {
    return value_(std::index_sequence_for<As...>{});
  }

  /** `true` if \f$ \exists{j \geq i},~\gamma(a_j) = \top^\flat \f$, `false` otherwise. */
  CUDA BInc is_top() const {
    return is_top_(std::index_sequence_for<As...>{});
  }

  /** `true` if \f$ \forall{j \geq i},~\gamma(a_j) = \bot^\flat \f$, `false` otherwise. */
  CUDA BDec is_bot() const {
    return is_bot_(std::index_sequence_for<As...>{});
  }

private:
  template<size_t i = 0>
  CUDA this_type& tell_(const this_type& other, BInc& has_changed) {
    if constexpr (i < n) {
      project<i>().tell(other.project<i>(), has_changed);
      return tell_<i+1>(other, has_changed);
    }
    else {
      return *this;
    }
  }

  template<size_t i = 0>
  CUDA this_type& dtell_(const this_type& other, BInc& has_changed) {
    if constexpr (i < n) {
      project<i>().dtell(other.project<i>(), has_changed);
      return dtell_<i+1>(other, has_changed);
    }
    else {
      return *this;
    }
  }

public:
  CUDA this_type& tell(const this_type& other, BInc& has_changed) { return tell_(other, has_changed); }
  CUDA this_type& dtell(const this_type& other, BInc& has_changed) { return dtell_(other, has_changed); }
  template<size_t i>
  CUDA this_type& tell(TypeOf<i>&& b, BInc& has_changed) {
    project<i>().tell(std::forward<TypeOf<i>>(b), has_changed);
    return *this;
  }
  template<size_t i>
  CUDA this_type& dtell(TypeOf<i>&& b, BInc& has_changed) {
    project<i>().dtell(std::forward<TypeOf<i>>(b), has_changed);
    return *this;
  }
  template<size_t i>
  CUDA this_type& tell(const TypeOf<i>& b, BInc& has_changed) {
    project<i>().tell(b, has_changed);
    return *this;
  }
  template<size_t i>
  CUDA this_type& dtell(const TypeOf<i>& b, BInc& has_changed) {
    project<i>().dtell(b, has_changed);
    return *this;
  }

  /** This is a non-commutative split, which splits on the first splittable abstract element (in the order of the template parameters). */
  template<class Alloc, size_t i = 0>
  CUDA DArray<this_type, Alloc> split(const Alloc& allocator = Alloc()) const {
    if constexpr(i < n) {
      auto split_i = project<i>().split(allocator);
      switch (split_i.size()) {
        case 0: return DArray<this_type, Alloc>();
        case 1: return split<Alloc, i+1>(allocator);
        default:
          DArray<this_type, Alloc> res(split_i.size(), *this, allocator);
          for(int j = 0; j < res.size(); ++j) {
            get<i>(res[j].val) = std::move(split_i[j]);
          }
          return std::move(res);
      }
    }
    else {
      return DArray<this_type, Alloc>(1, *this, allocator);
    }
  }

private:
  template<size_t... I>
  CUDA this_type clone_(std::index_sequence<I...>) const {
    return CartesianProduct(project<I>().clone()...);
  }
public:

  /** \return A copy of the current abstract element. */
  CUDA this_type clone() const {
    return clone_(std::index_sequence_for<As...>{});
  }

private:
  template<size_t i, class Allocator>
  CUDA TFormula<Allocator> deinterpret_(const LVar<Allocator>& x, TFormula<Allocator>::Sequence&& seq, const Allocator& allocator) const {
    if constexpr(i < n) {
      seq[i] = project<i>().deinterpret(x, allocator);
      return deinterpret_<i+1, Allocator>(x, std::move(seq), allocator);
    }
    else {
      return TFormula<Allocator>::make_nary(
        AND,
        std::forward<typename TFormula<Allocator>::Sequence>(seq),
        UNTYPED, EXACT, allocator);
    }
  }

public:
  template<class Allocator>
  CUDA TFormula<Allocator> deinterpret(const LVar<Allocator>& x, const Allocator& allocator = Allocator()) const {
    return deinterpret_<0, Allocator>(x, typename TFormula<Allocator>::Sequence(n), allocator);
  }

  template<class Allocator, size_t i = 0>
  CUDA void print(const LVar<Allocator>& x) const {
    if constexpr(i < n) {
      ::print(project<i>());
      if constexpr(i < n - 1) {
        printf("\n");
        print<i+1>();
      }
    }
  }
};

/// Similar to `cp.template project<i>()`, just to avoid the ".template" syntax.
template<size_t i, class... As>
CUDA const typename CartesianProduct<As...>::TypeOf<i>&
project(const CartesianProduct<As...>& cp) {
  return cp.template project<i>();
}

namespace impl {
  template<class... Ls, class... Ks, size_t... I>
  CUDA CartesianProduct<typename join_t<Ls, Ks>::type...> join_(
    const CartesianProduct<Ls...>& a, const CartesianProduct<Ks...>& b,
    std::index_sequence<I...>)
  {
    using R = CartesianProduct<typename join_t<Ls, Ks>::type...>;
    return R(join(project<I>(a), project<I>(b))...);
  }

  template<class... Ls, class... Ks, size_t... I>
  CUDA CartesianProduct<typename meet_t<Ls, Ks>::type...> meet_(
    const CartesianProduct<Ls...>& a, const CartesianProduct<Ks...>& b,
    std::index_sequence<I...>)
  {
    using R = CartesianProduct<typename meet_t<Ls, Ks>::type...>;
    return R(meet(project<I>(a), project<I>(b))...);
  }

  template<class O, class... Ls, class... Ks, size_t... I>
  CUDA typename leq_t<O, CartesianProduct<Ls...>, CartesianProduct<Ks...>>::type leq_(
    const CartesianProduct<Ls...>& a, const CartesianProduct<Ks...>& b,
    std::index_sequence<I...>)
  {
    return land(leq<typename O::TypeOf<I>>(project<I>(a), project<I>(b))...);
  }

  template<class O, class... Ls, class... Ks, size_t... I>
  CUDA typename lt_t<O, CartesianProduct<Ls...>, CartesianProduct<Ks...>>::type lt_(
    const CartesianProduct<Ls...>& a, const CartesianProduct<Ks...>& b,
    std::index_sequence<I...>)
  {
    return land(lt<typename O::TypeOf<I>>(project<I>(a), project<I>(b))...);
  }
}

/** \f$ (a_1, \ldots, a_n) \sqcup (b_1, \ldots, b_n) = (a_1 \sqcup_1 b_1, \ldots, a_n \sqcup_n b_n) \f$ */
template<class... Ls, class... Ks>
CUDA CartesianProduct<typename join_t<Ls, Ks>::type...> join(
  const CartesianProduct<Ls...>& a, const CartesianProduct<Ks...>& b)
{
  return impl::join_(a, b, std::index_sequence_for<Ls...>{});
}

/** \f$ (a_1, \ldots, a_n) \sqcap (b_1, \ldots, b_n) = (a_1 \sqcap_1 b_1, \ldots, a_n \sqcap_n b_n) \f$ */
template<class... Ls, class... Ks>
CUDA CartesianProduct<typename meet_t<Ls, Ks>::type...> meet(
  const CartesianProduct<Ls...>& a, const CartesianProduct<Ks...>& b)
{
  return impl::meet_(a, b, std::index_sequence_for<Ls...>{});
}

/** \f$ (a_1, \ldots, a_n) \sqcup (\bot_1, \ldots, b_i, \ldots, \bot_n) = (a_1, \ldots, a_i \sqcup_i b_i, \ldots, a_n) \f$ */
template<size_t i, class... Ls>
CUDA CartesianProduct<Ls...> join(const CartesianProduct<Ls...>& a,
  const typename CartesianProduct<Ls...>::TypeOf<i>& b)
{
  CartesianProduct<Ls...> r(a);
  BInc unused_ = BInc::bot();
  r.template tell<i>(b, unused_);
  return std::move(r);
}

/** \f$ (a_1, \ldots, a_n) \sqcap (\bot_1, \ldots, b_i, \ldots, \bot_n) = (a_1, \ldots, a_i \sqcap_i b_i, \ldots, a_n) \f$ */
template<size_t i, class... Ls>
CUDA CartesianProduct<Ls...> meet(const CartesianProduct<Ls...>& a,
  const typename CartesianProduct<Ls...>::TypeOf<i>& b)
{
  CartesianProduct<Ls...> r(a);
  BInc unused_ = BInc::bot();
  r.template dtell<i>(b, unused_);
  return std::move(r);
}

/** \f$ (a_1, \ldots, a_n) \leq (b_1, \ldots, b_n) \f$ holds when \f$ \forall{i \leq n},~a_i \leq_i b_i \f$. */
template<class O, class... Ls, class... Ks>
CUDA typename leq_t<O, CartesianProduct<Ls...>, CartesianProduct<Ks...>>::type leq(
  const CartesianProduct<Ls...>& a,
  const CartesianProduct<Ks...>& b)
{
  return impl::leq_<O>(a, b, std::index_sequence_for<Ls...>{});
}

template<class O, class... Ls, class... Ks>
CUDA typename lt_t<O, CartesianProduct<Ls...>, CartesianProduct<Ks...>>::type lt(
  const CartesianProduct<Ls...>& a,
  const CartesianProduct<Ks...>& b)
{
  return impl::lt_<O>(a, b, std::index_sequence_for<Ls...>{});
}

} // namespace lala

#endif
