// Copyright 2021 Pierre Talbot

#ifndef CARTESIAN_PRODUCT_HPP
#define CARTESIAN_PRODUCT_HPP

#include "thrust/optional.h"
#include "utility.hpp"
#include "darray.hpp"
#include "ast.hpp"
#include "tuple.hpp"
#include "variant.hpp"

namespace lala {

/** The Cartesian product abstract domain is a _domain transformer_ combining several abstract domains.
Concretization function: \f$ \gamma((a_1, \ldots, a_n)) \sqcap_{i \leq n} \gamma_i(a_i) \f$. */
template<class...As>
class CartesianProduct {
  template<size_t i>
  using TypeOf = typename battery::tuple_element<i, battery::tuple<As...>>::type;
  constexpr static size_t n = battery::tuple_size<battery::tuple<As...>>{};
public:
  using Allocator = typename TypeOf<0>::Allocator;
  using this_type = CartesianProduct<As...>;

  /** We suppose the underlying value type is the same for all components.
      `ValueType` is the value type of the first component. */
  using ValueType = typename TypeOf<0>::ValueType;
private:
  battery::tuple<As...> val;

public:
  CUDA CartesianProduct(const As&... as): val(battery::make_tuple(as...)) {}
  CUDA CartesianProduct(As&&... as): val(battery::make_tuple(std::forward<As>(as)...)) {}
  CUDA CartesianProduct(typename As::ValueType... vs): val(battery::make_tuple(As(vs)...)) {}
  CUDA CartesianProduct(const this_type& other): val(other.val) {}

  /** Similar to \f$[\![\mathit{true}]\!]\f$. */
  CUDA static this_type bot() {
    return CartesianProduct(As::bot()...);
  }

  /** Similar to \f$[\![\mathit{false}]\!]\f$. */
  CUDA static this_type top() {
    return CartesianProduct(As::top()...);
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

  CUDA battery::tuple<As...>& value() {
    return val;
  }

  CUDA const battery::tuple<As...>& value() const {
    return val;
  }

private:
  template<size_t i>
  CUDA TypeOf<i>& project() {
    return get<i>(val);
  }

public:
  template<size_t i>
  CUDA const TypeOf<i>& project() const {
    return get<i>(val);
  }

  /** `true` if \f$ \exists{j \geq i},~\gamma(a_j) = \top^\flat \f$, `false` otherwise. */
  template<size_t i = 0>
  CUDA bool is_top() const {
    if constexpr (i < n) {
      return project<i>().is_top() || is_top<i+1>();
    }
    else {
      return false;
    }
  }

  /** `true` if \f$ \forall{j \geq i},~\gamma(a_j) = \bot^\flat \f$, `false` otherwise. */
  template<size_t i = 0>
  CUDA bool is_bot() const {
    if constexpr (i < n) {
      return project<i>().is_bot() && is_bot<i+1>();
    }
    else {
      return true;
    }
  }

private:
  template<size_t i = 0>
  CUDA this_type& join_(const this_type& other) {
    if constexpr (i < n) {
      project<i>().join(get<i>(other.val));
      return join_<i+1>(other);
    }
    else {
      return *this;
    }
  }

  template<size_t i = 0>
  CUDA this_type& meet_(const this_type& other) {
    if constexpr (i < n) {
      project<i>().meet(get<i>(other.val));
      return meet_<i+1>(other);
    }
    else {
      return *this;
    }
  }

  template<size_t i = 0>
  CUDA this_type& tell_(const this_type& other, bool& has_changed) {
    if constexpr (i < n) {
      project<i>().tell(get<i>(other.val), has_changed);
      return tell_<i+1>(other, has_changed);
    }
    else {
      return *this;
    }
  }

  template<size_t i = 0>
  CUDA this_type& dtell_(const this_type& other, bool& has_changed) {
    if constexpr (i < n) {
      project<i>().dtell(get<i>(other.val), has_changed);
      return dtell_<i+1>(other, has_changed);
    }
    else {
      return *this;
    }
  }

public:
  /** \f$ (a_1, \ldots, a_n) \sqcup (b_1, \ldots, b_n) = (a_1 \sqcup_1 b_1, \ldots, a_n \sqcup_n b_n) \f$ */
  CUDA this_type& join(const this_type& other) {
    return join_(other);
  }

  /** \f$ (a_1, \ldots, a_n) \sqcap (b_1, \ldots, b_n) = (a_1 \sqcap_1 b_1, \ldots, a_n \sqcap_n b_n) \f$ */
  CUDA this_type& meet(const this_type& other) {
    return meet_(other);
  }

  /** \f$ (a_1, \ldots, a_n) \sqcup (\bot_1, \ldots, b_i, \ldots, \bot_n) = (a_1, \ldots, a_i \sqcup_i b_i, \ldots, a_n) \f$ */
  template<size_t i>
  CUDA this_type& join(TypeOf<i>&& b) {
    project<i>().join(std::forward<TypeOf<i>>(b));
    return *this;
  }

  /** \f$ (a_1, \ldots, a_n) \sqcap (\bot_1, \ldots, b_i, \ldots, \bot_n) = (a_1, \ldots, a_i \sqcap_i b_i, \ldots, a_n) \f$ */
  template<size_t i>
  CUDA this_type& meet(TypeOf<i>&& b) {
    project<i>().meet(std::forward<TypeOf<i>>(b));
    return *this;
  }

  CUDA this_type& tell(const this_type& other, bool& has_changed) { return tell_(other, has_changed); }
  CUDA this_type& dtell(const this_type& other, bool& has_changed) { return dtell_(other, has_changed); }
  template<size_t i>
  CUDA this_type& tell(TypeOf<i>&& b, bool& has_changed) {
    project<i>().tell(std::forward<TypeOf<i>>(b), has_changed);
    return *this;
  }
  template<size_t i>
  CUDA this_type& dtell(TypeOf<i>&& b, bool& has_changed) {
    project<i>().dtell(std::forward<TypeOf<i>>(b), has_changed);
    return *this;
  }

  /** \f$ (a_1, \ldots, a_n) \leq (b_1, \ldots, b_n) \f$ holds when \f$ \forall{i \leq n},~a_i \leq_i b_i \f$. */
  template<size_t i = 0>
  CUDA bool order(const this_type& other) const {
    if constexpr (i < n) {
      bool is_leq = true;
      is_leq = project<i>().order(get<i>(other.val));
      return is_leq && order<i+1>(other);
    }
    else {
      return true;
    }
  }

  /** This is a non-commutative split, which splits on the first splittable abstract element (in the order of the template parameters). */
  template<typename Alloc = Allocator, size_t i = 0>
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

  /** Reset the abstract elements \f$ (a_1,\ldots,a_n) \f$ to the one of `other`. */
  template<size_t i = 0>
  CUDA void reset(const this_type& other) {
    if constexpr(i < n) {
      project<i>().reset(other.project<i>());
      reset<i+1>(other);
    }
  }

private:
  template<size_t i, class... Bs>
  CUDA this_type clone_(Bs&&... bs) const {
    if constexpr(i < n) {
      return clone_<i+1>(bs..., project<i>().clone());
    }
    else {
      return CartesianProduct(std::forward<Bs>(bs)...);
    }
  }
public:

  /** \return A copy of the current abstract element. */
  CUDA this_type clone() const {
    return clone_<0>();
  }

private:
  template<size_t i, typename Alloc = Allocator>
  CUDA TFormula<Alloc> deinterpret_(const LVar<Allocator>& x, TFormula<Allocator>::Sequence&& seq, const Alloc& allocator) const {
    if constexpr(i < n) {
      seq[i] = project<i>().deinterpret(x, allocator);
      return deinterpret_<i+1, Alloc>(x, std::move(seq), allocator);
    }
    else {
      return TFormula<Allocator>::make_nary(
        AND,
        std::forward<typename TFormula<Allocator>::Sequence>(seq),
        UNTYPED, EXACT, allocator);
    }
  }

public:
  template<typename Alloc = Allocator>
  CUDA TFormula<Alloc> deinterpret(const LVar<Allocator>& x, const Alloc& allocator = Alloc()) const {
    return deinterpret_<0, Alloc>(x, typename TFormula<Allocator>::Sequence(n), allocator);
  }

  template<size_t i = 0>
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

template<class...As>
CUDA bool operator==(const CartesianProduct<As...>& lhs, const CartesianProduct<As...>& rhs) {
  return lhs.value() == rhs.value();
}

template<class...As>
CUDA bool operator!=(const CartesianProduct<As...>& lhs, const CartesianProduct<As...>& rhs) {
  return !(lhs.value() == rhs.value());
}

} // namespace lala

#endif
