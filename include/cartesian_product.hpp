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
  using LogicalElement = battery::tuple<thrust::optional<As>...>;
private:
  battery::tuple<As...> value;

public:
  CUDA CartesianProduct(const As&... as): value(battery::make_tuple(as...)) {}
  CUDA CartesianProduct(As&&... as): value(battery::make_tuple(std::forward<As>(as)...)) {}
  CUDA CartesianProduct(const this_type& other): value(other.value) {}

  /** Similar to \f$[\![\mathit{true}]\!]\f$. */
  CUDA static this_type bot() {
    return CartesianProduct(As::bot()...);
  }

  /** Similar to \f$[\![\mathit{false}]\!]\f$. */
  CUDA static this_type top() {
    return CartesianProduct(As::top()...);
  }

  template<size_t i, typename Formula>
  CUDA thrust::optional<LogicalElement> interpret_one(Approx appx, const Formula& f) {
    auto one = get<i>(value).interpret(appx, f);
    if(one.has_value()) {
      LogicalElement res;
      get<i>(res) = std::move(one).value();
      return res;
    }
    else {
      return {};
    }
  }

private:
  template<size_t i = 0, typename Formula>
  CUDA thrust::optional<LogicalElement> interpret_all(Approx appx, const Formula& f, LogicalElement res, bool empty) {
    if constexpr(i == n) {
      if(empty) {
        return {};
      }
      else {
        return thrust::optional(std::move(res));
      }
    }
    else {
      auto one = get<i>(value).interpret(appx, f);
      if(one.has_value()) {
        empty = false;
        get<i>(res) = std::move(one).value();
      }
      return interpret_all<i+1>(appx, f, res, empty);
    }
  }

public:
  template<typename Formula>
  CUDA thrust::optional<LogicalElement> interpret(Approx appx, const Formula& f) {
    return interpret_all(appx, f, LogicalElement(), true);
  }

  CUDA battery::tuple<As...>& data() {
    return value;
  }

  CUDA const battery::tuple<As...>& data() const {
    return value;
  }

  template<size_t i>
  CUDA TypeOf<i>& project() {
    return get<i>(value);
  }

  template<size_t i>
  CUDA const TypeOf<i>& project() const {
    return get<i>(value);
  }

  /** `true` if \f$ \exists{j \geq i},~a_j = \top_j \f$, `false` otherwise. */
  template<size_t i = 0>
  CUDA bool is_top() const {
    if constexpr (i < n) {
      return project<i>().is_top() || is_top<i+1>();
    }
    else {
      return false;
    }
  }

  /** `true` if \f$ \forall{j \geq i},~a_j = \bot_j \f$, `false` otherwise. */
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
  CUDA this_type& join_(const LogicalElement& other) {
    if constexpr (i < n) {
      if(get<i>(other).has_value()) {
        project<i>().join(get<i>(other).value());
      }
      return join_<i+1>(other);
    }
    else {
      return *this;
    }
  }
  template<size_t i = 0>
  CUDA this_type& meet_(const LogicalElement& other) {
    if constexpr (i < n) {
      if(get<i>(other).has_value()) {
        project<i>().meet(get<i>(other).value());
      }
      return meet_<i+1>(other);
    }
    else {
      return *this;
    }
  }

public:
  /** \f$ (a_1, \ldots, a_n) \sqcup (b_1, \ldots, b_n) = (a_1 \sqcup_1 b_1, \ldots, a_n \sqcup_n b_n) \f$ */
  CUDA this_type& join(const LogicalElement& other) {
    return join_(other);
  }

  /** \f$ (a_1, \ldots, a_n) \sqcap (b_1, \ldots, b_n) = (a_1 \sqcap_1 b_1, \ldots, a_n \sqcap_n b_n) \f$ */
  CUDA this_type& meet(const LogicalElement& other) {
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

  template<size_t i = 0>
  CUDA bool refine() {
    if constexpr (i < n) {
      bool has_changed = project<i>().refine();
      has_changed |= refine<i+1>();
      return has_changed;
    }
    else {
      return false;
    }
  }

  /** \f$ (a_1, \ldots, a_n) \models \varphi \f$ is defined as \f$ (a_1, \ldots, a_n) \geq [\![\varphi]\!] \f$. */
  template<size_t i = 0>
  CUDA bool entailment(const LogicalElement& other) const {
    if constexpr (i < n) {
      bool is_entailed = true;
      if(get<i>(other).has_value()) {
        is_entailed = project<i>().entailment(get<i>(other).value());
      }
      return is_entailed && entailment<i+1>(other);
    }
    else {
      return true;
    }
  }

  /** This is a non-commutative split, which splits on the first splittable abstract element (in the order of the template parameters). */
  template<typename Alloc = Allocator, size_t i = 0>
  CUDA DArray<LogicalElement, Alloc> split(const Alloc& allocator = Alloc()) const {
    if constexpr(i < n) {
      auto split_i = project<i>().split(allocator);
      switch (split_i.size()) {
        case 0:
          return DArray<LogicalElement, Alloc>();
        case 1:
          return split<Alloc, i+1>(allocator);
        default:
          DArray<LogicalElement, Alloc> res(split_i.size(), allocator);
          for(int j = 0; j < res.size(); ++j) {
            get<i>(res[j]) = std::move(split_i[j]);
          }
          return std::move(res);
      }
    }
    else {
      return DArray<LogicalElement, Alloc>(1, allocator);
    }
  }

  /** Reset the abstract elements \f$(a_1,\ldots,a_n)\f$ to the one of `other`. */
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
  CUDA TFormula<Allocator> deinterpret_(TFormula<Allocator>::Sequence&& seq, const Allocator& allocator) const {
    if constexpr(i < n) {
      seq[i] = project<i>().deinterpret(allocator);
      return deinterpret_<i+1, Alloc>(std::move(seq), allocator);
    }
    else {
      return TFormula<Allocator>::make_nary(
        AND,
        std::forward<typename TFormula<Allocator>::Sequence>(seq),
        UNTYPED, allocator);
    }
  }

public:
  template<typename Alloc = Allocator>
  CUDA TFormula<Allocator> deinterpret(const Allocator& allocator = Allocator()) const {
    return deinterpret_<0, Alloc>(typename TFormula<Allocator>::Sequence(n), allocator);
  }

  template<size_t i = 0>
  CUDA void print() const {
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
  return lhs.data() == rhs.data();
}

template<class...As>
CUDA bool operator!=(const CartesianProduct<As...>& lhs, const CartesianProduct<As...>& rhs) {
  return !(lhs.data() == rhs.data());
}

} // namespace lala

#endif