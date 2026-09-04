// Copyright 2022 Pierre Talbot

#ifndef LALA_CORE_ARITH_BOUND_HPP
#define LALA_CORE_ARITH_BOUND_HPP

#include <type_traits>
#include <utility>
#include <cmath>
#include <iostream>
#include "../logic/logic.hpp"
#include "pre_flb.hpp"
#include "pre_fub.hpp"
#include "pre_zlb.hpp"
#include "pre_zub.hpp"
#include "../b.hpp"
#include "battery/memory.hpp"

/** A pre-abstract universe is a lattice (with usual operations join, order, ...) equipped with a simple logical interpretation function and a next/prev functions.
    We consider totally ordered pre-abstract universes with a downset semantics.
    For any lattice \f$ L \f$, we consider an element \f$ a \in L \f$ to represent all the concrete elements equal to or below it.
    This set is called the downset of \f$ a \f$ and is denoted \f$ \mathord{\downarrow}{a} \f$.
    The concretization function \f$ \gamma \f$ formalizes this idea: \f$ \gamma(a) = \{x \mapsto b \;|\; b \in \mathord{\downarrow}{a} \cap U \} \f$ where \f$ U \f$ is the universe of discourse.
    The intersection with \f$ U \f$ is necessary to remove potential elements in the abstract universe that are not in the concrete universe of discourse (e.g., \f$ -\infty, \infty \f$ below).

    The downset semantics associates each element of a lattice to its concrete downset.
    It is possible to decide that each element is associated to the concrete upset instead.
    Doing so will reverse our usage of the lattice-theoretic operations (join instead of meet, <= instead of >=, etc.).
    Instead of considering the upset semantics, it is more convenient to consider the downset semantics of the dual lattice.

    Example:
      * The lattice of increasing integer \f$ \mathit{ZUB} = \langle \{-\infty, \ldots, -2, -1, 0, 1, 2, \ldots, \infty\}, \leq \rangle \f$ is ordered by the natural arithmetic comparison operator, it represents an upper bound on the set of integers represented.
        Using the downset semantics, we can represent simple constraints such as \f$ x \leq 3 \f$, in which case the downset \f$ \mathord{\downarrow}{3} = \{\ldots, 1, 2, 3\} \f$ represents all the values of \f$ x \f$ satisfying the constraints \f$ x \leq 3 \f$, that is, the solutions of the constraints.
      * By taking the upset semantics of \f$ \mathit{ZUB} \f$, we can represent constraints such as \f$ x \geq 3 \f$.
      * Alternatively, we can take the dual lattice of decreasing integers \f$ \mathit{ZLB} = \langle \{\infty, \ldots, 2, 1, 0, -1, -2, \ldots, -\infty\}, \geq \rangle \f$.
        The downset semantics of \f$ \mathit{ZLB} \f$ corresponds to the upset semantics of \f$ \mathit{ZUB} \f$.

  From a pre-abstract universe, we obtain an abstract universe using the `Universe` class below.
  We also define various aliases to abstract universes such as `ZLB`, `ZUB`, etc.
*/

namespace lala {

template<class PreUniverse, class Mem>
class FlatUniverse;

template<class PreUniverse, class Mem>
class ArithBound;

/** Lattice of integer lower bounds. */
template<class VT, class Mem>
using ZLB = ArithBound<PreZLB<VT>, Mem>;

/** Lattice of integer upper bounds. */
template<class VT, class Mem>
using ZUB = ArithBound<PreZUB<VT>, Mem>;

/** Lattice of floating-point lower bounds. */
template<class VT, class Mem>
using FLB = ArithBound<PreFLB<VT>, Mem>;

/** Lattice of floating-point upper bounds. */
template<class VT, class Mem>
using FUB = ArithBound<PreFUB<VT>, Mem>;

/** Aliases for lattice allocated on the stack (as local variable) and accessed by only one thread.
 * To make things simpler, the underlying type is also chosen (when required). */
namespace local {
  using ZLB = ::lala::ZLB<int, battery::local_memory>;
  using ZUB = ::lala::ZUB<int, battery::local_memory>;
  using FLB = ::lala::FLB<double, battery::local_memory>;
  using FUB = ::lala::FUB<double, battery::local_memory>;
}

namespace impl {
  template<class T>
  struct is_arith_bound {
    static constexpr bool value = false;
  };

  template<class PreUniverse, class Mem>
  struct is_arith_bound<ArithBound<PreUniverse, Mem>> {
    static constexpr bool value = true;
  };

  template <class T>
  inline constexpr bool is_arith_bound_v = is_arith_bound<T>::value;
}


template <class A, class R = A>
R project_fun(Sig fun, const A& a, const A& b) {
  R r{};
  r.project(fun, a, b);
  return r;
}

template <class A, class R = A>
R project_fun(Sig fun, const A& a) {
  R r{};
  r.project(fun, a);
  return r;
}

/** This function is useful when we need to convert a value to its dual.
    The dual is the upset of the current element, therefore, if we have \f$ x <= 10 \f$, the dual is given by the formula \f$ x >= 10 \f$ interpreted in the dual lattice.
    In that case, it just changes the type of the lattice without changing the value.
    A difference occurs on the bottom and top element.
    Indeed, by our representation of bot and top, the bottom value in a lattice L equals the top value in its dual, but we need them to remain the same, so the dual of `L::bot()` is `LDual::bot()`.*/
template <class LDual, class L>
CUDA constexpr LDual dual_bound(const L& x) {
  if(x.is_bot()) return LDual::bot();
  if(x.is_top()) return LDual::top();
  return LDual(x.value());
}

template<class PreUniverse, class Mem>
class ArithBound
{
  using U = PreUniverse;
public:
  using pre_universe = PreUniverse;
  using value_type = typename pre_universe::value_type;
  using memory_type = Mem;
  using this_type = ArithBound<pre_universe, memory_type>;
  using dual_type = ArithBound<typename pre_universe::dual_type, memory_type>;

  template<class M>
  using this_type2 = ArithBound<pre_universe, M>;

  using local_type = this_type2<battery::local_memory>;

  template<class M>
  using flat_type = FlatUniverse<typename pre_universe::upper_bound_type, M>;
  using local_flat_type = flat_type<battery::local_memory>;

  constexpr static const bool is_abstract_universe = true;
  constexpr static const bool sequential = Mem::sequential;
  constexpr static const bool is_totally_ordered = pre_universe::is_totally_ordered;
  constexpr static const bool preserve_bot = pre_universe::preserve_bot;
  constexpr static const bool preserve_top = pre_universe::preserve_top;
  constexpr static const bool preserve_join = pre_universe::preserve_join;
  constexpr static const bool preserve_meet = pre_universe::preserve_meet;
  constexpr static const bool injective_concretization = pre_universe::injective_concretization;
  constexpr static const bool preserve_concrete_covers = pre_universe::preserve_concrete_covers;
  constexpr static const bool is_lower_bound = pre_universe::is_lower_bound;
  constexpr static const bool is_upper_bound = pre_universe::is_upper_bound;
  constexpr static const char* name = pre_universe::name;

  constexpr static const bool is_arithmetic = pre_universe::is_arithmetic;

  static_assert(is_totally_ordered, "The underlying pre-universe must be totally ordered.");
  static_assert(is_arithmetic, "The underlying pre-universe must be arithmetic (e.g. integers, floating-point numbers).");

  /** A pre-interpreted formula `x >= value` ready to use.
   * This is mainly for optimization purpose to avoid calling `interpret` each time we need it. */
  CUDA static constexpr this_type geq_k(value_type k) {
    if constexpr(is_lower_bound) {
      return this_type(k);
    }
    else {
      static_assert(is_lower_bound,
        "The pre-interpreted formula x >= k is only available over abstract universe modelling lower bounds.");
    }
  }

  CUDA static constexpr this_type leq_k(value_type k) {
    if constexpr(is_upper_bound) {
      return this_type(k);
    }
    else {
      static_assert(is_upper_bound,
        "The pre-interpreted formula x <= k is only available over abstract universe modelling upper bounds.");
    }
  }

  using atomic_type = memory_type::template atomic_type<value_type>;
private:
  atomic_type val;

public:
  /** Similar to \f$[\![\mathit{false}]\!]\f$ if `preserve_bot` is true. */
  CUDA static constexpr local_type bot() { return local_type(U::bot()); }

  /** Similar to \f$[\![\mathit{true}]\!]\f$ if `preserve_top` is true. */
  CUDA static constexpr local_type top() { return local_type(U::top()); }
  /** Initialize an upset universe to top. */
  CUDA constexpr ArithBound(): val(U::top()) {}
  /** Similar to \f$[\![x \leq_A i]\!]\f$ for any name `x` where \f$ \leq_A \f$ is the lattice order. */
  CUDA constexpr ArithBound(value_type x): val(x) {}
  constexpr ArithBound(const this_type& other) = default;
  constexpr ArithBound(this_type&& other) = default;

  template <class M>
  CUDA constexpr ArithBound(const this_type2<M>& other): ArithBound(other.value()) {}

  /** The assignment operator can only be used in a sequential context.
   * It is monotone but not extensive. */
  template <class M>
  CUDA constexpr this_type& operator=(const this_type2<M>& other) {
    memory_type::store(val, other.value());
    return *this;
  }

  constexpr this_type& operator=(const this_type& other) = default;

  CUDA constexpr value_type value() const { return memory_type::load(val); }

  CUDA constexpr atomic_type& atomic() { return val; }

  // This is dangerous because a conversion to `value_type` can be done implicitly, and overloaded operators <, >, ... can be used on the underlying value_type instead of the abstract universe.
  CUDA constexpr operator value_type() const { return value(); }

  /** \return `true` whenever \f$ a = \top \f$, `false` otherwise.
   * @parallel @order-preserving @increasing
  */
  CUDA constexpr local::B is_top() const {
    return value() == U::top();
  }

  /** \return `true` whenever \f$ a = \bot \f$, `false` otherwise.
   * @parallel @order-preserving @decreasing
  */
  CUDA constexpr local::B is_bot() const {
    return value() == U::bot();
  }

  CUDA constexpr void join_top() {
    memory_type::store(val, U::top());
  }

  template<class M1>
  CUDA constexpr bool join(const this_type2<M1>& other) {
    value_type r1 = value();
    value_type r2 = other.value();
    if(U::strict_order(r1, r2)) {
      memory_type::store(val, r2);
      return true;
    }
    return false;
  }

  CUDA constexpr void meet_bot() {
    memory_type::store(val, U::bot());
  }

  template<class M1>
  CUDA constexpr bool meet(const this_type2<M1>& other) {
    value_type r1 = value();
    value_type r2 = other.value();
    if(U::strict_order(r2, r1)) {
      memory_type::store(val, r2);
      return true;
    }
    return false;
  }

  /** \return \f$ x <op> i \f$ where `x` is a variable's name, `i` the current value and `<op>` depends on the underlying universe.
  If `U` preserves top, `true` is returned whenever \f$ a = \top \f$, if it preserves bottom `false` is returned whenever \f$ a = \bot \f$.
  We always return an exact approximation, hence for any formula \f$ \llbracket \varphi \rrbracket = a \f$, we must have \f$ a =  \llbracket \rrbracket a \llbracket \rrbracket \f$ where \f$ \rrbracket a \llbracket \f$ is the deinterpretation function.


  /** Under-approximates the current element \f$ a \f$ w.r.t. \f$ \rrbracket a \llbracket \f$ into `ua`.
   * For this abstract universe, it always returns `true` since the current element \f$ a \f$ is an exact representation of \f$ \rrbracket a \llbracket \f$. */
  CUDA constexpr bool extract(local_type& ua) const {
    ua.val = value();
    return true;
  }

  /** Print the current element. */
  CUDA NI void print() const {
    if(is_bot()) {
      printf("\u22A5");
    }
    else if(is_top()) {
      printf("\u22A4");
    }
    else {
      ::battery::print(value());
    }
  }

private:



public:



public:
  CUDA static constexpr local_type next(const this_type2<Mem>& a) {
    return local_type(pre_universe::next(a.value()));
  }

  CUDA static constexpr local_type prev(const this_type2<Mem>& a) {
    return local_type(pre_universe::prev(a.value()));
  }

  /** Unary function of type `fun: FlatUniverse -> ArithBound`.
   * If `a` is `bot`, we meet with bottom in-place.
   * Otherwise, we apply the function `fun` to `a` and meet the result.
   * \remark The result of the function is always over-approximated (or exact when possible).
  */
  CUDA constexpr void project(Sig fun, const local_flat_type& a) {
    if(a.is_bot()) { meet_bot(); }
    else if(!a.is_top()) {
      meet(local_type(pre_universe::project(fun, a)));
    }
  }

  /** Binary functions of type `project: FlatUniverse x FlatUniverse -> ArithBound`.
   * If `a` or `b` is `bot`, we meet bottom in-place.
   * Otherwise, we meet `fun(a,b)` in-place.
   * \remark The result of the function is always over-approximated (or exact when possible).
   * \remark If the function `fun` is partial (e.g. division), we expect the arguments `a` and `b` to be in the domain of `fun` (e.g. not equal to 0).
   */
  CUDA constexpr void project(Sig fun, const local_flat_type& a, const local_flat_type& b) {
    if(a.is_bot() || b.is_bot()) { meet_bot(); }
    else if(!a.is_top() && !b.is_top()) {
      meet(local_type(pre_universe::project(fun, a.value(), b.value())));
    }
  }

  /** In this universe, the non-trivial order-preserving functions are {min, max, +}. */
  CUDA static constexpr bool is_trivial_fun(Sig fun) {
    return fun != MIN && fun != MAX && fun != ADD && (is_upper_bound || fun != ABS);
  }

  /** The functions that are order-preserving on the natural order of the universe of discourse, and its dual. */
  CUDA static constexpr bool is_order_preserving_fun(Sig fun) {
    return fun == ADD || fun == MIN || fun == MAX || (is_lower_bound && fun == ABS);
  }

  CUDA constexpr void project(Sig fun, const local_type &a, const local_type &b) {
    if (a.is_bot() || b.is_bot()) { meet_bot(); return; }
    if(fun == MIN || fun == MAX || fun == ADD) {
      meet(local_type(pre_universe::project(fun, a.value(), b.value())));
    }
  }

  CUDA constexpr void project(Sig fun, const local_type &a) {
    if (a.is_bot()) { meet_bot(); return; }
    if (!a.is_top()) {
      meet(project(fun, a.value()));
    }
  }

  template<class Pre2, class Mem2>
  friend class ArithBound;
};

// Lattice operators

template<class Pre, class M1, class M2>
CUDA constexpr ArithBound<Pre, battery::local_memory> fjoin(const ArithBound<Pre, M1>& a, const ArithBound<Pre, M2>& b) {
  return Pre::join(a.value(), b.value());
}

template<class Pre, class M1, class M2>
CUDA constexpr ArithBound<Pre, battery::local_memory> fmeet(const ArithBound<Pre, M1>& a, const ArithBound<Pre, M2>& b) {
  return Pre::meet(a.value(), b.value());
}

template<class Pre, class M1, class M2>
CUDA constexpr bool operator<=(const ArithBound<Pre, M1>& a, const ArithBound<Pre, M2>& b) {
  return Pre::order(a.value(), b.value());
}

template<class Pre, class M1, class M2>
CUDA constexpr bool operator<(const ArithBound<Pre, M1>& a, const ArithBound<Pre, M2>& b) {
  return Pre::strict_order(a.value(), b.value());
}

template<class Pre, class M1, class M2>
CUDA constexpr bool operator>=(const ArithBound<Pre, M1>& a, const ArithBound<Pre, M2>& b) {
  return Pre::order(b.value(), a.value());
}

template<class Pre, class M1, class M2>
CUDA constexpr bool operator>(const ArithBound<Pre, M1>& a, const ArithBound<Pre, M2>& b) {
  return Pre::strict_order(b.value(), a.value());
}

template<class Pre, class M1, class M2>
CUDA constexpr bool operator==(const ArithBound<Pre, M1>& a, const ArithBound<Pre, M2>& b) {
  return a.value() == b.value();
}

template<class Pre, class M1, class M2>
CUDA constexpr bool operator!=(const ArithBound<Pre, M1>& a, const ArithBound<Pre, M2>& b) {
  return a.value() != b.value();
}

template<class Pre, class M>
std::ostream& operator<<(std::ostream &s, const ArithBound<Pre, M> &a) {
  if(a.is_bot()) {
    s << "\u22A5";
  }
  else if(a.is_top()) {
    s << "\u22A4";
  }
  else {
    s << a.value();
  }
  return s;
}

} // namespace lala

#endif
