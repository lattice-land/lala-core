// Copyright 2022 Pierre Talbot

#ifndef LALA_CORE_PRE_FUB_HPP
#define LALA_CORE_PRE_FUB_HPP

#include "../logic/logic.hpp"

namespace lala {

template <class VT>
struct PreFLB;

/** `PreFUB` is a pre-abstract universe \f$ \langle \mathbb{F}\setminus\{NaN\}, \leq \rangle \f$ totally ordered by the floating-point arithmetic comparison operator.
    We work on a subset of floating-point numbers without NaN.
    It is used to represent (and possibly approximate) constraints of the form \f$ x \leq k \f$ where \f$ k \f$ is a real number.
*/
template<class VT>
struct PreFUB {
  using this_type = PreFUB<VT>;
  using dual_type = PreFLB<VT>;
  using value_type = VT;
  using lower_bound_type = dual_type;
  using upper_bound_type = this_type;

  constexpr static const bool is_totally_ordered = true;
  constexpr static const bool preserve_bot = true;
  constexpr static const bool preserve_top = true;
  constexpr static const bool preserve_join = true;
  constexpr static const bool preserve_meet = true;
  /** Note that -0 and +0 are treated as the same element. */
  constexpr static const bool injective_concretization = true;
  constexpr static const bool preserve_concrete_covers = false;
  constexpr static const bool is_lower_bound = false;
  constexpr static const bool is_upper_bound = true;
  constexpr static const char* name = "FUB";
  constexpr static const bool is_arithmetic = true;
  CUDA constexpr static value_type zero() { return 0.0; }
  CUDA constexpr static value_type one() { return 1.0; }

private:
  template<bool diagnose, bool is_tell, class F>
  CUDA NI static bool interpret(const F& f, value_type& k, IDiagnostics& diagnostics) {
    if(f.is(F::Z)) {
      auto z = f.z();
      // We do not consider the min and max values of integers to be infinities when they are part of the logical formula.
      if constexpr(is_tell) {
        k = battery::ru_cast<value_type, decltype(z), false>(z);
      }
      else {
        k = battery::rd_cast<value_type, decltype(z), false>(z);
      }
      return true;
    }
    else if(f.is(F::R)) {
      if constexpr(is_tell) {
        k = battery::ru_cast<value_type>(battery::get<1>(f.r()));
      }
      else {
        k = battery::rd_cast<value_type>(battery::get<0>(f.r()));
      }
      return true;
    }
    RETURN_INTERPRETATION_ERROR("Only a constant of sort `Int` or `Real` can be interpreted by a floating-point abstract universe.")
  }

public:
  /** Interpret a constant in the lattice of increasing floating-point numbers `FInc` according to the downset semantics.
      Interpretations:
        * Formulas of kind `F::Z` might be over-approximated (if the integer cannot be represented in a floating-point number because it is too large).
        * Formulas of kind `F::R` might be over-approximated to the upper bound of the interval (if the real number is represented by an interval [lb..ub] where lb != ub).
        * Other kind of formulas are not supported. */
  template<bool diagnose, class F>
  CUDA static bool interpret_tell(const F& f, value_type& k, IDiagnostics& diagnostics) {
    return interpret<diagnose, true>(f, k, diagnostics);
  }

  /** Same as `interpret_tell` but the constant is under-approximated instead. */
  template<bool diagnose, class F>
  CUDA static bool interpret_ask(const F& f, value_type& k, IDiagnostics& diagnostics) {
    return interpret<diagnose, false>(f, k, diagnostics);
  }

  /** Verify if the type of a variable, introduced by an existential quantifier, is compatible with the current abstract universe.
      Interpretations:
        * Variables of type `Int` are always over-approximated (\f$ \mathbb{Z} \subseteq \gamma(\top) \f$).
        * Variables of type `Real` are represented exactly (only initially because \f$ \mathbb{R} = \gamma(\top) \f$). */
  template<bool diagnose, class F, bool dualize = false>
  CUDA NI static bool interpret_type(const F& f, value_type& k, IDiagnostics& diagnostics) {
    assert(f.is(F::E));
    const auto& vname = battery::get<0>(f.exists());
    const auto& cty = battery::get<1>(f.exists());
    if(cty.is_int()) {
      k = dualize ? bot() : top();
      RETURN_INTERPRETATION_WARNING("Variable `" + vname + "` of sort `Int` is over-approximated in a floating-point abstract universe.");
    }
    else if(cty.is_real()) {
      k = dualize ? bot() : top();
      return true;
    }
    else {
      RETURN_INTERPRETATION_ERROR("Variable `" + vname + "` can only be of sort `Real`, or be over-approximated if the sort is `Bool` or `Int`.");
    }
  }

  /** Given a floating-point value, create a logical constant representing that value.
   * The constant is represented by a singleton interval of `double` [v..v].
   * Note that the lattice order has no influence here.
   * \pre `v != bot()` and `v != top()`.
  */
  template<class F>
  CUDA static F deinterpret(const value_type& v) {
    return F::make_real(v, v);
  }

  /** The logical predicate symbol corresponding to the order of this pre-universe.
      We have \f$ a \leq_\mathit{FInc} b \Leftrightarrow a \leq b \f$.
      \return The logical symbol `LEQ`. */
  CUDA static constexpr Sig sig_order() { return LEQ; }

  /** The logical predicate symbol corresponding to the strict order of this pre-universe.
      We have \f$ a <_\mathit{FInc} b \Leftrightarrow a < b \f$.
      \return The logical symbol `LT`. */
  CUDA static constexpr Sig sig_strict_order() { return LT; }

  /** \f$ \bot \f$ is represented by the floating-point negative infinity value. */
  CUDA static constexpr value_type bot() {
    return battery::limits<value_type>::neg_inf();
  }

  /** \f$ \top \f$  is represented by the floating-point positive infinity value. */
  CUDA static constexpr value_type top() {
    return battery::limits<value_type>::inf();
  }

  /** \return \f$ x \sqcup y \f$ defined as \f$ \mathit{max}(x, y) \f$. */
  CUDA static constexpr value_type join(value_type x, value_type y) { return battery::max(x, y); }

  /** \return \f$ x \sqcap y \f$ defined as \f$ \mathit{min}(x, y) \f$. */
  CUDA static constexpr value_type meet(value_type x, value_type y) { return battery::min(x, y); }

  /** \return \f$ \mathit{true} \f$ if \f$ x \leq_\mathit{FInc} y \f$ where the order \f$ \leq_\mathit{FInc} \f$ is the natural arithmetic ordering, otherwise returns \f$ \mathit{false} \f$. */
  CUDA static constexpr bool order(value_type x, value_type y) { return x <= y; }

  /** \return \f$ \mathit{true} \f$ if \f$ x <_\mathit{FInc} y \f$ where the order \f$ <_\mathit{ZInc} \f$ is the natural arithmetic ordering, otherwise returns \f$ \mathit{false} \f$. */
  CUDA static constexpr bool strict_order(value_type x, value_type y) { return x < y; }

  /** From a lattice perspective, this function returns an element \f$ y \f$ such that \f$ y \f$ is a cover of \f$ x \f$.

    \return The next value of \f$ x \f$ in the floating-point increasing chain \f$ -\infty, \ldots, prev(-2.0), -2.0, next(-2.0), \ldots, \infty \f$ is the next representable value of \f$ x \f$ when \f$ x \not\in \{\infty, -\infty\} \f$ and \f$ x \f$ otherwise.
      Note that \f$ 0 \f$ is considered to be represented by a single value. */
  CUDA static constexpr value_type next(value_type x) {
    if(x == bot() || x == top()) {
      return x;
    }
    if(x == value_type{-0.0}) {
      return battery::nextafter(value_type{+0.0}, top());
    }
    return battery::nextafter(x, top());
  }

  /** From a lattice perspective, this function returns an element \f$ y \f$ such that \f$ x \f$ is a cover of \f$ y \f$.

    \return The previous value of \f$ x \f$ in the floating-point increasing chain \f$ -\infty, \ldots, prev(-2.0), -2.0, next(-2.0), \ldots, \infty \f$ is the previous representable value of \f$ x \f$ when \f$ x \not\in \{\infty, -\infty\} \f$ and \f$ x \f$ otherwise.
      \f$ 0 \f$ is considered to be represented by a single value. */
  CUDA static constexpr value_type prev(value_type x) {
    if(x == bot() || x == top()) {
      return x;
    }
    if(x == value_type{+0.0}) {
      return battery::nextafter(value_type{-0.0}, bot());
    }
    return battery::nextafter(x, bot());
  }

  CUDA static constexpr value_type project(Sig fun, value_type x) {
    switch(fun) {
      case NEG: return -x;
      default: return top();
    }
  }

  CUDA static constexpr value_type project(Sig fun, value_type x, value_type y) {
    switch(fun) {
      case ADD: return battery::add_up(x, y);
      case SUB: return battery::sub_up(x, y);
      case MUL: return battery::mul_up(x, y);
      case DIV: return battery::div_up(x, y);
      case MIN: return battery::min(x, y);
      case MAX: return battery::max(x, y);
      case EQ: return x == y;
      case NEQ: return x != y;
      case LEQ: return x <= y;
      case GEQ: return x >= y;
      case LT: return x < y;
      case GT: return x >= y;
      default: return top();
    }
  }
};

} // namespace lala

#endif
