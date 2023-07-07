// Copyright 2022 Pierre Talbot

#ifndef LALA_CORE_PRE_FINC_HPP
#define LALA_CORE_PRE_FINC_HPP

#include "../logic/logic.hpp"

namespace lala {

template <class VT>
struct PreFDec;

/** `PreFInc` is a pre-abstract universe \f$ \langle \mathbb{F}\setminus\{NaN\}, \leq \rangle \f$ totally ordered by the floating-point arithmetic comparison operator.
    We work on a subset of floating-point numbers without NaN.
    It is used to represent (and possibly approximate) constraints of the form \f$ x \geq k \f$ where \f$ k \f$ is a real number.
*/
template<class VT>
struct PreFInc {
  using this_type = PreFInc<VT>;
  using dual_type = PreFDec<VT>;
  using value_type = VT;
  using increasing_type = this_type;

  constexpr static const bool is_totally_ordered = true;
  constexpr static const bool preserve_bot = true;
  constexpr static const bool preserve_top = true;
  /** Note that -0 and +0 are treated as the same element. */
  constexpr static const bool injective_concretization = true;
  constexpr static const bool preserve_concrete_covers = false;
  constexpr static const bool complemented = false;
  constexpr static const bool increasing = true;
  constexpr static const char* name = "FInc";
  constexpr static const bool is_arithmetic = true;
  CUDA constexpr static value_type zero() { return 0.0; }
  CUDA constexpr static value_type one() { return 1.0; }

  template<class F>
  using iresult = IResult<value_type, F>;

private:
  template<bool is_tell, class F>
  CUDA static iresult<F> interpret(const F& f) {
    if(f.is(F::Z)) {
      auto z = f.z();
      // We do not consider the min and max values of integers to be infinities when they are part of the logical formula.
      if constexpr(is_tell) {
        return iresult<F>(battery::rd_cast<value_type, decltype(z), false>(z));
      }
      else {
        return iresult<F>(battery::ru_cast<value_type, decltype(z), false>(z));
      }
    }
    else if(f.is(F::R)) {
      if constexpr(is_tell) {
        return iresult<F>(battery::rd_cast<value_type>(battery::get<0>(f.r())));
      }
      else {
        return iresult<F>(battery::ru_cast<value_type>(battery::get<1>(f.r())));
      }
    }
    return iresult<F>(IError<F>(true, name, "Only a constant of sort `Int` or `Real` can be interpreted by a floating-point abstract universe.", f));
  }

public:
  /** Interpret a constant in the lattice of increasing floating-point numbers `FInc` according to the upset semantics (see universe.hpp for explanation).
      Interpretations:
        * Formulas of kind `F::Z` might be over-approximated (if the integer cannot be represented in a floating-point number because it is too large).
        * Formulas of kind `F::R` might be over-approximated to the lower bound of the interval (if the real number is represented by an interval [lb..ub] where lb != ub).
        * Other kind of formulas are not supported. */
  template<class F>
  CUDA static iresult<F> interpret_tell(const F& f) {
    return interpret<true>(f);
  }

  /** Same as `interpret_tell` but the constant is under-approximated instead. */
  template<class F>
  CUDA static iresult<F> interpret_ask(const F& f) {
    return interpret<false>(f);
  }

  /** Verify if the type of a variable, introduced by an existential quantifier, is compatible with the current abstract universe.
      Interpretations:
        * Variables of type `Int` are always over-approximated (\f$ \mathbb{Z} \subseteq \gamma(\bot) \f$).
        * Variables of type `Real` are represented exactly (only initially because \f$ \mathbb{R} = \gamma(\bot) \f$). */
  template<class F>
  CUDA static iresult<F> interpret_type(const F& f) {
    assert(f.is(F::E));
    const auto& vname = battery::get<0>(f.exists());
    const auto& cty = battery::get<1>(f.exists());
    if(cty.is_int()) {
      return iresult<F>(bot(), IError<F>(false, name, "Variable `" + vname + "` of sort `Int` is over-approximated in a floating-point abstract universe.", f));
    }
    else if(cty.is_real()) {
      return iresult<F>(bot());
    }
    else {
      return iresult<F>(IError<F>(true, name, "Variable `" + vname + "` can only be of sort `Real`, or be over-approximated if the sort is `Bool` or `Int`.", f));
    }
  }

  /** Given a floating-point value, create a logical constant representing that value.
   * The constant is represented by a singleton interval of `double` [v..v].
   * Note that the lattice order has no influence here.
   * \precondition `v != bot()` and `v != top()`.
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
    return battery::limits<value_type>::bot();
  }

  /** \f$ \top \f$  is represented by the floating-point positive infinity value. */
  CUDA static constexpr value_type top() {
    return battery::limits<value_type>::top();
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

  CUDA static constexpr bool is_supported_fun(Sig sig) {
    switch(sig) {
      case NEG:
      case ABS:
      case MIN:
      case MAX:
      case EQ:
      case NEQ:
      case LEQ:
      case GEQ:
      case LT:
      case GT:
      case ADD:
      case SUB:
      case MUL:
      case DIV:
        return true;
      default: return false;
    }
  }

  template<Sig sig>
  CUDA static constexpr value_type fun(value_type x) {
    static_assert(is_supported_fun(sig), "Unsupported unary function.");
    // Negation and absolute functions are exact in floating-point arithmetic.
    switch(sig) {
      case NEG: return -x;
      case ABS: return abs(x);
      default: assert(0); return x;
    }
  }

  template<Sig sig>
  CUDA static constexpr value_type fun(value_type x, value_type y) {
    static_assert(is_supported_fun(sig), "Unsupported binary function.");
    switch(sig) {
      case ADD: return battery::add_down(x, y);
      case SUB: return battery::sub_down(x, y);
      case MUL: return battery::mul_down(x, y);
      case DIV: return battery::div_down(x, y);
      case MIN: return battery::min(x, y);
      case MAX: return battery::max(x, y);
      case EQ: return x == y;
      case NEQ: return x != y;
      case LEQ: return x <= y;
      case GEQ: return x >= y;
      case LT: return x < y;
      case GT: return x >= y;
      default: assert(0); return x;
    }
  }
};

} // namespace lala

#endif