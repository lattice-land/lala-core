// Copyright 2022 Pierre Talbot

#ifndef PRE_FINC_HPP
#define PRE_FINC_HPP

#include "../logic/logic.hpp"
#include "chain_pre_dual.hpp"

namespace lala {

/** `PreFInc` is a pre-abstract universe \f$ \langle \mathbb{F}\setminus\{NaN\}, \leq \rangle \f$ totally ordered by the floating-point arithmetic comparison operator.
    We work on a subset of floating-point numbers without NaN.
    It is used to represent (and possibly approximate) constraints of the form \f$ x \geq k \f$ where \f$ k \f$ is a real number.
*/
template<class VT>
struct PreFInc {
  using this_type = PreFInc<VT>;
  using reverse_type = ChainPreDual<this_type>;
  using value_type = VT;

  constexpr static const bool is_totally_ordered = true;
  constexpr static const bool preserve_bot = true;
  constexpr static const bool preserve_top = true;
  /** Note that -0 and +0 are treated as the same element. */
  constexpr static const bool injective_concretization = true;
  constexpr static const bool preserve_inner_covers = false;
  constexpr static const bool complemented = false;
  constexpr static const bool increasing = true;
  constexpr static const char* name = "FInc";
  constexpr static const char* dual_name = "FDec";
  constexpr static const value_type zero = 0.0;
  constexpr static const value_type one = 1.0;

  template<class F>
  using iresult = IResult<value_type, F>;

  /** Interpret a constant in the lattice of increasing floating-point numbers `FInc` according to the upset semantics (see universe.hpp for explanation).
      Interpretations:
        * Formulas of kind `F::Z` might be approximated (if the integer cannot be represented in a floating-point number because it is too large).
        * Formulas of kind `F::R` might be approximated (the real number is represented by an interval [lb..ub]):
            1. UNDER: \f$ x >= ub \f$ .
            2. OVER: \f$ x >= lb \f$.
            3. EXACT: Only possible if \f$ lb == ub \f$ and \f$ lb \f$ is representable in the type `value_type`.
        * Formulas of kind `F::S` are not supported. */
  template<class F>
  CUDA static iresult<F> interpret(const F& f, Approx appx) {
    if(f.is(F::Z)) {
      auto z = f.z();
      if(z == bot() || z == top()) {
        return iresult<F>(IError<F>(true, name, "Constant of type `CType::Int` with the minimal or maximal representable value of the underlying integer type. We use those values to model negative and positive infinities. Example: Suppose we use a byte type, `x >= 256` is interpreted as `x >= INF` which is always false and thus is different from the intended constraint.", f));
      }
      auto lb = rd_cast<value_type>(z);
      auto ub = ru_cast<value_type>(z);
      if(lb == ub) {
        return iresult<F>(std::move(lb));
      }
      switch(appx) {
        case UNDER: return iresult<F>(IError<F>(false, name, "Constant of type `CType::Int` under-approximated as floating-point number.", f));
        case OVER: return iresult<F>(IError<F>(false, name, "Constant of type `CType::Int` over-approximated as floating-point number.", f));
        default:
          assert(appx == EXACT);
          return iresult<F>(IError<F>(true, name, "Constant of type `CType::Int` cannot be interpreted exactly because it does not have an exact representation as a floating-point number (it is probably too large).", f));
      }
    }
    else if(f.is(F::R)) {
      auto lb = rd_cast<value_type>(battery::get<0>(f.r()));
      auto ub = ru_cast<value_type>(battery::get<1>(f.r()));
      if(lb == ub) {
        return iresult<F>(std::move(lb));
      }
      else {
        switch(appx) {
          case UNDER: return iresult<F>(std::move(ub));
          case OVER: return iresult<F>(std::move(lb));
          default:
            assert(appx == EXACT);
            return iresult<F>(IError<F>(true, name, "Constant of type `CType::Real` cannot be exactly interpreted by a floating-point number because the approximation of the constant is imprecise.", f));
        }
      }
    }
    else if(f.is(F::B)) {
      return iresult<F>(f.b() ? 1 : 0);
    }
    return iresult<F>(IError<F>(true, name, "Only constant of types `CType::Bool`, `CType::Int` and `CType::Real` can be interpreted by an integer-type.", f));
  }

  /** Verify if the type of a variable, introduced by an existential quantifier, is compatible with the current abstract universe.
      Interpretations:
        * Variables of type `CType::Bool` are always over-approximated (\f$ \mathbb{B} \subseteq \gamma(\bot) \f$).
        * Variables of type `CType::Int` are always over-approximated (\f$ \mathbb{Z} \subseteq \gamma(\bot) \f$).
        * Variables of type `CType::Real` are represented exactly (only initially because \f$ \mathbb{R} = \gamma(\bot) \f$. */
  template<class F>
  CUDA static iresult<F> interpret_type(const F& f) {
    assert(f.is(F::E));
    const auto& vname = battery::get<0>(f.exists());
    const auto& cty = battery::get<1>(f.exists());
    if(cty.is_bool() && f.is_over()) {
      return iresult<F>(bot(), IError<F>(false, name, "Variable `" + vname + "` of type `CType::Bool` is over-approximated in a floating-point abstract universe.", f));
    }
    else if(cty.is_int() && f.is_over()) {
      return iresult<F>(bot(), IError<F>(false, name, "Variable `" + vname + "` of type `CType::Int` is over-approximated in a floating-point abstract universe.", f));
    }
    else if(cty.is_real()) {
      return iresult<F>(bot());
    }
    else {
      return iresult<F>(IError<F>(true, name, "Variable `" + vname + "` can only be of type `CType::Real`, or be over-approximated if the type is `CType::Bool` or `CType::Int`.", f));
    }
  }

  /** The logical predicate symbol corresponding to the order of this pre-universe.
      We have \f$ a \leq_\mathit{FInc} b \Leftrightarrow a \leq b \f$.
      \return The logical symbol `LEQ`. */
  CUDA static constexpr Sig sig_order() { return LEQ; }
  CUDA static constexpr Sig dual_sig_order() { return GEQ; }

  /** The logical predicate symbol corresponding to the strict order of this pre-universe.
      We have \f$ a <_\mathit{FInc} b \Leftrightarrow a < b \f$.
      \return The logical symbol `LT`. */
  CUDA static constexpr Sig sig_strict_order() { return LT; }
  CUDA static constexpr Sig dual_sig_strict_order() { return GT; }

  /** \f$ \bot \f$ is represented by the floating-point negative infinity value. */
  CUDA static constexpr value_type bot() {
    return battery::Limits<value_type>::bot();
  }

  /** \f$ \top \f$  is represented by the floating-point positive infinity value. */
  CUDA static constexpr value_type top() {
    return battery::Limits<value_type>::top();
  }

  /** \return \f$ x \sqcup y \f$ defined as \f$ \mathit{max}(x, y) \f$. */
  CUDA static constexpr value_type join(value_type x, value_type y) { return battery::max(x, y); }

  /** \return \f$ x \sqcap y \f$ defined as \f$ \mathit{min}(x, y) \f$. */
  CUDA static constexpr value_type meet(value_type x, value_type y) { return battery::min(x, y); }

  /** \return \f$ \mathit{true} \f$ if \f$ x \leq_\mathit{FInc} y \f$ where the order \f$ \leq_\mathit{FInc} \f$ is the natural arithmetic ordering, otherwise returns \f$ \mathit{false} \f$. */
  CUDA static constexpr bool order(value_type x, value_type y) { return x <= y; }

  /** \return \f$ \mathit{true} \f$ if \f$ x <_\mathit{FInc} y \f$ where the order \f$ <_\mathit{ZInc} \f$ is the natural arithmetic ordering, otherwise returns \f$ \mathit{false} \f$. */
  CUDA static constexpr bool strict_order(value_type x, value_type y) { return x < y; }

  CUDA static constexpr bool has_unique_next(value_type x) { return true; }
  CUDA static constexpr bool has_unique_prev(value_type x) { return true; }

  /** From a lattice perspective, this function returns an element \f$ y \f$ such that \f$ y \f$ is a cover of \f$ x \f$.

    \return The next value of \f$ x \f$ in the floating-point increasing chain \f$ -\infty, \ldots, prev(-2.0), -2.0, next(-2.0), \ldots, \infty \f$ is the next representable value of \f$ x \f$ when \f$ x \not\in \{\infty, -\infty\} \f$ and \f$ x \f$ otherwise.
      Note that \f$ 0 \f$ is considered to be represented by a single value. */
  CUDA static value_type next(value_type x) {
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
  CUDA static value_type prev(value_type x) {
    if(x == bot() || x == top()) {
      return x;
    }
    if(x == value_type{+0.0}) {
      return battery::nextafter(value_type{-0.0}, bot());
    }
    return battery::nextafter(x, bot());
  }

  CUDA static constexpr bool is_supported_fun(Approx appx, Sig sig) {
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
        return true;
      case ADD:
      case SUB:
      case MUL:
      case DIV:
        return appx != EXACT;
      default: return false;
    }
  }

  template<Approx appx, Sig sig>
  CUDA static constexpr value_type fun(value_type x) {
    static_assert(is_supported_fun(appx, sig), "Unsupported unary function.");
    // Negation and absolute function are exact functions in floating-point arithmetic.
    switch(sig) {
      case NEG: return -x;
      case ABS: return abs(x);
      default: assert(0); return x;
    }
  }

  template<Approx appx, Sig sig>
  CUDA static constexpr value_type fun(value_type x, value_type y) {
    static_assert(is_supported_fun(appx, sig), "Unsupported binary function.");
    switch(sig) {
      case ADD: return appx == UNDER ? battery::add_up(x, y) : battery::add_down(x, y);
      case SUB: return appx == UNDER ? battery::sub_up(x, y) : battery::sub_down(x, y);
      case MUL: return appx == UNDER ? battery::mul_up(x, y) : battery::mul_down(x, y);
      case DIV: return appx == UNDER ? battery::div_up(x, y) : battery::div_down(x, y);
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