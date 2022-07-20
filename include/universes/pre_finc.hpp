// Copyright 2022 Pierre Talbot

#ifndef PRE_ZINC_HPP
#define PRE_ZINC_HPP

namespace lala {

/** `PreFInc` is a pre-abstract universe \f$ \langle \mathbb{F}\setminus\{NaN\}, \leq \rangle \f$ totally ordered by the floating-point arithmetic comparison operator.
    We work on a subset of floating-point numbers without NaN.
    It is used to represent (and possibly approximate) constraints of the form \f$ x \geq k \f$ where \f$ k \f$ is a real number.
*/
template<class VT>
struct PreFInc {
  using this_type = PreFInc<VT>;
  using reverse_type = PreDual<this_type>;
  using value_type = VT;

  constexpr static bool preverse_bot = true;
  constexpr static bool preverse_top = true;
  /** Note that -0 and +0 are treated as the same element. */
  constexpr static bool injective_concretization = true;
  constexpr static bool preverse_inner_covers = false;
  constexpr static bool complemented = false;

  template<class F>
  using iresult = IResult<value_type, typename F::allocator_type>;

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
        return iresult<F>(error_tag, value_type{}, "Constant of type `CType::Int` with the minimal or maximal representable value of the underlying integer type. We use those values to model negative and positive infinities. Example: Suppose we use a byte type, `x >= 256` is interpreted as `x >= INF` which is always false and thus is different from the intended constraint.");
      }
      auto lb = rd_cast<value_type>(z);
      auto ub = ru_cast<value_type>(z);
      if(lb == ub) {
        return iresult<F>(lb);
      }
      switch(appx) {
        case UNDER: return iresult<F>(warning_tag, ub, "Constant of type `CType::Int` under-approximated as floating-point number.");
        case OVER: return iresult<F>(warning_tag, lb, "Constant of type `CType::Int` over-approximated as floating-point number.");
        default:
          assert(appx == EXACT);
          return iresult<F>(error_tag, value_type{}, "Constant of type `CType::Int` cannot be interpreted exactly because it does not have an exact representation as a floating-point number (it is probably too large).");
      }
    }
    else if(f.is(F::R)) {
      auto lb = rd_cast<value_type>(battery::get<0>(f.r()));
      auto ub = ru_cast<value_type>(battery::get<1>(f.r()));
      if(lb == ub) {
        return iresult<F>(lb);
      }
      else {
        switch(appx) {
          case UNDER: return iresult<F>(ub);
          case OVER: return iresult<F>(lb);
          default:
            assert(appx == EXACT);
            return iresult<F>(error_tag, lb, "Constant of type `CType::Real` cannot be exactly interpreted by a floating-point number because the approximation of the constant is imprecise.");
        }
      }
    }
    return iresult<F>(error_tag, 0, "Only constant of types `CType::Int` and `CType::Real` can be interpreted by an integer-type.");
  }

  /** Verify if the type of a variable, introduced by an existential quantifier, is compatible with the current abstract universe.
      Interpretations:
        * Variables of type `CType::Int` are always over-approximated.
        * Variables of type `CType::Real` are always over-approximated. */
  template<class F>
  CUDA static iresult<F> interpret_type(const F& f) {
    assert(f.is(F::E));
    const auto& vname = battery::get<0>(f.exists());
    const auto& cty = battery::get<1>(f.exists());
    if((cty.tag == CType::Int || cty.tag == CType::Real) && f.approx() == OVER) {
      return iresult<F>(bot());
    }
    else {
      return iresult<F>(error_tag, bot(), "Variable `" + vname + "` can only be of type CType::Int or CType::Real and be over-approximated in a floating-point universe.");
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

  /** \$f \bot \f$ is represented by the floating-point negative infinity value. */
  CUDA static constexpr value_type bot() {
    return battery::Limits<value_type>::bot();
  }

  /** \$f \top \f$  is represented by the floating-point positive infinity value. */
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
};

} // namespace lala

#endif