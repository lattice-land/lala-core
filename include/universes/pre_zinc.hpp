// Copyright 2022 Pierre Talbot

#ifndef PRE_ZINC_HPP
#define PRE_ZINC_HPP

namespace lala {

/** `PreZInc` is a pre-abstract universe \f$ \langle \{-\infty, \ldots, -2, -1, 0, 1, 2, \ldots, \infty\}, \leq \rangle \f$ totally ordered by the natural arithmetic comparison operator.
    It is used to represent constraints of the form \f$ x \geq k \f$ where \f$ k \f$ is an integer.
*/
template<class VT>
struct PreZInc {
  using this_type = PreZInc<VT>;
  using reverse_type = PreDual<this_type>;
  using value_type = VT;

  /** `true` if \f$ \gamma(\bot) = \bot^\flat \f$. */
  constexpr static bool preverse_bot = true;

  /** `true` if \f$ \gamma(\top) = \top^\flat \f$. */
  constexpr static bool preverse_top = true;

  /** The concretization is injective when each abstract element maps to a distinct concrete element.
      This is important for the correctness of `prev` and `next` because we suppose \f$ \gamma(x) != \gamma(\mathit{next}(x)) \f$ when \f$ x \neq \bot \land x \neq \top \f$. */
  constexpr static bool injective_concretization = true;

  /** `true` if inner covers are preserved in the concrete domain, \emph{i.e.}, \f$ \gamma(\mathit{next}(x)) \f$ is a cover of \f$ \gamma(x) \f$.
      An inner cover is a cover where bottom and top are not considered. */
  constexpr static bool preverse_inner_covers = true;

  /** `true` if for all element \f$ x \in A \f$, there exists an element \f$ \lnot x \in A \f$ such that \f$ x \sqcup \lnot x = \top \f$ and \f$ x \sqcap \lnot x = \bot \f$. */
  constexpr static bool complemented = false;

  template<class F>
  using iresult = IResult<value_type, typename F::allocator_type>;

  /** Interpret a constant in the lattice of increasing integers according to the upset semantics (see universe.hpp for explanation).
      Overflows are not verified (issue #1).
      Interpretations:
        * Formulas of kind `F::Z` are interpreted exactly.
        * Formulas of kind `F::R` are under- or over-approximated, unless the real number represents an integer.
            Note that a constant is interpreted as in the constraint \f$ x >= 2.5 \f$ where \f$ x \f$ has an integer type, and thus the constraint being equivalent to \f$ x >= 3 \f$.
            If the approximation of the real constant is, e.g., \f$ [2.9..3.1] \f$, i.e., with an integer within the bounds, then the approximation matters:
              * UNDER: we consider \f$ x >= 4 \f$, we can't consider \f$ x >= 3 \f$ since the real number might be \f$ 3.01 \f$.
              * OVER: we consider \f$ x >= 3 \f$.
              * EXACT: we can't interpret the constraint exactly due to the approximated constant.
        * Formulas of kind `F::S` are not supported. */
  template<class F>
  CUDA static iresult<F> interpret(const F& f, Approx appx) {
    if(f.is(F::Z)) {
      auto z = f.z();
      if(z == bot() || z == top()) {
        return iresult<F>(error_tag, z, "Constant of type `CType::Int` with the minimal or maximal representable value of the underlying integer type. We use those values to model negative and positive infinities. Example: Suppose we use a byte type, `x >= 256` is interpreted as `x >= INF` which is always false and thus is different from the intended constraint.");
      }
      return iresult<F>(z);
    }
    else if(f.is(F::R)) {
      auto lb = rd_cast<value_type>(battery::get<0>(f.r()));
      auto ub = ru_cast<value_type>(battery::get<1>(f.r()));
      if(lb == ub) {
        return iresult<F>(lb);
      }
      else {
        switch(appx) {
          case UNDER:
            return iresult<F>(warning_tag, ub, "Constant of type `CType::Real` under-approximated by an integer.");
          case OVER:
            return iresult<F>(warning_tag, lb, "Constant of type `CType::Real` over-approximated by an integer.");
          default:
            assert(appx == EXACT);
            return iresult<F>(error_tag, 0, "Non-integer constant of type `CType::Real` cannot be exactly interpreted by an integer.");
        }
      }
    }
    return iresult<F>(error_tag, 0, "Only constant of types `CType::Int` and `CType::Real` can be interpreted by an integer-type.");
  }

  /** Verify if the type of a variable, introduced by an existential quantifier, is compatible with the current abstract universe.
      Interpretations:
        * Variables of type `CType::Int` are interpreted exactly (under- and overflow are not considered).
        * Variables of type `CType::Real` can be under-approximated as integers, but not over-approximated. */
  template<class F>
  CUDA static iresult<F> interpret_type(const F& f) {
    assert(f.is(F::E));
    const auto& vname = battery::get<0>(f.exists());
    const auto& cty = battery::get<1>(f.exists());
    switch(cty.tag) {
      case Int: return iresult<F>(bot());
      case Real:
        switch(appx) {
          case UNDER: return iresult<F>(warning_tag, bot(), "Real variable `" + vname + "` under-approximated by an integer.");
          case OVER: return iresult<F>(error_tag, bot(), "Real variable `" + vname + "` cannot be over-approximated by an integer.");
          default:
            assert(appx == EXACT);
            return iresult<F>(error_tag, bot(), "Real variable `" + vname + "` cannot be exactly represented by an integer.");
        }
      default:
        return iresult<F>(error_tag, bot(), "The type of `" + vname + "` can only be `CType::Int` or `CType::Real` when under-approximated.");
    }
  }

  /** The logical predicate symbol corresponding to the order of this pre-universe.
      We have \f$ a \leq_\mathit{ZInc} b \Leftrightarrow a \leq b \f$.
      \return The logical symbol `LEQ`. */
  CUDA static constexpr Sig sig_order() { return LEQ; }
  CUDA static constexpr Sig dual_sig_order() { return GEQ; }

  /** The logical predicate symbol corresponding to the strict order of this pre-universe.
      We have \f$ a <_\mathit{ZInc} b \Leftrightarrow a < b \f$.
      \return The logical symbol `LT`. */
  CUDA static constexpr Sig sig_strict_order() { return LT; }
  CUDA static constexpr Sig dual_sig_strict_order() { return GT; }

  /** \$f \bot \f$ is represented by the minimal representable value of the underlying value type. */
  CUDA static constexpr value_type bot() {
    return battery::Limits<value_type>::bot();
  }

  /** \$f \top \f$ is represented by the maximal representable value of the underlying value type. */
  CUDA static constexpr value_type top() {
    return battery::Limits<value_type>::top();
  }

  /** \return \f$ x \sqcup y \f$ defined as \f$ \mathit{max}(x, y) \f$. */
  CUDA static constexpr value_type join(value_type x, value_type y) { return battery::max(x, y); }

  /** \return \f$ x \sqcap y \f$ defined as \f$ \mathit{min}(x, y) \f$. */
  CUDA static constexpr value_type meet(value_type x, value_type y) { return battery::min(x, y); }

  /** \return \f$ \mathit{true} \f$ if \f$ x \leq_\mathit{ZInc} y \f$ where the order \f$ \leq_\mathit{ZInc} \f$ is the natural arithmetic ordering, otherwise returns \f$ \mathit{false} \f$. */
  CUDA static constexpr bool order(value_type x, value_type y) { return x <= y; }

  /** \return \f$ \mathit{true} \f$ if \f$ x <_\mathit{ZInc} y \f$ where the order \f$ <_\mathit{ZInc} \f$ is the natural arithmetic ordering, otherwise returns \f$ \mathit{false} \f$. */
  CUDA static constexpr bool strict_order(value_type x, value_type y) { return x < y; }

  /** `true` if the element \f$ x \f$ has a unique cover in the abstract universe. */
  CUDA static constexpr bool has_unique_next(value_type x) { return true; }

  /** `true` if the element \f$ x \f$ covers a unique element in the abstract universe. */
  CUDA static constexpr bool has_unique_prev(value_type x) { return true; }

  /**  From a lattice perspective, this function returns an element \f$ y \f$ such that \f$ y \f$ is a cover of \f$ x \f$.

    \return The next value of \f$ x \f$ in the discrete increasing chain \f$ -\infty, \ldots, -2, -1, 0, 1, \ldots \infty \f$ is \f$ x + 1 \f$ when \f$ x \not\in \{\infty, -\infty\} \f$ and \f$ x \f$ otherwise. */
  CUDA static constexpr value_type next(value_type x) {
    return x + (x != top() && x != bot());
  }

  /** From a lattice perspective, this function returns an element \f$ y \f$ such that \f$ x \f$ is a cover of \f$ y \f$.

   \return The previous value of \f$ x \f$ in the discrete increasing chain \f$ -\infty, \ldots, -2, -1, 0, 1, \ldots \infty \f$ is \f$ x - 1 \f$ when \f$ x \not\in \{\infty, -\infty\} \f$ and \f$ x \f$ otherwise. */
  CUDA static constexpr value_type prev(value_type x, Approx appx) {
    return x - (x != top() && x != bot());
  }
};

} // namespace lala

#endif