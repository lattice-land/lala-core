// Copyright 2022 Pierre Talbot

#ifndef PRE_ZINC_HPP
#define PRE_ZINC_HPP

#include "../logic/logic.hpp"
#include "chain_pre_dual.hpp"

namespace lala {

/** `PreZInc` is a pre-abstract universe \f$ \langle \{-\infty, \ldots, -2, -1, 0, 1, 2, \ldots, \infty\}, \leq \rangle \f$ totally ordered by the natural arithmetic comparison operator.
    It is used to represent constraints of the form \f$ x \geq k \f$ where \f$ k \f$ is an integer.
*/
template<class VT>
struct PreZInc {
  using this_type = PreZInc<VT>;
  using reverse_type = ChainPreDual<this_type>;
  using value_type = VT;

  static_assert(std::is_integral_v<value_type>, "PreZInc only works over integer types.");

  constexpr static const bool is_totally_ordered = true;

  /** `true` if \f$ \gamma(\bot) = \bot^\flat \f$. */
  constexpr static const bool preserve_bot = true;

  /** `true` if \f$ \gamma(\top) = \top^\flat \f$. */
  constexpr static const bool preserve_top = true;

  /** The concretization is injective when each abstract element maps to a distinct concrete element.
      This is important for the correctness of `prev` and `next` because we suppose \f$ \gamma(x) != \gamma(\mathit{next}(x)) \f$ when \f$ x \neq \bot \land x \neq \top \f$. */
  constexpr static const bool injective_concretization = true;

  /** `true` if inner covers are preserved in the concrete domain, i.e., \f$ \gamma(\mathit{next}(x)) \f$ is a cover of \f$ \gamma(x) \f$.
      An inner cover is a cover where bottom and top are not considered. */
  constexpr static const bool preserve_inner_covers = true;

  /** `true` if for all element \f$ x \in A \f$, there exists an element \f$ \lnot x \in A \f$ such that \f$ x \sqcup \lnot x = \top \f$ and \f$ x \sqcap \lnot x = \bot \f$. */
  constexpr static const bool complemented = false;

  /** `true` if the natural order of the universe of discourse coincides with the lattice order of this pre-universe, `false` if it is reversed. */
  constexpr static const bool increasing = true;

  constexpr static const char* name = "ZInc";
  constexpr static const char* dual_name = "ZDec";

  constexpr static const value_type zero = 0;
  constexpr static const value_type one = 1;

  template<class F>
  using iresult = IResult<value_type, F>;

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
        * Formulas of kind `F::S` are not supported.

      We consider various cases of interpretation, depending on the initial sort of \f$ x \f$, the approximation kind and the constant on the right side.

      I. The sort is \f$ \mathbb{Z} \f$.
      ==================================

        * \f$ [\![ x:\mathbb{Z} \geq k:\mathbb{Z} ]\!] = k \f$.
        * \f$ [\![ x:\mathbb{Z} \geq false:\mathbb{B} ]\!] = 0 \f$.
        * \f$ [\![ x:\mathbb{Z} \geq true:\mathbb{B} ]\!] = 1 \f$.
        * \f$ [\![ x:\mathbb{Z} \geq [l..u]:\mathbb{R} ]\!]_o = \lceil l \rceil \f$. Note that all elements in \f$ [l..\lceil l \rceil[\f$ do not belong to \mathbb{Z}, so they can be safely ignored (even for an over-approximation).
        * \f$ [\![ x:\mathbb{Z} \geq [l..u]:\mathbb{R} ]\!]_u = \lceil u \rceil \f$.
        * \f$ [\![ x:\mathbb{Z} \geq [l..u]:\mathbb{R} ]\!]_e = l \f$ iff \f$ \lfloor l \rfloor = \lceil u \rceil \f$.

      II. The sort is \f$ \mathbb{B} \f$.
      ===================================

        * \f$ [\![ x:\mathbb{B} \geq k:\mathbb{Z} ]\!] = 0 \f$ iff \f$ k \leq 0 \f$.
        * \f$ [\![ x:\mathbb{B} \geq 1:\mathbb{Z} ]\!] = 1 \f$.
        * \f$ [\![ x:\mathbb{B} \geq k:\mathbb{Z} ]\!] = \top \f$ iff \f$ k > 1 \f$.
        * \f$ [\![ x:\mathbb{B} \geq false:\mathbb{B} ]\!] = 0 \f$.
        * \f$ [\![ x:\mathbb{B} \geq true:\mathbb{B} ]\!] = 1 \f$.
        * \f$ [\![ x:\mathbb{B} \geq [l..u]:\mathbb{R} ]\!]_o = \lceil l \rceil \f$. Note that all elements in \f$ [l..\lceil l \rceil[\f$ do not belong to \mathbb{Z}, so they can be safely ignored (even for an over-approximation).
        * \f$ [\![ x:\mathbb{B} \geq [l..u]:\mathbb{R} ]\!]_u = \lceil u \rceil \f$.
        * \f$ [\![ x:\mathbb{B} \geq [l..u]:\mathbb{R} ]\!]_e = l \f$ iff \f$ \lfloor l \rfloor = \lceil u \rceil \f$.
      */
  template<class F, class Sort, bool dualize = false>
  CUDA static iresult<F> interpret(const F& f, const Sort& sort, Approx appx) {
    if(f.is(F::Z) && sort.is_int()) {
      auto z = f.z();
      if(z == bot() || z == top()) {
        return iresult<F>(IError<F>(true, name, "Constant of sort `Int` with the minimal or maximal representable value of the underlying integer type. We use those values to model negative and positive infinities. Example: Suppose we use a byte type, `x >= 256` is interpreted as `x >= INF` which is always false and thus is different from the intended constraint.", f));
      }
      return iresult<F>(z);
    }
    else if(f.is(F::R) && sort.is_int()) {
      auto lb = battery::rd_cast<value_type>(battery::get<0>(f.r()));
      auto ub = battery::ru_cast<value_type>(battery::get<1>(f.r()));
      if(lb == ub) {
        return iresult<F>(std::move(lb));
      }
      else {
        switch(appx) {
          case UNDER: return iresult<F>(std::move(ub), IError<F>(false, name, "Constant of sort `Real` under-approximated by an integer.", f));
          case OVER: return iresult<F>(std::move(lb), IError<F>(false, name, "Constant of sort `Real` over-approximated by an integer.", f));
          default:
            assert(appx == EXACT);
            return iresult<F>(IError<F>(true, name, "Non-integer constant of sort `Real` cannot be exactly interpreted by an integer.", f));
        }
      }
    }
    else if(f.is(F::B) && sort.is_int()) {
      return iresult<F>(value_type(f.b() ? one : zero));
    }
    return iresult<F>(IError<F>(true, name, "Only constant of sorts `Bool`, `Int` and `Real` can be interpreted by an integer-type.", f));
  }

  /** Verify if the type of a variable, introduced by an existential quantifier, is compatible with the current abstract universe.
      Interpretations:
        * Variables of type `CType::Bool` are over-approximated (\f$ \mathbb{B} \subseteq \gamma(\bot) \f$).
        * Variables of type `CType::Int` are interpreted exactly (\f$ \mathbb{Z} = \gamma(\bot) \f$).
        * Variables of type `CType::Real` can only be under-approximated (\f$ \mathbb{R} \supseteq \gamma(\bot) \f$) */
  template<class F>
  CUDA static iresult<F> interpret_type(const F& f) {
    assert(f.is(F::E));
    const auto& vname = battery::get<0>(f.exists());
    const auto& sort = battery::get<1>(f.exists());
    if(sort.is_int()) {
      return iresult<F>(bot());
    }
    else if(sort.is_real()) {
      switch(f.approx()) {
        case UNDER:
          return iresult<F>(bot(), IError<F>(false, name, "Real variable `" + vname + "` under-approximated by an integer.", f));
        case OVER:
          return iresult<F>(IError<F>(true, name, "Real variable `" + vname + "` cannot be over-approximated by an integer.", f));
        default:
          assert(f.is_exact());
          return iresult<F>(IError<F>(true, name, "Real variable `" + vname + "` cannot be exactly represented by an integer.", f));
      }
    }
    else if(sort.is_bool() && f.is_over()) {
      return iresult<F>(bot(), IError<F>(false, name, "Boolean variable `" + vname + "` over-approximated by an integer.", f));
    }
    else {
      return iresult<F>(IError<F>(true, name, "The type of `" + vname + "` can only be `CType::Int`, under-approximated `CType::Real`, or over-approximated `CType::Bool`.", f));
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

  /** \f$ \bot \f$ is represented by the minimal representable value of the underlying value type. */
  CUDA static constexpr value_type bot() {
    return battery::Limits<value_type>::bot();
  }

  /** \f$ \top \f$ is represented by the maximal representable value of the underlying value type. */
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
  CUDA static constexpr value_type prev(value_type x) {
    return x - (x != top() && x != bot());
  }

  CUDA static constexpr bool is_supported_fun(Approx appx, Sig sig) {
    switch(sig) {
      case NEG:
      case ABS:
      case ADD:
      case SUB:
      case MUL:
      case TDIV:
      case TMOD:
      case FDIV:
      case FMOD:
      case CDIV:
      case CMOD:
      case EDIV:
      case EMOD:
      case POW:
      case MIN:
      case MAX:
      case EQ:
      case NEQ:
      case LEQ:
      case GEQ:
      case LT:
      case GT: return true;
      default: return false;
    }
  }

  template<Approx appx, Sig sig>
  CUDA static constexpr value_type fun(value_type x) {
    static_assert(sig == NEG || sig == ABS, "Unsupported unary function.");
    switch(sig) {
      case NEG: return -x;
      case ABS: return abs(x);
      default: assert(0); return x;
    }
  }

  template<Approx appx, Sig sig>
  CUDA static constexpr value_type fun(value_type x, value_type y) {
    static_assert(
      sig == ADD || sig == SUB || sig == MUL || sig == TDIV || sig == TMOD || sig == FDIV || sig == FMOD || sig == CDIV || sig == CMOD || sig == EDIV || sig == EMOD || sig == POW || sig == MIN || sig == MAX || sig == EQ || sig == NEQ || sig == LEQ || sig == GEQ || sig == LT || sig == GT,
      "Unsupported binary function.");
    switch(sig) {
      case ADD: return x + y;
      case SUB: return x - y;
      case MUL: return x * y;
      // Truncated division and modulus, by default in C++.
      case TDIV: return x / y;
      case TMOD: return x % y;
      // Floor division and modulus, see (Leijend D. (2003). Division and Modulus for Computer Scientists).
      case FDIV: return x / y - (battery::signum(x % y) == -battery::signum(y));
      case FMOD: return x % y + y * (battery::signum(x % y) == -battery::signum(y));
      // Ceil division and modulus.
      case CDIV: return x / y + (battery::signum(x % y) == battery::signum(y));
      case CMOD: return x % y - y * (battery::signum(x % y) == battery::signum(y));
      // Euclidean division and modulus, see (Leijend D. (2003). Division and Modulus for Computer Scientists).
      case EDIV: return x / y - ((x % y >= 0) ? 0 : battery::signum(y));
      case EMOD: return x % y + y * ((x % y >= 0) ? 0 : battery::signum(y));
      case POW: return battery::ipow(x, y);
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