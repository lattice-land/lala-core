// Copyright 2021 Pierre Talbot

/** A pre-abstract universe is a lattice (with usual operations join, order, ...) equipped with a simple logical interpretation function and a next/prev functions.
    We consider pre-abstract universes with an upset semantics.
    For any lattice \f$ L \f$, we consider an element \f$ a \in L \f$ to represent all the concrete elements equal to or above it.
    This set is called the upset of \f$ a \f$ and is denoted \f$ \mathord{\uparrow}{a} \f$.
    The concretization function \f$ \gamma \f$ formalizes this idea: \f$ \gamma(a) = \{x \mapsto b \;|\; b \in \mathord{\uparrow}{a} \cap U \} \f$ where \f$ U \f$ is the universe of discourse.
    The intersection with \f$ U \f$ is necessary to remove potential elements in the abstract universe that are not in the concrete universe of discourse (e.g., \f$ -\infty, \infty \f$ below).

    The upset semantics associates each element of a lattice to its concrete upset.
    It is possible to decide that each element is associated to the concrete downset instead.
    Doing so will reverse our usage of the lattice-theoretic operations (join instead of meet, <= instead of >=, etc.).
    Instead of considering the upset semantics, it is more convenient to consider the downset semantics of the dual lattice.

    Example:
      * The lattice of increasing integer \f$ \mathit{ZInc} = \langle \{-\infty, \ldots, -2, -1, 0, 1, 2, \ldots, \infty\}, \leq \rangle \f$ is ordered by the natural arithmetic comparison operator.
        Using the upset semantics, we can represent simple constraints such as \f$ x \geq 3 \f$, in which case the upset \f$ \mathord{\uparrow}{3} = \{3, 4, \ldots\} \f$ represents all the values of \f$ x \f$ satisfying the constraints \f$ x \geq 3 \f$, that is, the solutions of the constraints.
      * By taking the downset semantics of \f$ \mathit{ZInc} \f$, we can represent constraints such as \f$ x \leq 3 \f$.
      * Alternatively, we can take the dual lattice of decreasing integers \f$ \mathit{ZDec} = \langle \{\infty, \ldots, 2, 1, 0, -1, -2, \ldots, -\infty\}, \geq \rangle \f$.
        The upset semantics of \f$ \mathit{ZDec} \f$ corresponds to the downset semantics of \f$ \mathit{ZInc} \f$.
*/

#ifndef PRE_Z_HPP
#define PRE_Z_HPP

enum class Sign {
  NEG,
  POS,
  BOTH
};

template<class VT, Sign sign>
struct PreZInc;

/** Depending on the `sign` template parameter, we obtain three kinds of discrete decreasing chains:
    (1) `Sign::BOTH`: we have \f$ \infty, \ldots, 2, 1, 0, -1, \ldots -\infty \f$.
    (2) `Sign::POS`: we have \f$ \infty, \ldots, 2, 1, 0, \top \f$.
    (3) `Sign::NEG`: we have \f$ 0, -1, \ldots -\infty \f$.

    The bottom and top elements are the first and last elements of the chain.
    The special element \f$ \top \f$ in the chain (2) is necessary to represent \f$ \mathit{false} \f$ in this pre-universe.
    In the chain (3), we assume the underlying universe of discourse is \f$ \mathbb{Z}^- \f$, and thus \f$ 0 \f$ is the interpretation of \f$ \mathit{true} \f$ in that universe.
*/
template<class VT, Sign sign = Sign::BOTH>
struct PreZDec {
  using reverse_type = PreZInc<VT, sign>;
  using value_type = VT;

  /** According to the upset semantics, interpret a constraint of the form \f$ x \leq k \f$ where \f$ \leq \f$ is the natural arithmetic order, and \f$ k \f$ a constant of type `value_type`.
      We have:
      (1) `Sign::BOTH`: \f$ \llbracket x \leq k \rrbracket = k \f$.
      (2) `Sign::POS`: \f$ \llbracket x \leq k \rrbracket = k \f$ if \f$ k \geq 0 \f$ and \f$ \top \f$ otherwise.
      (3) `Sign::NEG`: \f$ \llbracket x \leq k \rrbracket = k \f$ if \f$ k \leq 0 \f$ and \f$ \bot \f$ otherwise (note that \f$ \bot = 0 \f$).

      Example:
        1. \f$ x \leq -1 \f$ is interpreted to \f$ \top \f$ if the chain is positive (`Sign::POS`).
        2. \f$ x \leq 10 \f$ is interpreted to \f$ 0 \f$ if the chain is negative (`Sign::NEG`), this is because what we "really" interpret is \f$ x \leq 10 \land x \leq 0 \f$.
  */
  CUDA static constexpr value_type interpret(value_type k) {
    if constexpr(sign == Sign::POS) {
      return k >= 0 ? k : top();
    }
    else if constexpr(sign == Sign::NEG) {
      return k <= 0 ? k : bot();
    }
    return k;
  }

  /** The dual interpretation using the downset semantics.
      It interprets constraints of the form \f$ x \geq k \f$.
      It differs from `interpret` when the sign is positive or negative and `k` falls out of scope of the domain.
      In that case, we have:
        * `Sign::POS`: \f$ x \geq -1 \f$ interprets to \f$ \bot \f$ (or 0).
        * `Sign::NEG`: \f$ x \geq 1 \f$ interprets to \f$ \top \f$.
  */
  CUDA static constexpr value_type dual_interpret(value_type k) {
    if constexpr(sign == Sign::POS) {
      return k >= 0 ? k : bot();
    }
    else if constexpr(sign == Sign::NEG) {
      return k <= 0 ? k : top();
    }
    return k;
  }

  /** The logical predicate symbol corresponding to the order of this pre-universe.
      We have \f$ a \leq_\mathit{ZDec} b \Leftrightarrow a \geq b \f$.
      \return The logical symbol `GEQ`. */
  CUDA static constexpr Sig sig_order() { return GEQ; }
  CUDA static constexpr Sig dual_sig_order() { return LEQ; }

  /** The logical predicate symbol corresponding to the strict order of this pre-universe.
      We have \f$ a <_\mathit{ZDec} b \Leftrightarrow a > b \f$.
      \return The logical symbol `GT`. */
  CUDA static constexpr Sig sig_strict_order() { return GT; }
  CUDA static constexpr Sig dual_sig_strict_order() { return LT; }

  /** Assert that k is different from the minimum and maximum values of the underlying value type.
      Those values cannot be used as constant values in logical formulas since we attribute those a different meaning: they represent infinities. */
  CUDA static constexpr void check(value_type k) {
    assert(k != battery::Limits<value_type>::top() && k != battery::Limits<value_type>::bot());
  }

  /** For chains (1) and (2), \$f \bot \f$ is represented by the maximum representable value of the underlying value type.
      For chain (3), \f$ \bot = 0 \f$. */
  CUDA static constexpr value_type bot() {
    if constexpr(sign == Sign::NEG) {
      return 0;
    }
    return battery::Limits<value_type>::top();
  }

  /** \$f \top \f$ is represented by the minimum representable value of the underlying value type. */
  CUDA static constexpr value_type top() {
    return battery::Limits<value_type>::bot();
  }

  /** \return \f$ x \sqcup y \f$ defined as \f$ \mathit{min}(x, y) \f$. */
  CUDA static constexpr value_type join(value_type x, value_type y) { return battery::min(x, y); }

  /** \return \f$ x \sqcap y \f$ defined as \f$ \mathit{max}(x, y) \f$. */
  CUDA static constexpr value_type meet(value_type x, value_type y) { return battery::max(x, y); }

  /** \return \f$ \mathit{true} \f$ if \f$ x \leq_\mathit{ZDec} y \f$ where the order \f$ \leq_\mathit{ZDec} \f$ is the natural arithmetic ordering reversed, e.g., \f$ x \geq y \f$, otherwise returns \f$ \mathit{false} \f$. */
  CUDA static constexpr bool order(value_type x, value_type y) { return x >= y; }

  /** \return \f$ \mathit{true} \f$ if \f$ x <_\mathit{ZDec} y \f$ where the order \f$ <_\mathit{ZDec} \f$ is the natural arithmetic ordering reversed, e.g., \f$ x > y \f$, otherwise returns \f$ \mathit{false} \f$. */
  CUDA static constexpr bool strict_order(value_type x, value_type y) { return x > y; }

  /** \return The next value of \f$ x \f$ in a discrete decreasing chain \f$ \infty, \ldots, 2, 1, 0, -1, \ldots -\infty \f$ is \f$ x - 1 \f$ when \f$ x \not\in \{\infty, -\infty\} \f$ and \f$ x \f$ otherwise. */
  CUDA static constexpr value_type next(value_type i) {
    return i - (i != top() && i != bot());
  }

  /** \return The previous value of \f$ x \f$ in a discrete decreasing chain \f$ \infty, \ldots, 2, 1, 0, -1, \ldots -\infty \f$ is \f$ x + 1 \f$ when \f$ x \not\in \{\infty, -\infty\} \f$ and \f$ x \f$ otherwise. */
  CUDA static constexpr value_type prev(value_type i) {
    return i + (i != top() && i != bot());
  }
};

/** Depending on the `sign` template parameter, we obtain three kinds of discrete decreasing chains:
    (1) `Sign::BOTH`: we have \f$ \infty, \ldots, 2, 1, 0, -1, \ldots -\infty \f$.
    (2) `Sign::POS`: we have \f$ \infty, \ldots, 2, 1, 0, \top \f$.
    (3) `Sign::NEG`: we have \f$ 0, -1, \ldots -\infty \f$.

    The bottom and top elements are the first and last elements of the chain.
    The special element \f$ \top \f$ in the chain (2) is necessary to represent \f$ \mathit{false} \f$ in this pre-universe.
    In the chain (3), we assume the underlying universe of discourse is \f$ \mathbb{Z}^- \f$, and thus \f$ 0 \f$ is the interpretation of \f$ \mathit{true} \f$ in that universe.
*/
template<class VT, Sign sign = Sign::BOTH>
struct PreZInc {
  constexpr static bool increasing = false;
  constexpr static bool decreasing = true;
  using reverse_type = PreZInc<VT, sign>;
  using value_type = VT;

  /** Interpret a constraint of the form \f$ x \leq k \f$ where \f$ \leq \f$ is the natural arithmetic order, and \f$ k \f$ a constant of type `value_type`.
      We have:
      (1) `Sign::BOTH`: \f$ \llbracket x \leq k \rrbracket = k \f$.
      (2) `Sign::POS`: \f$ \llbracket x \leq k \rrbracket = k \f$ if \f$ k \geq 0 \f$ and \f$ \top \f$ otherwise.
      (3) `Sign::NEG`: \f$ \llbracket x \leq k \rrbracket = k \f$ if \f$ k \leq 0 \f$ and \f$ \bot \f$ otherwise (note that \f$ \bot = 0 \f$).

      Example:
        1. \f$ x \leq -1 \f$ is interpreted to \f$ \top \f$ if the chain is positive (`Sign::POS`).
        2. \f$ x \leq 10 \f$ is interpreted to \f$ 0 \f$ if the chain is negative (`Sign::NEG`), this is because what we "really" interpret is \f$ x \leq 10 \land x \leq 0 \f$.
  */
  CUDA static constexpr value_type interpret(value_type k) {
    if constexpr(sign == Sign::POS) {
      return k >= 0 ? k : top();
    }
    else if constexpr(sign == Sign::NEG) {
      return k <= 0 ? k : bot();
    }
    return k;
  }

  /** The logical predicate symbol corresponding to the order of this pre-universe.
      We have \f$ a \leq_\mathit{ZDec} b \Leftrightarrow a \geq b \f$.
      \return The logical symbol `GEQ`. */
  CUDA static constexpr Sig sig_order() { return GEQ; }

  /** The logical predicate symbol corresponding to the strict order of this pre-universe.
      We have \f$ a <_\mathit{ZDec} b \Leftrightarrow a > b \f$.
      \return The logical symbol `GT`. */
  CUDA static constexpr Sig sig_strict_order() { return GT; }

  /** Assert that k is different from the minimum and maximum values of the underlying value type.
      Those values cannot be used as constant values in logical formulas since we attribute those a different meaning: they represent infinities. */
  CUDA static constexpr void check(value_type k) {
    assert(strict_order(k, top()));
    if constexpr(sign != Sign::NEG) {
      assert(strict_order(bot(), k));
    }
  }

  /** For chains (1) and (2), \$f \bot \f$ is represented by the maximum representable value of the underlying value type.
      For chain (3), \f$ \bot = 0 \f$. */
  CUDA static constexpr value_type bot() {
    if constexpr(sign == Sign::NEG) {
      return 0;
    }
    return battery::Limits<value_type>::top();
  }

  /** \$f \top \f$ is represented by the minimum representable value of the underlying value type. */
  CUDA static constexpr value_type top() {
    return battery::Limits<value_type>::bot();
  }

  /** \return \f$ x \sqcup y \f$ defined as \f$ \mathit{min}(x, y) \f$. */
  CUDA static constexpr value_type join(value_type x, value_type y) { return battery::min(x, y); }

  /** \return \f$ x \sqcap y \f$ defined as \f$ \mathit{max}(x, y) \f$. */
  CUDA static constexpr value_type meet(value_type x, value_type y) { return battery::max(x, y); }

  /** \return \f$ \mathit{true} \f$ if \f$ x \leq_\mathit{ZDec} y \f$ where the order \f$ \leq_\mathit{ZDec} \f$ is the natural arithmetic ordering reversed, e.g., \f$ x \geq y \f$, otherwise returns \f$ \mathit{false} \f$. */
  CUDA static constexpr bool order(value_type x, value_type y) { return x >= y; }

  /** \return \f$ \mathit{true} \f$ if \f$ x <_\mathit{ZDec} y \f$ where the order \f$ <_\mathit{ZDec} \f$ is the natural arithmetic ordering reversed, e.g., \f$ x > y \f$, otherwise returns \f$ \mathit{false} \f$. */
  CUDA static constexpr bool strict_order(value_type x, value_type y) { return x > y; }

  /** The next value of \f$ x \f$ in a discrete decreasing chain \f$ \infty, \ldots, 2, 1, 0, -1, \ldots -\infty \f$ is \f$ x - 1 \f$ when \f$ x \not\in \{\infty, -\infty\} \f$ and \f$ x \f$ otherwise. */
  CUDA static constexpr value_type next(value_type i) {
    return i - (i != top() && i != bot());
  }
};


template<class VT, Sign sign = Sign::BOTH>
struct PreZInc {
  constexpr static bool increasing = true;
  constexpr static bool decreasing = false;
  using dual_type = PreZDec<VT, sign>;
  using value_type = VT;

  CUDA static value_type interpret_leq(long long int x) {

  }


  CUDA static value_type next(value_type i) {
    if(i == top() || (i == bot() && (sign == Sign::BOTH || sign == Sign::NEG))) {
      return i;
    }
    return i + 1;
  }
  CUDA static value_type bot() {
    if constexpr (sign == Sign::POS) {
      return value_type{};
    }
    else {
      return battery::Limits<value_type>::bot();
    }
  }
  CUDA static value_type top() {
    if constexpr (sign == Sign::NEG) {
      return value_type{};
    }
    else {
      return battery::Limits<value_type>::top();
    }
  }
  CUDA static value_type join(value_type x, value_type y) { return battery::max(x, y); }
  CUDA static value_type meet(value_type x, value_type y) { return battery::min(x, y); }
  CUDA static bool order(value_type x, value_type y) { return x <= y; }
  CUDA static bool strict_order(value_type x, value_type y) { return x < y; }
  CUDA static Sig sig_order() { return GEQ; }
  CUDA static Sig sig_strict_order() { return GT; }
  CUDA static void check(value_type i) {
    if constexpr(sign == Sign::BOTH) {
      assert(strict_order(bot(), i) && strict_order(i, top()));
    }
    else if constexpr(sign == Sign::NEG) {
      assert(strict_order(bot(), i) && order(i, top()));
    }
    else if constexpr(sign == Sign::POS) {
      assert(order(bot(), i) && strict_order(i, top()));
    }
  }
};
