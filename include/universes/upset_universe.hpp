// Copyright 2022 Pierre Talbot

#ifndef UPSET_UNIVERSE_HPP
#define UPSET_UNIVERSE_HPP

#include <type_traits>
#include <utility>
#include <cmath>
#include "thrust/optional.h"
#include "utility.hpp"
#include "ast.hpp"

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

  From a pre-abstract universe, we obtain an abstract universe using the `Universe` class below.
  We also define various aliases to abstract universes such as `ZInc`, `ZDec`, etc.
*/

namespace lala {

template<class PreUniverse, class Mem>
class UpsetUniverse;

/** Lattice of increasing integers.
Concretization function: \f$ \gamma(x) = \{_ \mapsto y \;|\; x \leq y\} \f$. */
template<class VT, class Mem>
using ZInc = UpsetUniverse<PreZInc<VT>, Mem>;

/** Lattice of decreasing integers.
Concretization function: \f$ \gamma(x) = \{_ \mapsto y \;|\; x \geq y\} \f$. */
template<class VT, class Mem>
using ZDec = UpsetUniverse<ChainPreDual<PreZInc<VT>>, Mem>;

/** Lattice of increasing floating-point numbers.
Concretization function: \f$ \gamma(x) = \{_ \mapsto y \;|\; y \in \mathbb{R}, x \leq y\} \f$. */
template<class VT, class Mem>
using FInc = UpsetUniverse<PreFInc<VT>, Mem>;

/** Lattice of decreasing floating-point numbers.
Concretization function: \f$ \gamma(x) = \{_ \mapsto y \;|\; y \in \mathbb{R}, x \geq y\} \f$. */
template<class VT, class Mem>
using FDec = UpsetUniverse<ChainPreDual<PreFInc<VT>>, Mem>;

/** Lattice of increasing Boolean where \f$ \mathit{false} \leq \mathit{true} \f$. */
template<class Mem>
using BInc = UpsetUniverse<PreBInc, Mem>;

/** Lattice of decreasing Boolean where \f$ \mathit{true} \leq \mathit{false} \f$. */
template<class Mem>
using BDec = UpsetUniverse<ChainPreDual<PreBInc>, Mem>;

/** Aliases for lattice allocated on the stack (as local variable) and accessed by only one thread.
 * To make things simpler, the underlying type is also chosen (when required). */
namespace local {
  using ZInc = ::lala::ZInc<int, battery::LocalMemory>;
  using ZDec = ::lala::ZDec<int, battery::LocalMemory>;
  using FInc = ::lala::FInc<double, battery::LocalMemory>;
  using FDec = ::lala::FDec<double, battery::LocalMemory>;
  using BInc = ::lala::BInc<battery::LocalMemory>;
  using BDec = ::lala::BDec<battery::LocalMemory>;
}

template<class PreUniverse, class Mem>
class UpsetUniverse
{
  using U = PreUniverse;
public:
  using pre_universe = PreUniverse;
  using value_type = typename pre_universe::value_type;
  using memory_type = Mem;
  using this_type = UpsetUniverse<pre_universe, memory_type>;
  using reverse_type = UpsetUniverse<typename pre_universe::reverse_type, memory_type>;

  template<class M>
  using this_type2 = UpsetUniverse<pre_universe, M>;

  template<class F>
  using iresult = IResult<this_type, F>;

  constexpr static bool is_totally_ordered = pre_universe::is_totally_ordered;
  constexpr static bool preserve_bot = pre_universe::preserve_bot;
  constexpr static bool preserve_top = pre_universe::preserve_top;
  constexpr static bool injective_concretization = pre_universe::injective_concretization;
  constexpr static bool preserve_inner_covers = pre_universe::preserve_inner_covers;
  constexpr static bool complemented = pre_universe::complemented;
  constexpr static const char* name = pre_universe::name;

private:
  using atomic_type = typename memory_type::atomic_type<value_type>;
  atomic_type val;

public:
  /** Similar to \f$[\![\mathit{true}]\!]\f$ if `preserve_bot` is true. */
  CUDA static this_type bot() {
    return this_type(U::bot());
  }

  /** Similar to \f$[\![\mathit{false}]\!]\f$ if `preserve_top` is true. */
  CUDA static this_type top() {
    return this_type(U::top());
  }

  CUDA dual_type dual() const {
    return dual_type(value());
  }

  /** Similar to \f$[\![x \geq_A i]\!]\f$ for any name `x` where \f$ \geq_A \f$ is the lattice order. */
  CUDA UpsetUniverse(value_type x): val(x) {}
  CUDA UpsetUniverse(const this_type& other): UpsetUniverse(other.value()) {}

  template <class M>
  CUDA UpsetUniverse(const this_type2<M>& other): UpsetUniverse(other.value()) {}

  CUDA value_type value() const { return memory_type::load(val); }

  CUDA value_type operator value_type() const { return value(); }

  /** `true` whenever \f$ a = \top \f$, `false` otherwise. */
  CUDA local::BInc is_top() const {
    return value() == U::top();
  }

  /** `true` whenever \f$ a = \bot \f$, `false` otherwise. */
  CUDA local::BDec is_bot() const {
    return value() == U::bot();
  }

  CUDA this_type& tell_top() {
    memory_type::store(val, U::top());
    return *this;
  }

  template<class M1, class M2>
  CUDA this_type& tell(const this_type2<M1>& other, BInc<M2>& has_changed) {
    U r1 = value();
    U r2 = is_totally_ordered ? other.value() : U::join(r1, other.value());
    if(U::strict_order(r1, r2)) {
      memory_type::store(val, r2);
      has_changed.tell_top();
    }
    return *this;
  }

  template<class M1>
  CUDA this_type& tell(const this_type2<M1>& other) {
    U r1 = value();
    U r2 = is_totally_ordered ? other.value() : U::join(r1, other.value());
    if(U::strict_order(r1, r2)) {
      memory_type::store(val, r2);
    }
    return *this;
  }

  CUDA this_type& dtell_bot() {
    memory_type::store(val, U::bot());
    return *this;
  }

  template<class M1, class M2>
  CUDA this_type& dtell(const this_type2<M1>& other, BInc<M2>& has_changed) {
    U r1 = value();
    U r2 = is_totally_ordered ? other.value() : U::meet(r1, other.value());
    if(U::strict_order(r2, r1)) {
      memory_type::store(val, r2);
      has_changed.tell_top();
    }
    return *this;
  }

  template<class M1>
  CUDA this_type& dtell(const this_type2<M1>& other) {
    U r1 = value();
    U r2 = is_totally_ordered ? other.value() : U::meet(r1, other.value());
    if(U::strict_order(r2, r1)) {
      memory_type::store(val, r2);
    }
    return *this;
  }

  /** \return \f$ x \geq i \f$ where `x` is a variable's name and `i` the current value.
  If `U` preserves bottom `true` is returned whenever \f$ a = \bot \f$, if it preserves top `false` is returned whenever \f$ a = \top \f$.
  We always return an exact approximation, hence for any formula \f$ \llbracket \varphi \rrbracket = a \f$, we must have \f$ a =  \llbracket \rrbracket a \llbracket \rrbracket \f$ where \f$ \rrbracket a \llbracket \f$ is the deinterpretation function. */
  template<class Allocator>
  CUDA TFormula<Allocator> deinterpret(const LVar<Allocator>& x, const Allocator& allocator = Allocator()) const {
    if(preserve_top && is_top()) {
      return TFormula<Allocator>::make_false();
    }
    else if(preserve_bot && is_bot()) {
      return TFormula<Allocator>::make_true();
    }
    return make_v_op_z(x, U::sig_order(), value(), UNTYPED, EXACT, allocator);
  }

  /** Under-approximates the current element \f$ a \f$ w.r.t. \f$ \rrbracket a \llbracket \f$ into `ua`.
   * For this abstract universe, it always returns `true` since the current element \f$ a \f$ is an exact representation of \f$ \rrbracket a \llbracket \f$. */
  CUDA bool extract(this_type2<battery::LocalMemory>& ua) const {
    ua.val = value();
    return true;
  }

  /** Print the current element. */
  CUDA void print() const {
    if(is_bot()) {
      printf("%c", 0x22A5);
    }
    else if(is_top()) {
      printf("%c", 0x22A4);
    }
    else {
      ::battery::print(value());
    }
  }

private:

  /** Interpret a formula of the form `x <sig> k`. */
  template<class F>
  CUDA static iresult<F> interpret_x_op_k(const F& f, const F& x, Sig sig, const F& k) {
    auto r = pre_universe::interpret(k, f.approx());
    if(!r.is_ok()) {
      return r;
    }
    else if(sig == U::sig_order()) {  // e.g., x <= 4 or x >= 4.24
      return r;
    }
    else if(sig == U::sig_strict_order()) {  // e.g., x < 4
      if(f.is_under() ||
         (preserve_inner_covers && pre_universe::has_unique_next(r.value())))
      {
        return r.map(pre_universe::next(r.value()));
      }
      else if(f.is_exact()) {
        auto r = IError<F>(true, name, "Exactly interpreting a strict relation, i.e. `x < k`, not supported.", f);
        if constexpr(!preserve_inner_covers) {
          r.add_suberror(true, name, "Inner covers are not preserved: there might be elements between k and next(k).", F::make_false());
        }
        if(!pre_universe::has_unique_next(r.value())) {
          r.add_suberror(true, name, "The cover is not unique: there are several incomparable elements satisfying next(k).", F::make_false());
        }
        return iresult<F>(r);
      }
      // In case of over-approximation, interpreting using `U::sig_order` is a correct option.
      return r;
    }
    // Under-approximation of `x != k` as `next(k)`.
    else if(f.is_under() && sig == NEQ) {
      return r.map(pre_universe::next(r.value()));
    }
    // Over-approximation of `x == k` as `k`.
    else if(f.is_over() && sig == EQ) {
      return r;
    }
    else {
      return iresult<F>(IError<F>(true, name, "The signature of the symbol `" + string_of_sig(sig) + "` is not supported: either the symbol is unknown, approximation kind is not supported or the type of the arguments of the symbols are not supported.", f));
    }
  }

public:

  /** Expects a predicate of the form `x <op> i` where `x` is any variable's name, and `i` an integer.
    - If `f.approx()` is EXACT: `op` can be `U::sig_order()` or `U::sig_strict_order()`.
    - If `f.approx()` is UNDER: `op` can be, in addition to exact, `!=`.
    - If `f.approx()` is OVER: `op` can be, in addition to exact, `==`.
    Existential formula \f$ \exists{x:T} \f$ can also be interpreted (only to bottom).
    - The interpretation depends on the abstract pre-universe.
    */
  template<class Formula>
  CUDA static iresult<this_type> interpret(const Formula& f) {
    if(f.is_true()) {
      if(preserve_bot || f.is_under()) {
        return bot();
      }
      else {
        return iresult<F>(IError<F>(true, name, "Bottom is not preserved, hence it cannot exactly interpret or over-approximate the formula `true`.", f));
      }
    }
    else if(f.is_false()) {
      if(preserve_top || f.is_over()) {
        return top();
      }
      else {
        return iresult<F>(IError<F>(true, name, "Top is not preserved, hence it cannot exactly interpret or under-approximate the formula `false`.", f));
      }
    }
    else if(f.is(Formula::E)) {
      return pre_universe::interpret_type(f);
    }
    else {
      if(f.is_binary()) {
        int idx_constant = f.seq(0).is_constant() ? 0 : (f.seq(1).is_constant() ? 1 : 100);
        int idx_variable = f.seq(0).is_variable() ? 0 : (f.seq(1).is_variable() ? 1 : 100);
        if(idx_constant + idx_variable != 1) {
          return iresult<F>(IError<F>(true, name, "Only binary formulas of the form `t1 <sig> t2` where if t1 is a constant and t2 is a variable (or conversely) are supported.", f));
        }
        const auto& k = f.seq(idx_constant);
        const auto& x = f.seq(idx_variable);
        Sig sig = idx_constant == 0 ? converse_comparison(f.sig()) : f.sig();
        return interpret_x_op_k(f, x, sig, k);
      }
      else {
        return iresult<F>(IError<F>(true, name, "Only binary constraints are supported.", f));
      }
    }
  }

  CUDA static constexpr bool is_supported_fun(Approx appx, Sig sig) {
    return pre_universe::is_supported_fun(appx, sig);
  }

private:
  template <class T, class = int>
  struct has_preserve_bot {
    static constexpr bool value = false;
  };

  template <class T>
  struct has_preserve_bot<T, decltype(T::preserve_bot, int)> {
    static constexpr bool value = T::preserve_bot;
  };

  template <class T>
  inline constexpr bool preserve_bot_v = has_preserve_bot<T>::value;

  template <class T, class = int>
  struct has_preserve_top {
    static constexpr bool value = false;
  };

  template <class T>
  struct has_preserve_top<T, decltype(T::preserve_top, int)> {
    static constexpr bool value = T::preserve_top;
  };

  template <class T>
  inline constexpr bool preserve_top_v = has_preserve_top<T>::value;

public:
  template<Approx appx, Sig sig, class A>
  CUDA static constexpr this_type2<battery::LocalMemory> fun(A a) {
    if constexpr(preserve_top_v<A>) {
      if(a == A::top())
        return this_type2<battery::LocalMemory>::top();
    }
    if constexpr(preserve_bot_v<A>) {
      if(a == A::bot())
        return this_type2<battery::LocalMemory>::bot();
    }
    return pre_universe::template fun<appx, sig>(a);
  }

  template<Approx appx, Sig sig, class A, class B>
  CUDA static constexpr this_type2<battery::LocalMemory> fun(A a, B b) {
    if constexpr(preserve_top_v<A>) {
      if(a.is_top())
        return this_type2<battery::LocalMemory>::top();
    }
    if constexpr(preserve_top_v<B>) {
      if(b.is_top())
        return this_type2<battery::LocalMemory>::top();
    }
    if constexpr(preserve_bot_v<A>) {
      if(a.is_bot())
        return this_type2<battery::LocalMemory>::bot();
    }
    if constexpr(preserve_bot_v<B>) {
      if(b.is_bot())
        return this_type2<battery::LocalMemory>::bot();
    }
    return pre_universe::template fun<appx, sig>(arg);
  }

  // Lattice operators

  template<class Pre2, class Mem2>
  friend class UpsetUniverse;

  template<class M1, class M2>
  CUDA friend this_type2<battery::LocalMemory> join(const this_type2<M1>& a, const this_type2<M2>& b) {
    return Pre::join(a, b);
  }

  template<class M1, class M2>
  CUDA friend this_type2<battery::LocalMemory> meet(const this_type2<M1>& a, const this_type2<M2>& b) {
    return Pre::meet(a, b);
  }

  template<class M1, class M2>
  CUDA friend bool operator<=(const this_type2<M1>& a, const this_type2<M2>& b) {
    return Pre::order(a, b);
  }

  template<class M1, class M2>
  CUDA friend bool operator<(const this_type2<M1>& a, const this_type2<M1>& b) {
    return Pre::strict_order(a, b);
  }

  template<class M1, class M2>
  CUDA friend bool operator>=(const this_type2<M1>& a, const this_type2<M2>& b) {
    return Pre::order(b, a);
  }

  template<class M1, class M2>
  CUDA friend bool operator>(const this_type2<M1>& a, const this_type2<M1>& b) {
    return Pre::strict_order(b, a);
  }

  template<class M1, class M2>
  CUDA friend bool operator==(const this_type2<M1>& a, const this_type2<M2>& b) {
    return a == b;
  }

  template<class M1, class M2>
  CUDA friend bool operator!=(const this_type2<M1>& a, const this_type2<M1>& b) {
    return a != b;
  }
};


} // namespace lala

#endif
