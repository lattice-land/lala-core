// Copyright 2022 Pierre Talbot

#ifndef LALA_CORE_PRIMITIVE_UPSET_HPP
#define LALA_CORE_PRIMITIVE_UPSET_HPP

#include <type_traits>
#include <utility>
#include <cmath>
#include <iostream>
#include "../logic/logic.hpp"
#include "pre_binc.hpp"
#include "pre_finc.hpp"
#include "pre_zinc.hpp"
#include "pre_bdec.hpp"
#include "pre_fdec.hpp"
#include "pre_zdec.hpp"
#include "../b.hpp"
#include "battery/memory.hpp"

/** A pre-abstract universe is a lattice (with usual operations join, order, ...) equipped with a simple logical interpretation function and a next/prev functions.
    We consider totally ordered pre-abstract universes with an upset semantics.
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
class FlatUniverse;

template<class PreUniverse, class Mem>
class PrimitiveUpset;

/** Lattice of increasing integers.
Concretization function: \f$ \gamma(x) = \{_ \mapsto y \;|\; x \leq y\} \f$. */
template<class VT, class Mem>
using ZInc = PrimitiveUpset<PreZInc<VT>, Mem>;

/** Lattice of decreasing integers.
Concretization function: \f$ \gamma(x) = \{_ \mapsto y \;|\; x \geq y\} \f$. */
template<class VT, class Mem>
using ZDec = PrimitiveUpset<PreZDec<VT>, Mem>;

/** Lattice of increasing floating-point numbers.
Concretization function: \f$ \gamma(x) = \{_ \mapsto y \;|\; y \in \mathbb{R}, x \leq y\} \f$. */
template<class VT, class Mem>
using FInc = PrimitiveUpset<PreFInc<VT>, Mem>;

/** Lattice of decreasing floating-point numbers.
Concretization function: \f$ \gamma(x) = \{_ \mapsto y \;|\; y \in \mathbb{R}, x \geq y\} \f$. */
template<class VT, class Mem>
using FDec = PrimitiveUpset<PreFDec<VT>, Mem>;

/** Lattice of increasing Boolean where \f$ \mathit{false} \leq \mathit{true} \f$. */
template<class Mem>
using BInc = PrimitiveUpset<PreBInc, Mem>;

/** Lattice of decreasing Boolean where \f$ \mathit{true} \leq \mathit{false} \f$. */
template<class Mem>
using BDec = PrimitiveUpset<PreBDec, Mem>;

/** Aliases for lattice allocated on the stack (as local variable) and accessed by only one thread.
 * To make things simpler, the underlying type is also chosen (when required). */
namespace local {
  using ZInc = ::lala::ZInc<int, battery::local_memory>;
  using ZDec = ::lala::ZDec<int, battery::local_memory>;
  using FInc = ::lala::FInc<double, battery::local_memory>;
  using FDec = ::lala::FDec<double, battery::local_memory>;
  using BInc = ::lala::BInc<battery::local_memory>;
  using BDec = ::lala::BDec<battery::local_memory>;
}

namespace impl {
  template<class T>
  struct is_primitive_upset {
    static constexpr bool value = false;
  };

  template<class PreUniverse, class Mem>
  struct is_primitive_upset<PrimitiveUpset<PreUniverse, Mem>> {
    static constexpr bool value = true;
  };

  template <class T>
  inline constexpr bool is_primitive_upset_v = is_primitive_upset<T>::value;
}

/** This function is useful when we need to convert a value to its dual.
    The dual is the downset of the current element, therefore, if we have \f$ x >= 10 \f$, the dual is given by the formula \f$ x <= 10 \f$ interpreted in the dual lattice.
    In that case, it just changes the type of the lattice without changing the value.
    A difference occurs on the bottom and top element.
    Indeed, by our representation of bot and top, the bottom value in a lattice L equals the top value in its dual, but we need them to remain the same, so the dual of `L::bot()` is `LDual::bot()`.*/
template <class LDual, class L>
CUDA constexpr LDual dual(const L& x) {
  if(x.is_bot()) return LDual::bot();
  if(x.is_top()) return LDual::top();
  return LDual(x.value());
}

template<class PreUniverse, class Mem>
class PrimitiveUpset
{
  using U = PreUniverse;
public:
  using pre_universe = PreUniverse;
  using value_type = typename pre_universe::value_type;
  using memory_type = Mem;
  using this_type = PrimitiveUpset<pre_universe, memory_type>;
  using dual_type = PrimitiveUpset<typename pre_universe::dual_type, memory_type>;

  template<class M>
  using this_type2 = PrimitiveUpset<pre_universe, M>;

  using local_type = this_type2<battery::local_memory>;

  template<class M>
  using flat_type = FlatUniverse<typename pre_universe::increasing_type, M>;
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
  constexpr static const bool complemented = pre_universe::complemented;
  constexpr static const bool increasing = pre_universe::increasing;
  constexpr static const char* name = pre_universe::name;

  constexpr static const bool is_arithmetic = pre_universe::is_arithmetic;

  static_assert(is_totally_ordered, "The underlying pre-universe must be totally ordered.");

  /** A pre-interpreted formula `x >= value` ready to use.
   * This is mainly for optimization purpose to avoid calling `interpret` each time we need it. */
  CUDA static constexpr this_type geq_k(value_type k) {
    if constexpr(increasing && is_arithmetic) {
      return this_type(k);
    }
    else {
      static_assert(increasing && is_arithmetic,
        "The pre-interpreted formula x >= k is only available over arithmetic universe such as integers, floating-point numbers and Boolean.\
        Moreover, the underlying abstract universe must be increasing, otherwise this formula is not supported.");
    }
  }

  CUDA static constexpr this_type leq_k(value_type k) {
    if constexpr(!increasing && is_arithmetic) {
      return this_type(k);
    }
    else {
      static_assert(!increasing && is_arithmetic,
        "The pre-interpreted formula x <= k is only available over arithmetic universe such as integers, floating-point numbers and Boolean.\
        Moreover, the underlying abstract universe must be decreasing, otherwise this formula is not supported.");
    }
  }

private:
  using atomic_type = memory_type::template atomic_type<value_type>;
  atomic_type val;

public:
  /** Similar to \f$[\![\mathit{true}]\!]\f$ if `preserve_bot` is true. */
  CUDA static constexpr local_type bot() { return local_type(); }

  /** Similar to \f$[\![\mathit{false}]\!]\f$ if `preserve_top` is true. */
  CUDA static constexpr local_type top() { return local_type(U::top()); }
  /** Initialize an upset universe to bottom. */
  CUDA constexpr PrimitiveUpset(): val(U::bot()) {}
  /** Similar to \f$[\![x \geq_A i]\!]\f$ for any name `x` where \f$ \geq_A \f$ is the lattice order. */
  CUDA constexpr PrimitiveUpset(value_type x): val(x) {}
  CUDA constexpr PrimitiveUpset(const this_type& other): PrimitiveUpset(other.value()) {}
  constexpr PrimitiveUpset(this_type&& other) = default;

  template <class M>
  CUDA constexpr PrimitiveUpset(const this_type2<M>& other): PrimitiveUpset(other.value()) {}

  /** The assignment operator can only be used in a sequential context.
   * It is monotone but not extensive. */
  template <class M>
  CUDA constexpr this_type& operator=(const this_type2<M>& other) {
   if constexpr(sequential) {
      memory_type::store(val, other.value());
      return *this;
    }
    else {
      static_assert(sequential, "The operator= in `PrimitiveUpset` can only be used when the underlying memory is `sequential`.");
    }
  }

  CUDA constexpr this_type& operator=(const this_type& other) {
    if constexpr(sequential) {
      memory_type::store(val, other.value());
      return *this;
    }
    else {
      static_assert(sequential, "The operator= in `PrimitiveUpset` can only be used when the underlying memory is `sequential`.");
    }
  }

  CUDA constexpr value_type value() const { return memory_type::load(val); }

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

  /** \return \f$ x \geq i \f$ where `x` is a variable's name and `i` the current value.
  If `U` preserves bottom `true` is returned whenever \f$ a = \bot \f$, if it preserves top `false` is returned whenever \f$ a = \top \f$.
  We always return an exact approximation, hence for any formula \f$ \llbracket \varphi \rrbracket = a \f$, we must have \f$ a =  \llbracket \rrbracket a \llbracket \rrbracket \f$ where \f$ \rrbracket a \llbracket \f$ is the deinterpretation function.
  */
  template<class Env>
  CUDA NI TFormula<typename Env::allocator_type> deinterpret(AVar avar, const Env& env) const {
    using F = TFormula<typename Env::allocator_type>;
    if(preserve_top && is_top()) {
      return F::make_false();
    }
    else if(preserve_bot && is_bot()) {
      return F::make_true();
    }
    return F::make_binary(
      deinterpret<F>(),
      U::sig_order(),
      F::make_avar(avar),
      UNTYPED, env.get_allocator());
  }

  /** Deinterpret the current value to a logical constant. */
  template<class F>
  CUDA NI F deinterpret() const {
    return pre_universe::template deinterpret<F>(value());
  }

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
  /** Interpret a formula of the form `k <sig> x`. */
  template<bool diagnose = false, class F, class M2>
  CUDA NI static bool interpret_tell_k_op_x(const F& f, const F& k, Sig sig, this_type2<M2>& tell, IDiagnostics& diagnostics) {
    value_type value = pre_universe::bot();
    bool res = pre_universe::template interpret_tell<diagnose>(k, value, diagnostics);
    if(res) {
      if(sig == EQ || sig == U::sig_order()) {  // e.g., x <= 4 or x >= 4.24
        tell.join(local_type(value));
      }
      else if(sig == U::sig_strict_order()) {  // e.g., x < 4 or x > 4.24
        if constexpr(preserve_concrete_covers) {
          tell.join(local_type(pre_universe::next(value)));
        }
        else {
          tell.join(local_type(value));
        }
      }
      else {
        RETURN_INTERPRETATION_ERROR("The symbol `" + LVar<typename F::allocator_type>(string_of_sig(sig)) + "` is not supported in the tell language of this universe.");
      }
    }
    return res;
  }

  /** Interpret a formula of the form `k <sig> x`. */
  template<bool diagnose = false, class F, class M2>
  CUDA NI static bool interpret_ask_k_op_x(const F& f, const F& k, Sig sig, this_type2<M2>& tell, IDiagnostics& diagnostics) {
    value_type value = pre_universe::bot();
    bool res = pre_universe::template interpret_ask<diagnose>(k, value, diagnostics);
    if(res) {
      if(sig == U::sig_order()) {
        tell.join(local_type(value));
      }
      else if(sig == NEQ || sig == U::sig_strict_order()) {
        // We could actually do a little bit better in the case of FInc/FDec.
        // If the real number `k` is approximated by `[f, g]`, it actually means `]f, g[` so we could safely choose `r` since it already under-approximates `k`.
        tell.join(local_type(pre_universe::next(value)));
      }
      else {
        RETURN_INTERPRETATION_ERROR("The symbol `" + LVar<typename F::allocator_type>(string_of_sig(sig)) + "` is not supported in the ask language of this universe.");
      }
    }
    return res;
  }

  template<bool diagnose = false, class F, class M2>
  CUDA NI static bool interpret_tell_set(const F& f, const F& k, this_type2<M2>& tell, IDiagnostics& diagnostics) {
    const auto& set = k.s();
    if(set.size() == 0) {
      tell.join_top();
      return true;
    }
    value_type meet_s = pre_universe::top();
    constexpr int bound_index = increasing ? 0 : 1;
    // We interpret each component of the set and take the meet of all the results.
    for(int i = 0; i < set.size(); ++i) {
      auto bound = battery::get<bound_index>(set[i]);
      value_type set_element = pre_universe::bot();
      bool res = pre_universe::template interpret_tell<diagnose>(bound, set_element, diagnostics);
      if(!res) {
        return false;
      }
      meet_s = pre_universe::meet(meet_s, set_element);
    }
    tell.join(local_type(meet_s));
    return true;
  }

public:
  /** Expects a predicate of the form `x <op> k`, `k <op> x` or `x in k`, where `x` is any variable's name, and `k` a constant.
   * The symbol `<op>` is expected to be `U::sig_order()`, `U::sig_strict_order()` and `=`.
   * Existential formula \f$ \exists{x:T} \f$ can also be interpreted (only to bottom) depending on the underlying pre-universe.
   */
  template<bool diagnose = false, class F, class Env, class M2>
  CUDA NI static bool interpret_tell(const F& f, const Env&, this_type2<M2>& tell, IDiagnostics& diagnostics) {
    if(f.is(F::E)) {
      typename U::value_type val;
      bool res = pre_universe::template interpret_type<diagnose>(f, val, diagnostics);
      if(res) {
        tell.join(local_type(val));
      }
      return res;
    }
    else {
      if(f.is_binary()) {
        int idx_constant = f.seq(0).is_constant() ? 0 : (f.seq(1).is_constant() ? 1 : 100);
        int idx_variable = f.seq(0).is_variable() ? 0 : (f.seq(1).is_variable() ? 1 : 100);
        if(idx_constant + idx_variable != 1) {
          RETURN_INTERPRETATION_ERROR("Only binary formulas of the form `t1 <sig> t2` where if t1 is a constant and t2 is a variable (or conversely) are supported.")
        }
        const auto& k = f.seq(idx_constant);
        if(f.sig() == IN) {
          if(idx_constant == 0) { // `k in x` is equivalent to `{k} \subseteq x`.
            RETURN_INTERPRETATION_ERROR("The formula `k in x` is not supported in this abstract universe (`x in k` is supported).")
          }
          else { // `x in k` is equivalent to `x >= meet k` where `>=` is the lattice order `U::sig_order()`.
            return interpret_tell_set<diagnose>(f, k, tell, diagnostics);
          }
        }
        else if(is_comparison(f)) {
          Sig sig = idx_constant == 1 ? converse_comparison(f.sig()) : f.sig();
          return interpret_tell_k_op_x<diagnose>(f, k, sig, tell, diagnostics);
        }
        else {
          RETURN_INTERPRETATION_ERROR("This symbol is not supported.")
        }
      }
      else {
        RETURN_INTERPRETATION_ERROR("Only binary constraints are supported.")
      }
    }
  }

  /** Expects a predicate of the form `x <op> k` or `k <op> x`, where `x` is any variable's name, and `k` a constant.
   * The symbol `<op>` is expected to be `U::sig_order()`, `U::sig_strict_order()` or `!=`.
   */
  template<bool diagnose = false, class F, class Env, class M2>
  CUDA NI static bool interpret_ask(const F& f, const Env&, this_type2<M2>& ask, IDiagnostics& diagnostics) {
    if(f.is_binary()) {
      int idx_constant = f.seq(0).is_constant() ? 0 : (f.seq(1).is_constant() ? 1 : 100);
      int idx_variable = f.seq(0).is_variable() ? 0 : (f.seq(1).is_variable() ? 1 : 100);
      if(idx_constant + idx_variable != 1) {
        RETURN_INTERPRETATION_ERROR("Only binary formulas of the form `t1 <sig> t2` where if t1 is a constant and t2 is a variable (or conversely) are supported.");
      }
      const auto& k = f.seq(idx_constant);
      if(is_comparison(f)) {
        Sig sig = idx_constant == 1 ? converse_comparison(f.sig()) : f.sig();
        return interpret_ask_k_op_x<diagnose>(f, k, sig, ask, diagnostics);
      }
      else {
        RETURN_INTERPRETATION_ERROR("This symbol is not supported.");
      }
    }
    else {
      RETURN_INTERPRETATION_ERROR("Only binary constraints are supported.");
    }
  }

  template<IKind kind, bool diagnose = false, class F, class Env, class M2>
  CUDA NI static bool interpret(const F& f, const Env& env, this_type2<M2>& value, IDiagnostics& diagnostics) {
    if constexpr(kind == IKind::TELL) {
      return interpret_tell<diagnose>(f, env, value, diagnostics);
    }
    else {
      return interpret_ask<diagnose>(f, env, value, diagnostics);
    }
  }

public:
  CUDA static constexpr local_type next(const this_type2<Mem>& a) {
    return local_type(pre_universe::next(a.value()));
  }

  CUDA static constexpr local_type prev(const this_type2<Mem>& a) {
    return local_type(pre_universe::prev(a.value()));
  }

  /** Unary function of type `fun: FlatUniverse -> PrimitiveUpset`.
   * If `a` is `top`, we join top in-place.
   * Otherwise, we apply the function `fun` to `a` and join the result.
   * \remark The result of the function is always over-approximated (or exact when possible).
  */
  CUDA constexpr void project(Sig fun, const local_flat_type& a) {
    if(a.is_top()) { join_top(); }
    else if(!a.is_bot()) {
      join(local_type(pre_universe::project(fun, a)));
    }
  }

  /** Binary functions of type `project: FlatUniverse x FlatUniverse -> PrimitiveUpset`.
   * If `a` or `b` is `top`, we join top in-place.
   * Otherwise, we join `fun(a,b)` in-place.
   * \remark The result of the function is always over-approximated (or exact when possible).
   * \remark If the function `fun` is partial (e.g. division), we expect the arguments `a` and `b` to be in the domain of `fun` (e.g. not equal to 0).
   */
  CUDA constexpr void project(Sig fun, const local_flat_type& a, const local_flat_type& b) {
    if(a.is_top() || b.is_top()) { join_top(); }
    else if(!a.is_bot() && !b.is_bot()) {
      join(local_type(pre_universe::project(fun, a.value(), b.value())));
    }
  }

  /** In this universe, the non-trivial order-preserving functions are {min, max, +}. */
  CUDA static constexpr bool is_trivial_fun(Sig fun) {
    return fun != MIN && fun != MAX && fun != ADD;
  }

  /** The functions that are order-preserving on the natural order of the universe of discourse, and its dual. */
  CUDA static constexpr bool is_order_preserving_fun(Sig fun) {
    return fun == ADD || fun == MIN || fun == MAX || (increasing && fun == ABS);
  }

  CUDA constexpr void project(Sig fun, const local_type &a, const local_type &b) {
    if(fun == ADD) {
      project(ADD, local_flat_type(a), local_flat_type(b));
    }
    else {
      if (a.is_top() || b.is_top()) { join_top(); return; }
      switch(fun) {
        case MIN: { increasing ? join(fmeet(a, b)) : join(fjoin(a, b)); return; }
        case MAX: { increasing ? join(fjoin(a, b)) : join(fmeet(a, b)); return; }
      }
    }
  }

  CUDA constexpr void project(Sig fun, const local_type &a) {
    if (a.is_top()) { join_top(); return; }
    if(fun == ABS && increasing) {
      join(fjoin(a, local_type(pre_universe::zero())));
    }
  }

  template<class Pre2, class Mem2>
  friend class PrimitiveUpset;
};

// Lattice operators

template<class Pre, class M1, class M2>
CUDA constexpr PrimitiveUpset<Pre, battery::local_memory> fjoin(const PrimitiveUpset<Pre, M1>& a, const PrimitiveUpset<Pre, M2>& b) {
  return Pre::join(a, b);
}

template<class Pre, class M1, class M2>
CUDA constexpr PrimitiveUpset<Pre, battery::local_memory> fmeet(const PrimitiveUpset<Pre, M1>& a, const PrimitiveUpset<Pre, M2>& b) {
  return Pre::meet(a, b);
}

template<class Pre, class M1, class M2>
CUDA constexpr bool operator<=(const PrimitiveUpset<Pre, M1>& a, const PrimitiveUpset<Pre, M2>& b) {
  return Pre::order(a, b);
}

template<class Pre, class M1, class M2>
CUDA constexpr bool operator<(const PrimitiveUpset<Pre, M1>& a, const PrimitiveUpset<Pre, M2>& b) {
  return Pre::strict_order(a, b);
}

template<class Pre, class M1, class M2>
CUDA constexpr bool operator>=(const PrimitiveUpset<Pre, M1>& a, const PrimitiveUpset<Pre, M2>& b) {
  return Pre::order(b, a);
}

template<class Pre, class M1, class M2>
CUDA constexpr bool operator>(const PrimitiveUpset<Pre, M1>& a, const PrimitiveUpset<Pre, M2>& b) {
  return Pre::strict_order(b, a);
}

template<class Pre, class M1, class M2>
CUDA constexpr bool operator==(const PrimitiveUpset<Pre, M1>& a, const PrimitiveUpset<Pre, M2>& b) {
  return a.value() == b.value();
}

template<class Pre, class M1, class M2>
CUDA constexpr bool operator!=(const PrimitiveUpset<Pre, M1>& a, const PrimitiveUpset<Pre, M2>& b) {
  return a.value() != b.value();
}

template<class Pre, class M>
std::ostream& operator<<(std::ostream &s, const PrimitiveUpset<Pre, M> &upset) {
  if(upset.is_bot()) {
    s << "\u22A5";
  }
  else if(upset.is_top()) {
    s << "\u22A4";
  }
  else {
    s << upset.value();
  }
  return s;
}

} // namespace lala

#endif
