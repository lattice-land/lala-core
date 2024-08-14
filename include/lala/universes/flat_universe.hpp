// Copyright 2023 Pierre Talbot

#ifndef LALA_CORE_FLAT_UNIVERSE_HPP
#define LALA_CORE_FLAT_UNIVERSE_HPP

#include "primitive_upset.hpp"

namespace lala {

template<class PreUniverse, class Mem>
class FlatUniverse;

/** Lattice of flat integers. */
template<class VT, class Mem>
using ZFlat = FlatUniverse<PreZInc<VT>, Mem>;

/** Lattice of flat floating-point numbers. */
template<class VT, class Mem>
using FFlat = FlatUniverse<PreFInc<VT>, Mem>;

/** Aliases for lattice allocated on the stack (as local variable) and accessed by only one thread.
 * To make things simpler, the underlying type is also chosen (when required). */
namespace local {
  using ZFlat = ::lala::ZFlat<int, battery::local_memory>;
  using FFlat = ::lala::FFlat<double, battery::local_memory>;
}

template<class PreUniverse, class Mem>
class FlatUniverse
{
  using U = PreUniverse;
public:
  using pre_universe = PreUniverse;
  using value_type = typename pre_universe::value_type;
  using memory_type = Mem;
  using this_type = FlatUniverse<pre_universe, memory_type>;
  template <class M> using this_type2 = FlatUniverse<pre_universe, M>;
  using local_type = this_type2<battery::local_memory>;

  static_assert(pre_universe::increasing);
  static_assert(pre_universe::preserve_bot && pre_universe::preserve_top,
    "The Flat lattice construction reuse the bottom and top elements of the pre-universe.\
    Therefore, it must preserve bottom and top.");

  constexpr static const bool is_abstract_universe = true;
  constexpr static const bool sequential = Mem::sequential;
  constexpr static const bool is_totally_ordered = false;
  constexpr static const bool preserve_bot = true;
  constexpr static const bool preserve_top = true;
  constexpr static const bool preserve_join = true;
  constexpr static const bool preserve_meet = false;
  constexpr static const bool injective_concretization = pre_universe::injective_concretization;
  constexpr static const bool preserve_concrete_covers = true;
  constexpr static const bool complemented = false;
  constexpr static const char* name = "Flat";
  constexpr static const bool is_arithmetic = pre_universe::is_arithmetic;

  /** A pre-interpreted formula `x = value` ready to use.
   * This is mainly for optimization purpose to avoid calling `interpret` each time we need it. */
  CUDA static constexpr local_type eq_k(value_type k) {
    return local_type(k);
  }

private:
  using atomic_type = memory_type::template atomic_type<value_type>;
  atomic_type val;

public:
  /** Similar to \f$[\![\mathit{true}]\!]\f$. */
  CUDA static constexpr local_type bot() {
    return local_type();
  }

  /** Similar to \f$[\![\mathit{false}]\!]\f$. */
  CUDA static constexpr local_type top() {
    return local_type(U::top());
  }
  /** Initialize to the bottom of the flat lattice. */
  CUDA constexpr FlatUniverse(): val(U::bot()) {}
  /** Similar to \f$[\![x = k]\!]\f$ for any name `x` where \f$ = \f$ is the equality. */
  CUDA constexpr FlatUniverse(value_type k): val(k) {}
  CUDA constexpr FlatUniverse(const this_type& other): FlatUniverse(other.value()) {}
  CUDA constexpr FlatUniverse(this_type&& other): val(std::move(other.val)) {}

  template <class M>
  CUDA constexpr FlatUniverse(const this_type2<M>& other): FlatUniverse(other.value()) {}

  template <class M>
  CUDA constexpr FlatUniverse(const PrimitiveUpset<pre_universe, M>& other)
    : FlatUniverse(other.value()) {}

  template <class M>
  CUDA constexpr FlatUniverse(const PrimitiveUpset<typename pre_universe::dual_type, M> &other)
    : FlatUniverse(dual<PrimitiveUpset<pre_universe, battery::local_memory>>(other)) {}

  /** The assignment operator can only be used in a sequential context.
   * It is monotone but not extensive. */
  template <class M>
  CUDA constexpr this_type& operator=(const this_type2<M>& other) {
   if constexpr(sequential) {
      memory_type::store(val, other.value());
      return *this;
    }
    else {
      static_assert(sequential, "The operator= in `FlatUniverse` can only be used when the underlying memory is `sequential`.");
    }
  }

  CUDA constexpr this_type& operator=(const this_type& other) {
    if constexpr(sequential) {
      memory_type::store(val, other.value());
      return *this;
    }
    else {
      static_assert(sequential, "The operator= in `FlatUniverse` can only be used when the underlying memory is `sequential`.");
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
    if(other.is_bot() || *this == other || is_top()) {
      return false;
    }
    if(is_bot()) {
      memory_type::store(val, other.value());
    }
    else {
      join_top();
    }
    return true;
  }

  CUDA constexpr void meet_bot() {
    memory_type::store(val, U::bot());
  }

  template<class M1>
  CUDA constexpr bool meet(const this_type2<M1>& other) {
    if(is_bot() || *this == other || other.is_top()) {
      return false;
    }
    if(is_top()) {
      memory_type::store(val, other.value());
    }
    else {
      meet_bot();
    }
    return true;
  }

  /** \return \f$ x = k \f$ where `x` is a variable's name and `k` the current value.
  `true` is returned whenever \f$ a = \bot \f$, and `false` is returned whenever \f$ a = \top \f$.
  We always return an exact approximation, hence for any formula \f$ \llbracket \varphi \rrbracket = a \f$, we must have \f$ a =  \llbracket \rrbracket a \llbracket \rrbracket \f$ where \f$ \rrbracket a \llbracket \f$ is the deinterpretation function. */
  template<class Env>
  CUDA NI TFormula<typename Env::allocator_type> deinterpret(AVar avar, const Env& env) const {
    using F = TFormula<typename Env::allocator_type>;
    if(is_top()) {
      return F::make_false();
    }
    else if(is_bot()) {
      return F::make_true();
    }
    return F::make_binary(
      F::make_avar(avar),
      EQ,
      deinterpret<F>(),
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

public:
  /** Expects a predicate of the form `x = k` or `k = x`, where `x` is any variable's name, and `k` a constant.
      Existential formula \f$ \exists{x:T} \f$ can also be interpreted (only to bottom). */
  template<bool diagnose = false, class F, class Env, class M2>
  CUDA NI static bool interpret_tell(const F& f, const Env& env, this_type2<M2>& tell, IDiagnostics& diagnostics) {
    if(f.is(F::E)) {
      value_type k;
      bool res = pre_universe::template interpret_type<diagnose>(f, k, diagnostics);
      if(res) {
        tell.join(local_type(k));
      }
      return res;
    }
    else {
      if(f.is_binary() && f.sig() == EQ) {
        int idx_constant = f.seq(0).is_constant() ? 0 : (f.seq(1).is_constant() ? 1 : 100);
        int idx_variable = f.seq(0).is_variable() ? 0 : (f.seq(1).is_variable() ? 1 : 100);
        if(idx_constant + idx_variable == 1) {
          const auto& k = f.seq(idx_constant);
          const auto& x = f.seq(idx_variable);
          value_type t;
          if(pre_universe::template interpret_tell<diagnose>(k, t, diagnostics)) {
            value_type a;
            if(pre_universe::template interpret_ask<diagnose>(k, a, diagnostics)) {
              if(a == t) {
                tell.join(local_type(t));
                return true;
              }
              else {
                RETURN_INTERPRETATION_ERROR("The constant has no exact interpretation which is required in this abstract universe.");
              }
            }
          }
          return false;
        }
      }
      RETURN_INTERPRETATION_ERROR(
        "Tell interpretation only supports existential quantifier and binary formulas of the form `t1 = t2` where t1 is a constant and t2 is a variable (or conversely).");
    }
  }

  /** Same as `interpret_tell` without the support for existential quantifier. */
  template<bool diagnose = false, class F, class Env, class M2>
  CUDA NI static bool interpret_ask(const F& f, const Env& env, this_type2<M2>& ask, IDiagnostics& diagnostics) {
    if(f.is(F::E)) {
      RETURN_INTERPRETATION_ERROR("Ask interpretation only supports binary formulas of the form `t1 = t2` where t1 is a constant and t2 is a variable (or conversely).")
    }
    return interpret_tell<diagnose>(f, env, ask, diagnostics);
  }

  template<IKind kind, bool diagnose = false, class F, class Env, class M2>
  CUDA NI static bool interpret(const F& f, const Env& env, this_type2<M2>& value, IDiagnostics& diagnostics) {
    if constexpr(kind == IKind::ASK) {
      return interpret_ask<diagnose>(f, env, value, diagnostics);
    }
    else {
      return interpret_tell<diagnose>(f, env, value, diagnostics);
    }
  }

  CUDA static constexpr bool is_supported_fun(Sig fun) {
    return pre_universe::is_supported_fun(fun);
  }

public:
  /** In-place projection of the result of the unary function `fun(a)`. */
  CUDA constexpr void project(Sig fun, const local_type& a) {
    assert(is_supported_fun(fun));
    if(a.is_top()) {
      join_top();
    }
    else if(!a.is_bot()) {
      join(local_type(pre_universe::project(fun, a.value())));
    }
  }

  /** In-place projection of the result of the binary function `fun(a, b)`. */
  CUDA constexpr void project(Sig fun, const local_type& a, const local_type& b) {
    assert(is_supported_fun(fun));
    if(a.is_top() || b.is_top()) {
      join_top();
    }
    else if(!a.is_bot() && !b.is_bot()) {
      if constexpr(is_arithmetic) {
        if(is_division(fun) && b == pre_universe::zero()) {
          join_top();
          return;
        }
      }
      join(local_type(pre_universe::project(fun, a.value(), b.value())));
    }
  }

  template<class Pre2, class Mem2>
  friend class FlatUniverse;
};

// Lattice operators

template<class Pre>
CUDA constexpr FlatUniverse<Pre, battery::local_memory> fjoin(const FlatUniverse<Pre, battery::local_memory>& a, const FlatUniverse<Pre, battery::local_memory>& b) {
  if(a == b) {
    return a;
  }
  else if(a.is_bot() || b.is_top()) {
    return b;
  }
  else if(b.is_bot() || a.is_top()) {
    return a;
  }
  else {
    return FlatUniverse<Pre, battery::local_memory>::top();
  }
}

template<class Pre>
CUDA constexpr FlatUniverse<Pre, battery::local_memory> fmeet(const FlatUniverse<Pre, battery::local_memory>& a, const FlatUniverse<Pre, battery::local_memory>& b) {
  if(a == b) {
    return a;
  }
  else if(a.is_bot() || b.is_top()) {
    return a;
  }
  else if(b.is_bot() || a.is_top()) {
    return b;
  }
  else {
    return FlatUniverse<Pre, battery::local_memory>::bot();
  }
}

template<class Pre, class M1, class M2>
CUDA constexpr bool operator<=(const FlatUniverse<Pre, M1>& a, const FlatUniverse<Pre, M2>& b) {
  return a.is_bot() || b.is_top() || a == b;
}

template<class Pre, class M1, class M2>
CUDA constexpr bool operator<(const FlatUniverse<Pre, M1>& a, const FlatUniverse<Pre, M2>& b) {
  return (a.is_bot() || b.is_top()) && a != b;
}

template<class Pre, class M1, class M2>
CUDA constexpr bool operator>=(const FlatUniverse<Pre, M1>& a, const FlatUniverse<Pre, M2>& b) {
  return b <= a;
}

template<class Pre, class M1, class M2>
CUDA constexpr bool operator>(const FlatUniverse<Pre, M1>& a, const FlatUniverse<Pre, M2>& b) {
  return b < a;
}

template<class Pre, class M1, class M2>
CUDA constexpr bool operator==(const FlatUniverse<Pre, M1>& a, const FlatUniverse<Pre, M2>& b) {
  return a.value() == b.value();
}

template<class Pre, class M1, class M2>
CUDA constexpr bool operator!=(const FlatUniverse<Pre, M1>& a, const FlatUniverse<Pre, M2>& b) {
  return a.value() != b.value();
}

template<class Pre, class M>
std::ostream& operator<<(std::ostream &s, const FlatUniverse<Pre, M> &a) {
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
