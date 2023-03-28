// Copyright 2023 Pierre Talbot

#ifndef FLAT_UNIVERSE_HPP
#define FLAT_UNIVERSE_HPP

#include "upset_universe.hpp"

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
  using ZFlat = ::lala::ZFlat<int, battery::LocalMemory>;
  using FFlat = ::lala::FFlat<double, battery::LocalMemory>;
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

  template<class M>
  using this_type2 = FlatUniverse<pre_universe, M>;

  template<class F>
  using iresult = IResult<this_type, F>;

  static_assert(pre_universe::preserve_bot && pre_universe::preserve_top,
    "The Flat lattice construction reuse the bottom and top elements of the pre-universe.\
    Therefore, it must preserve bottom and top.");

  constexpr static const bool sequential = Mem::sequential;
  constexpr static const bool is_totally_ordered = false;
  constexpr static const bool preserve_bot = true;
  constexpr static const bool preserve_top = true;
  constexpr static const bool injective_concretization = pre_universe::injective_concretization;
  constexpr static const bool preserve_inner_covers = false;
  constexpr static const bool complemented = false;
  constexpr static const char* name = "Flat";
  constexpr static const bool is_arithmetic = pre_universe::is_arithmetic;

  /** A pre-interpreted formula `x = value` ready to use.
   * This is mainly for optimization purpose to avoid calling `interpret` each time we need it. */
  CUDA static constexpr this_type eq_k(value_type k) {
    return this_type(k);
  }

private:
  using atomic_type = typename memory_type::atomic_type<value_type>;
  atomic_type val;

public:
  /** Similar to \f$[\![\mathit{true}]\!]\f$. */
  CUDA static constexpr this_type bot() {
    return this_type();
  }

  /** Similar to \f$[\![\mathit{false}]\!]\f$. */
  CUDA static constexpr this_type top() {
    return this_type(U::top());
  }
  /** Initialize an upset universe to bottom. */
  CUDA constexpr FlatUniverse(): val(U::bot()) {}
  /** Similar to \f$[\![x = k]\!]\f$ for any name `x` where \f$ = \f$ is the equality. */
  CUDA constexpr FlatUniverse(value_type k): val(k) {}
  CUDA constexpr FlatUniverse(const this_type& other): FlatUniverse(other.value()) {}
  CUDA constexpr FlatUniverse(this_type&& other): val(std::move(other.val)) {}

  template <class M>
  CUDA constexpr FlatUniverse(const this_type2<M>& other): FlatUniverse(other.value()) {}

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

  /** `true` whenever \f$ a = \top \f$, `false` otherwise. */
  CUDA constexpr local::BInc is_top() const {
    return value() == U::top();
  }

  /** `true` whenever \f$ a = \bot \f$, `false` otherwise. */
  CUDA constexpr local::BDec is_bot() const {
    return value() == U::bot();
  }

  CUDA constexpr this_type& tell_top() {
    memory_type::store(val, U::top());
    return *this;
  }

  template<class M1, class M2>
  CUDA constexpr this_type& tell(const this_type2<M1>& other, BInc<M2>& has_changed) {
    if(other.is_bot() || *this == other || is_top()) {
      return *this;
    }
    has_changed.tell_top();
    if(is_bot()) {
      memory_type::store(val, other.value());
      return *this;
    }
    else {
      return tell_top();
    }
  }

  template<class M1>
  CUDA constexpr this_type& tell(const this_type2<M1>& other) {
    if(other.is_bot() || *this == other || is_top()) {
      return *this;
    }
    if(is_bot()) {
      memory_type::store(val, other.value());
      return *this;
    }
    else {
      return tell_top();
    }
  }

  CUDA constexpr this_type& dtell_bot() {
    memory_type::store(val, U::bot());
    return *this;
  }

  template<class M1, class M2>
  CUDA constexpr this_type& dtell(const this_type2<M1>& other, BInc<M2>& has_changed) {
    if(is_bot() || *this == other || other.is_top()) {
      return *this;
    }
    has_changed.tell_top();
    if(is_top()) {
      memory_type::store(val, other.value());
      return *this;
    }
    else {
      return dtell_bot();
    }
  }

  template<class M1>
  CUDA constexpr this_type& dtell(const this_type2<M1>& other) {
    if(is_bot() || *this == other || other.is_top()) {
      return *this;
    }
    if(is_top()) {
      memory_type::store(val, other.value());
      return *this;
    }
    else {
      return dtell_bot();
    }
  }

  /** \return \f$ x = k \f$ where `x` is a variable's name and `k` the current value.
  `true` is returned whenever \f$ a = \bot \f$, and `false` is returned whenever \f$ a = \top \f$.
  We always return an exact approximation, hence for any formula \f$ \llbracket \varphi \rrbracket = a \f$, we must have \f$ a =  \llbracket \rrbracket a \llbracket \rrbracket \f$ where \f$ \rrbracket a \llbracket \f$ is the deinterpretation function. */
  template<class Env>
  CUDA TFormula<typename Env::allocator_type> deinterpret(AVar avar, const Env& env) const {
    using allocator_t = typename Env::allocator_type;
    if(is_top()) {
      return TFormula<allocator_t>::make_false();
    }
    else if(is_bot()) {
      return TFormula<allocator_t>::make_true();
    }
    return make_v_op_z(avar, EQ, value(), avar.aty(), EXACT, env.get_allocator());
  }

  /** Under-approximates the current element \f$ a \f$ w.r.t. \f$ \rrbracket a \llbracket \f$ into `ua`.
   * For this abstract universe, it always returns `true` since the current element \f$ a \f$ is an exact representation of \f$ \rrbracket a \llbracket \f$. */
  CUDA constexpr bool extract(this_type2<battery::LocalMemory>& ua) const {
    ua.val = value();
    return true;
  }

  /** Print the current element. */
  CUDA void print() const {
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
  template <class F>
  CUDA static iresult<F> interpret_false(const F& f) {
    if(preserve_top || f.is_over()) {
      return iresult<F>(top());
    }
    else {
      return iresult<F>(IError<F>(true, name, "Top is not preserved, hence it cannot exactly interpret or under-approximate formulas equivalent to `false`.", f));
    }
  }

  template<class F>
  CUDA static iresult<F> interpret_true(const F& f) {
    if(preserve_bot || f.is_under()) {
      return bot();
    }
    else {
      return iresult<F>(IError<F>(true, name, "Bottom is not preserved, hence it cannot exactly interpret or over-approximate formula equivalent to `true`.", f));
    }
  }

public:

  /** Expects a predicate of the form `x = k` or `k = x`, where `x` is any variable's name, and `k` a constant.
    The only interpretation mode possible is `EXACT`.
    Existential formula \f$ \exists{x:T} \f$ can also be interpreted (only to bottom). */
  template<class F, class Env>
  CUDA static iresult<F> interpret(const F& f, const Env& env) {
    if(f.is_true()) {
      return interpret_true(f);
    }
    else if(f.is_false()) {
      return interpret_false(f);
    }
    else if(f.is(F::E)) {
      return pre_universe::interpret_type(f);
    }
    else {
      if(f.is_binary() && f.sig() == EQ && f.approx() == EXACT) {
        int idx_constant = f.seq(0).is_constant() ? 0 : (f.seq(1).is_constant() ? 1 : 100);
        int idx_variable = f.seq(0).is_variable() ? 0 : (f.seq(1).is_variable() ? 1 : 100);
        if(idx_constant + idx_variable == 1) {
          const auto& k = f.seq(idx_constant);
          const auto& x = f.seq(idx_variable);
          auto r = pre_universe::interpret(k, k.sort().value(), EXACT);
          return r;
        }
      }
      return iresult<F>(IError<F>(true, name,
        "Only exact interpretation of binary formulas of the form `t1 = t2` where if t1 is a constant and t2 is a variable (or conversely) are supported.", f));
    }
  }

  /** By construction, the flat lattice only supports exact function.
   * Indeed, an under-approximation systematically leads to top, and an over-approximation systematically leads to bottom. */
  CUDA static constexpr bool is_supported_fun(Approx appx, Sig sig) {
    return appx == EXACT && pre_universe::is_supported_fun(appx, sig);
  }

public:
  /** Unary function over `value_type`. */
  template<Approx appx, Sig sig, class M1>
  CUDA static constexpr this_type2<battery::LocalMemory> fun(const this_type2<M1>& u) {
    static_assert(is_supported_fun(appx, sig));
    if(u.is_top()) {
      return this_type2<battery::LocalMemory>::top();
    }
    else if(u.is_bot()) {
      return this_type2<battery::LocalMemory>::bot();
    }
    return pre_universe::template fun<appx, sig>(u.value());
  }

  /** Binary functions over `value_type`. */
  template<Approx appx, Sig sig, class M1, class M2>
  CUDA static constexpr this_type2<battery::LocalMemory> fun(const this_type2<M1>& a, const this_type2<M2>& b) {
    static_assert(is_supported_fun(appx, sig));
    if(a.is_top() || b.is_top()) {
      return this_type2<battery::LocalMemory>::top();
    }
    else if(a.is_bot() || b.is_bot()) {
      return this_type2<battery::LocalMemory>::bot();
    }
    if constexpr(is_division(sig) && is_arithmetic) {
      if(b.value() == pre_universe::zero()) {
        return this_type2<battery::LocalMemory>::top();
      }
    }
    return pre_universe::template fun<appx, sig>(a, b);
  }

  template<class Pre2, class Mem2>
  friend class FlatUniverse;
};

// Lattice operators

template<class Pre, class M1, class M2>
CUDA constexpr FlatUniverse<Pre, battery::LocalMemory> join(const FlatUniverse<Pre, M1>& a, const FlatUniverse<Pre, M2>& b) {
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
    return FlatUniverse<Pre, battery::LocalMemory>::top();
  }
}

template<class Pre, class M1, class M2>
CUDA constexpr FlatUniverse<Pre, battery::LocalMemory> meet(const FlatUniverse<Pre, M1>& a, const FlatUniverse<Pre, M2>& b) {
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
    return FlatUniverse<Pre, battery::LocalMemory>::bot();
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
