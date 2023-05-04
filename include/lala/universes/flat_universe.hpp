// Copyright 2023 Pierre Talbot

#ifndef FLAT_UNIVERSE_HPP
#define FLAT_UNIVERSE_HPP

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

  template<class F>
  using iresult = IResult<local_type, F>;
  static_assert(pre_universe::increasing);
  static_assert(pre_universe::preserve_bot && pre_universe::preserve_top,
    "The Flat lattice construction reuse the bottom and top elements of the pre-universe.\
    Therefore, it must preserve bottom and top.");

  constexpr static const bool is_abstract_universe = true;
  constexpr static const bool sequential = Mem::sequential;
  constexpr static const bool is_totally_ordered = false;
  constexpr static const bool preserve_bot = true;
  constexpr static const bool preserve_top = true;
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
  using atomic_type = typename memory_type::atomic_type<value_type>;
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
    return make_v_op_z(avar, EQ, value(), avar.aty(), env.get_allocator());
  }

  /** Under-approximates the current element \f$ a \f$ w.r.t. \f$ \rrbracket a \llbracket \f$ into `ua`.
   * For this abstract universe, it always returns `true` since the current element \f$ a \f$ is an exact representation of \f$ \rrbracket a \llbracket \f$. */
  CUDA constexpr bool extract(local_type& ua) const {
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

public:
  /** Expects a predicate of the form `x = k` or `k = x`, where `x` is any variable's name, and `k` a constant.
      Existential formula \f$ \exists{x:T} \f$ can also be interpreted (only to bottom). */
  template<class F, class Env>
  CUDA static iresult<F> interpret_tell(const F& f, const Env& env) {
    if(f.is_true()) {
      return bot();
    }
    else if(f.is_false()) {
      return top();
    }
    else if(f.is(F::E)) {
      return pre_universe::interpret_type(f);
    }
    else {
      if(f.is_binary() && f.sig() == EQ) {
        int idx_constant = f.seq(0).is_constant() ? 0 : (f.seq(1).is_constant() ? 1 : 100);
        int idx_variable = f.seq(0).is_variable() ? 0 : (f.seq(1).is_variable() ? 1 : 100);
        if(idx_constant + idx_variable == 1) {
          const auto& k = f.seq(idx_constant);
          const auto& x = f.seq(idx_variable);
          auto t = pre_universe::interpret_tell(k);
          auto a = pre_universe::interpret_ask(k);
          if(t.has_value()) {
            if(a.has_value()) {
              if(a.value() == t.value()) {
                return t;
              }
              else {
                return std::move(iresult<F>(IError<F>(true, name, "The constant has no exact interpretation which is required in this abstract universe.", f))
                  .join_errors(std::move(t))
                  .join_errors(std::move(a)));
              }
            }
            else {
              return a;
            }
          }
          else {
            return t;
          }
        }
      }
      return iresult<F>(IError<F>(true, name,
        "Only interpretations of existential quantifier and binary formulas of the form `t1 = t2` where if t1 is a constant and t2 is a variable (or conversely) are supported.", f));
    }
  }

  /** Same as `interpret_tell` without the support for existential quantifier. */
  template<class F, class Env>
  CUDA static iresult<F> interpret_ask(const F& f, const Env& env) {
    if(f.is(F::E)) {
      return iresult<F>(IError<F>(true, name,
        "Only interpretation of binary formulas of the form `t1 = t2` where if t1 is a constant and t2 is a variable (or conversely) are supported.", f));
    }
    return interpret_tell(f, env);
  }

  CUDA static constexpr bool is_supported_fun(Sig sig) {
    return pre_universe::is_supported_fun(sig);
  }

public:
  /** Unary function over `value_type`. */
  template<Sig sig, class M1>
  CUDA static constexpr local_type fun(const this_type2<M1>& u) {
    static_assert(is_supported_fun(sig));
    auto a = u.value();
    if(U::top() == a) {
      return local_type::top();
    }
    else if(U::bot() == a) {
      return local_type::bot();
    }
    return pre_universe::template fun<sig>(a);
  }

  /** Binary functions over `value_type`. */
  template<Sig sig, class M1, class M2>
  CUDA static constexpr local_type fun(const this_type2<M1>& l, const this_type2<M2>& k) {
    static_assert(is_supported_fun(sig));
    auto a = l.value();
    auto b = k.value();
    if(U::top() == a || U::top() == b) {
      return local_type::top();
    }
    else if(U::bot() == a || U::bot() == b) {
      return local_type::bot();
    }
    if constexpr(is_division(sig) && is_arithmetic) {
      if(b == pre_universe::zero()) {
        return local_type::top();
      }
    }
    return pre_universe::template fun<sig>(a, b);
  }

  template<class Pre2, class Mem2>
  friend class FlatUniverse;
};

// Lattice operators

template<class Pre, class M1, class M2>
CUDA constexpr FlatUniverse<Pre, battery::local_memory> join(const FlatUniverse<Pre, M1>& a, const FlatUniverse<Pre, M2>& b) {
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

template<class Pre, class M1, class M2>
CUDA constexpr FlatUniverse<Pre, battery::local_memory> meet(const FlatUniverse<Pre, M1>& a, const FlatUniverse<Pre, M2>& b) {
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
