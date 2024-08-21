// Copyright 2024 Pierre Talbot

#ifndef LALA_CORE_NBITSET_HPP
#define LALA_CORE_NBITSET_HPP

#include "arith_bound.hpp"
#include "battery/bitset.hpp"

namespace lala {

/** This class represents a set of integer values with a fixed-size bitset.
 * In order to have well-defined arithmetic operations preserving bottom and top elements, the first and last bits (written L and R below) of the bitset are reserved.
 * The meaning of L is to include all negative integers and the meaning of R is to include all integers greater than the size of the bitset.
 * Given a bitset \f$ Lb_0b_1...b_nR \f$ of size n + 3, the concretization function is given as follows:
 * \f$ \gamma(Lb_0b_1...b_nR) = \{ i \in \mathbb{Z} \mid 0 \leq i \leq n \land b_i = 1 \} \cup \{ i \in \mathbb{Z} \;|\; i < 0 \land L = 1 \} \cup \{ i \in \mathbb{Z} \;|\; i > n \land R = 1 \} \f$
 */
template <size_t N, class Mem, class T = unsigned long long>
class NBitset
{
public:
  using memory_type = Mem;
  using bitset_type = battery::bitset<N, Mem, T>;
  using this_type = NBitset<N, Mem, T>;
  template <class M> using this_type2 = NBitset<N, M, T>;
  using local_type = this_type2<battery::local_memory>;

  using LB = local::ZLB;
  using UB = local::ZUB;
  using value_type = typename LB::value_type;

  template <size_t N2, class Mem2, class T2>
  friend class NBitset;

  constexpr static const bool is_abstract_universe = true;
  constexpr static const bool sequential = Mem::sequential;
  constexpr static const bool is_totally_ordered = false;
  constexpr static const bool preserve_bot = true;
  constexpr static const bool preserve_top = true;
  constexpr static const bool preserve_join = true;
  constexpr static const bool preserve_meet = true;
  constexpr static const bool injective_concretization = true;
  constexpr static const bool preserve_concrete_covers = false;
  constexpr static const bool complemented = true;
  constexpr static const char* name = "NBitset";

private:
  bitset_type bits;

  struct bot_constructor_tag {};
  CUDA constexpr NBitset(bot_constructor_tag) {}

public:
  /** Initialize to top (all bits at `1`). */
  CUDA constexpr NBitset() {
    bits.set();
  }

  CUDA constexpr static this_type from_set(const battery::vector<int>& values) {
    this_type b(bot());
    for(int i = 0; i < values.size(); ++i) {
      b.bits.set(battery::min(static_cast<int>(N)-1, battery::max(values[i]+1,0)), true);
    }
    return b;
  }

  CUDA constexpr NBitset(const this_type& other): NBitset(other.bits) {}
  constexpr NBitset(this_type&&) = default;

  /** Given a value \f$ x \in U \f$ where \f$ U \f$ is the universe of discourse, we initialize a singleton bitset \f$ 0_0..1_{x+1}...0_n \f$. */
  CUDA constexpr NBitset(value_type x) {
    bits.set(battery::min(static_cast<int>(N)-1, battery::max(0, x+1)));
  }

  CUDA constexpr NBitset(value_type lb, value_type ub): bits(
    battery::min(static_cast<int>(N)-1, battery::max(lb+1,0)),
    battery::min(static_cast<int>(N)-1, battery::max(ub+1, 0)))
  {}

  template<class M>
  CUDA constexpr NBitset(const this_type2<M>& other): bits(other.bits) {}

  template<class M>
  CUDA constexpr NBitset(this_type2<M>&& other): bits(std::move(other.bits)) {}

  template<class M>
  CUDA constexpr NBitset(const battery::bitset<N, M, T>& bits): bits(bits) {}

  /** The assignment operator can only be used in a sequential context.
   * It is monotone but not extensive. */
  template <class M>
  CUDA constexpr this_type& operator=(const this_type2<M>& other) {
    if constexpr(sequential) {
      bits = other.bits;
      return *this;
    }
    else {
      static_assert(sequential, "The operator= in `NBitset` can only be used when the underlying memory is `sequential`.");
    }
  }

  CUDA constexpr this_type& operator=(const this_type& other) {
    if constexpr(sequential) {
      bits = other.bits;
      return *this;
    }
    else {
      static_assert(sequential, "The operator= in `NBitset` can only be used when the underlying memory is `sequential`.");
    }
  }

  /** Pre-interpreted formula `x == 0`. */
  CUDA constexpr static local_type eq_zero() { return local_type(0); }
  /** Pre-interpreted formula `x == 1`. */
  CUDA constexpr static local_type eq_one() { return local_type(1); }

  CUDA constexpr static local_type bot() { return NBitset(bot_constructor_tag{}); }
  CUDA constexpr static local_type top() { return NBitset(); }
  CUDA constexpr local::B is_top() const { return bits.all(); }
  CUDA constexpr local::B is_bot() const { return bits.none(); }
  CUDA constexpr const bitset_type& value() const { return bits; }

private:
  template<bool diagnose, class F, class Env, class M>
  CUDA NI static bool interpret_existential(const F& f, const Env& env, this_type2<M>& k, IDiagnostics& diagnostics) {
    const auto& sort = battery::get<1>(f.exists());
    if(sort.is_int()) {
      return true;
    }
    else if(sort.is_bool()) {
      k.meet(local_type(0,1));
      return true;
    }
    else {
      const auto& vname = battery::get<0>(f.exists());
      RETURN_INTERPRETATION_ERROR(("NBitset only supports variables of type `Int` or `Bool`, but `" + vname + "` has another sort."));
    }
  }

  template<bool diagnose, bool negated, class F, class M>
  CUDA NI static bool interpret_tell_set(const F& f, const F& k, this_type2<M>& tell, IDiagnostics& diagnostics) {
    using sort_type = Sort<typename F::allocator_type>;
    std::optional<sort_type> sort = f.seq(1).sort();
    if(sort.has_value() &&
       (sort.value() == sort_type(sort_type::Set, sort_type(sort_type::Int))
     || sort.value() == sort_type(sort_type::Set, sort_type(sort_type::Bool))))
    {
      const auto& set = f.seq(1).s();
      local_type join_s(bot_constructor_tag{});
      bool over_appx = false;
      for(int i = 0; i < set.size(); ++i) {
        int l = battery::get<0>(set[i]).to_z();
        int u = battery::get<1>(set[i]).to_z();
        join_s.join(local_type(l, u));
        if(l < 0 || u >= join_s.bits.size() - 2) {
          over_appx = true;
        }
      }
      if constexpr(negated) {
        join_s = join_s.complement();
        // In any case it must be set to true: if no element is below zero, then some elements in the negation are; and if some elements are below zero it's not all of them.
        join_s.bits.set(0, true);
        join_s.bits.set(join_s.bits.size()-1, true);
      }
      tell.meet(join_s);
      if(over_appx) {
        RETURN_INTERPRETATION_WARNING("Constraint `x in S` is over-approximated because some elements of `S` fall outside the bitset.");
      }
      return true;
    }
    else {
      RETURN_INTERPRETATION_ERROR("NBitset only supports membership (`x in S`) where `S` is a set of integers.");
    }
  }

  template<bool diagnose, class F, class M>
  CUDA NI static bool interpret_tell_x_op_k(const F& f, logic_int k, Sig sig, this_type2<M>& tell, IDiagnostics& diagnostics) {
    if(sig == LT) {
      return interpret_tell_x_op_k<diagnose>(f, k-1, LEQ, tell, diagnostics);
    }
    else if(sig == GT) {
      return interpret_tell_x_op_k<diagnose>(f, k+1, GEQ, tell, diagnostics);
    }
    else if(k < 0 || k >= tell.bits.size() - 2) {
      if((k == -1 && sig == LEQ) || (k == tell.bits.size() - 2 && sig == GEQ)) {
        // this is fine because x <= -1 and x >= n-2 can be represented exactly.
      }
      else {
        INTERPRETATION_WARNING("Constraint `x <op> k` is over-approximated because `k` is not representable in the bitset. Note that for a bitset of size `n`, the only values representable exactly are in the interval `[0, n-3]` because two bits are used to represent all negative values and all values exceeding the size of the bitset.");
        // If it is NEQ, we can't give a better approximation than top.
        if(sig == NEQ) {
          return true;
        }
      }
    }
    switch(sig) {
      case EQ: tell.meet(local_type(k, k)); break;
      case NEQ: tell.meet(local_type(k, k).complement()); break;
      case LEQ: tell.meet(local_type(-1, k)); break;
      case GEQ: tell.meet(local_type(k, tell.bits.size())); break;
      default: RETURN_INTERPRETATION_ERROR("This symbol is not supported.");
    }
    return true;
  }

  template<bool diagnose, bool negated, class F, class Env, class M>
  CUDA NI static bool interpret_binary(const F& f, const Env& env, this_type2<M>& tell, IDiagnostics& diagnostics) {
    if(f.sig() == IN) {
      return interpret_tell_set<diagnose, negated>(f, f.seq(1), tell, diagnostics);
    }
    else if(f.seq(1).is(F::Z) || f.seq(1).is(F::B)) {
      return interpret_tell_x_op_k<diagnose>(f, f.seq(1).to_z(), f.sig(), tell, diagnostics);
    }
    else {
      RETURN_INTERPRETATION_ERROR("Only integer and Boolean constants are supported in NBitset.");
    }
  }

public:
  /** Support the following language where all constants `k` are integer or Boolean values:
   *   * `var x:Z`
   *   * `var x:B`
   *   * `x <op> k` where `k` is an integer constant and `<op>` in {==, !=, <, <=, >, >=}.
   *   * `x in S` where `S` is a set of integers.
   * It can be over-approximated if the element `k` falls out of the bitset. */
  template<bool diagnose = false, class F, class Env, class M>
  CUDA NI static bool interpret_tell(const F& f, const Env& env, this_type2<M>& tell, IDiagnostics& diagnostics) {
    using sort_type = Sort<typename F::allocator_type>;
    if(f.is(F::E)) {
      return interpret_existential<diagnose>(f, env, tell, diagnostics);
    }
    else if(f.is_unary() && f.sig() == NOT && f.seq(0).is_binary()) {
      return interpret_binary<diagnose, true>(f.seq(0), env, tell, diagnostics);
    }
    else if(f.is_binary() && f.seq(0).is_variable() && f.seq(1).is_constant()) {
      return interpret_binary<diagnose, false>(f, env, tell, diagnostics);
    }
    else {
      RETURN_INTERPRETATION_ERROR("Only binary formulas of the form `x <sig> k` where if x is a variable and k is a constant are supported. We also supports existential quantifier and membership in a set of integers (x in S).");
    }
  }

  /** Support the same language than the "tell language" without existential. */
  template<bool diagnose = false, class F, class Env, class M>
  CUDA NI static bool interpret_ask(const F& f, const Env& env, this_type2<M>& k, IDiagnostics& diagnostics) {
    local_type b = local_type::top();
    auto nf = negate(f);
    if(!nf.has_value()) {
      RETURN_INTERPRETATION_ERROR("Could not negate the formula in order to interpret_ask it.");
    }
    if(f.is(F::E)) {
      RETURN_INTERPRETATION_ERROR("Existential quantification is not supported in ask interpretation.");
    }
    if(interpret_tell<diagnose>(nf.value(), env, b, diagnostics)) {
      k.meet(b.complement());
      return true;
    }
    else {
      return false;
    }
  }

  template<IKind kind, bool diagnose = false, class F, class Env, class M>
  CUDA NI static bool interpret(const F& f, const Env& env, this_type2<M>& k, IDiagnostics& diagnostics) {
    if constexpr(kind == IKind::ASK) {
      return interpret_ask<diagnose>(f, env, k, diagnostics);
    }
    else {
      return interpret_tell<diagnose>(f, env, k, diagnostics);
    }
  }

  CUDA constexpr LB lb() const {
    value_type l = bits.countr_zero();
    return l == 0 ? LB::top() :
      (l == bits.size() ? LB::bot() : LB::geq_k(l-1));
  }

  CUDA constexpr UB ub() const {
    value_type r = bits.countl_zero();
    return r == 0 ? UB::top() :
      (r == bits.size() ? UB::bot() : UB::leq_k(bits.size() - r - 2));
  }

  CUDA constexpr local_type complement() const {
    local_type c(bits);
    c.bits.flip();
    return c;
  }

  CUDA constexpr void join_top() {
    bits.set();
  }

  template<class A>
  CUDA constexpr bool join_lb(const A& lb) {
    return join(local_type(lb.value(), bits.size()));
  }

  template<class A>
  CUDA constexpr bool join_ub(const A& ub) {
    return join(local_type(-1, ub.value()));
  }

  template<class M>
  CUDA constexpr bool join(const this_type2<M>& other) {
    if(!other.bits.is_subset_of(bits)) {
      bits |= other.bits;
      return true;
    }
    return false;
  }

  CUDA constexpr void meet_bot() {
    bits.reset();
  }

  template<class M>
  CUDA constexpr bool meet(const this_type2<M>& other) {
    if(!bits.is_subset_of(other.bits)) {
      bits &= other.bits;
      return true;
    }
    return false;
  }

  template <class M>
  CUDA constexpr bool extract(this_type2<M>& ua) const {
    ua.bits = bits;
    return true;
  }

  template<class Env, class Allocator = typename Env::allocator_type>
  CUDA TFormula<Allocator> deinterpret(AVar x, const Env& env, const Allocator& allocator = Allocator()) const {
    using F = TFormula<Allocator>;
    if(is_bot()) {
      return F::make_false();
    }
    else if(is_top()) {
      return F::make_true();
    }
    else {
      typename F::Sequence seq{allocator};
      if(bits.test(0)) {
        seq.push_back(F::make_binary(F::make_avar(x), LEQ, F::make_z(-1), UNTYPED, allocator));
      }
      if(bits.test(bits.size()-1)) {
        seq.push_back(F::make_binary(F::make_avar(x), GEQ, F::make_z(bits.size()-2), UNTYPED, allocator));
      }
      logic_set<F> logical_set(allocator);
      for(int i = 1; i < bits.size()-1; ++i) {
        if(bits.test(i)) {
          int l = i - 1;
          for(i = i + 1; i < bits.size()-1 && bits.test(i); ++i) {}
          int u = i - 2;
          logical_set.push_back(battery::make_tuple(F::make_z(l), F::make_z(u)));
        }
      }
      if(logical_set.size() > 0) {
        seq.push_back(F::make_binary(F::make_avar(x), IN, F::make_set(std::move(logical_set)), UNTYPED, allocator));
      }
      if(seq.size() == 1) {
        return std::move(seq[0]);
      }
      else {
        return F::make_nary(OR, std::move(seq));
      }
    }
  }

  /** Deinterpret the current value to a logical constant.
   * The lower bound is deinterpreted, and it is up to the user to check that interval is a singleton.
  */
  template<class F>
  CUDA NI F deinterpret() const {
    return lb().template deinterpret<F>();
  }

  CUDA NI void print() const {
    printf("{");
    bool comma_needed = false;
    if(bits.test(0)) {
      printf(".., -1");
      comma_needed = true;
    }
    for(int i = 1; i < bits.size() - 1; ++i) {
      if(bits.test(i)) {
        if(comma_needed) { printf(", "); }
        printf("%d", i-1);
        comma_needed = true;
      }
    }
    if(bits.test(bits.size()-1)) {
      if(comma_needed) { printf(", "); }
      printf("%d, ..", static_cast<int>(bits.size())-2);
    }
    printf("}");
  }

  CUDA NI constexpr static bool is_trivial_fun(Sig sig) {
    switch(sig) {
      case ABS:
      case NEG: return false;
      default: return true;
    }
  }

public:
  CUDA constexpr void neg(const local_type& x) {
    // if `x` represents all negative numbers, then the negation is all positive numbers.
    if(x.bits.test(0)) {
      if(x.bits.count() == 1) {
        bits.set(0, false);
      }
    }
    else if(x.bits.count() == 0) {
      meet_bot();
    }
    else {
      meet(local_type(-1));
    }
  }

  CUDA constexpr void abs(const local_type& x) {
    // If the first bit is set, it means all negative numbers are represented, so it only constrains the current value to be positive. Otherwise, we just take the meet with `x`.
    if(x.bits.test(0)) {
      bits.set(0, false);
    }
    else {
      meet(x);
    }
  }

  CUDA constexpr void project(Sig fun, const local_type& x)  {
    switch(fun) {
      case NEG: neg(x); break;
      case ABS: abs(x); break;
    }
  }

  CUDA constexpr void additive_inverse(const local_type& x) {
    printf("%% additive_inverse is unsupported\n");
    int* ptr = nullptr;
    ptr[1] = 193;
  }

  CUDA constexpr void fun(Sig fun, const local_type& x, const local_type& y) {
    printf("%% binary functions %s are unsupported\n", string_of_sig(fun));
    int* ptr = nullptr;
    ptr[1] = 193;
  }

  CUDA constexpr local_type width() const {
    if(bits.test(0) || bits.test(bits.size() - 1)) { return top(); }
    else { return local_type(bits.count()); }
  }

  /** \return The median value of the bitset. */
  CUDA constexpr local_type median() const {
    if(is_bot()) { return local_type::bot(); }
    int total = bits.count();
    int current = 0;
    for(int i = 0; i < bits.size(); ++i) {
      if(bits.test(i)) {
        ++current;
        if(current == total/2 || total == 1) {
          return local_type(i-1);
        }
      }
    }
    return local_type::bot();
  }
};

// Lattice operations

template<size_t N, class M1, class M2, class T>
CUDA constexpr NBitset<N, battery::local_memory, T> fjoin(const NBitset<N, M1, T>& a, const NBitset<N, M2, T>& b)
{
  return NBitset<N, battery::local_memory, T>(a.value() | b.value());
}

template<size_t N, class M1, class M2, class T>
CUDA constexpr NBitset<N, battery::local_memory, T> fmeet(const NBitset<N, M1, T>& a, const NBitset<N, M2, T>& b)
{
  return NBitset<N, battery::local_memory, T>(a.value() & b.value());
}

template<size_t N, class M1, class M2, class T>
CUDA constexpr bool operator<=(const NBitset<N, M1, T>& a, const NBitset<N, M2, T>& b)
{
  return a.value().is_subset_of(b.value());
}

template<size_t N, class M1, class M2, class T>
CUDA constexpr bool operator<(const NBitset<N, M1, T>& a, const NBitset<N, M2, T>& b)
{
  return a.value().is_proper_subset_of(b.value());
}

template<size_t N, class M1, class M2, class T>
CUDA constexpr bool operator>=(const NBitset<N, M1, T>& a, const NBitset<N, M2, T>& b)
{
  return b <= a;
}

template<size_t N, class M1, class M2, class T>
CUDA constexpr bool operator>(const NBitset<N, M1, T>& a, const NBitset<N, M2, T>& b)
{
  return b < a;
}

template<size_t N, class M1, class M2, class T>
CUDA constexpr bool operator==(const NBitset<N, M1, T>& a, const NBitset<N, M2, T>& b)
{
  return a.value() == b.value();
}

template<size_t N, class M1, class M2, class T>
CUDA constexpr bool operator!=(const NBitset<N, M1, T>& a, const NBitset<N, M2, T>& b)
{
  return a.value() != b.value();
}

template<size_t N, class M, class T>
std::ostream& operator<<(std::ostream &s, const NBitset<N, M, T> &a) {
  s << "{";
  bool comma_needed = false;
  if(a.value().test(0)) {
    s << ".., -1";
    comma_needed = true;
  }
  for(int i = 1; i < a.value().size() - 1; ++i) {
    if(a.value().test(i)) {
      if(comma_needed) { s << ", "; }
      s << (i-1);
      comma_needed = true;
    }
  }
  if(a.value().test(a.value().size()-1)) {
    if(comma_needed) { s << ", "; }
    s << a.value().size()-2 << ", ..";
  }
  s << "}";
  return s;
}

} // end namespace lala

#endif
