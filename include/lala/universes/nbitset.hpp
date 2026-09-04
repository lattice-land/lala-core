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
  constexpr static const bool is_arithmetic = true;
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
    bits = other.bits;
    return *this;
  }

  CUDA constexpr this_type& operator=(const this_type& other) {
    bits = other.bits;
    return *this;
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

  /** The two extremal bits of the bitset stand for "some value below the representable range" and
   * "some value above it". Joining them is the only write an interpretation needs to perform on
   * the raw bits, so we expose it as an operation rather than exposing the bitset itself. */
  CUDA constexpr this_type& join_out_of_range() {
    bits.set(0, true);
    bits.set(bits.size() - 1, true);
    return *this;
  }

  /** Number of bits, i.e. the number of values representable by this bitset (including the two
   * out-of-range flags). Values in `[0, capacity()-3]` are represented exactly. */
  CUDA constexpr static int capacity() { return N; }

private:




public:



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

  template<class A>
  CUDA constexpr bool meet_lb(const A& lb) {
    return meet(local_type(lb.value(), bits.size()));
  }

  template<class A>
  CUDA constexpr bool meet_ub(const A& ub) {
    return meet(local_type(-1, ub.value()));
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

  CUDA constexpr void project(Sig fun, const local_type& x, const local_type& y) {
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
