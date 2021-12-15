// Copyright 2021 Pierre Talbot

#ifndef Z_HPP
#define Z_HPP

#include "thrust/optional.h"
#include "utility.hpp"
#include "darray.hpp"
#include "ast.hpp"

namespace lala {

template<typename VT>
struct ZIncUniverse {
  using ValueType = VT;
  static ValueType next(ValueType i) { return i + 1; }
  static ValueType bot() { return Limits<ValueType>::bot(); }
  static ValueType top() { return Limits<ValueType>::top(); }
  static ValueType join(ValueType x, ValueType y) { return max(x, y); }
  static ValueType meet(ValueType x, ValueType y) { return min(x, y); }
  static bool order(ValueType x, ValueType y) { return x <= y; }
  static bool strict_order(ValueType x, ValueType y) { return x < y; }
  static Sig sig_order() { return GEQ; }
  static Sig sig_strict_order() { return GT; }
};

template<typename VT>
struct ZDecUniverse {
  using ValueType = VT;
  static ValueType next(ValueType i) { return i - 1; }
  static ValueType bot() { return Limits<ValueType>::top(); }
  static ValueType top() { return Limits<ValueType>::bot(); }
  static ValueType join(ValueType x, ValueType y) { return min(x, y); }
  static ValueType meet(ValueType x, ValueType y) { return max(x, y); }
  static bool order(ValueType x, ValueType y) { return x >= y; }
  static bool strict_order(ValueType x, ValueType y) { return x > y; }
  static Sig sig_order() { return LEQ; }
  static Sig sig_strict_order() { return LT; }
};

template<typename ZUniverse, typename Alloc>
class ZTotalOrder {
public:
  using ValueType = typename ZUniverse::ValueType;
  using Allocator = Alloc;
  using this_type = ZTotalOrder<ZUniverse, Alloc>;
private:
  using U = ZUniverse;

  ValueType val;

  CUDA ZTotalOrder() {}
public:
  /** Similar to \f$[\![\mathit{true}]\!]\f$. */
  CUDA static this_type bot() {
    this_type a;
    a.val = U::bot();
    return a;
  }

  /** Similar to \f$[\![\mathit{false}]\!]\f$. */
  CUDA static this_type top() {
    this_type a;
    a.val = U::top();
    return a;
  }

  template<typename T, typename U>
  using IsConvertible = std::enable_if_t<std::is_convertible_v<T, U>, bool>;

  /** Similar to \f$[\![x \geq_A i]\!]\f$ for any name `x` where \f$ \geq_A \f$ is the lattice order. */
  template<typename VT2, IsConvertible<VT2, ValueType> = true>
  CUDA explicit ZTotalOrder(VT2 i): val(static_cast<ValueType>(i)) {
    assert(U::strict_order(U::bot(), i) && U::strict_order(i, U::top()));
  }

  CUDA const ValueType& value() const { return val; }
  CUDA explicit operator ValueType() const { return val; }

  /** Expects a predicate of the form `x <op> i` where `x` is any variable's name, and `i` an integer.
    - If `appx` is EXACT: `op` can be `U::sig_order()` or `U::sig_strict_order()`.
    - If `appx` is UNDER: `op` can be, in addition to exact, `!=`.
    - If `appx` is OVER: `op` can be, in addition to exact, `==`.
    */
  template<typename Formula>
  CUDA static thrust::optional<this_type> interpret(const Formula& f) {
    if(f.is_true()) {
      return bot();
    }
    else if(f.is_false()) {
      return top();
    }
    if(is_v_op_z(f, U::sig_order())) {      // e.g., x <= 4
      return this_type(f.seq(1).z());
    }
    else if(is_v_op_z(f, U::sig_strict_order())) {  // e.g., x < 4
      return this_type(U::next(f.seq(1).z()));
    }
    // Under-approximation of `x != 4` as `next(4)`.
    else if(is_v_op_z(f, NEQ) && f.approx() == UNDER) {
      return this_type(U::next(f.seq(1).z()));
    }
    // Over-approximation of `x == 4` as `4`.
    else if(is_v_op_z(f, EQ) && f.approx() == OVER) {
      return this_type(f.seq(1).z());
    }
    return {};
  }

  template<typename B>
  CUDA bool operator==(const B& other) const {
    return val == other.value();
  }

  template<typename B>
  CUDA bool operator!=(const B& other) const {
    return val != other.value();
  }

  /** `true` whenever \f$ a = \top \f$, `false` otherwise. */
  CUDA bool is_top() const {
    return val == U::top();
  }

  /** `true` whenever \f$ a = \bot \f$, `false` otherwise. */
  CUDA bool is_bot() const {
    return val == U::bot();
  }

  /** \f$ a \sqcup b \f$ is defined by `U::join`. */
  CUDA this_type& join(const this_type& other) {
    this->val = U::join(other.val, this->val);
    return *this;
  }

  /** \f$ a \sqcap b \f$ is defined by `U::meet`. */
  CUDA this_type& meet(const this_type& other) {
    this->val = U::meet(other.val, this->val);
    return *this;
  }

  CUDA this_type& tell(const this_type& other, bool& has_changed) {
    if(U::strict_order(this->val, other.val)) {
      this->val = other.val;
      has_changed = true;
    }
    return *this;
  }

  CUDA this_type& dtell(const this_type& other, bool& has_changed) {
    if(U::strict_order(other.val, this->val)) {
      this->val = other.val;
      has_changed = true;
    }
    return *this;
  }

  /** \f$ a \leq b\f$ is defined by `U::order`. */
  CUDA bool order(const this_type& other) const {
    return U::order(this->val, other.val);
  }

  /** \return \f$ x \geq i \f$ where `x` is a variable's name and `i` the integer value.
  `true` is returned whenever \f$ a = \bot \f$ and `false` whenever \f$ a = \top \f$. */
  template<typename Allocator = Alloc>
  CUDA TFormula<Allocator> deinterpret(const LVar<Allocator>& x, const Allocator& allocator = Allocator()) const {
    if(is_top()) {
      return TFormula<Allocator>::make_false();
    }
    else if(is_bot()) {
      return TFormula<Allocator>::make_true();
    }
    return make_v_op_z(x, U::sig_order(), val, EXACT, allocator);
  }

  template<typename Allocator = Alloc>
  CUDA DArray<this_type, Allocator> split(const Allocator& allocator = Allocator()) const {
    if(is_top()) {
      return DArray<this_type, Allocator>();
    }
    else {
      return DArray<this_type, Allocator>(1, *this, allocator);
    }
  }

  /** Reset the internal counter to the one of `other`. */
  CUDA void reset(const this_type& other) {
    val = other.value();
  }

  /** \return A copy of the current abstract element. */
  CUDA this_type clone() const { return *this; }

  /** Print the current element. */
  CUDA void print() const {
    if(is_bot()) {
      printf("%c", 0x22A5);
    }
    else if(is_top()) {
      printf("%c", 0x22A4);
    }
    else if(val >= 0) {
      printf("%llu", (unsigned long long int) val);
    }
    else {
      printf("%lld", (long long int) val);
    }
  }
};

/** The lattice of increasing integers.
Concretization function: \f$ \gamma(x) = \{_ \mapsto y \;|\; x \leq y\} \f$. */
template<typename VT, typename Alloc>
using ZInc = ZTotalOrder<ZIncUniverse<VT>, Alloc>;

/** The lattice of decreasing integers.
Concretization function: \f$ \gamma(x) = \{_ \mapsto y \;|\; x \geq y\} \f$. */
template<typename VT, typename Alloc>
using ZDec = ZTotalOrder<ZDecUniverse<VT>, Alloc>;

} // namespace lala

#endif
