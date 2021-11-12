// Copyright 2021 Pierre Talbot

#ifndef Z_HPP
#define Z_HPP

#include "thrust/optional.h"
#include "utility.hpp"
#include "darray.hpp"
#include "ast.hpp"

namespace lala {

/** The lattice of increasing integers.
Concretization function: \f$ \gamma(x) = \{_ \mapsto y \;|\; x \leq y\} \f$. */
template<typename VT, typename Alloc>
class ZInc {
public:
  using ValueType = VT;
  using Allocator = Alloc;
  using this_type = ZInc<ValueType, Allocator>;
private:
  VT val;

  CUDA ZInc() {}
public:
  /** Similar to \f$[\![\mathit{true}]\!]\f$. */
  CUDA static ZInc bot() {
    ZInc zi;
    zi.val = Limits<ValueType>::bot();
    return zi;
  }

  /** Similar to \f$[\![\mathit{false}]\!]\f$. */
  CUDA static ZInc top() {
    ZInc zi;
    zi.val = Limits<ValueType>::top();
    return zi;
  }

  CUDA const ValueType& value() const { return val; }
  CUDA explicit operator ValueType() const { return val; }

  template<typename T, typename U>
  using IsConvertible = std::enable_if_t<std::is_convertible_v<T, U>, bool>;

  /** Similar to \f$[\![x \geq i]\!]\f$ for any name `x`. */
  template<typename VT2, IsConvertible<VT2, ValueType> = true>
  CUDA explicit ZInc(VT2 i): val(static_cast<ValueType>(i)) {
    assert(i > bot().val && i < top().val);
  }

  template<typename B>
  CUDA bool operator==(const B& other) const {
    return val == other.value();
  }

  template<typename B>
  CUDA bool operator!=(const B& other) const {
    return val != other.value();
  }

  /** Expects a predicate of the form `x <op> i` where `x` is any variable's name, and `i` an integer.
    - If `appx` is EXACT: `op` can be >= or >.
    - If `appx` is UNDER: `op` can be >=, > or !=.
    - If `appx` is OVER: `op` can be >=, > or ==.
    */
  template<typename Formula>
  CUDA static thrust::optional<this_type> interpret(Approx appx, const Formula& f) {
    if(f.is_true()) {
      return this_type(bot().value());
    }
    else if(f.is_false()) {
      return this_type(top().value());
    }
    if(is_v_op_z(f, GEQ)) {      // x >= 4
      return this_type(f.seq(1).z());
    }
    else if(is_v_op_z(f, GT)) {  // x > 4
      return this_type(f.seq(1).z() + 1);
    }
    // Under-approximation of `x != 4` as `5`.
    else if(is_v_op_z(f, NEQ) && appx == UNDER) {
      return this_type(f.seq(1).z() + 1);
    }
    // Over-approximation of `x == 4` as `4`.
    else if(is_v_op_z(f, EQ) && appx == OVER) {
      return this_type(f.seq(1).z());
    }
    return {};
  }

  /** `true` whenever \f$ a = \top \f$, `false` otherwise. */
  CUDA bool is_top() const {
    return val == Limits<ValueType>::top();
  }

  /** `true` whenever \f$ a = \bot \f$, `false` otherwise. */
  CUDA bool is_bot() const {
    return val == Limits<ValueType>::bot();
  }

  /** \f$ a \sqcup b = \mathit{max}(a,b) \f$. */
  CUDA this_type& join(const this_type& other) {
    val = max(other.val, val);
    return *this;
  }

  /** \f$ a \sqcap b = \mathit{min}(a,b) \f$. */
  CUDA this_type& meet(const this_type& other) {
    val = min(other.val, val);
    return *this;
  }

  /** \f$ a \leq b\f$ is defined by the natural arithmetic order. */
  CUDA bool order(const this_type& other) const {
    return val <= other.val;
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
    val = other.val;
  }

  /** \return A copy of the current abstract element. */
  CUDA this_type clone() const {
    return *this;
  }

  /** \return \f$ x \geq i \f$ where `x` is a variable's name and `i` the integer value.
  `true` is returned whenever \f$ a = \bot \f$ and `false` whenever \f$ a = \top \f$. */
  template<typename Allocator = Alloc>
  CUDA TFormula<Allocator> deinterpret(AVar x, const Allocator& allocator = Allocator()) const {
    if(is_top()) {
      return TFormula<Allocator>::make_false();
    }
    else if(is_bot()) {
      return TFormula<Allocator>::make_true();
    }
    return make_v_op_z(x, GEQ, val, allocator);
  }

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

/** The lattice of decreasing integers.
Concretization function: \f$ \gamma(x) = \{_ \mapsto y \;|\; x \geq y\} \f$. */
template<typename VT, typename Alloc>
class ZDec {
public:
  using ValueType = VT;
  using Allocator = Alloc;
  using this_type = ZDec<ValueType, Allocator>;
private:
  VT val;

  CUDA ZDec() {}
public:
  /** Similar to \f$[\![\mathit{true}]\!]\f$. */
  CUDA static ZDec bot() {
    ZDec zd;
    zd.val = Limits<ValueType>::top();
    return zd;
  }

  /** Similar to \f$[\![\mathit{false}]\!]\f$. */
  CUDA static ZDec top() {
    ZDec zd;
    zd.val = Limits<ValueType>::bot();
    return zd;
  }

  CUDA const ValueType& value() const { return val; }
  CUDA explicit operator ValueType() const { return val; }

  template<typename T, typename U>
  using IsConvertible = std::enable_if_t<std::is_convertible_v<T, U>, bool>;

  /** Similar to \f$[\![x \geq i]\!]\f$ for any name `x`. */
  template<typename VT2, IsConvertible<VT2, ValueType> = true>
  CUDA explicit ZDec(VT2 i): val(static_cast<ValueType>(i)) {
    assert(i > top().val && i < bot().val);
  }

  template<typename B>
  CUDA bool operator==(const B& other) const {
    return val == other.value();
  }

  template<typename B>
  CUDA bool operator!=(const B& other) const {
    return val != other.value();
  }

  /** Expects a predicate of the form `x <op> i` where `x` is any variable's name, and `i` an integer.
    - If `appx` is EXACT: `op` can be >= or >.
    - If `appx` is UNDER: `op` can be >=, > or !=.
    - If `appx` is OVER: `op` can be >=, > or ==.
    */
  template<typename Formula>
  CUDA static thrust::optional<this_type> interpret(Approx appx, const Formula& f) {
    if(f.is_true()) {
      return this_type(bot().value());
    }
    else if(f.is_false()) {
      return this_type(top().value());
    }
    if(is_v_op_z(f, LEQ)) {      // x <= 4
      return this_type(f.seq(1).z());
    }
    else if(is_v_op_z(f, LT)) {  // x < 4
      return this_type(f.seq(1).z() - 1);
    }
    // Under-approximation of `x != 4` as `3`.
    else if(is_v_op_z(f, NEQ) && appx == UNDER) {
      return this_type(f.seq(1).z() - 1);
    }
    // Over-approximation of `x == 4` as `4`.
    else if(is_v_op_z(f, EQ) && appx == OVER) {
      return this_type(f.seq(1).z());
    }
    return {};
  }

  /** `true` whenever \f$ a = \top \f$, `false` otherwise. */
  CUDA bool is_top() const {
    return val == Limits<ValueType>::bot();
  }

  /** `true` whenever \f$ a = \bot \f$, `false` otherwise. */
  CUDA bool is_bot() const {
    return val == Limits<ValueType>::top();
  }

  /** \f$ a \sqcup b = \mathit{min}(a,b) \f$. */
  CUDA this_type& join(const this_type& other) {
    val = min(other.val, val);
    return *this;
  }

  /** \f$ a \sqcap b = \mathit{max}(a,b) \f$. */
  CUDA this_type& meet(const this_type& other) {
    val = max(other.val, val);
    return *this;
  }

  /**  \f$ a \leq_{ZDec} b\f$ is defined by the inverse arithmetic order \f$ \geq \f$. */
  CUDA bool order(const this_type& other) const {
    return val >= other.val;
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
    val = other.val;
  }

  /** \return A copy of the current abstract element. */
  CUDA this_type clone() const {
    return *this;
  }

  /** \return \f$ x \leq i \f$ where `x` is a variable's name and `i` the integer value.
  `true` is returned whenever \f$ a = \bot \f$ and `false` whenever \f$ a = \top \f$. */
  template<typename Allocator = Alloc>
  CUDA TFormula<Allocator> deinterpret(AVar x, const Allocator& allocator = Allocator()) const {
    if(is_top()) {
      return TFormula<Allocator>::make_false();
    }
    else if(is_bot()) {
      return TFormula<Allocator>::make_true();
    }
    return make_v_op_z(x, LEQ, val, allocator);
  }

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

} // namespace lala

#endif