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
  VT value;

  CUDA ZInc() {}
public:
  using LogicalElement = this_type;

  /** Similar to \f$[\![\mathit{true}]\!]\f$. */
  CUDA static ZInc bot() {
    ZInc zi;
    zi.value = Limits<ValueType>::bot();
    return zi;
  }

  /** Similar to \f$[\![\mathit{false}]\!]\f$. */
  CUDA static ZInc top() {
    ZInc zi;
    zi.value = Limits<ValueType>::top();
    return zi;
  }

  CUDA explicit operator ValueType() const { return value; }

  template<typename T, typename U>
  using IsConvertible = std::enable_if_t<std::is_convertible_v<T, U>, bool>;

  /** Similar to \f$[\![x \geq i]\!]\f$ for any name `x`. */
  template<typename VT2, IsConvertible<VT2, ValueType> = true>
  CUDA explicit ZInc(VT2 i): value(static_cast<ValueType>(i)) {
    assert(i > bot().value && i < top().value);
  }

  template<typename Allocator2>
  CUDA ZInc(const ZInc<ValueType, Allocator2>& i): value(ValueType(i)) {}
  CUDA ZInc(const this_type& i): value(i.value) {}

  CUDA bool operator==(const this_type& other) const {
    return value == other.value;
  }

  /** Expects a predicate of the form `x <op> i` where `x` is any variable's name, and `i` an integer.
    - If `appx` is EXACT: `op` can be >= or >.
    - If `appx` is UNDER: `op` can be >=, > or !=.
    - If `appx` is OVER: `op` can be >=, > or ==.
    */
  template<typename Formula>
  CUDA thrust::optional<LogicalElement> interpret(Approx appx, const Formula& f) {
    if(f.is_true()) {
      return bot();
    }
    else if(f.is_false()) {
      return top();
    }
    if(is_v_op_z(f, GEQ)) {      // x >= 4
      return ZInc(f.seq(1).z());
    }
    else if(is_v_op_z(f, GT)) {  // x > 4
      return ZInc(f.seq(1).z() + 1);
    }
    // Under-approximation of `x != 4` as `5`.
    else if(is_v_op_z(f, NEQ) && appx == UNDER) {
      return ZInc(f.seq(1).z() + 1);
    }
    // Over-approximation of `x == 4` as `4`.
    else if(is_v_op_z(f, EQ) && appx == OVER) {
      return ZInc(f.seq(1).z());
    }
    return {};
  }

  /** `true` whenever \f$ a = \top \f$, `false` otherwise. */
  CUDA bool is_top() const {
    return value == Limits<ValueType>::top();
  }

  /** `true` whenever \f$ a = \bot \f$, `false` otherwise. */
  CUDA bool is_bot() const {
    return value == Limits<ValueType>::bot();
  }

  /** \f$ a \sqcup b = \mathit{max}(a,b) \f$. */
  CUDA this_type& join(const LogicalElement& other) {
    value = max(other.value, value);
    return *this;
  }

  /** \f$ a \sqcap b = \mathit{min}(a,b) \f$. */
  CUDA this_type& meet(const LogicalElement& other) {
    value = min(other.value, value);
    return *this;
  }

  /** Has no effect.
     \return always `false`. */
  CUDA bool refine() { return false; }

  /** \f$ a \models \varphi \f$ is defined as \f$ a \geq [\![\varphi]\!] \f$. */
  CUDA bool entailment(const LogicalElement& other) const {
    return value >= ValueType(other);
  }

  template<typename Allocator = Alloc>
  CUDA DArray<LogicalElement, Allocator> split(const Allocator& allocator = Allocator()) const {
    if(is_top()) {
      return DArray<LogicalElement, Allocator>();
    }
    else {
      return DArray<LogicalElement, Allocator>(1, *this, allocator);
    }
  }

  /** Reset the internal counter to the one of `other`. */
  CUDA void reset(const this_type& other) {
    value = other.value;
  }

  /** \return A copy of the current abstract element. */
  CUDA this_type clone() const {
    return *this;
  }

  /** \return \f$ _ \geq i \f$ where `_` is an arbitrary variable's name and `i` the integer value.
  `true` is returned whenever \f$ a = \bot \f$ and `false` whenever \f$ a = \top \f$. */
  template<typename Allocator = Alloc>
  CUDA TFormula<Allocator> deinterpret(const Allocator& allocator = Allocator()) const {
    if(is_top()) {
      return TFormula<Allocator>::make_false();
    }
    else if(is_bot()) {
      return TFormula<Allocator>::make_true();
    }
    return make_v_op_z(0, GEQ, value, allocator);
  }

  /** Print the current element. */
  CUDA void print() const {
    if(is_bot()) {
      printf("%c", 0x22A5);
    }
    else if(is_top()) {
      printf("%c", 0x22A4);
    }
    else if(value >= 0) {
      printf("%llu", (unsigned long long int) value);
    }
    else {
      printf("%lld", (long long int) value);
    }
  }

  template <typename VT2, typename Alloc2>
  friend class ZDec;
};

/** The lattice of decreasing integers.
Concretization function: \f$ \gamma(x) = \{_ \mapsto y \;|\; x \geq y\} \f$. */
template<typename VT, typename Alloc>
class ZDec {
public:
  /** The dual lattice of ZDec.
   * Note, however, that the interpretation function is not dually equivalent.
   * This is still not very clear what is the dual of an abstract domain as a whole. */
  using DualType = ZInc<VT, Alloc>;
  using ValueType = typename DualType::ValueType;
  using Allocator = typename DualType::Allocator;
  using this_type = ZDec<ValueType, Allocator>;
private:
  DualType dual;

  CUDA ZDec() {}
  CUDA ZDec(DualType dual): dual(dual) {}
public:
  using LogicalElement = this_type;

  /** Similar to \f$[\![\mathit{true}]\!]\f$. */
  CUDA static ZDec bot() {
    return ZDec(DualType::top());
  }

  /** Similar to \f$[\![\mathit{false}]\!]\f$. */
  CUDA static ZDec top() {
    return ZDec(DualType::bot());
  }

  CUDA explicit operator ValueType() const { return static_cast<ValueType>(dual); }

  /** Similar to \f$[\![x \geq i]\!]\f$ for any name `x`. */
  template<typename VT2>
  CUDA ZDec(VT2 i): dual(i) {}

  template<typename Allocator2>
  CUDA ZDec(const ZDec<ValueType, Allocator2>& i): dual(i.dual) {}
  CUDA ZDec(const this_type& i): dual(i.dual) {}

  CUDA bool operator==(const this_type& other) const {
    return dual == other.dual;
  }

  /** Expects a predicate of the form `x <op> i` where `x` is any variable's name, and `i` an integer.
    - If `appx` is EXACT: `op` can be <= or <.
    - If `appx` is UNDER: `op` can be <=, < or !=.
    - If `appx` is OVER: `op` can be <=, < or ==.
    */
  template<typename Formula>
  CUDA thrust::optional<LogicalElement> interpret(Approx appx, const Formula& f) {
    if(f.is_true()) {
      return bot();
    }
    else if(f.is_false()) {
      return top();
    }
    if(is_v_op_z(f, LEQ)) {      // x <= 4
      return ZDec(f.seq(1).z());
    }
    else if(is_v_op_z(f, LT)) {  // x < 4
      return ZDec(f.seq(1).z() - 1);
    }
    // Under-approximation of `x != 4` as `3`.
    else if(is_v_op_z(f, NEQ) && appx == UNDER) {
      return ZDec(f.seq(1).z() - 1);
    }
    // Over-approximation of `x == 4` as `4`.
    else if(is_v_op_z(f, EQ) && appx == OVER) {
      return ZDec(f.seq(1).z());
    }
    return {};
  }

  /** `true` whenever \f$ a = \top \f$, `false` otherwise. */
  CUDA bool is_top() const {
    return dual.is_bot();
  }

  /** `true` whenever \f$ a = \bot \f$, `false` otherwise. */
  CUDA bool is_bot() const {
    return dual.is_top();
  }

  /** \f$ a \sqcup b = \mathit{min}(a,b) \f$. */
  CUDA this_type& join(const LogicalElement& other) {
    dual.meet(other.dual);
    return *this;
  }

  /** \f$ a \sqcap b = \mathit{max}(a,b) \f$. */
  CUDA this_type& meet(const LogicalElement& other) {
    dual.join(other.dual);
    return *this;
  }

  /** Has no effect.
     \return always `false`. */
  CUDA bool refine() { return false; }

  /** \f$ a \models \varphi \f$ is defined as \f$ a \leq [\![\varphi]\!] \f$. */
  CUDA bool entailment(const LogicalElement& other) const {
    return ValueType(dual) <= ValueType(other);
  }

  template<typename Allocator = Alloc>
  CUDA DArray<LogicalElement, Allocator> split(const Allocator& allocator = Allocator()) const {
    if(is_top()) {
      return DArray<LogicalElement, Allocator>();
    }
    else {
      return DArray<LogicalElement, Allocator>(1, *this, allocator);
    }
  }

  /** Reset the internal counter to the one of `other`. */
  CUDA void reset(const this_type& other) {
    dual.reset(other.dual);
  }

  /** \return A copy of the current abstract element. */
  CUDA this_type clone() const {
    return *this;
  }

  /** \return \f$ _ \leq i \f$ where `_` is an arbitrary variable's name and `i` the integer value.
  `true` is returned whenever \f$ a = \bot \f$ and `false` whenever \f$ a = \top \f$. */
  template<typename Allocator = Alloc>
  CUDA TFormula<Allocator> deinterpret(const Allocator& allocator = Allocator()) const {
    if(is_top()) {
      return TFormula<Allocator>::make_false();
    }
    else if(is_bot()) {
      return TFormula<Allocator>::make_true();
    }
    return make_v_op_z(0, LEQ, ValueType(dual), allocator);
  }

  /** Print the current element. */
  CUDA void print() const {
    if(is_bot()) {
      printf("%c", 0x22A5);
    }
    else if(is_top()) {
      printf("%c", 0x22A4);
    }
    else if(ValueType(dual) >= 0) {
      printf("%llu", (unsigned long long int) ValueType(dual));
    }
    else {
      printf("%lld", (long long int) ValueType(dual));
    }
  }
};

} // namespace lala

#endif