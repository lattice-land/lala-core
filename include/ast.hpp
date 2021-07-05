// Copyright 2021 Pierre Talbot

#ifndef AST_HPP
#define AST_HPP

#include "utility.hpp"
#include "darray.hpp"
#include "string.hpp"
#include "variant.hpp"
#include "thrust/optional.h"

namespace lala {

/** Each abstract domain is uniquely identified by an UID. */
using AType = int;
/** This value means a formula is not typed in a particular abstract domain and its type should be inferred. */
#define UNTYPED (-1)

/** A "logical variable" is just the name of the variable. */
template<typename Allocator>
using LVar = String<Allocator>;

/** We call an "abstract variable" the representation of this variable in an abstract domain.
It is simply an integer containing the UID of the abstract element and an internal integer variable identifier proper to the abstract domain.
The mapping between logical variables and abstract variables is maintained in `Environment` below.
An abstract variable always has a single name (or no name if it is not explicitly represented in the initial formula).
However, a logical variable can be represented by several abstract variables when the variable occurs in different domains. */
typedef int AVar;
#define AID(v) (v & ((1 << 8) - 1))
#define VID(v) (v >> 8)

CUDA AVar make_var(AType type, int var_id);

/** The approximation of a formula in an abstract domain w.r.t. the concrete domain.
* `UNDER`: An under-approximating element contains only solutions but not necessarily all.
* `OVER`: An over-approximating element contains all solutions but not necessarily only solutions.
* `EXACT`: An exact element is both under- and over-approximating; it exactly represents the set of solutions. */
enum Approx {
  UNDER,
  OVER,
  EXACT
};

/** The symbols of the terms and predicates of a first-order signature.
We also include the symbol of logical connectors, and a `RAW` variant for possible extension of the signature.
For genericity purpose, we sometimes reuse the same symbols over different universes of discourse such as LEQ for the inclusion \f$ X \subseteq Y \f$ in the universe of sets. */
enum Sig {
  ///@{
  NEG, ADD, SUB, MUL, DIV, MOD, POW,  ///< Terms
  ///@}
  ///@{
  EQ, LEQ, GEQ, NEQ, GT, LT,     ///< Predicates
  ///@}
  ///@{
  AND, OR, IMPLY, EQUIV, NOT,    ///< Formulas
  ///@}
};
}

template<>
CUDA void print(const lala::Sig& sig);

namespace lala {

/** `TFormula` represents the AST of a typed multi-sorted first-order logical formula.
In our context, the type of a formula is an integer representing the UID of an abstract domain in which the formula should to be interpreted.
This integer can take the value `UNTYPED` if the formula is not (yet) typed.
By default, the sorts of integer and real number are supported (although constants are bounded).
The supported symbols can be extended with the template parameter `ExtendedSig`.
This extended signature can also be used for representing exactly constant such as real numbers using a string.
The AST of a formula is represented by a variant, where each alternative is described below.
We represent everything at the same level (terms, formula, predicate, variable, constant).
This is general convenient when modelling to avoid creating intermediate boolean variables when reifying.
We can have `x + (x > y \/ y > x + 4)` and this expression is true if the value is != 0. */
template<typename Allocator, typename ExtendedSig = String<Allocator>>
class TFormula {
public:
  using this_type = TFormula<Allocator, ExtendedSig>;
  using Sequence = DArray<this_type, Allocator>;
  using Formula = Variant<
    long long int, ///< Constant in the domain of discourse that can be represented exactly.
    std::tuple<double, double>,    ///< A real represented as an interval \f$ [d[0]..d[1]] \f$. Indeed, we sometimes cannot use a single `double` because it cannot represent all real numbers, it is up to the abstract domain to under- or over-approximate it, or choose an exact representation such as rational.
    AVar,          ///< Abstract variable
    std::tuple<Sig, Sequence>,  ///< ADD, SUB, ..., EQ, ..., AND, .., NOT
    std::tuple<ExtendedSig, Sequence> ///< see above
  >;

  static constexpr int Z = 0;
  static constexpr int R = 1;
  static constexpr int V = 2;
  static constexpr int Seq = 3;
  static constexpr int ESeq = 4;

private:
  AType type_;
  Formula formula;

public:
  /** By default, we initialize the formula to `true`. */
  CUDA TFormula(): type_(UNTYPED), formula(Formula::template create<Z>(1)) {}
  CUDA TFormula(Formula&& formula): type_(UNTYPED), formula(std::forward<Formula>(formula)) {}
  CUDA TFormula(AType uid, Formula&& formula): type_(uid), formula(std::forward<Formula>(formula)) {}

  CUDA TFormula(const this_type& other): type_(other.type_), formula(other.formula) {}
  CUDA TFormula(this_type&& other): type_(other.type_), formula(std::move(other.formula)) {}

  CUDA this_type& operator=(this_type& rhs) {
    type_ = rhs.type_;
    formula = rhs.formula;
    return *this;
  }

  CUDA this_type& operator=(this_type&& rhs) {
    type_ = rhs.type_;
    formula = std::move(rhs.formula);
    return *this;
  }

  CUDA Formula& data() { return formula; }
  CUDA const Formula& data() const { return formula; }
  CUDA AType type() const { return type_; }
  CUDA void type_as(AType ty) {
    type_ = ty;
    if(is(V)) {
      v() = make_var(ty, VID(v()));
    }
  }

  /** The formula `true` is represented by the integer constant `1`. */
  CUDA static this_type make_true() { return TFormula(); }
  /** The formula `false` is represented by the integer constant `0`. */
  CUDA static this_type make_false() { return TFormula(Formula::template create<Z>(0)); }

  CUDA static this_type make_z(long long int i, AType atype = UNTYPED) {
    return this_type(atype, Formula::template create<Z>(i));
  }

  CUDA static this_type make_real(double lb, double ub, AType atype = UNTYPED) {
    return this_type(atype, Formula::template create<R>(std::make_tuple(lb, ub)));
  }

  /** The type of the formula is embedded in `v`. */
  CUDA static this_type make_avar(AVar v) {
    return this_type(AID(v), Formula::template create<V>(v));
  }

  CUDA static this_type make_avar(AType ty, int vid) {
    return make_avar(make_var(ty, vid));
  }

  CUDA static this_type make_nary(Sig sig, Sequence children, AType atype = UNTYPED, const Allocator& allocator = Allocator()) {
    return this_type(atype, Formula::template create<Seq>(std::make_tuple(sig, children)));
  }

  CUDA static this_type make_unary(Sig sig, TFormula child, AType atype = UNTYPED, const Allocator& allocator = Allocator()) {
    Sequence children(1, allocator);
    children[0] = std::move(child);
    return make_nary(sig, std::move(children), atype, allocator);
  }

  CUDA static this_type make_binary(TFormula lhs, Sig sig, TFormula rhs, AType atype = UNTYPED, const Allocator& allocator = Allocator()) {
    Sequence children(2, allocator);
    children[0] = std::move(lhs);
    children[1] = std::move(rhs);
    return make_nary(sig, std::move(children), atype, allocator);
  }


  CUDA static this_type make_nary(ExtendedSig esig, Sequence children, AType atype = UNTYPED, const Allocator& allocator = Allocator()) {
    return this_type(atype, Formula::create<ESeq>(std::make_tuple(std::move(esig), std::move(children))));
  }

  CUDA bool is(int kind) const {
    return formula.index() == kind;
  }

  CUDA bool is_true() const {
    return is(Z) && z() != 0;
  }

  CUDA bool is_false() const {
    return is(Z) && z() == 0;
  }

  CUDA long long int z() const {
    return get<Z>(formula);
  }

  CUDA const std::tuple<double, double>& r() const {
    return get<R>(formula);
  }

  CUDA AVar v() const {
    return get<V>(formula);
  }

  CUDA Sig sig() const {
    return std::get<0>(get<Seq>(formula));
  }

  CUDA const ExtendedSig& esig() const {
    return std::get<0>(get<ESeq>(formula));
  }

  CUDA const Sequence& seq() const {
    return std::get<1>(get<Seq>(formula));
  }

  CUDA const this_type& seq(size_t i) const {
    return seq()[i];
  }

  CUDA const Sequence& eseq() const {
    return std::get<1>(get<ESeq>(formula));
  }

  CUDA const this_type& eseq(size_t i) const {
    return eseq()[i];
  }

  CUDA long long int& z() {
    return get<Z>(formula);
  }

  CUDA std::tuple<double, double>& r() {
    return get<R>(formula);
  }

  CUDA AVar& v() {
    return get<V>(formula);
  }

  CUDA Sig& sig() {
    return std::get<0>(get<Seq>(formula));
  }

  CUDA ExtendedSig& esig() {
    return std::get<0>(get<ESeq>(formula));
  }

  CUDA Sequence& seq() {
    return std::get<1>(get<Seq>(formula));
  }

  CUDA this_type& seq(size_t i) {
    return seq()[i];
  }

  CUDA Sequence& eseq() {
    return std::get<1>(get<ESeq>(formula));
  }

  CUDA this_type& eseq(size_t i) {
    return eseq()[i];
  }


private:
  template<size_t n>
  CUDA void print_sequence() const {
    auto op = std::get<0>(get<n>(formula));
    auto children = std::get<1>(get<n>(formula));
    assert(children.size() > 0);
    if(children.size() == 1) {
      ::print(op);
      ::print(children[0]);
    }
    else {
      printf("(");
      for(int i = 0; i < children.size(); ++i) {
        ::print(children[i]);
        if(i < children.size() - 1) {
          printf(" ");
          ::print(op);
          printf(" ");
        }
      }
      printf(")");
    }
  }

public:
  CUDA void print(bool print_types = true) const {
    switch(formula.index()) {
      case Z: printf("%lld:long", z()); break;
      case R: printf("[%lf..%lf]", get<0>(r()), get<1>(r())); break;
      case V: printf("%d:var", v()); break;
      case Seq: print_sequence<Seq>(); break;
      case ESeq: print_sequence<ESeq>(); break;
      default: assert(0); break;
    }
    if(print_types) {
      printf(":%d", type_);
    }
  }
};

template<typename Allocator, typename ExtendedSig>
CUDA bool operator==(const TFormula<Allocator, ExtendedSig>& lhs, const TFormula<Allocator, ExtendedSig>& rhs) {
  return lhs.type() == rhs.type() && lhs.data() == rhs.data();
}

/** \return `true` if the formula `f` has the shape `variable op constant`, e.g., `x < 4`. */
template<typename Allocator, typename ExtendedSig>
CUDA bool is_v_op_z(const TFormula<Allocator, ExtendedSig>& f, Sig sig) {
  using F = TFormula<Allocator, ExtendedSig>;
  return f.is(F::Seq)
    && f.sig() == sig
    && f.seq(0).is(F::V)
    && f.seq(1).is(F::Z);
}

template<typename Allocator>
CUDA TFormula<Allocator> make_v_op_z(AVar x, Sig sig, long long int i, const Allocator& allocator = Allocator()) {
  typedef TFormula<Allocator> F;
  return F::make_binary(F::make_avar(x), sig, F::make_z(i), UNTYPED, allocator);
}

/** `SFormula` is a formula to be solved with a possible optimisation mode (MINIMIZE or MAXIMIZE), otherwise it will enumerate `n` satisfiable solutions, if any. */
template<typename Allocator>
struct SFormula {
  enum {
    MINIMIZE,
    MAXIMIZE,
    SATISFY
  } tag;

  union {
    LVar<Allocator> lv;  ///< The logical variable to optimize.
    AVar av;  ///< The abstract variable to optimize. (We use this one after the variable has been added to an abstract element).
    int num_sols; ///< How many solutions should we compute (SATISFY mode).
  };

  TFormula<Allocator> f;
};

/** A `VarEnv` is a variable environment mapping between logical variables and abstract variables.
This class is supposed to be used inside an abstract domain, to help with the conversion. */
template<typename Allocator>
class VarEnv {
  typedef LVar<Allocator> vname;
  typedef DArray<LVar<Allocator>, Allocator> env_type;

  AType uid;
  /** Given an abstract variable `v`, `avar2lvar[VID(v)]` is the name of the variable. */
  env_type avar2lvar;
  /** This is the number of variables in the environment. */
  size_t size_;

public:
  CUDA VarEnv(AType uid, int capacity): uid(uid), avar2lvar(capacity), size_(0) {}

  CUDA const vname& to_lvar(AVar av) const {
    assert(VID(av) < size_);
    return avar2lvar[VID(av)];
  }

  CUDA thrust::optional<AVar> to_avar(const vname& lv) const {
    AVar i = 0;
    for(; i < size_; ++i) {
      if(avar2lvar[i] == lv) {
        return make_var(uid, i);
      }
    }
    return {};
  }

  CUDA AVar add(vname lv) {
    assert(size() < capacity());
    avar2lvar[size_++] = std::move(lv);
    return make_var(uid, size_ - 1);
  }

  CUDA AType ad_uid() const {
    return uid;
  }

  CUDA size_t capacity() const {
    return avar2lvar.size();
  }

  CUDA size_t size() const {
    return size_;
  }
};

} // namespace lala
#endif
