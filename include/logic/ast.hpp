// Copyright 2021 Pierre Talbot

#ifndef AST_HPP
#define AST_HPP

#include "utility.hpp"
#include "vector.hpp"
#include "string.hpp"
#include "tuple.hpp"
#include "variant.hpp"
#include "shared_ptr.hpp"
#include "unique_ptr.hpp"
#include "thrust/optional.h"
#include "logic/types.hpp"

namespace lala {

/** A "logical variable" is just the name of the variable. */
template<class Allocator>
using LVar = battery::String<Allocator>;

/** We call an "abstract variable" the representation of a logical variable in an abstract domain.
It is a pair of integers `(aid, vid)` where `aid` is the UID of the abstract element and `vid` is an internal identifier of the variable inside the abstract element.
The mapping between logical variables and abstract variables is maintained in `VarEnv` below.
An abstract variable always has a single name (or no name if it is not explicitly represented in the initial formula).
However, a logical variable can be represented by several abstract variables when the variable occurs in different abstract elements. */
using AVar = int;
#define AID(v) (v & ((1 << 8) - 1))
#define VID(v) (v >> 8)

CUDA inline AVar make_var(AType atype, int var_id) {
  assert(atype >= 0);
  assert(var_id >= 0);
  assert(atype < (1 << 8));
  assert(var_id < (1 << 23));
  return (var_id << 8) | atype;
}

/** The approximation of a formula in an abstract domain w.r.t. the concrete domain. */
enum Approx {
  UNDER, ///< An under-approximating element contains only solutions but not necessarily all.
  OVER, ///< An over-approximating element contains all solutions but not necessarily only solutions.
  EXACT ///< An exact element is both under- and over-approximating; it exactly represents the set of solutions.
};

static constexpr Approx dapprox(Approx appx) {
  return appx == EXACT ? EXACT : (appx == UNDER ? OVER : UNDER);
}

/** A first-order signature is a triple \f$ (X, F, P) \f$ where \f$ X \f$ is the set of variables, \f$ F \f$ the set of function symbols and \f$ P \f$ the set of predicates.
  We represent \f$ X \f$ by strings (see `LVar`), while \f$ F \f$ and \f$ P \f$ are described in the following enumeration `Sig`.
  For programming conveniency, we suppose that logical connectors are included in the set of predicates and thus are in the signature as well.
  Finally, function symbols and predicates are at the "same level".
  Hence a predicate can occur as the argument of a function, which is convenient when modelling, consider for example a cardinality constraint: \f$ ((x > 4) + (y < 4) + (z = 3)) \neq 2 \f$.

  Symbols are sometimes overloaded across different universe of discourse.
  For instance, `ADD` can be used over integers, reals and even set of integers (pairwise addition).

  Division and modulus are defined as usual over continuous domains such as rational and real numbers.
  However, it gets more tricky when defined over discrete domains such as integers and floating-point numbers, since there is not a single definition of division and modulus.
  The various kinds of discrete divisions are explained in (Leijend D. (2003). Division and Modulus for Computer Scientists), and we add four definitions to the logical signature.
  There are several use-cases of modulus and division:
    * If you write a constraint model, you probably want to use euclidean division and modulus (EDIV, EMOD) as this is the most "mathematical" definition.
    * If you intend to model the semantics of a programming language, you should use the same kind of division as the one present in your programming language (most likely truncated division).
 */
enum Sig {
  ///@{
  NEG, ABS, ///< Unary arithmetic function symbol.
  ADD, SUB, MUL, POW, MIN, MAX,  ///< Binary arithmetic function symbol.
  DIV, MOD, ///< Division and modulus over continuous domains (e.g., floating-point numbers and rational).
  TDIV, TMOD, ///< Truncated division, present in most programming languages, is defined as \f$ a\,\mathbf{tdiv}\,b = \mathit{trunc}(a/b) \f$, i.e., it rounds towards zero. Modulus is defined as \f$ a\,\mathbf{tmod}\,b = a - b * (a\,\mathbf{tdiv}\,b) \f$.
  FDIV, FMOD, ///< Floor division (Knuth D. (1972). The Art of Computer Programming, Vol 1, Fundamental Algorithms), is defined as \f$ a\,\mathbf{fdiv}\,b = \lfloor a/b \rfloor \f$, i.e., it rounds towards negative infinity. Modulus is defined as \f$ a\,\mathbf{fmod}\,b = a - b * (a\,\mathbf{fdiv}\,b) \f$.
  CDIV, CMOD, ///< Ceil division is defined as \f$ a\,\mathbf{cdiv}\,b = \lceil a/b \rceil \f$. Modulus is defined as \f$ a\,\mathbf{cmod}\,b = a - b * (a\,\mathbf{cdiv}\,b) \f$.
  EDIV, EMOD, ///< Euclidean division (Boute T. R. (1992). The Euclidean definition of the functions div and mod). The properties satisfy by this division are: (1) \f$ a\,\mathbf{ediv}\,b \in \mathbb{Z} \f$, (2) \f$ a = b * (a\,\mathbf{ediv}\,b) + (a\,\mathbf{emod}\,b) \f$ and (3) \f$ 0 \leq a\,\mathbf{emod}\,b < |b|\f$. Further, note that Euclidean division satisfies \f$ a\,\mathbf{ediv}\,(-b) = -(a\,\mathbf{ediv}\,b) \f$ and \f$ a\,\mathbf{emod}\,(-b) = a\,\mathbf{emod}\,b \f$.
  ///@}
  UNION, INTERSECTION, COMPLEMENT, ///< Set functions.
  SUBSET, SUBSETEQ, SUPSET, SUPSETEQ, ///< Set inclusion predicates.
  IN, ///< Set membership predicate.
  CARD, ///< Cardinality function from set to integer.
  EQ, NEQ, ///< Equality relations.
  LEQ, GEQ, LT, GT, ///< Arithmetic comparison predicates.
  AND, OR, IMPLY, EQUIV, NOT, XOR,    ///< Logical connector.
  ITE ///< If-then-else
  ///@}
};

/** Returns the converse of a comparison operator <=, <, >, >=, =, !=. */
CUDA inline Sig converse_comparison(Sig sig) {
  switch(sig) {
    case LEQ: return GEQ;
    case GEQ: return LEQ;
    case GT: return LT;
    case LT: return GT;
    case EQ: return sig;
    case NEQ: return sig;
    default:
      assert(false); // converse not supported for all operators.
      break;
  }
  return sig;
}

CUDA inline const char* string_of_sig(Sig sig) {
  switch(sig) {
    case NEG: return "-";
    case ABS: return "abs";
    case ADD: return "+";
    case SUB: return "-";
    case MUL: return "*";
    case DIV: return "/";
    case MOD: return "%%";
    case POW: return "^";
    case MIN: return "min";
    case MAX: return "max";
    case UNION: return "\u222A";
    case INTERSECTION: return "\u2229";
    case COMPLEMENT: return "\\";
    case IN: return "\u2208";
    case SUBSET: return "\u2282";
    case SUBSETEQ: return "\u2286";
    case SUPSET: return "\u2283";
    case SUPSETEQ: return "\u2287";
    case CARD: return "card";
    case EQ: return "=";
    case NEQ: return "\u2260";
    case LEQ: return "\u2264";
    case GEQ: return "\u2265";
    case LT: return "<";
    case GT: return ">";
    case AND: return "\u2227";
    case OR: return "\u2228";
    case IMPLY: return "\u21D2";
    case EQUIV: return "\u21D4";
    case NOT: return "\u00AC";
    case XOR: return "\u2295";
    case ITE: return "ite";
    default:
      assert(false);
      return "<bug! unknown sig>";
    }
  }
}

namespace battery {
  template<>
  CUDA inline void print(const lala::Sig& sig) {
    printf("%s", lala::string_of_sig(sig));
  }
}

namespace lala {

/** `TFormula` represents the AST of a typed first-order logical formula.
In our context, the type of a formula is an integer representing the UID of an abstract domain in which the formula should to be interpreted.
It defaults to the value `UNTYPED` if the formula is not (yet) typed.
By default, the types of integer and real number are supported.
The supported symbols can be extended with the template parameter `ExtendedSig`.
This extended signature can also be used for representing exactly constant such as real numbers using a string.
The AST of a formula is represented by a variant, where each alternative is described below.
We represent everything at the same level (term, formula, predicate, variable, constant).
This is generally convenient when modelling to avoid creating intermediate boolean variables when reifying.
We can have `x + (x > y \/ y > x + 4) >= 1`.

Differently from first-order logic, the existential quantifier does not have a subformula, i.e., we write \f$ \exists{x:Int} \land \exists{y:Int} \land x < y\f$.
This semantics comes from "dynamic predicate logic" where a formula is interpreted in a context (here the abstract element).
(The exact connection of our framework to dynamic predicate logic is not yet perfectly clear.) */
template<class Allocator, class ExtendedSig = battery::String<Allocator>>
class TFormula {
public:
  using allocator_type = Allocator;
  using this_type = TFormula<Allocator, ExtendedSig>;
  using Sequence = battery::vector<this_type, Allocator>;
  using Existential = battery::tuple<LVar<Allocator>, CType>;
  using LogicSet = logic_set<this_type, allocator_type>;
  using Formula = battery::variant<
    logic_int, ///< Representation of integers. See above.
    logic_real, ///< Approximation of real numbers. See above.
    LogicSet, ///< Set of integers, reals or sets.
    AVar,            ///< Abstract variable
    LVar<Allocator>, ///< Logical variable
    Existential,     ///< Existential quantifier
    battery::tuple<Sig, Sequence>,  ///< ADD, SUB, ..., EQ, ..., AND, .., NOT
    battery::tuple<ExtendedSig, Sequence>  ///< see above
  >;

  /** Index of integers in the variant type `Formula` (called kind below). */
  static constexpr int Z = 0;

  /** Index of real numbers in the variant type `Formula` (called kind below). */
  static constexpr int R = Z + 1;

  /** Index of sets in the variant type `Formula` (called kind below). */
  static constexpr int S = R + 1;

  /** Index of abstract variables in the variant type `Formula` (called kind below). */
  static constexpr int V = S + 1;

  /** Index of logical variables in the variant type `Formula` (called kind below). */
  static constexpr int LV = V + 1;

  /** Index of existential quantifier in the variant type `Formula` (called kind below). */
  static constexpr int E = LV + 1;

  /** Index of n-ary operators in the variant type `Formula` (called kind below). */
  static constexpr int Seq = E + 1;

  /** Index of n-ary operators where the operator is an extended signature in the variant type `Formula` (called kind below). */
  static constexpr int ESeq = Seq + 1;

private:
  AType type_;
  Approx appx;
  Formula formula;

public:
  /** By default, we initialize the formula to `true`. */
  CUDA TFormula(): type_(UNTYPED), appx(EXACT), formula(Formula::template create<Z>(1)) {}
  CUDA TFormula(Formula&& formula): type_(UNTYPED), appx(EXACT), formula(std::move(formula)) {}
  CUDA TFormula(AType uid, Approx appx, Formula&& formula): type_(uid), formula(std::move(formula)), appx(appx) {}

  CUDA TFormula(const this_type& other): type_(other.type_), appx(other.appx), formula(other.formula) {}
  CUDA TFormula(this_type&& other): type_(other.type_), appx(other.appx), formula(std::move(other.formula)) {}

  template <class Alloc2, class ExtendedSig2>
  friend class TFormula;

  template <class Alloc2, class ExtendedSig2>
  CUDA TFormula(const TFormula<Alloc2, ExtendedSig2>& other, const Allocator& allocator = Allocator())
    : type_(other.type_), appx(other.appx), formula(Formula::template create<Z>(1))
  {
    switch(other.formula.index()) {
      case Z: formula = Formula::template create<Z>(other.z()); break;
      case R: formula = Formula::template create<R>(other.r()); break;
      case S: formula = Formula::template create<S>(LogicSet(other.s(), allocator));
        break;
      case V: formula = Formula::template create<V>(other.v()); break;
      case LV: formula = Formula::template create<LV>(LVar<Allocator>(other.lv(), allocator)); break;
      case E: formula = Formula::template create<E>(
        battery::make_tuple(
          LVar<Allocator>(battery::get<0>(other.exists()), allocator),
          battery::get<1>(other.exists())));
        break;
      case Seq:
        formula = Formula::template create<Seq>(
          battery::make_tuple(
            other.sig(),
            Sequence(other.seq(), allocator)));
          break;
      case ESeq:
        formula = Formula::template create<ESeq>(
          battery::make_tuple(
            ExtendedSig(other.esig(), allocator),
            Sequence(other.eseq(), allocator)));
          break;
      default: printf("print: formula not handled.\n"); assert(false); break;
    }
  }

  CUDA void swap(this_type& other) {
    ::battery::swap(type_, other.type_);
    ::battery::swap(appx, other.appx);
    ::battery::swap(formula, other.formula);
  }

  CUDA this_type& operator=(const this_type& rhs) {
    this_type(rhs).swap(*this);
    return *this;
  }

  CUDA this_type& operator=(this_type&& rhs) {
    this_type(std::move(rhs)).swap(*this);
    return *this;
  }

  CUDA Formula& data() { return formula; }
  CUDA const Formula& data() const { return formula; }
  CUDA AType type() const { return type_; }
  CUDA Approx approx() const { return appx; }
  CUDA Approx is_under() const { return appx == UNDER; }
  CUDA Approx is_over() const { return appx == OVER; }
  CUDA Approx is_exact() const { return appx == EXACT; }
  CUDA void type_as(AType ty) {
    type_ = ty;
    if(is(V)) {
      v() = make_var(ty, VID(v()));
    }
  }

  CUDA void approx_as(Approx a) {
    appx = a;
  }

  /** The formula `true` is represented by the integer constant `1`. */
  CUDA static this_type make_true() { return TFormula(); }

  /** The formula `false` is represented by the integer constant `0`. */
  CUDA static this_type make_false() { return TFormula(Formula::template create<Z>(0)); }

  CUDA static this_type make_z(logic_int i, AType atype = UNTYPED) {
    return this_type(atype, EXACT, Formula::template create<Z>(i));
  }

  /** Create a term representing a real number which is approximated by interval [lb..ub].
      By default the real number is supposedly over-approximated. */
  CUDA static this_type make_real(logic_real lb, logic_real ub, AType atype = UNTYPED, Approx a = OVER) {
    return this_type(atype, a, Formula::template create<R>(battery::make_tuple(lb, ub)));
  }

  CUDA static this_type make_set(LogicSet set, AType atype = UNTYPED, Approx a = OVER) {
    return this_type(atype, a, Formula::template create<S>(std::move(set)));
  }

  /** The type of the formula is embedded in `v`. */
  CUDA static this_type make_avar(AVar v, Approx a = EXACT) {
    return this_type(AID(v), a, Formula::template create<V>(v));
  }

  CUDA static this_type make_avar(AType ty, int vid, Approx a = EXACT) {
    return make_avar(make_var(ty, vid), a);
  }

  CUDA static this_type make_lvar(AType ty, LVar<Allocator> lvar, Approx a = EXACT) {
    return this_type(ty, a, Formula::template create<LV>(std::move(lvar)));
  }

  CUDA static this_type make_exists(AType ty, LVar<Allocator> lvar, CType ctype, Approx a = EXACT, const Allocator& allocator = Allocator()) {
    return this_type(ty, a, Formula::template create<E>(battery::make_tuple(std::move(lvar), std::move(ctype))));
  }

  CUDA static this_type make_nary(Sig sig, Sequence children, AType atype = UNTYPED, Approx a = EXACT, const Allocator& allocator = Allocator()) {
    return this_type(atype, a, Formula::template create<Seq>(battery::make_tuple(sig, std::move(children))));
  }

  CUDA static this_type make_unary(Sig sig, TFormula child, AType atype = UNTYPED, Approx a = EXACT, const Allocator& allocator = Allocator()) {
    return make_nary(sig, Sequence({std::move(child)}, allocator), atype, a, allocator);
  }

  CUDA static this_type make_binary(TFormula lhs, Sig sig, TFormula rhs, AType atype = UNTYPED, Approx a = EXACT, const Allocator& allocator = Allocator()) {
    return make_nary(sig, Sequence({std::move(lhs), std::move(rhs)}, allocator), atype, a, allocator);
  }

  CUDA static this_type make_nary(ExtendedSig esig, Sequence children, AType atype = UNTYPED, Approx a = EXACT, const Allocator& allocator = Allocator()) {
    return this_type(atype, a, Formula::template create<ESeq>(battery::make_tuple(std::move(esig), std::move(children))));
  }

  CUDA int index() const {
    return formula.index();
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

  CUDA bool is_constant() const {
    return is(Z) || is(R) || is(S);
  }

  CUDA bool is_variable() const {
    return is(LV) || is(V);
  }

  CUDA bool is_binary() const {
    return is(Seq) && seq().size() == 2;
  }

  CUDA logic_int z() const {
    return battery::get<Z>(formula);
  }

  CUDA const logic_real& r() const {
    return battery::get<R>(formula);
  }

  CUDA const LogicSet& s() const {
    return battery::get<S>(formula);
  }

  CUDA AVar v() const {
    return battery::get<V>(formula);
  }

  CUDA const LVar<Allocator>& lv() const {
    return battery::get<LV>(formula);
  }

  CUDA const Existential& exists() const {
    return battery::get<E>(formula);
  }

  CUDA Sig sig() const {
    return battery::get<0>(battery::get<Seq>(formula));
  }

  CUDA const ExtendedSig& esig() const {
    return battery::get<0>(battery::get<ESeq>(formula));
  }

  CUDA const Sequence& seq() const {
    return battery::get<1>(battery::get<Seq>(formula));
  }

  CUDA const this_type& seq(size_t i) const {
    return seq()[i];
  }

  CUDA const Sequence& eseq() const {
    return battery::get<1>(battery::get<ESeq>(formula));
  }

  CUDA const this_type& eseq(size_t i) const {
    return eseq()[i];
  }

  CUDA logic_int& z() {
    return battery::get<Z>(formula);
  }

  CUDA logic_real& r() {
    return battery::get<R>(formula);
  }

  CUDA LogicSet& s() {
    return battery::get<S>(formula);
  }

  CUDA AVar& v() {
    return battery::get<V>(formula);
  }

  CUDA Sig& sig() {
    return battery::get<0>(battery::get<Seq>(formula));
  }

  CUDA ExtendedSig& esig() {
    return battery::get<0>(battery::get<ESeq>(formula));
  }

  CUDA Sequence& seq() {
    return battery::get<1>(battery::get<Seq>(formula));
  }

  CUDA this_type& seq(size_t i) {
    return seq()[i];
  }

  CUDA Sequence& eseq() {
    return battery::get<1>(battery::get<ESeq>(formula));
  }

  CUDA this_type& eseq(size_t i) {
    return eseq()[i];
  }

private:
  template<size_t n>
  CUDA void print_sequence(bool print_atype) const {
    const auto& op = battery::get<0>(battery::get<n>(formula));
    const auto& children = battery::get<1>(battery::get<n>(formula));
    assert(children.size() > 0);
    if constexpr(n == Seq) {
      if(children.size() == 1) {
        if(op == ABS) printf("|");
        else if(op == CARD) printf("#(");
        else if(op != SQR) ::battery::print(op);
        children[0].print(print_atype);
        if(op == ABS) printf("|");
        else if(op == CARD) printf(")");
        else if(op == SQR) printf("^2");
        return;
      }
    }
    printf("(");
    for(int i = 0; i < children.size(); ++i) {
      children[i].print(print_atype);
      if(i < children.size() - 1) {
        printf(" ");
        ::battery::print(op);
        printf(" ");
      }
    }
    printf(")");
  }

public:
  CUDA void print(bool print_atype = true) const {
    switch(formula.index()) {
      case Z:
        printf("%lld", z());
        break;
      case R:
        printf("[%lf..%lf]", battery::get<0>(r()), battery::get<1>(r()));
        break;
      case S:
        printf("{");
        for(int i = 0; i < s().size(); ++i) {
          const auto& lb = battery::get<0>(s()[i]);
          const auto& ub = battery::get<1>(s()[i]);
          if(lb == ub) {
            lb.print(print_atype);
          }
          else {
            printf("[");
            lb.print(print_atype);
            printf("..");
            ub.print(print_atype);
            printf("]");
          }
          if(i < s().size() - 1) {
            printf(", ");
          }
        }
        printf("}");
        break;
      case V:
        printf("var(%d)", v());
        break;
      case LV:
        lv().print();
        break;
      case E: {
        if(print_atype) { printf("("); }
        const auto& e = exists();
        printf("var ");
        battery::get<0>(e).print();
        printf(":");
        battery::get<1>(e).print();
        if(print_atype) { printf(")"); }
        else { printf(", "); }
        break;
      }
      case Seq: print_sequence<Seq>(print_atype); break;
      case ESeq: print_sequence<ESeq>(print_atype); break;
      default: printf("print: formula not handled.\n"); assert(false); break;
    }
    if(print_atype) {
      printf(":%d", type_);
    }
  }
};

template<typename Allocator, typename ExtendedSig>
CUDA bool operator==(const TFormula<Allocator, ExtendedSig>& lhs, const TFormula<Allocator, ExtendedSig>& rhs) {
  return lhs.type() == rhs.type() && lhs.approx() == rhs.approx() && lhs.data() == rhs.data();
}

enum Mode {
  MINIMIZE,
  MAXIMIZE,
  SATISFY
};

/** `SFormula` is a formula to be solved with a possible optimisation mode (MINIMIZE or MAXIMIZE), otherwise it will enumerate `n` satisfiable solutions, if any. */
template<class Allocator>
class SFormula {
public:
  using F = TFormula<Allocator>;

private:
  Mode mode_;

  using ModeData = battery::variant<
    LVar<Allocator>,  ///< The logical variable to optimize.
    AVar,   ///< The abstract variable to optimize. (We use this one after the variable has been added to an abstract element).
    size_t  ///< How many solutions should we compute (SATISFY mode).
  >;

  ModeData mode_data;

  F f;

public:
  /** Create a formula for which we want to find one or more solutions. */
  CUDA SFormula(F f, size_t num_sols = 1):
    mode_(SATISFY), f(std::move(f)), mode_data(ModeData::template create<2>(num_sols)) {}

  /** Create a formula for which we want to find the best solution minimizing or maximizing an objective variable. */
  CUDA SFormula(F f, Mode mode, LVar<Allocator> to_optimize):
    mode_(mode), f(std::move(f)), mode_data(ModeData::template create<0>(std::move(to_optimize)))
  {
    assert(mode_ != SATISFY);
  }

  template <class Alloc2>
  friend class SFormula;

  template <class Alloc2>
  CUDA SFormula(const SFormula<Alloc2>& other, const Allocator& allocator = Allocator()):
    mode_(other.mode_), mode_data(ModeData::template create<2>(0)), f(other.f, allocator)
  {
    switch(other.mode_data.index()) {
      case 0: mode_data = ModeData::template create<0>(
        LVar<Allocator>(battery::get<0>(other.mode_data), allocator)); break;
      case 1: mode_data = ModeData::template create<1>(battery::get<1>(other.mode_data)); break;
      case 2: mode_data = ModeData::template create<2>(battery::get<2>(other.mode_data)); break;
      default: assert(0);
    }
  }

  CUDA Mode mode() const {
    return mode_;
  }

  CUDA const LVar<Allocator>& optimization_lvar() const {
    assert(mode_ != SATISFY);
    return battery::get<0>(mode_data);
  }

  CUDA AVar optimization_avar() const {
    assert(mode_ != SATISFY);
    return battery::get<1>(mode_data);
  }

  CUDA const F& formula() const {
    return f;
  }

  CUDA F& formula() {
    return f;
  }

  CUDA size_t num_sols() const {
    assert(mode_ == SATISFY);
    return battery::get<2>(mode_data);
  }

  CUDA void convert_optimization_var(AVar a) {
    mode_data = ModeData::template create<1>(a);
  }
};

} // namespace lala
#endif
