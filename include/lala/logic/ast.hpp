// Copyright 2021 Pierre Talbot

#ifndef LALA_CORE_AST_HPP
#define LALA_CORE_AST_HPP

#include "battery/utility.hpp"
#include "battery/vector.hpp"
#include "battery/string.hpp"
#include "battery/tuple.hpp"
#include "battery/variant.hpp"
#include "battery/shared_ptr.hpp"
#include "battery/unique_ptr.hpp"
#include "thrust/optional.h"
#include "sort.hpp"

namespace lala {

/** A "logical variable" is just the name of the variable. */
template<class Allocator>
using LVar = battery::string<Allocator>;

/** We call an "abstract variable" the representation of a logical variable in an abstract domain.
It is a pair of integers `(aty, vid)` where `aty` is the UID of the abstract element and `vid` is an internal identifier of the variable inside the abstract element.
The mapping between logical variables and abstract variables is maintained in an environment (see `env.hpp`).
An abstract variable always has a single name (or no name if it is not explicitly represented in the initial formula).
But a logical variable can be represented by several abstract variables when the variable occurs in different abstract elements. */
class AVar {
  int value;
public:
  CUDA constexpr AVar(): value(-1) {}
  constexpr AVar(const AVar&) = default;
  constexpr AVar(AVar&&) = default;
  constexpr AVar& operator=(const AVar&) = default;
  constexpr AVar& operator=(AVar&&) = default;
  CUDA constexpr bool operator==(const AVar& other) const {
    return value == other.value;
  }
  CUDA constexpr bool operator!=(const AVar& other) const {
    return value != other.value;
  }

  CUDA constexpr AVar(AType atype, int var_id): value((var_id << 8) | atype) {
    assert(atype >= 0);
    assert(var_id >= 0);
    assert(atype < (1 << 8));
    assert(var_id < (1 << 23));
  }

  CUDA constexpr bool is_untyped() const {
    return value == -1;
  }

  CUDA constexpr int aty() const {
    return is_untyped() ? -1 : (value & ((1 << 8) - 1));
  }

  CUDA constexpr int vid() const {
    return is_untyped() ? -1 : value >> 8;
  }

  CUDA constexpr void type_as(AType aty) {
    value = (vid() << 8) | aty;
  }
};

/** A first-order signature is a triple \f$ (X, F, P) \f$ where \f$ X \f$ is the set of variables, \f$ F \f$ the set of function symbols and \f$ P \f$ the set of predicates.
  We represent \f$ X \f$ by strings (see `LVar`), while \f$ F \f$ and \f$ P \f$ are described in the following enumeration `Sig`.
  For programming conveniency, we suppose that logical connectors are included in the set of predicates and thus are in the signature as well.
  Finally, function symbols and predicates are at the "same level".
  Hence a predicate can occur as the argument of a function, which is convenient when modelling, consider for example a cardinality constraint: \f$ ((x > 4) + (y < 4) + (z = 3)) \neq 2 \f$.

  Symbols are sometimes overloaded across different universe of discourse.
  For instance, `ADD` can be used over integers, reals and even set of integers (pairwise addition).

  Division and modulus are defined as usual over continuous domains such as rational and real numbers.
  However, it gets more tricky when defined over discrete domains such as integers and floating-point numbers, since there is not a single definition of division and modulus.
  The various kinds of discrete divisions are explained in (Leijend D. (2003). Division and Modulus for Computer Scientists), and we add four of those definitions to the logical signature.
  There are several use-cases of modulus and division:
    * If you write a constraint model, you probably want to use euclidean division and modulus (EDIV, EMOD) as this is the most "mathematical" definition.
    * If you intend to model the semantics of a programming language, you should use the same kind of division as the one present in your programming language (most likely truncated division).
 */
enum Sig {
  ///@{
  NEG, ABS, ///< Unary arithmetic function symbols.
  ADD, SUB, MUL, POW, MIN, MAX,  ///< Binary arithmetic function symbols.
  SQRT, EXP, LN, ///< Square root, natural exponential and natural logarithm function symbols.
  NROOT, LOG, ///< nth root and logarithm to base (both binary function symbols).
  SIN, COS, TAN, ASIN, ACOS, ATAN, SINH, COSH, TANH, ASINH, ACOSH, ATANH, ///< Trigonometric unary function symbols.
  DIV, MOD, ///< Division and modulus over continuous domains (e.g., floating-point numbers and rational).
  TDIV, TMOD, ///< Truncated division, present in most programming languages, is defined as \f$ a\,\mathbf{tdiv}\,b = \mathit{trunc}(a/b) \f$, i.e., it rounds towards zero. Modulus is defined as \f$ a\,\mathbf{tmod}\,b = a - b * (a\,\mathbf{tdiv}\,b) \f$.
  FDIV, FMOD, ///< Floor division (Knuth D. (1972). The Art of Computer Programming, Vol 1, Fundamental Algorithms), is defined as \f$ a\,\mathbf{fdiv}\,b = \lfloor a/b \rfloor \f$, i.e., it rounds towards negative infinity. Modulus is defined as \f$ a\,\mathbf{fmod}\,b = a - b * (a\,\mathbf{fdiv}\,b) \f$.
  CDIV, CMOD, ///< Ceil division is defined as \f$ a\,\mathbf{cdiv}\,b = \lceil a/b \rceil \f$. Modulus is defined as \f$ a\,\mathbf{cmod}\,b = a - b * (a\,\mathbf{cdiv}\,b) \f$.
  EDIV, EMOD, ///< Euclidean division (Boute T. R. (1992). The Euclidean definition of the functions div and mod). The properties satisfy by this division are: (1) \f$ a\,\mathbf{ediv}\,b \in \mathbb{Z} \f$, (2) \f$ a = b * (a\,\mathbf{ediv}\,b) + (a\,\mathbf{emod}\,b) \f$ and (3) \f$ 0 \leq a\,\mathbf{emod}\,b < |b|\f$. Further, note that Euclidean division satisfies \f$ a\,\mathbf{ediv}\,(-b) = -(a\,\mathbf{ediv}\,b) \f$ and \f$ a\,\mathbf{emod}\,(-b) = a\,\mathbf{emod}\,b \f$.
  UNION, INTERSECTION, DIFFERENCE, SYMMETRIC_DIFFERENCE, COMPLEMENT, ///< Set functions.
  SUBSET, SUBSETEQ, SUPSET, SUPSETEQ, ///< Set inclusion predicates.
  IN, ///< Set membership predicate.
  CARD, ///< Cardinality function from set to integer.
  HULL, ///< Unary function performing the convex hull of a set, e.g., \f$ \mathit{hull}(s) = \{x \;|\; \mathit{min}(s) \leq x \leq \mathit{max}(s) \} \f$.
  CONVEX, ///< Unary predicate, requiring \f$ s = \mathit{hull}(s) \f$.
  EQ, NEQ, ///< Equality relations.
  LEQ, GEQ, LT, GT, ///< Arithmetic comparison predicates. When applied to set, it corresponds to the lexicographic ordering of the sorted set according the underlying natural ordering of the elements in the set.
  AND, OR, IMPLY, EQUIV, NOT, XOR, ///< Logical connector.
  ITE, ///< If-then-else
  MAXIMIZE, ///< Unary "meta-predicate" indicating that its argument must be maximized, according to the increasing ordering of the underlying universe of discourse. This is not a predicate because it is defined on the solutions space of the whole formulas.
  MINIMIZE ///< Same as MAXIMIZE, but for minimization.
  ///@}
};

CUDA inline const char* string_of_sig(Sig sig) {
  switch(sig) {
    case NEG: return "-";
    case ABS: return "abs";
    case ADD: return "+";
    case SUB: return "-";
    case MUL: return "*";
    case DIV: return "/";
    case TDIV: return "tdiv";
    case FDIV: return "fdiv";
    case CDIV: return "cdiv";
    case EDIV: return "ediv";
    case MOD: return "%%";
    case TMOD: return "tmod";
    case FMOD: return "fmod";
    case CMOD: return "cmod";
    case EMOD: return "emod";
    case POW: return "^";
    case SQRT: return "sqrt";
    case EXP: return "exp";
    case LN: return "ln";
    case NROOT: return "nroot";
    case LOG: return "log";
    case SIN: return "sin";
    case COS: return "cos";
    case TAN: return "tan";
    case ASIN: return "asin";
    case ACOS: return "acos";
    case ATAN: return "atan";
    case SINH: return "sinh";
    case COSH: return "cosh";
    case TANH: return "tanh";
    case ASINH: return "asinh";
    case ACOSH: return "acosh";
    case ATANH: return "atanh";
    case MIN: return "min";
    case MAX: return "max";
    case UNION: return "\u222A";
    case INTERSECTION: return "\u2229";
    case DIFFERENCE: return "\\";
    case SYMMETRIC_DIFFERENCE: return "\u2296";
    case COMPLEMENT: return "complement";
    case IN: return "\u2208";
    case SUBSET: return "\u2282";
    case SUBSETEQ: return "\u2286";
    case SUPSET: return "\u2283";
    case SUPSETEQ: return "\u2287";
    case CARD: return "card";
    case HULL: return "hull";
    case CONVEX: return "convex";
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
    case MAXIMIZE: return "maximize";
    case MINIMIZE: return "minimize";
    default:
      assert(false);
      return "<bug! unknown sig>";
    }
  }

  CUDA inline constexpr bool is_prefix(Sig sig) {
    return sig == ABS || sig == SQRT || sig == EXP || sig == LN || sig == NROOT || sig == LOG || sig == SIN || sig == COS || sig == TAN || sig == ASIN || sig == ACOS || sig == ATAN || sig == SINH || sig == COSH || sig == TANH || sig == ASINH || sig == ACOSH || sig == ATANH || sig == MIN || sig == MAX || sig == COMPLEMENT || sig == CARD || sig == HULL || sig == CONVEX || sig == ITE || sig == MAXIMIZE || sig == MINIMIZE;
  }

  CUDA inline constexpr bool is_division(Sig sig) {
    return sig == DIV || sig == TDIV || sig == EDIV || sig == FDIV || sig == CDIV;
  }

  CUDA inline constexpr bool is_modulo(Sig sig) {
    return sig == MOD || sig == TMOD || sig == EMOD || sig == FMOD || sig == CMOD;
  }

  CUDA inline constexpr bool is_associative(Sig sig) {
    return sig == ADD || sig == MUL || sig == AND || sig == OR || sig == EQUIV || sig == XOR
      || sig == UNION || sig == INTERSECTION || sig == MAX || sig == MIN;
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
template<class Allocator, class ExtendedSig = battery::string<Allocator>>
class TFormula {
public:
  using allocator_type = Allocator;
  using this_type = TFormula<Allocator, ExtendedSig>;
  using Sequence = battery::vector<this_type, Allocator>;
  using Existential = battery::tuple<LVar<Allocator>, Sort<Allocator>>;
  using LogicSet = logic_set<this_type, allocator_type>;
  using Formula = battery::variant<
    logic_bool, ///< Representation of Booleans.
    logic_int, ///< Representation of integers.
    logic_real, ///< Approximation of real numbers.
    LogicSet, ///< Set of Booleans, integers, reals or sets.
    AVar,            ///< Abstract variable
    LVar<Allocator>, ///< Logical variable
    Existential,     ///< Existential quantifier
    battery::tuple<Sig, Sequence>,  ///< ADD, SUB, ..., EQ, ..., AND, .., NOT
    battery::tuple<ExtendedSig, Sequence>  ///< see above
  >;

  /** Index of Booleans in the variant type `Formula` (called kind below). */
  static constexpr int B = 0;

  /** Index of integers in the variant type `Formula` (called kind below). */
  static constexpr int Z = B + 1;

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
  Formula formula;

public:
  /** By default, we initialize the formula to `true`. */
  CUDA TFormula(): type_(UNTYPED), formula(Formula::template create<B>(true)) {}
  CUDA TFormula(Formula&& formula): type_(UNTYPED), formula(std::move(formula)) {}
  CUDA TFormula(AType uid, Formula&& formula): type_(uid), formula(std::move(formula)) {}

  CUDA TFormula(const this_type& other): type_(other.type_), formula(other.formula) {}
  CUDA TFormula(this_type&& other): type_(other.type_), formula(std::move(other.formula)) {}

  template <class Alloc2, class ExtendedSig2>
  friend class TFormula;

  template <class Alloc2, class ExtendedSig2>
  CUDA TFormula(const TFormula<Alloc2, ExtendedSig2>& other, const Allocator& allocator = Allocator())
    : type_(other.type_), formula(Formula::template create<B>(true))
  {
    switch(other.formula.index()) {
      case B: formula = Formula::template create<B>(other.b()); break;
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
  CUDA void type_as(AType ty) {
    type_ = ty;
    if(is(V)) {
      v().type_as(ty);
    }
  }

  CUDA constexpr bool is_untyped() const {
    return type() == UNTYPED;
  }

  CUDA thrust::optional<Sort<allocator_type>> sort() const {
    using sort_t = Sort<allocator_type>;
    switch(formula.index()) {
      case B: return sort_t(sort_t::Bool);
      case Z: return sort_t(sort_t::Int);
      case R: return sort_t(sort_t::Real);
      case S: {
        if(s().empty()) {
          return {}; // Impossible to get the sort of the empty set.
        }
        for(int i = 0; i < s().size(); ++i) {
          auto subsort = battery::get<0>(s()[i]).sort();
          if(!subsort.has_value()) {
            subsort = battery::get<1>(s()[i]).sort();
          }
          if(subsort.has_value()) {
            return sort_t(sort_t::Set, std::move(*subsort) , s().get_allocator());
          }
        }
        break;
      }
      case E: return battery::get<1>(exists());
    }
    return {};
  }

  CUDA static this_type make_true() { return TFormula(); }
  CUDA static this_type make_false() { return TFormula(Formula::template create<B>(false)); }

  CUDA static this_type make_bool(logic_bool b, AType atype = UNTYPED) {
    return this_type(atype, Formula::template create<B>(b));
  }

  CUDA static this_type make_z(logic_int i, AType atype = UNTYPED) {
    return this_type(atype, Formula::template create<Z>(i));
  }

  /** Create a term representing a real number which is approximated by interval [lb..ub].
      By default the real number is supposedly over-approximated. */
  CUDA static this_type make_real(double lb, double ub, AType atype = UNTYPED) {
    return this_type(atype, Formula::template create<R>(battery::make_tuple(lb, ub)));
  }

  CUDA static this_type make_real(logic_real r, AType atype = UNTYPED) {
    return this_type(atype, Formula::template create<R>(r));
  }

  CUDA static this_type make_set(LogicSet set, AType atype = UNTYPED) {
    return this_type(atype, Formula::template create<S>(std::move(set)));
  }

  /** The type of the formula is embedded in `v`. */
  CUDA static this_type make_avar(AVar v) {
    return this_type(v.aty(), Formula::template create<V>(v));
  }

  CUDA static this_type make_avar(AType ty, int vid) {
    return make_avar(AVar(ty, vid));
  }

  CUDA static this_type make_lvar(AType ty, LVar<Allocator> lvar) {
    return this_type(ty, Formula::template create<LV>(std::move(lvar)));
  }

  CUDA static this_type make_exists(AType ty, LVar<Allocator> lvar, Sort<Allocator> ctype) {
    return this_type(ty, Formula::template create<E>(battery::make_tuple(std::move(lvar), std::move(ctype))));
  }

  /** If `flatten` is `true` it will try to merge the children together to avoid nested formula. */
  CUDA static this_type make_nary(Sig sig, Sequence children, AType atype = UNTYPED, bool flatten=true)
  {
    Sequence seq;
    for(size_t i = 0; i < children.size(); ++i) {
      if(flatten && children[i].is(Seq) && children[i].sig() == sig && children[i].type() == atype) {
        for(size_t j = 0; j < children[i].seq().size(); ++j) {
          seq.push_back(std::move(children[i].seq(j)));
        }
      }
      else {
        seq.push_back(std::move(children[i]));
      }
    }
    return this_type(atype, Formula::template create<Seq>(battery::make_tuple(sig, std::move(seq))));
  }

  CUDA static this_type make_unary(Sig sig, TFormula child, AType atype = UNTYPED, const Allocator& allocator = Allocator()) {
    return make_nary(sig, Sequence({std::move(child)}, allocator), atype);
  }

  CUDA static this_type make_binary(TFormula lhs, Sig sig, TFormula rhs, AType atype = UNTYPED, const Allocator& allocator = Allocator(), bool flatten=true) {
    return make_nary(sig, Sequence({std::move(lhs), std::move(rhs)}, allocator), atype, flatten);
  }

  CUDA static this_type make_nary(ExtendedSig esig, Sequence children, AType atype = UNTYPED) {
    return this_type(atype, Formula::template create<ESeq>(battery::make_tuple(std::move(esig), std::move(children))));
  }

  CUDA int index() const {
    return formula.index();
  }

  CUDA bool is(int kind) const {
    return formula.index() == kind;
  }

  CUDA bool is_true() const {
    return (is(B) && b()) || (is(Z) && z() != 0);
  }

  CUDA bool is_false() const {
    return (is(B) && !b()) || (is(Z) && z() == 0);
  }

  CUDA bool is_constant() const {
    return is(B) || is(Z) || is(R) || is(S);
  }

  CUDA bool is_variable() const {
    return is(LV) || is(V);
  }

  CUDA bool is_binary() const {
    return is(Seq) && seq().size() == 2;
  }

  CUDA logic_bool b() const {
    return battery::get<B>(formula);
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

  CUDA bool is_logical() const {
    if(is(Seq)) {
      Sig s = sig();
      return s == AND || s == OR || s == IMPLY
        || s == EQUIV || s == NOT || s == XOR || s == ITE;
    }
    return false;
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

  CUDA logic_bool& b() {
    return battery::get<B>(formula);
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

  CUDA this_type map_sig(Sig sig) const {
    assert(is(Seq));
    this_type f = *this;
    f.sig() = sig;
    return std::move(f);
  }

  CUDA this_type map_atype(AType aty) const {
    this_type f = *this;
    f.type_as(aty);
    return std::move(f);
  }
private:
  template<size_t n>
  CUDA void print_sequence(bool print_atype, bool top_level = false) const {
    const auto& op = battery::get<0>(battery::get<n>(formula));
    const auto& children = battery::get<1>(battery::get<n>(formula));
    if(children.size() == 0) {
      ::battery::print(op);
      return;
    }
    if constexpr(n == Seq) {
      if(children.size() == 1) {
        if(op == ABS) printf("|");
        else if(op == CARD) printf("#(");
        else printf("%s(", string_of_sig(op));
        children[0].print_impl(print_atype);
        if(op == ABS) printf("|");
        else if(op == CARD) printf(")");
        else printf(")");
        return;
      }
    }
    bool isprefix = true;
    if constexpr(n == Seq) {
      isprefix = is_prefix(op);
    }
    if(isprefix) {
      ::battery::print(op);
    }
    printf("(");
    for(int i = 0; i < children.size(); ++i) {
      children[i].print_impl(print_atype);
      if(i < children.size() - 1) {
        if(!isprefix) {
          printf(" ");
          ::battery::print(op);
          printf(" ");
        }
        else {
          printf(", ");
        }
      }
      if constexpr(n == Seq) {
        if(op == AND && top_level) {
          printf("\n");
        }
      }
    }
    printf(")");
  }

  CUDA void print_impl(bool print_atype = true, bool top_level = false) const {
    switch(formula.index()) {
      case B:
        printf("%s", b() ? "true" : "false");
        break;
      case Z:
        printf("%lld", z());
        break;
      case R: {
        const auto& r_lb = battery::get<0>(r());
        const auto& r_ub = battery::get<1>(r());
        if(r_lb == r_ub) {
          printf("%.50lf", r_lb);
        }
        else {
          printf("[%.50lf..%.50lf]", r_lb, r_ub);
        }
        break;
      }
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
        printf("var(%d, %d)", v().aty(), v().vid());
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
        break;
      }
      case Seq: print_sequence<Seq>(print_atype, top_level); break;
      case ESeq: print_sequence<ESeq>(print_atype, top_level); break;
      default: printf("print: formula not handled.\n"); assert(false); break;
    }
    if(print_atype) {
      printf(":%d", type_);
    }
  }

public:
  CUDA void print(bool print_atype = true) const {
    print_impl(print_atype, true);
  }
};

template<typename Allocator, typename ExtendedSig>
CUDA bool operator==(const TFormula<Allocator, ExtendedSig>& lhs, const TFormula<Allocator, ExtendedSig>& rhs) {
  return lhs.type() == rhs.type() && lhs.data() == rhs.data();
}

} // namespace lala
#endif
