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

namespace lala {

/** Each abstract domain is uniquely identified by an UID.
    We call it an _abstract type_.
    Each formula (and recursively, its subformulas) is assigned to an abstract type indicating in what abstract domain this formula should be interpreted. */
using AType = int;

/** This value means a formula is not typed in a particular abstract domain and its type should be inferred. */
#define UNTYPED (-1)

/** The dependencies list of the abstract domains DAG when copying abstract domains.
 * It should probably be somewhere else than in AST, but for a lack of better place it is here now... */
template<class Alloc = battery::StandardAllocator>
class AbstractDeps
{
  struct dep_erasure {
    CUDA virtual ~dep_erasure() {}
  };

  template <class A>
  struct dep_holder : dep_erasure {
    battery::shared_ptr<A, Alloc> a;
    CUDA dep_holder(A* ptr, const Alloc& alloc): a(ptr, alloc) {}
    CUDA virtual ~dep_holder() {}
  };

  battery::vector<battery::unique_ptr<dep_erasure, Alloc>, Alloc> deps;

public:
  using allocator_type = Alloc;

  CUDA AbstractDeps(const Alloc& alloc = Alloc()): deps(alloc) {}

  CUDA size_t size() const {
    return deps.size();
  }

  template<class A>
  CUDA battery::shared_ptr<A, Alloc> extract(AType uid) {
    assert(uid != UNTYPED);
    assert(deps.size() > uid);
    assert(deps[uid]);
    return static_cast<dep_holder<A>*>(deps[uid].get())->a;
  }

  template<class A, class FromAlloc>
  CUDA battery::shared_ptr<A, Alloc> clone(const battery::shared_ptr<A, FromAlloc>& a)
  {
    if(!a) {
      return nullptr;
    }
    assert(a->uid() != UNTYPED); // Abstract domain must all have a unique identifier to be copied.
    // If the dependency is not in the list, we copy it and add it.
    if(deps.size() <= a->uid() || !static_cast<bool>(deps[a->uid()])) {
      deps.resize(battery::max((int)deps.size(), a->uid()+1));
      Alloc to_alloc = deps.get_allocator();
      deps[a->uid()] = battery::unique_ptr<dep_erasure, Alloc>(
        new(to_alloc) dep_holder<A>(
          new(to_alloc) A(*a, *this),
          to_alloc),
        to_alloc);
      // NOTE: Since we are copying a DAG, `A(*a, *this)` or one of its dependency cannot create `deps[a->uid()]`.
    }
    return extract<A>(a->uid());
  }

  CUDA allocator_type get_allocator() const {
    return deps.get_allocator();
  }
};

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

  Symbols are overloaded across different abstract universes.
  Therefore, a logical formula can have different semantics depending on the abstract universe in which it is interpreted.
  The semantics of the predicates (from `JOIN` to `GT` below) are relative to a lattice order given to the universe of discourse.
  In the following, we suppose \f$ L \f$ is a universe of discourse with a lattice structure.
  Given a universe of discourse, we take its "natural increasing lattice order", e.g., for integers it is the lattice of increasing integers, and for set the lattice of increasing set (ordered by subset inclusion, i.e., \f$ X \leq Y \Leftrightarrow X \subseteq Y \f$).
 */
enum Sig {
  ///@{
  NEG, ABS, SQR, ///< Unary arithmetic function symbol.
  ADD, SUB, MUL, DIV, MOD, POW,  ///< Binary arithmetic function symbol.
  ///@}
  JOIN,   ///< The join operator \f$ x \sqcup y \f$ (function \f$\sqcup: L \times L \to L \f$). For instance, on the lattice of increasing integers it is the max function, on the lattice of increasing sets it is the union.
  MEET,   ///< The meet operator \f$ x \sqcap y \f$ (function \f$\sqcap: L \times L \to L \f$). For instance, on the lattice of increasing integers it is the min function, on the lattice of increasing sets it is the intersection.
  COMPLEMENT, ///< The complement operator \f$ \lnot x \f$ (function \f$\lnot: L \to L \f$) such that \f$ x \sqcup \lnot x = \top \f$ and \f$ x \sqcap \lnot x = \bot \f$. For instance, the complement does not exist on the lattice of increasing integers, and it is the set complement on the lattice of increasing set.
  LEQ,   ///< The lattice order of the underlying universe of discourse (predicate \f$\leq: L \times L \f$).
  GEQ,   ///< \f$ x \geq y \Leftrightarrow y \leq x \f$ (predicate \f$\geq: L \times L \f$).
  EQ,    ///< \f$ x = y \Leftrightarrow x \leq y \land x \geq y \f$ (predicate \f$=: L \times L \f$).
  NEQ,   ///< \f$ x \neq y \Leftrightarrow \lnot (x = y) \f$ (predicate \f$\neq: L \times L \f$).
  LT,    ///< \f$ x < y \Leftrightarrow x \leq y \land x \neq y \f$ (predicate \f$<: L \times L \f$).
  GT,    ///< \f$ x > y \Leftrightarrow x \geq y \land x \neq y \f$ (predicate \f$>: L \times L \f$).
  CARD,  ///< \f$ #(x) \f$ is a function \f$ #: L \to ZDec \f$ which maps an abstract element to the cardinality of the set \f$ \gamma(x) \f$; note that it can map to \f$ \infty \f$ for elements with infinite cardinality.
  ///@{
  AND, OR, IMPLY, EQUIV, NOT, XOR    ///< Logical connector.
  ///@}
};
}

namespace battery {
  template<>
  CUDA inline void print(const lala::Sig& sig) {
    using namespace lala;
    switch(sig) {
      case NEG: printf("-"); break;
      case ADD: printf("+"); break;
      case SUB: printf("-"); break;
      case MUL: printf("*"); break;
      case DIV: printf("/"); break;
      case MOD: printf("%%"); break;
      case POW: printf("^"); break;
      case EQ: printf("="); break;
      case LEQ: printf("<="); break;
      case GEQ: printf(">="); break;
      case NEQ: printf("!="); break;
      case GT: printf(">"); break;
      case LT: printf("<"); break;
      case AND: printf("/\\"); break;
      case OR: printf("\\/"); break;
      case IMPLY: printf("=>"); break;
      case EQUIV: printf("<=>"); break;
      case NOT: printf("!"); break;
      case XOR: printf("xor"); break;
      case JOIN: printf("|_|"); break;
      case MEET: printf("|-|"); break;
      default:
        assert(false);
        break;
    }
  }
}

namespace lala {

/** The concrete type of variables introduced by existential quantification.
    More concrete types could be added later. */
struct CType {
  enum Tag {
    Int,
    Real,
    Set
  };

  Tag tag;
  thrust::optional<CType> sub;

  CType(Tag tag): tag(tag) {
    assert(tag != Set);
  }

  CType(Tag tag, CType sub): tag(tag), sub(sub) {
    assert(tag == Set);
    assert(sub.has_value());
  }

  CType(const CType&) = default;
  CType(CType&&) = default;

  void print() const {
    switch(tag) {
      case Int: printf("Z"); break;
      case Real: printf("R"); break;
      case Set: printf("S("); sub.print(); print(")"); break;
    }
  }
};

struct error_tag {};
struct warning_tag {};

/** This class is used in abstract domains to represent the result of an interpretation.
    If the abstract domain cannot interpret the formula, it must explain why.
    This is similar to compilation errors in compiler. */
template <class T, class Alloc>
struct IResult {
  using vector_type = battery::vector<battery::string<Alloc>, Alloc>;
  T data;
  vector_type errors;
  vector_type warnings;

  CUDA void print_array(const vector_type& array, const char* name) {
    for(int i = 0; i < array.size(); ++i) {
      printf("%s", name);
      array[i].print();
      printf("\n");
    }
  }

public:
  CUDA IResult(T&& data): data(std::move(data)) {}

  CUDA IResult(error_tag, T&& data, battery::string<Alloc>&& error): data(std::move(data)) {
    errors.push_back(std::move(error));
  }

  CUDA IResult(warning_tag, T&& data, battery::string<Alloc>&& warning): data(std::move(data)) {
    warnings.push_back(std::move(warning));
  }

  CUDA bool is_ok() const {
    return errors.size() == 0;
  }

  CUDA const T& value() const {
    return data;
  }

  CUDA T& value() {
    return data;
  }

  CUDA void print_diagnostics() const {
    print_array(errors, "[error]");
    print_array(warnings, "[warning]");
  }
}

/** The type of integers used in logic formulas.
    Integers are represented by the set \f$ \{-\infty, \infty\} \cup Z (\text{ with} Z \subset \mathbb{Z}) \f$.
    The minimal and maximal values of `logic_int` represents \f$ -\infty \f$ and \f$ \infty \f$ respectively. */
using logic_int = long long int;

/** The type of real numbers used in logic formulas.
    Real numbers are approximated by the set \f$ \mathbb{F} \times \mathbb{F} \f$.
    When a real number \f$ r \in \mathbb{R} \f$ is also a floating-point number, then it is represented by \f$ (r, r) \f$, otherwise it is represented by \f$ (\lfloor r \rfloor, \lceil r \rceil) \f$ such that \f$ \lfloor r \rfloor < r < \lceil r \rceil \f$ and there is no floating-point number \f$ f \f$ such that \f$ \lfloor r \rfloor < f < \lceil r \rceil \f$. */
using logic_real = battery::tuple<double, double>;

/** A set is parametric in a universe of discourse.
    For instance, `logic_set<logic_int>` is a set of integers.
    Sets are defined in extension: we explicitly list the values belonging to the set.
    To avoid using too much memory with large sets, we use an interval representation, e.g., \f$ \{1..3, 5..5, 10..12\} = \{1, 2, 3, 5, 10, 11, 12\} \f$.
    When sets occur in intervals, they are ordered by set inclusion, e.g., \f$ \{\{1..2\}..\{1..4\}\} = \{\{1,2\}, \{1,2,3\}, \{1,2,4\}, \{1,2,3,4\}\} \f$. */
template<class F, class Allocator>
using logic_set = battery::vector<battery::tuple<F, F>, Allocator>;

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

/** \return `true` if the formula `f` has the shape `variable op constant`, e.g., `x < 4`. */
template<typename Allocator, typename ExtendedSig>
CUDA bool is_v_op_z(const TFormula<Allocator, ExtendedSig>& f, Sig sig) {
  using F = TFormula<Allocator, ExtendedSig>;
  return f.is(F::Seq)
    && f.sig() == sig
    && (f.seq(0).is(F::LV) || f.seq(0).is(F::V))
    && f.seq(1).is(F::Z);
}

template<typename Allocator>
CUDA TFormula<Allocator> make_v_op_z(LVar<Allocator> v, Sig sig, logic_int z, AType aty = UNTYPED, Approx a = EXACT, const Allocator& allocator = Allocator()) {
  using F = TFormula<Allocator>;
  return F::make_binary(F::make_lvar(UNTYPED, std::move(v)), sig, F::make_z(z), UNTYPED, a, allocator);
}

namespace impl {
  template<typename Allocator, typename ExtendedSig>
  CUDA const TFormula<Allocator, ExtendedSig>& var_in_impl(const TFormula<Allocator, ExtendedSig>& f, bool& found);

  template<size_t n, typename Allocator, typename ExtendedSig>
  CUDA const TFormula<Allocator, ExtendedSig>& find_var_in_seq(const TFormula<Allocator, ExtendedSig>& f, bool& found) {
    const auto& children = battery::get<1>(battery::get<n>(f.data()));
    for(int i = 0; i < children.size(); ++i) {
      const auto& subformula = var_in_impl(children[i], found);
      if(found) {
        return subformula;
      }
    }
    return f;
  }

  template<typename Allocator, typename ExtendedSig>
  CUDA const TFormula<Allocator, ExtendedSig>& var_in_impl(const TFormula<Allocator, ExtendedSig>& f, bool& found)
  {
    using F = TFormula<Allocator, ExtendedSig>;
    switch(f.index()) {
      case F::Z:
      case F::R:
        return f;
      case F::V:
      case F::E:
      case F::LV:
        found = true;
        return f;
      case F::Seq: return find_var_in_seq<F::Seq>(f, found);
      case F::ESeq: return find_var_in_seq<F::ESeq>(f, found);
      default: printf("var_in: formula not handled.\n"); assert(false); return f;
    }
  }

  template<size_t n, class F>
  CUDA int num_vars_in_seq(const F& f) {
    const auto& children = battery::get<1>(battery::get<n>(f.data()));
    int total = 0;
    for(int i = 0; i < children.size(); ++i) {
      total += num_vars(children[i]);
    }
    return total;
  }
}

/** \return The first variable occurring in the formula, or any other subformula if the formula does not contain a variable.
    It returns either a logical variable, an abstract variable or a quantifier. */
template<typename Allocator, typename ExtendedSig>
CUDA const TFormula<Allocator, ExtendedSig>& var_in(const TFormula<Allocator, ExtendedSig>& f) {
  bool found = false;
  return impl::var_in_impl(f, found);
}

/** \return The number of variables occurring in the formula `F` including existential quantifier, logical variables and abstract variables.
 * Each occurrence of a variable is added up (duplicates are counted). */
template<class F>
CUDA int num_vars(const F& f)
{
  switch(f.index()) {
    case F::Z:
    case F::R:
      return 0;
    case F::V:
    case F::E:
    case F::LV:
      return 1;
    case F::Seq: return impl::num_vars_in_seq<F::Seq>(f);
    case F::ESeq: return impl::num_vars_in_seq<F::ESeq>(f);
    default: printf("num_vars: formula not handled.\n"); assert(false); return 0;
  }
}

template<class F>
CUDA AType type_of_conjunction(const typename F::Sequence& seq) {
  AType ty = UNTYPED;
  for(int i = 0; i < seq.size(); ++i) {
    if(seq[i].type() == UNTYPED) {
      return UNTYPED;
    }
    else if(ty == UNTYPED) {
      ty = seq[i].type();
    }
    else if(ty != seq[i].type()) {
      return UNTYPED;
    }
  }
  return ty;
}

/** Given a conjunctive formula `f` of the form \f$ c_1 \land ... \land c_n \f$, it returns a pair \f$ \langle c_i \land .. \land c_j, c_k \land ... \land c_l \rangle \f$ such that the first component contains all formulas with the type `ty`, and the second component, all other formulas. */
template<class F>
CUDA battery::tuple<F,F> extract_ty(const F& f, AType ty) {
  assert(f.is(F::Seq));
  const auto& seq = f.seq();
  typename F::Sequence fty;
  typename F::Sequence other;
  for(int i = 0; i < seq.size(); ++i) {
    // In case of nested conjunction.
    if(seq[i].is(F::Seq) && seq[i].sig() == AND) {
      auto r = extract_ty(seq[i], ty);
      auto& fty_ = battery::get<0>(r).seq();
      auto& other_ = battery::get<1>(r).seq();
      for(int i = 0; i < fty_.size(); ++i) {
        fty.push_back(std::move(fty_[i]));
      }
      for(int i = 0; i < other_.size(); ++i) {
        other.push_back(std::move(other_[i]));
      }
    }
    else if(seq[i].type() == ty) {
      fty.push_back(seq[i]);
    }
    else {
      other.push_back(seq[i]);
    }
  }
  AType other_ty = type_of_conjunction<F>(other);
  return battery::make_tuple(
    F::make_nary(AND, std::move(fty), ty, f.approx()),
    F::make_nary(AND, std::move(other), other_ty, f.approx()));
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

/** A `VarEnv` is a variable environment mapping between logical variables and abstract variables.
This class is supposed to be used inside an abstract domain, to help with the conversion. */
template<class Allocator>
class VarEnv {
public:
  using allocator_type = Allocator;
  using LName = LVar<allocator_type>;
  using this_type = VarEnv<Allocator>;

private:
  AType uid_;
  /** Given an abstract variable `v`, `avar2lvar[VID(v)]` is the name of the variable. */
  battery::vector<LName, allocator_type> avar2lvar;

public:
  CUDA VarEnv(AType uid, const Allocator& allocator = Allocator())
   : uid_(uid), avar2lvar(allocator) {}

  CUDA VarEnv(AType uid, int capacity, const Allocator& allocator = Allocator())
   : uid_(uid), avar2lvar(allocator)
  {
    avar2lvar.reserve(capacity);
  }

  CUDA VarEnv(VarEnv&& other): uid_(other.uid_), avar2lvar(std::move(other.avar2lvar)) {}

  template<class Alloc2>
  CUDA VarEnv(const VarEnv<Alloc2>& other, const Allocator& allocator = Allocator())
   : uid_(other.uid_), avar2lvar(other.avar2lvar, allocator) {}

  CUDA this_type& operator=(const this_type& other) {
    uid_ = other.uid_;
    avar2lvar = other.avar2lvar;
    return *this;
  }

  CUDA AType uid() const {
    return uid_;
  }

  CUDA size_t capacity() const {
    return avar2lvar.capacity();
  }

  CUDA size_t size() const {
    return avar2lvar.size();
  }

  CUDA const LName& operator[](size_t i) const {
    assert(i < size());
    return avar2lvar[i];
  }

  CUDA const LName& to_lvar(AVar av) const {
    assert(VID(av) < size());
    return avar2lvar[VID(av)];
  }

  CUDA thrust::optional<AVar> to_avar(const LName& lv) const {
    for(int i = 0; i < size(); ++i) {
      if(avar2lvar[i] == lv) {
        return make_var(uid(), i);
      }
    }
    return {};
  }

  CUDA AVar add(LName lv) {
    avar2lvar.push_back(std::move(lv));
    return make_var(uid(), size() - 1);
  }

  CUDA void reserve(int new_cap) {
    avar2lvar.reserve(new_cap);
  }
};

/** Given a formula `f` and an environment, return the first abstract variable (`AVar`) occurring in `f` or `{}` if `f` has no variable in `env`. */
template <class F, class Allocator>
CUDA thrust::optional<AVar> var_in(const F& f, const VarEnv<Allocator>& env) {
  const auto& g = var_in(f);
  switch(g.index()) {
    case F::V:
      return g.v();
    case F::E:
      return env.to_avar(battery::get<0>(g.exists()));
    case F::LV:
      return env.to_avar(g.lv());
    default:
      return {};
  }
}

} // namespace lala
#endif
