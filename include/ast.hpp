// Copyright 2021 Pierre Talbot

#ifndef AST_HPP
#define AST_HPP

#include "utility.hpp"
#include "darray.hpp"
#include "string.hpp"
#include "tuple.hpp"
#include "variant.hpp"
#include "thrust/optional.h"

using namespace battery;

namespace lala {

/** Each abstract domain is uniquely identified by an UID.
    We call it an _abstract type_.
    Each formula (and recursively, its subformulas) is assigned to an abstract type indicating in what abstract domain this formula should be interpreted. */
using AType = int;

/** This value means a formula is not typed in a particular abstract domain and its type should be inferred. */
#define UNTYPED (-1)

/** The concrete type of variables introduced by existential quantification.
    More concrete types could be added later. */
enum CType {
  Int,
  Real
};

/** A "logical variable" is just the name of the variable. */
template<typename Allocator>
using LVar = String<Allocator>;

/** We call an "abstract variable" the representation of a logical variable in an abstract domain.
It is a pair of integers `(aid, vid)` where `aid` is the UID of the abstract element and `vid` is an internal identifier of the variable inside the abstract element.
The mapping between logical variables and abstract variables is maintained in `VarEnv` below.
An abstract variable always has a single name (or no name if it is not explicitly represented in the initial formula).
However, a logical variable can be represented by several abstract variables when the variable occurs in different abstract elements. */
typedef int AVar;
#define AID(v) (v & ((1 << 8) - 1))
#define VID(v) (v >> 8)

CUDA AVar make_var(AType type, int var_id);

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

  Symbols are overloaded across different abstract domains.
  Therefore, a logical formula can have different semantics depending on the abstract domain in which it is interpreted.
  This is where the type of the formula becomes important in order to force the formula to be interpreted in the right abstract domain, as intended.

  Another peculiarity of this signature is that the semantics of the predicates (from `JOIN` to `GT` below) are relative to a lattice order given to the universe of discourse.

  Finally, function symbols and predicates are at the "same level".
  Hence a predicate can occur as the argument of a function, which is convenient when modelling, consider for example a cardinality constraint: \f$ ((x > 4) + (y < 4) + (z = 3)) \neq 2 \f$.

  In the following, we suppose \f$ L \f$ is a universe of discourse with a lattice structure.
 */
enum Sig {
  ///@{
  NEG, ABS, SQR, ///< Unary arithmetic function symbol.
  ADD, SUB, MUL, DIV, MOD, POW,  ///< Binary arithmetic function symbol.
  ///@}
  JOIN,   ///< The join operator \f$ x \sqcup y \f$ (function \f$\sqcup: L \times L \to L \f$). For instance, on the lattice of increasing integers it is the max function, on the lattice of increasing sets it is the union.
  MEET,   ///< The meet operator \f$ x \sqcap y \f$ (function \f$\sqcap: L \times L \to L \f$). For instance, on the lattice of increasing integers it is the min function, on the lattice of increasing sets it is the intersection.
  COMPLEMENT, ///< The complement operator \f$ \lnot x \f$ (function \f$\lnot: L \to L \f$) such that \f$ x \sqcup \lnot x = \top \f$ and \f$ x \sqcap \lnot x = \bot \f$. For instance, the complement does not exist on the lattice of increasing integers, and it is the set complement on the lattice of increasing or decreasing sets.
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
  CUDA void print(const lala::Sig& sig);
}

namespace lala {

/** `TFormula` represents the AST of a typed first-order logical formula.
In our context, the type of a formula is an integer representing the UID of an abstract domain in which the formula should to be interpreted.
This integer can take the value `UNTYPED` if the formula is not (yet) typed.
By default, the types of integer and real number are supported, but constants are bounded.
The supported symbols can be extended with the template parameter `ExtendedSig`.
This extended signature can also be used for representing exactly constant such as real numbers using a string.
The AST of a formula is represented by a variant, where each alternative is described below.
We represent everything at the same level (term, formula, predicate, variable, constant).
This is generally convenient when modelling to avoid creating intermediate boolean variables when reifying.
We can have `x + (x > y \/ y > x + 4)` and this expression is true if the value is != 0.

Differently from first-order logic, the existential quantifier does not have a subformula, i.e., we write \f$ \exists{x:Int} \land \exists{y:Int} \land x < y\f$.
This semantics comes from `dynamic predicate logic` where a formula is interpreted in a context (here the abstract element).
(The exact connection of our framework to dynamic predicate logic is not yet perfectly clear.) */
template<typename Allocator, typename ExtendedSig = String<Allocator>>
class TFormula {
public:
  using this_type = TFormula<Allocator, ExtendedSig>;
  using Sequence = DArray<this_type, Allocator>;
  using Existential = battery::tuple<LVar<Allocator>, CType>;
  using Formula = Variant<
    long long int, ///< Constant in the domain of discourse that can be represented exactly.
    battery::tuple<double, double>,    ///< A real represented as an interval \f$ [d[0]..d[1]] \f$. Indeed, we sometimes cannot use a single `double` because it cannot represent all real numbers, it is up to the abstract domain to under- or over-approximate it, or choose an exact representation such as rational.
    AVar,            ///< Abstract variable
    LVar<Allocator>, ///< Logical variable
    Existential,     ///< Existential quantifier
    battery::tuple<Sig, Sequence>,              ///< ADD, SUB, ..., EQ, ..., AND, .., NOT
    battery::tuple<ExtendedSig, Sequence>       ///< see above
  >;

  /** Index of integers in the variant type `Formula` (called kind below). */
  static constexpr int Z = 0;

  /** Index of real numbers in the variant type `Formula` (called kind below). */
  static constexpr int R = 1;

  /** Index of abstract variables in the variant type `Formula` (called kind below). */
  static constexpr int V = 2;

  /** Index of logical variables in the variant type `Formula` (called kind below). */
  static constexpr int LV = 3;

  /** Index of existential quantifier in the variant type `Formula` (called kind below). */
  static constexpr int E = 4;

  /** Index of n-ary operators in the variant type `Formula` (called kind below). */
  static constexpr int Seq = 5;

  /** Index of n-ary operators where the operator is an extended signature in the variant type `Formula` (called kind below). */
  static constexpr int ESeq = 6;

private:
  AType type_;
  Approx appx;
  Formula formula;

public:
  /** By default, we initialize the formula to `true`. */
  CUDA TFormula(): type_(UNTYPED), appx(EXACT), formula(Formula::template create<Z>(1)) {}
  CUDA TFormula(Formula&& formula): type_(UNTYPED), appx(EXACT), formula(std::forward<Formula>(formula)) {}
  CUDA TFormula(AType uid, Approx appx, Formula&& formula): type_(uid), formula(std::forward<Formula>(formula)), appx(appx) {}

  CUDA TFormula(const this_type& other): type_(other.type_), appx(other.appx), formula(other.formula) {}
  CUDA TFormula(this_type&& other): type_(other.type_), appx(other.appx), formula(std::move(other.formula)) {}

  CUDA this_type& operator=(this_type& rhs) {
    type_ = rhs.type_;
    appx = rhs.appx;
    formula = rhs.formula;
    return *this;
  }

  CUDA this_type& operator=(this_type&& rhs) {
    type_ = rhs.type_;
    appx = rhs.appx;
    formula = std::move(rhs.formula);
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

  CUDA static this_type make_z(long long int i, AType atype = UNTYPED) {
    return this_type(atype, EXACT, Formula::template create<Z>(i));
  }

  /** Create a term representing a real number which is approximated by interval [lb..ub].
      By default the real number is supposedly over-approximated. */
  CUDA static this_type make_real(double lb, double ub, AType atype = UNTYPED, Approx a = OVER) {
    return this_type(atype, a, Formula::template create<R>(battery::make_tuple(lb, ub)));
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
    return this_type(ty, a, Formula::template create<E>(battery::make_tuple(std::move(lvar), ctype)));
  }

  CUDA static this_type make_nary(Sig sig, Sequence children, AType atype = UNTYPED, Approx a = EXACT, const Allocator& allocator = Allocator()) {
    return this_type(atype, a, Formula::template create<Seq>(battery::make_tuple(sig, std::move(children))));
  }

  CUDA static this_type make_unary(Sig sig, TFormula child, AType atype = UNTYPED, Approx a = EXACT, const Allocator& allocator = Allocator()) {
    Sequence children(1, allocator);
    children[0] = std::move(child);
    return make_nary(sig, std::move(children), atype, a, allocator);
  }

  CUDA static this_type make_binary(TFormula lhs, Sig sig, TFormula rhs, AType atype = UNTYPED, Approx a = EXACT, const Allocator& allocator = Allocator()) {
    Sequence children(2, allocator);
    children[0] = std::move(lhs);
    children[1] = std::move(rhs);
    return make_nary(sig, std::move(children), atype, a, allocator);
  }

  CUDA static this_type make_nary(ExtendedSig esig, Sequence children, AType atype = UNTYPED, Approx a = EXACT, const Allocator& allocator = Allocator()) {
    return this_type(atype, a, Formula::create<ESeq>(battery::make_tuple(std::move(esig), std::move(children))));
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

  CUDA long long int z() const {
    return get<Z>(formula);
  }

  CUDA const battery::tuple<double, double>& r() const {
    return get<R>(formula);
  }

  CUDA AVar v() const {
    return get<V>(formula);
  }

  CUDA const LVar<Allocator>& lv() const {
    return get<LV>(formula);
  }

  CUDA const Existential& exists() const {
    return get<E>(formula);
  }

  CUDA Sig sig() const {
    return battery::get<0>(get<Seq>(formula));
  }

  CUDA const ExtendedSig& esig() const {
    return battery::get<0>(get<ESeq>(formula));
  }

  CUDA const Sequence& seq() const {
    return battery::get<1>(get<Seq>(formula));
  }

  CUDA const this_type& seq(size_t i) const {
    return seq()[i];
  }

  CUDA const Sequence& eseq() const {
    return battery::get<1>(get<ESeq>(formula));
  }

  CUDA const this_type& eseq(size_t i) const {
    return eseq()[i];
  }

  CUDA long long int& z() {
    return get<Z>(formula);
  }

  CUDA battery::tuple<double, double>& r() {
    return get<R>(formula);
  }

  CUDA AVar& v() {
    return get<V>(formula);
  }

  CUDA Sig& sig() {
    return battery::get<0>(get<Seq>(formula));
  }

  CUDA ExtendedSig& esig() {
    return battery::get<0>(get<ESeq>(formula));
  }

  CUDA Sequence& seq() {
    return battery::get<1>(get<Seq>(formula));
  }

  CUDA this_type& seq(size_t i) {
    return seq()[i];
  }

  CUDA Sequence& eseq() {
    return get<1>(get<ESeq>(formula));
  }

  CUDA this_type& eseq(size_t i) {
    return eseq()[i];
  }

private:
  template<size_t n>
  CUDA void print_sequence(bool print_atype) const {
    const auto& op = get<0>(get<n>(formula));
    const auto& children = get<1>(get<n>(formula));
    assert(children.size() > 0);
    if(children.size() == 1) {
      if(op == ABS) printf("|");
      else if(op == CARD) printf("#(");
      else if(op != SQR) ::print(op);
      children[0].print(print_atype);
      if(op == ABS) printf("|");
      else if(op == CARD) printf(")");
      else if(op == SQR) printf("^2");
    }
    else {
      printf("(");
      for(int i = 0; i < children.size(); ++i) {
        children[i].print(print_atype);
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
  CUDA void print(bool print_atype = true) const {
    switch(formula.index()) {
      case Z:
        printf("%lld", z());
        break;
      case R:
        printf("[%lf..%lf]", get<0>(r()), get<1>(r()));
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
        printf("\u2203");
        get<0>(e).print();
        switch(get<1>(e)) {
          case Int: printf(":Z"); break;
          case Real: printf(":R"); break;
          default: printf("print: concrete type (CType) not handled.\n"); assert(false); break;
        }
        printf(", ");
        get<1>(e).print(print_atype);
        if(print_atype) { printf(")"); }
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
CUDA TFormula<Allocator> make_v_op_z(LVar<Allocator> v, Sig sig, long long int z, Approx a = EXACT, const Allocator& allocator = Allocator()) {
  using F = TFormula<Allocator>;
  return F::make_binary(F::make_lvar(UNTYPED, std::move(v)), sig, F::make_z(z), UNTYPED, a, allocator);
}

namespace impl {
  template<typename Allocator, typename ExtendedSig>
  CUDA const TFormula<Allocator, ExtendedSig>& var_in_impl(const TFormula<Allocator, ExtendedSig>& f, bool& found);

  template<size_t n, typename Allocator, typename ExtendedSig>
  CUDA const TFormula<Allocator, ExtendedSig>& find_var_in_seq(const TFormula<Allocator, ExtendedSig>& f, bool& found) {
    const auto& children = get<1>(get<n>(f.data()));
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
}

/** \return The first variable occurring in the formula, or any other subformula if the formula does not contain a variable.
    It returns either a logical variable, an abstract variable or a quantifier. */
template<typename Allocator, typename ExtendedSig>
CUDA const TFormula<Allocator, ExtendedSig>& var_in(const TFormula<Allocator, ExtendedSig>& f) {
  bool found = false;
  return impl::var_in_impl(f, found);
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
public:
  using LName = LVar<Allocator>;

private:
  using EnvType = DArray<LName, Allocator>;

  AType uid;
  /** Given an abstract variable `v`, `avar2lvar[VID(v)]` is the name of the variable. */
  EnvType avar2lvar;
  /** This is the number of variables in the environment. */
  size_t size_;

public:
  CUDA VarEnv(AType uid, int capacity): uid(uid), avar2lvar(capacity), size_(0) {}

  CUDA AType ad_uid() const {
    return uid;
  }

  CUDA size_t capacity() const {
    return avar2lvar.size();
  }

  CUDA size_t size() const {
    return size_;
  }

  CUDA const LName& to_lvar(AVar av) const {
    assert(VID(av) < size_);
    return avar2lvar[VID(av)];
  }

  CUDA thrust::optional<AVar> to_avar(const LName& lv) const {
    AVar i = 0;
    for(; i < size_; ++i) {
      if(avar2lvar[i] == lv) {
        return make_var(uid, i);
      }
    }
    return {};
  }

  CUDA AVar add(LName lv) {
    assert(size() < capacity());
    avar2lvar[size_++] = std::move(lv);
    return make_var(uid, size_ - 1);
  }

  CUDA void reserve(int new_cap) {
    if(new_cap > capacity()) {
      EnvType avar2lvar2 = EnvType(new_cap);
      for(int i = 0; i < avar2lvar.size(); ++i) {
        avar2lvar2[i] = std::move(avar2lvar[i]);
      }
      avar2lvar = avar2lvar2;
    }
  }
};

/** Given a formula `f` and an environment, return the first abstract variable (`AVar`) occurring in `f` or `{}` if `f` has no variable in `env`. */
template <typename F, typename Allocator>
CUDA thrust::optional<AVar> var_in(const F& f, const VarEnv<Allocator>& env) {
  const auto& g = var_in(f);
  switch(g.index()) {
    case F::V:
      return g.v();
    case F::E:
      return env.to_avar(get<0>(g.exists()));
    case F::LV:
      return env.to_avar(g.lv());
    default:
      return {};
  }
}

} // namespace lala
#endif
