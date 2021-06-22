// Copyright 2021 Pierre Talbot

#ifndef AST_HPP
#define AST_HPP

#include "utility.hpp"
#include "darray.hpp"
#include "string.hpp"
#include "thrust/optional.h"

namespace lala {

/** Each abstract domain is uniquely identified by an UID. */
typedef int AD_UID;
/** This value means a formula is not typed in a particular abstract domain and its type should be inferred. */
#define UNTYPED_AD (-1)

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

CUDA AVar make_var(int ad_uid, int var_id);

/** The approximation of a formula in an abstract domain w.r.t. the concrete domain.
* `UNDER`: An under-approximating element contains only solutions but not necessarily all.
* `OVER`: An over-approximating element contains all solutions but not necessarily only solutions.
* `EXACT`: An exact element is both under- and over-approximating; it exactly represents the set of solutions. */
enum Approx {
  UNDER,
  OVER,
  EXACT
};

/** We represent everything at the same level (terms, formula, predicate, variable, constant).
This is general convenient when modelling to avoid creating intermediate boolean variables when reifying.
We can have `x + (x > y \/ y > x + 4)` and this expression is true if the value is != 0. */
template<typename Allocator>
struct Formula {
  typedef Formula<Allocator> this_type;
  typedef DArray<this_type, Allocator> sequence;

  AD_UID ad_uid;

  enum Type {
    ///@{
    LONG, REAL,                    ///< Constant in the domain of discourse that can be represented exactly.
    ///@}
    AVAR,                          ///< Abstract variable
    ///@{
    NEG, ADD, SUB, MUL, DIV, MOD, POW,  ///< Terms
    ///@}
    ///@{
    EQ, LEQ, GEQ, NEQ, GT, LT,     ///< Predicates
    ///@}
    ///@{
    TRUE, FALSE, AND, OR, IMPLY, EQUIV, NOT,    ///< Formulas
    ///@}
    RAW                            ///< General tag for extension purposes.
  };

  Type tag;

  /** The name of the variable, term, function or predicate is represented by a string.
  This struct can also be used for representing constant such as real numbers.
  NOTE: We sometimes cannot use a float type because it cannot represent all real numbers, it is up to the abstract domain to under- or over-approximate it, or choose an exact representation such as rational. */
  struct Raw {
    String<Allocator> name;
    sequence children;

    Raw(String<Allocator> name, sequence children):
      name(std::move(name)), children(std::move(children)) {}
  };

  union {
    long long int i;    ///< LONG
    double d[2];    ///< REAL represented as an interval [rlb..rub]
    AVar v;   ///< AVAR
    sequence children; ///< ADD, SUB, ..., EQ, ..., AND, .., NOT
    Raw raw;  ///< LVar, global constraints, predicates, real numbers, ...
  };

private:
  CUDA void clear() {
    switch(tag) {
      case LONG: case REAL: case AVAR: break;
      case RAW:
        raw.~Raw();
        break;
      case NEG: case ADD: case SUB: case MUL: case DIV: case MOD: case POW:
      case EQ: case LEQ: case GEQ: case NEQ: case GT: case LT:
      case TRUE: case FALSE: case AND: case OR: case IMPLY: case EQUIV: case NOT:
        children.~sequence();
        break;
      default:
        printf("tag=%d / avar = %d\n", tag, AVAR);
        assert(false);
        break;
    }
    tag = LONG;
    i = 0;
  }

  CUDA void init_from(this_type&& other) {
    switch(tag) {
      case LONG: i = other.i; break;
      case REAL: d[0] = other.d[0]; d[1] = other.d[1]; break;
      case AVAR: v = other.v; break;
      case RAW: new(&raw) Raw(std::move(other.raw)); break;
      case NEG: case ADD: case SUB: case MUL: case DIV: case MOD: case POW:
      case EQ: case LEQ: case GEQ: case NEQ: case GT: case LT:
      case TRUE: case FALSE: case AND: case OR: case IMPLY: case EQUIV: case NOT:
        new(&children) sequence(std::move(other.children));
        break;
      default:
        assert(false);
        break;
    }
    other.tag = LONG;
    other.i = 0;
  }
public:

  /** By default, we initialize a constant 0 of type LONG. */
  CUDA Formula(): ad_uid(UNTYPED_AD), tag(LONG), i(0) {}
  CUDA Formula(const this_type& other):
    ad_uid(other.ad_uid), tag(other.tag)
  {
    switch(tag) {
      case LONG: i = other.i; break;
      case REAL: d[0] = other.d[0]; d[1] = other.d[1]; break;
      case AVAR: v = other.v; break;
      case RAW: new(&raw) Raw(other.raw); break;
      case NEG: case ADD: case SUB: case MUL: case DIV: case MOD: case POW:
      case EQ: case LEQ: case GEQ: case NEQ: case GT: case LT:
      case TRUE: case FALSE: case AND: case OR: case IMPLY: case EQUIV: case NOT:
        new(&children) sequence(other.children);
        break;
      default:
        assert(false);
        break;
    }
  }

  CUDA this_type& operator=(this_type other) {
    clear();
    ad_uid = other.ad_uid;
    tag = other.tag;
    init_from(std::move(other));
    return *this;
  }

  CUDA Formula(this_type&& other): ad_uid(other.ad_uid), tag(other.tag) {
    init_from(std::forward<this_type>(other));
  }

  CUDA static this_type make_long(AD_UID ad_uid, long long int i) {
    this_type f;
    f.ad_uid = ad_uid;
    f.tag = LONG;
    f.i = i;
    return std::move(f);
  }

  CUDA static this_type make_avar(AD_UID ad_uid, AVar v) {
    this_type f;
    f.ad_uid = ad_uid;
    f.tag = AVAR;
    f.v = v;
    return std::move(f);
  }

  CUDA static this_type make_true() {
    this_type f;
    f.tag = TRUE;
    new(&f.children) sequence;
    return f;
  }

  CUDA static this_type make_false() {
    this_type f;
    f.tag = FALSE;
    new(&f.children) sequence;
    return f;
  }

  CUDA Formula(AD_UID ad_uid, double lb, double ub):
    ad_uid(ad_uid), tag(REAL)
  {
    d[0] = lb;
    d[1] = ub;
  }

  CUDA Formula(AD_UID ad_uid, Type type, Formula sub, const Allocator& allocator = Allocator()):
    ad_uid(ad_uid), tag(type)
  {
    new(&children) sequence(1, allocator);
    children[0] = std::move(sub);
  }

  CUDA Formula(AD_UID ad_uid, Type type, Formula left, Formula right, const Allocator& allocator = Allocator()):
    ad_uid(ad_uid), tag(type)
  {
    new(&children) sequence(2, allocator);
    children[0] = std::move(left);
    children[1] = std::move(right);
  }

  CUDA Formula(AD_UID ad_uid, Type type, sequence children):
    ad_uid(ad_uid), tag(type)
  {
    new(&children) sequence(std::move(children));
  }

  CUDA Formula(AD_UID ad_uid, String<Allocator> name, sequence children):
    ad_uid(ad_uid), tag(RAW)
  {
    new(&raw) Raw(std::move(name), std::move(children));
  }

  CUDA ~Formula() {
    clear();
  }

  CUDA void print() const {
    const char* op = nullptr;
    const sequence* children_ptr = &children;
    switch(tag) {
      case LONG: printf("%lld:long", i); break;
      case REAL: printf("[%lf..%lf]", d[0], d[1]); break;
      case AVAR: printf("%d:var", v); break;
      case RAW: op = raw.name.data(); children_ptr = &raw.children; break;
      case NEG: printf("-("); children[0].print(); printf(")"); break;
      case TRUE: printf("true"); break;
      case FALSE: printf("false"); break;
      case ADD: op = "+"; break;
      case SUB: op = "-"; break;
      case MUL: op = "*"; break;
      case DIV: op = "/"; break;
      case MOD: op = "%"; break;
      case POW: op = "^"; break;
      case EQ: op = "="; break;
      case LEQ: op = "<="; break;
      case GEQ: op = ">="; break;
      case NEQ: op = "!="; break;
      case GT: op = ">"; break;
      case LT: op = "<"; break;
      case AND: op = "/\\"; break;
      case OR: op = "\\/"; break;
      case IMPLY: op = "=>"; break;
      case EQUIV: op = "<=>"; break;
      case NOT: op = "!"; break;
      default:
        assert(false);
        break;
    }
    if(op != nullptr) {
      printf("(");
      for(int i = 0; i < children_ptr->size(); ++i) {
        (*children_ptr)[i].print();
        if(i < children_ptr->size() - 1)
          printf(" %s ", op);
      }
      printf(")");
    }
  }
};

template<typename Allocator>
CUDA bool operator==(const Formula<Allocator>& lhs, const Formula<Allocator>& rhs) {
  if(lhs.tag != rhs.tag) return false;
  if(lhs.ad_uid != rhs.ad_uid) return false;
  typedef Formula<Allocator> F;
  switch(lhs.tag) {
    case F::LONG: return lhs.i == rhs.i;
    case F::REAL: return lhs.d[0] == rhs.d[0] && lhs.d[1] == rhs.d[1];
    case F::AVAR: return lhs.v == rhs.v;
    case F::RAW: return lhs.raw.name == rhs.raw.name && lhs.raw.children == rhs.raw.children;
    case F::NEG: case F::ADD: case F::SUB: case F::MUL: case F::DIV: case F::MOD: case F::POW:
    case F::EQ: case F::LEQ: case F::GEQ: case F::NEQ: case F::GT: case F::LT:
    case F::TRUE: case F::FALSE: case F::AND: case F::OR: case F::IMPLY: case F::EQUIV: case F::NOT:
      return lhs.children == rhs.children;
    default:
      assert(false);
      break;
  }
}

#define SHAPE(f,a,b,c) (f.tag == (a) && f.children[0].tag == (b) && f.children[1].tag == (c))

template<typename Allocator>
CUDA Formula<Allocator> make_x_op_i(typename Formula<Allocator>::Type op, AVar x, long long int i, const Allocator& allocator = Allocator()) {
  typedef Formula<Allocator> F;
  return F(UNTYPED_AD, op, F::make_avar(UNTYPED_AD, x), F::make_long(UNTYPED_AD, i), allocator);
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

  Formula<Allocator> f;
};

/** A `VarEnv` is a variable environment mapping between logical variables and abstract variables.
This class is supposed to be used inside an abstract domain, to help with the conversion. */
template<typename Allocator>
class VarEnv {
  typedef LVar<Allocator> vname;
  typedef DArray<LVar<Allocator>, Allocator> env_type;

  AD_UID uid;
  /** Given an abstract variable `v`, `avar2lvar[VID(v)]` is the name of the variable. */
  env_type avar2lvar;
  /** This is the number of variables in the environment. */
  size_t size_;

public:
  CUDA VarEnv(AD_UID uid, int capacity): uid(uid), avar2lvar(capacity), size_(0) {}

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

  CUDA AD_UID ad_uid() const {
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
