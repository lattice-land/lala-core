// Copyright 2021 Pierre Talbot

#include "ast.hpp"
#include <cassert>

namespace lala {

CUDA AVar make_var(AType atype, int var_id) {
  assert(atype >= 0);
  assert(var_id >= 0);
  assert(atype < (1 << 8));
  assert(var_id < (1 << 23));
  return (var_id << 8) | atype;
}
}
namespace battery {
template<>
CUDA void print(const lala::Sig& sig) {
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
    default:
      assert(false);
      break;
  }
}
}