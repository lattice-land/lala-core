#ifndef LALA_CORE_BINARIZE_HPP
#define LALA_CORE_BINARIZE_HPP

#include "ast.hpp"

namespace lala
{
  template <class F>
  CUDA NI bool must_binarize(const F &f)
  {

    // if it is a "constraint" for the store
    if (!f.is_binary())
    {
      return true;
    }
    auto symbol = f.sig() == EQ || f.sig() == LEQ || f.sig() == GEQ;
    auto variable_constant = f.seq(0).is_variable() && f.seq(1).is_constant();
    auto constant_variable = f.seq(0).is_constant() && f.seq(1).is_variable();
    return !(symbol && (variable_constant || constant_variable));
  }
  template <class F>
  CUDA NI bool is_supported(const F &f)
  {
    return f.is(F::E) || (f.is(F::Seq) && f.sig() != MINIMIZE && f.sig() != MAXIMIZE);
  }

  template <class F, class Allocator>
  struct Binarizer
  {
    battery::vector<F, Allocator> conjunction;
    battery::vector<F, Allocator> existentials;
    std::unordered_map<int, LVar<Allocator>> constant2lvar;
    // std::unordered_map<LVar<Allocator>, F> lvar2formula;
    unsigned int counter_aux = 0;

  private:
    CUDA NI LVar<Allocator> make_aux_with_name(std::string name)
    {
      return LVar<Allocator>(name);
    }
    CUDA NI LVar<Allocator> make_aux_lvar(std::string prefix = "t_")
    {
      return make_aux_with_name(prefix + std::to_string(counter_aux++));
    }
    CUDA NI const F introduce_var(LVar<Allocator> &lvar)
    {
      auto var = F::make_lvar(UNTYPED, lvar);
      this->existentials.push_back(F::make_exists(UNTYPED, lvar, lala::Sort<Allocator>::Int));
      return var;
    }

    CUDA NI const F introduce_z_var(std::string prefix = "z_")
    {
      auto lvar = this->make_aux_lvar(prefix);
      auto var = F::make_lvar(UNTYPED, lvar);
      this->existentials.push_back(F::make_exists(UNTYPED, lvar, lala::Sort<Allocator>::Int));
      return var;
    }
    CUDA NI const F introduce_bool_var(std::string prefix = "b_")
    {
      auto lvar = this->make_aux_lvar(prefix);
      auto var = F::make_lvar(UNTYPED, lvar);
      this->existentials.push_back(F::make_exists(UNTYPED, lvar, lala::Sort<Allocator>::Bool));
      return var;
    }
    CUDA NI const F binarize_constant(const F &f)
    {
      auto index = f.is(F::Z) ? f.z() : (int)f.b();
      // if the constant is already a logical variable, we return it.
      if (this->constant2lvar.find(index) != this->constant2lvar.end())
      {
        return F::make_lvar(UNTYPED, this->constant2lvar[index]);
      }
      if (f.is(F::Z))
      {
        auto suffix = f.z() < 0 ? "minus_" + std::to_string(abs(f.z())) : std::to_string(f.z());
        auto lvar = make_aux_with_name("c_" + suffix);
        auto var = introduce_var(lvar);
        this->constant2lvar[f.z()] = lvar;
        this->conjunction.push_back(F::make_binary(var, EQ, f));
        return var;
      }else if(f.is(F::B)){
        auto suffix = std::to_string((int)f.b());
        auto lvar = make_aux_with_name("c_" + suffix);
        auto var = introduce_var(lvar);
        this->constant2lvar[(int)f.b()] = lvar;
        this->conjunction.push_back(F::make_binary(var, EQ, f));
        return var;
      }
    }
    CUDA NI F internal_binarize(const F &f)
    {
      if (f.is_variable())
      {
        return f;
      }
      else if (f.is(F::Z) || f.is(F::B))
      {
        return binarize_constant(f);
      }
      else if (f.is(F::E))
      {
        this->existentials.push_back(f);
        auto lv = battery::get<0>(f.exists());
        auto var = F::make_lvar(UNTYPED, lv);
        return var;
      }
      else if (f.is_unary())
      {
        auto sub = internal_binarize(f.seq(0));
        if (f.sig() == NEG)
        {
          auto result = introduce_z_var();
          this->conjunction.push_back(F::make_binary(result, EQ, F::make_binary(internal_binarize(F::make_z(0)), SUB, sub)));
          return result;
        }
        else if (f.sig() == ABS)
        {
          auto result = introduce_z_var();
          this->conjunction.push_back(F::make_binary(result, EQ, F::make_nary(MAX, {sub, internal_binarize(F::make_unary(NEG, sub))}, UNTYPED, false)));
          return result;
        }
        else if (f.sig() == NOT)
        {
          return internal_binarize(F::make_binary(sub, EQ, F::make_z(0)));
        }
        else
        {
          fprintf(stderr, "Unary operator %s not supported\n", string_of_sig(f.sig()));
          assert(false);
        }
      }
      else if (f.is_binary())
      {

        if (f.sig() == IN)
        {
          battery::vector<F, Allocator> disjunction;
          for (int i = 0; i < f.seq(1).s().size(); i++)
          {
            auto set = f.seq(1).s()[i];
            auto geq = F::make_binary(f.seq(0), GEQ, battery::get<0>(set));
            auto leq = F::make_binary(f.seq(0), LEQ, battery::get<1>(set));
            disjunction.push_back(F::make_binary(geq, AND, leq));
          }

          if (disjunction.size() == 1)
          {
            return internal_binarize(disjunction[0]);
          }
          else
          {
            return internal_binarize(F::make_nary(OR, std::move(disjunction)));
          }
        }

        auto t1 = internal_binarize(f.seq(0));
        auto t2 = internal_binarize(f.seq(1));

        if (f.is_logical() || f.sig() == MIN || f.sig() == MAX)
        {
          // transform to min/max form

          if (f.sig() == AND || f.sig() == MIN)
          {
            auto result = introduce_bool_var();
            this->conjunction.push_back(F::make_binary(result, EQ, F::make_nary(MIN, {t1, t2})));
            return result;
          }
          else if (f.sig() == OR || f.sig() == MAX)
          {
            auto result = introduce_bool_var();
            this->conjunction.push_back(F::make_binary(result, EQ, F::make_nary(MAX, {t1, t2})));
            return result;
          }
          else if (f.sig() == IMPLY)
          {
            return internal_binarize(F::make_binary(F::make_unary(NEG, t1), OR, t2));
          }
          else if (f.sig() == EQUIV)
          {
            return internal_binarize(F::make_binary(F::make_binary(t1, IMPLY, t2), AND, F::make_binary(t2, IMPLY, t1)));
          }
          else if (f.sig() == XOR)
          {
            return internal_binarize(F::make_binary(F::make_binary(t1, OR, t2), AND, F::make_binary(F::make_unary(NOT, t1), OR, F::make_unary(NOT, t2))));
          }
          else
          {
            printf("Logical operator %s not supported\n", string_of_sig(f.sig()));
            f.print(false);
            printf("\n");
            assert(false);
          }
        }
        else if (f.is_comparison())
        {
          if (f.sig() == EQ || f.sig() == LEQ || f.sig() == LT)
          {
            auto result = introduce_bool_var();
            this->conjunction.push_back(F::make_binary(result, EQ, F::make_binary(t1, f.sig(), t2)));
            return result;
          }
          else if (f.sig() == NEQ)
          {
            auto c_0 = internal_binarize(F::make_z(0));
            this->conjunction.push_back(F::make_binary(c_0, EQ, F::make_binary(t1, EQ, t2)));
            return c_0; // c_0 is false
          }
          else if (f.sig() == GT)
          {
            return internal_binarize(F::make_binary(F::make_z(0), EQ, F::make_binary(t1, LT, t2)));
          }
          else if (f.sig() == GEQ)
          {
            return internal_binarize(F::make_binary(t2, LEQ, t1));
          }
          else
          {
            printf("Comparison operator not supported\n");
            f.print(false);
            printf("\n");
          }
        }
        else if (f.sig() == ADD || f.sig() == SUB || f.sig() == MUL || is_division(f.sig()) || is_modulo(f.sig()))
        {
          auto result = introduce_z_var();
          this->conjunction.push_back(F::make_binary(result, EQ, F::make_binary(t1, f.sig(), t2)));
          return result;
        }
        else
        {
          printf("Formula not supported \n");
          f.print(false);
          printf("\n");
          assert(false);
        }
      }
      else if (f.is(F::Seq) && f.seq().size() > 2)
      {

        auto tmp = internal_binarize(F::make_binary(f.seq(0), f.sig(), f.seq(1)));
        for (int i = 2; i < f.seq().size(); i++)
        {
          tmp = internal_binarize(F::make_binary(tmp, f.sig(), f.seq(i)));
        }
        return tmp;
      }
      else
      {
        printf("Formula not supported\n");
        f.print(false);
        printf("\n");
        assert(false);
      }
    }

  public:
    Binarizer() = default;

    /**
     * Binarize a formula.
     * For example for the formula \f$x² + y - z/2 \leq w + w\f$, the binarized formula generates the following formula :
     * \f[
     * x² = t_1 \land
     * z/2 = t_2 \land
     * y - t_2 = t_3 \land
     * t_1 + t_3 = t_4 \land
     * w + w = t_5 \land
     * t_4 \leq t_5
     * \f]
     * @tparam F the formula type
     * @param f the formula to binarize
     * @return the binarized formula
     */
    CUDA NI F binarize(const F &f)
    {
      if (f.is(F::Seq))
      {
        auto seq = f.seq();
        for (int i = 0; i < seq.size(); ++i)
        {
          if (must_binarize(seq[i]) && is_supported(seq[i]))
          {
            internal_binarize(seq[i]);
          }
          else
          {
            this->conjunction.push_back(seq[i]);
          }
        }
      }
      else
      {
        if (must_binarize(f) && is_supported(f))
        {
          internal_binarize(f);
        }
        else
        {
          this->conjunction.push_back(f);
        }
      }

      auto all_formula = std::move(this->existentials);
      for (int i = 0; i < this->conjunction.size(); ++i)
      {
        all_formula.push_back(this->conjunction[i]);
      }
      return F::make_nary(AND, std::move(all_formula), UNTYPED, false);
    }
  };

}

#endif // LALA_CORE_BINARIZE_HPP