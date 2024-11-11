#include <gtest/gtest.h>
#include "battery/allocator.hpp"
#include "lala/logic/logic.hpp"
#include "lala/logic/binarize.hpp"
#include "lala/flatzinc_parser.hpp"
#include "lala/vstore.hpp"
#include "lala/simplifier.hpp"
#include "lala/interval.hpp"
#include "lala/fixpoint.hpp"
#include "abstract_testing.hpp"
#include <optional>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <regex>

using namespace lala;
using namespace battery;
namespace fs = std::filesystem;
using Itv = Interval<local::ZLB>;
using IStore = VStore<Itv, standard_allocator>;

template <class F>
bool contains(const F &f, const F &must_contain)
{
    for (int i = 0; i < f.seq().size(); ++i)
    {
        if (f.seq(i) == must_contain)
        {
            return true;
        }
    }
    return false;
}

bool is_PIR_symbol(Sig &sig)
{
    return sig == ADD || sig == SUB || sig == MUL || is_division(sig) || is_modulo(sig) || sig == EQ || sig == LT || sig == LEQ || sig == MAX || sig == MIN;
}

void test_binarize(
    std::pair<std::string, std::optional<std::string>> formulas)
{
    VarEnv<standard_allocator> env;
    // Cannot be interpreted in IStore, but after applying Simplifier, it can be interpreted.
    auto f1 = *parse_flatzinc<standard_allocator>(std::get<0>(formulas));
    if (f1.seq().size() < 100)
    {
        printf("\n Orignal formula : ");
        f1.print(false);
        printf("\n");
    }

    std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();
    Binarizer<F, standard_allocator> b;
    auto binarized = b.binarize(f1);
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    if (binarized.seq().size() < 100)
    {
        printf("Binarized formula : ");
        binarized.print(false);
        printf("\n");
    }

    printf("Time difference = %ld\n", std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());

    printf("NB variables original formula : %d\n", num_quantified_vars(f1));
    printf("NB constraints original formula : %d\n", f1.seq().size() - num_quantified_vars(f1));
    printf("NB variables binarized formula : %d\n", num_quantified_vars(binarized));
    printf("NB constraints binarized formula : %d\n", binarized.seq().size() - num_quantified_vars(binarized));

    if (std::get<1>(formulas).has_value())
    {
        auto f2 = *parse_flatzinc<standard_allocator>(std::get<1>(formulas).value());
        if (f2.seq().size() < 100)
        {
            printf("\n Expected formula : ");
            f2.print(false);
            printf("\n");
        }

        if (std::get<1>(formulas).has_value())
        {
            EXPECT_EQ(binarized.seq().size(), f2.seq().size());

            for (int i = 0; i < binarized.seq().size(); ++i)
            {
                // printf("We search '");
                // binarized.seq(i).print(false);
                // printf("' in formula\n");
                EXPECT_TRUE(contains(f2, binarized.seq(i)));
            }
        }
    }
    else
    {
        EXPECT_TRUE(binarized.seq().size() > 0); // it is a sequence
    }

    // std::unordered_set<F> variables;
    // std::unordered_set<F> use_variables;

    for (int i = 0; i < binarized.seq().size(); ++i)
    {
        auto f = binarized.seq(i);
        // test the form of the formula
        if (f.is(F::E) || !must_binarize(f) || !is_supported(f))
        {
            // variables.insert(f);
            continue;
        }

        EXPECT_TRUE(f.sig() == EQ);

        auto first_kind = f.seq(0).is_variable() && f.seq(1).is_constant();
        auto second_kind = f.seq(0).is_constant() && f.seq(1).is_variable();
        auto third_kind = f.seq(0).is_variable() && (f.seq(1).is(F::Seq) && f.seq(1).is_binary() && is_PIR_symbol(f.seq(1).sig()));
        auto fourth_kind = f.seq(1).is_variable() && (f.seq(0).is(F::Seq) && f.seq(0).is_binary() && is_PIR_symbol(f.seq(0).sig()));
        EXPECT_TRUE(first_kind || second_kind || third_kind || fourth_kind);

        // use_variables.insert(first_kind || third_kind ? f.seq(0) : f.seq(1));
        // if(third_kind){
        //     variables.insert(f.seq(1).seq(0));
        //     variables.insert(f.seq(1).seq(1));
        // }else if(fourth_kind){
        //     variables.insert(f.seq(0).seq(0));
        //     variables.insert(f.seq(0).seq(1));
        // }
    }

    // EXPECT_EQ(variables.size(), use_variables.size());
}

std::vector<std::pair<std::string, std::optional<std::string>>> get_test_cases(const std::string &test_directory)
{
    std::vector<std::pair<std::string, std::optional<std::string>>> test_cases;
    std::regex test_file_pattern(R"(.*\.fzn)");

    std::filesystem::path current_path = std::filesystem::current_path();
    std::filesystem::path back("..");

    for (const auto &entry : fs::directory_iterator(current_path / back / back / test_directory))
    {
        if (entry.is_regular_file())
        {
            const std::string filename = entry.path().filename().string();

            // Vérifie si le fichier correspond au format t<n>.input.fzn
            if (std::regex_match(filename, test_file_pattern) && filename.find(".output.fzn") == std::string::npos)
            {
                std::string input_file = entry.path().string();

                // Remplace ".input.fzn" par ".output.fzn" pour trouver le fichier de sortie correspondant
                std::string output_file = input_file;

                if (output_file.find(".input.fzn") != std::string::npos)
                {
                    output_file.replace(output_file.find(".input.fzn"), 10, ".output.fzn");
                }
                else
                {
                    output_file = "";
                }
                // Vérifie que le fichier de sortie existe
                if (fs::exists(output_file))
                {
                    test_cases.emplace_back(input_file, output_file);
                }
                else
                {
                    test_cases.emplace_back(input_file, std::nullopt);
                }
            }
        }
    }
    return test_cases;
}

class BinarizeTest : public ::testing::TestWithParam<std::pair<std::string, std::optional<std::string>>>
{
};

std::string testNameGenerator(const ::testing::TestParamInfo<std::pair<std::string, std::optional<std::string>>> &info)
{
    // Extraire le nom du fichier sans l'extension pour un nom plus lisible
    std::string filename = fs::path(std::get<0>(info.param)).stem().stem().string();
    filename.erase(std::remove(filename.begin(), filename.end(), '-'), filename.end());
    filename.erase(std::remove(filename.begin(), filename.end(), '_'), filename.end());
    filename.erase(std::remove(filename.begin(), filename.end(), '.'), filename.end());

    return filename;
}

TEST_P(BinarizeTest, RunTest)
{
    auto test_case = GetParam();
    test_binarize(test_case);
}

INSTANTIATE_TEST_SUITE_P(
    DomainTest,                                                   
    BinarizeTest,                                                 
    ::testing::ValuesIn(get_test_cases("tests/data/fzn/domain")), 
    testNameGenerator                                             
);

INSTANTIATE_TEST_SUITE_P(
    BinaryConstraintTest,                                         
    BinarizeTest,                                                 
    ::testing::ValuesIn(get_test_cases("tests/data/fzn/binary")), 
    testNameGenerator                                             
);

INSTANTIATE_TEST_SUITE_P(
    NaryTest,                                                   
    BinarizeTest,                                              
    ::testing::ValuesIn(get_test_cases("tests/data/fzn/nary")),
    testNameGenerator                                           
);

INSTANTIATE_TEST_SUITE_P(
    WordpressInstancesTest,                                          
    BinarizeTest,                                                    
    ::testing::ValuesIn(get_test_cases("tests/data/fzn/wordpress")), 
    testNameGenerator                                                
);

INSTANTIATE_TEST_SUITE_P(
    MZN2022InstancesTest,                                                     
    BinarizeTest,                                                             
    ::testing::ValuesIn(get_test_cases("../turbo/benchmarks/data/mzn2022/")), 
    testNameGenerator                                                         
);

INSTANTIATE_TEST_SUITE_P(
    EasyInstancesTest,                                                
    BinarizeTest,                                                     
    ::testing::ValuesIn(get_test_cases("../turbo/benchmarks/data/")), 
    testNameGenerator                                                 
);