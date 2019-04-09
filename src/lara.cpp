// ===========================================================================
//                LaRA: Lagrangian Relaxed structural Alignment
// ===========================================================================
// Copyright (c) 2016-2019, Jörg Winkler, Freie Universität Berlin
// Copyright (c) 2016-2019, Gianvito Urgese, Politecnico di Torino
// Copyright (c) 2006-2019, Knut Reinert, Freie Universität Berlin
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
// * Neither the name of Jörg Winkler, Gianvito Urgese, Knut Reinert,
//   the FU Berlin or the Politecnico di Torino nor the names of
//   its contributors may be used to endorse or promote products derived
//   from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL KNUT REINERT OR THE FU BERLIN BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
// OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
// DAMAGE.

//#ifdef WITH_OPENMP
//#include <omp.h>
//#endif
//
//#include "data_types.hpp"
//#include "io.hpp"
//#include "lagrange.hpp"
//#include "parameters.hpp"
//#include "subgradient_solver.hpp"
//
//int main (int argc, char const ** argv)
//{
//    // Parse arguments and options.
//    lara::Parameters params(argc, argv);
//    if (params.status != lara::Status::CONTINUE)
//        return params.status == lara::Status::EXIT_OK ? 0 : 1;
//
//    // Read input files and prepare structured sequences.
//    lara::InputStorage store(params);
//    size_t const problem_size = store.size() * (store.size() - 1) / 2;
//    _LOG(1, "Attempting to solve " << problem_size << " structural alignments with max. " << params.num_threads
//                                   << " threads." << std::endl);
//    _LOG(2, store << std::endl);
//    lara::OutputTCoffeeLibrary tcLib(store);
//    lara::SubgradientSolverMulti solverMulti(store, params);
//    solverMulti.solve(tcLib);
//
//    if (problem_size > 1ul)
//        tcLib.print(params.outFile);
//}

#include <numeric>
#include <vector>

#include <seqan/simd.h>

#include "score.hpp"
int main (int argc, char const ** argv)
{
    (void) argc;
    (void) argv;

    using ScoreType = seqan::Score<int32_t, seqan::RnaStructureScore>;
    using AlignmentType = seqan::Align<std::vector<unsigned>, seqan::ArrayGaps>;

    // Input Sequences
    seqan::Rna5String seq1 = "GGGGCCTTAG";
    seqan::Rna5String seq2 = "GGGCTCGTAG";

    // Score for (seq1, seq2)
    ScoreType scA;
    scA.data_gap_open = -24;
    scA.data_gap_extend = -8;
    scA.dim = seqan::length(seq2);
    scA.matrix = std::vector<int32_t>
    {
        26, 26, 26, 17, 20, 17, 26, -2, -174762, -174762,
        26, 26, 26, 17, 20, 17, 26, -2, -2, -174762,
        26, 26, 26, 17, 20, 17, 26, -2, -2, 3,
        26, 26, 26, 12, 20, 17, 26, -2, -2, 3,
        17, 17, 17, 30, 17, 26, 17, -1, -3, -6,
        17, 17, 17, 26, 26, 22, 17, -1, -3, 2,
        19, 19, 19, 21, 25, 25, 15, 3, -1, 7,
        17, 17, 17, 19, 24, 19, 17, 3, -1, 9,
        -174762, 20, 20, 18, 20, 18, 24, -1, 5, 10,
        -174762, 25, 25, 16, 19, 16, 25, -2, -2, 3
    };



    seqan::Rna5String seq3 = "GGUACGACC"; // 9
    seqan::Rna5String seq4 = "GUCACGAC";  // 8

    // Score for (seq3, seq4)
    ScoreType scB;
    scB.data_gap_open = -24;
    scB.data_gap_extend = -8;
    scB.dim = seqan::length(seq4); // 8
    scB.matrix = std::vector<int32_t>
    {
    //   G   U   C   A   C   G   A   C
        10,  8, -6, -2, -6,  3, -2, -6, // G
        18,  3, -6, -2, -6,  3, -2, -6, // G
         7, 18, -1, -1, -1, -2, -1, -1, // U
        -2, -1, -3,  5, -3, -2,  5, -3, // A
        -6, -1,  3, -3,  3, -6, -3,  3, // C
         3, -2, -6, -2, -6,  3, -2, -6, // G
        -2, -1, -3,  5, -3, -2, 11,  7, // A
        -6, -1,  3, -3,  3, -6, 12, 10, // C
        -6, -1,  3, -3,  3, -6,  7, 18  // C
    };

    // Integer Sequences
    std::vector<unsigned> iseq1(seqan::length(seq1));
    std::vector<unsigned> iseq2(seqan::length(seq2));
    std::vector<unsigned> iseq3(seqan::length(seq3));
    std::vector<unsigned> iseq4(seqan::length(seq4));
    std::iota(iseq1.begin(), iseq1.end(), 0u);
    std::iota(iseq2.begin(), iseq2.end(), 0u);
    std::iota(iseq3.begin(), iseq3.end(), 0u);
    std::iota(iseq4.begin(), iseq4.end(), 0u);

    // Alignment (seq1, seq2)
    AlignmentType alignA;
    seqan::resize(seqan::rows(alignA), 2);
    seqan::assignSource(seqan::row(alignA, 0), iseq1);
    seqan::assignSource(seqan::row(alignA, 1), iseq2);
    std::cerr << "start globalAlignment" << std::endl;
    int32_t resA = seqan::globalAlignment(alignA, scA, seqan::AffineGaps());
    std::cerr << "finish globalAlignment" << std::endl;


    // Alignment (seq3, seq4)
    AlignmentType alignB;
    seqan::resize(seqan::rows(alignB), 2);
    seqan::assignSource(seqan::row(alignB, 0), iseq3);
    seqan::assignSource(seqan::row(alignB, 1), iseq4);

    std::cerr << "start globalAlignment" << std::endl;
    int32_t resB = seqan::globalAlignment(alignB, scB, seqan::AffineGaps());
    std::cerr << "finish globalAlignment" << std::endl;






    // SIMD VERSION

    //using SimdVectorType = typename seqan::SimdVector<int32_t>::Type;
    using SimdScoreType = seqan::Score<int32_t, seqan::RnaStructureScoreSimd>;
    std::pair<size_t, size_t> simd_dim{std::max(seqan::length(seq1), seqan::length(seq3)),  // 10
                                       std::max(seqan::length(seq2), seqan::length(seq4))}; // 10

    SimdScoreType scST;
    scST.data_gap_open = -24; //seqan::createVector<SimdVectorType>(-24);
    scST.data_gap_extend = -8; //seqan::createVector<SimdVectorType>(-8);
    scST.dim = simd_dim.second;

    resize(scST.matrix, simd_dim.first * simd_dim.second); // 10 * 10
    for (size_t idx = 0ul; idx < length(scST.matrix); ++idx)
    {
        size_t i = idx / scST.dim;
        size_t j = idx % scST.dim;
        seqan::fillVector(scST.matrix[idx],
                          (j < scA.dim && i < seqan::length(seq1) ? scA.matrix[i * scA.dim + j] : 0u),
                          (j < scB.dim && i < seqan::length(seq3) ? scB.matrix[i * scB.dim + j] : 0u),
                          0u, 0u);

    }

    seqan::StringSet<seqan::Gaps<std::vector<unsigned>, seqan::ArrayGaps>> gapsH;
    seqan::StringSet<seqan::Gaps<std::vector<unsigned>, seqan::ArrayGaps>> gapsV;
    seqan::reserve(gapsH, 2);
    seqan::reserve(gapsV, 2);

    appendValue(gapsH, seqan::Gaps<std::vector<unsigned>, seqan::ArrayGaps>{iseq1});
    appendValue(gapsV, seqan::Gaps<std::vector<unsigned>, seqan::ArrayGaps>{iseq2});
    appendValue(gapsH, seqan::Gaps<std::vector<unsigned>, seqan::ArrayGaps>{iseq3});
    appendValue(gapsV, seqan::Gaps<std::vector<unsigned>, seqan::ArrayGaps>{iseq4});

    std::cerr << "start SimdGlobalAlignment" << std::endl;
    seqan::String<int32_t> resST = seqan::globalAlignment(gapsH, gapsV, scST);
    std::cerr << "finish SimdGlobalAlignment" << std::endl;

    // Print results
    using RowIteratorType = seqan::Iterator<seqan::Row<AlignmentType>::Type>::Type;

    std::cerr << resA << "\n" << seqan::length(seqan::row(alignA, 0)) << " " << seqan::length(seqan::row(alignA, 1)) << std::endl;
    for (RowIteratorType it = seqan::begin(seqan::row(alignA, 0)); it != seqan::end(seqan::row(alignA, 0)); ++it)
        std::cerr << seq1[*it];
    std::cerr << std::endl;
    for (RowIteratorType it = seqan::begin(seqan::row(alignA, 1)); it != seqan::end(seqan::row(alignA, 1)); ++it)
        std::cerr << seq2[*it];
    std::cerr << std::endl;

    std::cerr << resB << "\n" << seqan::length(seqan::row(alignB, 0)) << " " << seqan::length(seqan::row(alignB, 1)) << std::endl;
    for (RowIteratorType it = seqan::begin(seqan::row(alignB, 0)); it != seqan::end(seqan::row(alignB, 0)); ++it)
        std::cerr << (seqan::isGap(it) ? seqan::gapValue<char>() : (char)(seq3[*it]));
    std::cerr << std::endl;
    for (RowIteratorType it = seqan::begin(seqan::row(alignB, 1)); it != seqan::end(seqan::row(alignB, 1)); ++it)
        std::cerr << (seqan::isGap(it) ? seqan::gapValue<char>() : (char)(seq4[*it]));
    std::cerr << std::endl;


    std::cerr << "\nSIMD ALIGNMENTS" << std::endl;
    std::cerr << resST[0] << "\n" << seqan::length(gapsH[0]) << " " << seqan::length(gapsV[0]) << std::endl;
    for (RowIteratorType it = seqan::begin(gapsH[0]); it != seqan::end(gapsH[0]); ++it)
        std::cerr << (seqan::isGap(it) ? seqan::gapValue<char>() : (char)(seq1[*it]));
    std::cerr << std::endl;
    for (RowIteratorType it = seqan::begin(gapsV[0]); it != seqan::end(gapsV[0]); ++it)
        std::cerr << (seqan::isGap(it) ? seqan::gapValue<char>() : (char)(seq2[*it]));
    std::cerr << std::endl;

    std::cerr << resST[1] << "\n" << seqan::length(gapsH[1]) << " " << seqan::length(gapsV[1]) << std::endl;
    for (RowIteratorType it = seqan::begin(gapsH[1]); it != seqan::end(gapsH[1]); ++it)
        std::cerr << (seqan::isGap(it) ? seqan::gapValue<char>() : (char)(seq3[*it]));
    std::cerr << std::endl;
    for (RowIteratorType it = seqan::begin(gapsV[1]); it != seqan::end(gapsV[1]); ++it)
        std::cerr << (seqan::isGap(it) ? seqan::gapValue<char>() : (char)(seq4[*it]));
    std::cerr << std::endl;
}
