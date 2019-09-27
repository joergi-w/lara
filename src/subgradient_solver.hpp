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

#pragma once

/*!\file subgradient_solver.hpp
 * \brief This file contains the Subgradient Solver for LaRA.
 */

#ifdef WITH_OPENMP
#include <omp.h>
#endif

#include <chrono>
#include <iostream>
#include <iterator>
#include <list>
#include <ostream>
#include <utility>
#include <vector>

#include <seqan/sequence.h>
#include <seqan/simd.h>

#include "data_types.hpp"
#include "lagrange.hpp"
#include "parameters.hpp"
#include "score.hpp"

namespace lara
{

class SubgradientSolver
{
public:
    Lagrange lagrange;
//    float stepSizeFactor;
//    float bestLowerBound;
//    float bestUpperBound;
//    float currentLowerBound;
//    float currentUpperBound;
//    size_t nondecreasingRounds;
//    unsigned remainingIterations;
    PosPair sequenceIndices;

    std::vector<float> subgradient{};
    std::vector<float> dual{};
    std::list<size_t> subgradientIndices{};

    SubgradientSolver(PosPair indices,
                      InputStorage const & store,
                      Parameters & params,
                      RnaScoreType * score,
                      size_t seqIdx):
        lagrange(store[indices.first], store[indices.second], params, score, seqIdx),
//        stepSizeFactor{params.stepSizeFactor},
//        bestLowerBound{negInfinity},
//        bestUpperBound{posInfinity},
//        currentLowerBound{negInfinity},
//        currentUpperBound{posInfinity},
//        nondecreasingRounds{0ul},
//        remainingIterations{params.numIterations},
        sequenceIndices{indices}
    {
        subgradient.resize(lagrange.getDimension());
        dual.resize(subgradient.size());
    }

    SubgradientSolver()                                      = delete;
    SubgradientSolver(SubgradientSolver const &)             = default;
    SubgradientSolver(SubgradientSolver &&)                  = default;
    SubgradientSolver & operator=(SubgradientSolver const &) = default;
    SubgradientSolver & operator=(SubgradientSolver &&)      = default;
    ~SubgradientSolver()                                     = default;
};

#ifdef SEQAN_SIMD_ENABLED
class SubgradientSolverMulti
{
private:
    using ScoreVec = typename seqan::SimdVector<ScoreType>::Type;
    using UIntVec = typename seqan::SimdVector<uint32_t>::Type;
    // parameters
    InputStorage const & store;
    Parameters & params;

    // sort alignment order for the first sequence length
    struct longer_seq
    {
        InputStorage const & store;

        bool operator()(PosPair lhs, PosPair rhs) const
        {
            if (seqan::length(store[lhs.first].sequence) > seqan::length(store[rhs.first].sequence))
                return true;
            else if (seqan::length(store[lhs.first].sequence) == seqan::length(store[rhs.first].sequence))
                return seqan::length(store[lhs.second].sequence) >= seqan::length(store[rhs.second].sequence);
            else
                return false;
        }
    };

    std::set<PosPair, longer_seq> inputPairs;
    std::vector<SubgradientSolver> solvers;

    typedef struct _BoundInfo
    {
        ScoreVec bestLower;
        ScoreVec bestUpper;
        ScoreVec currentLower;
        ScoreVec currentUpper;
        UIntVec nondecreasing;
        UIntVec stepFactor;
        UIntVec remainingIterations;

        explicit _BoundInfo(float stepSizeFactor, unsigned numIterations)
        {
            bestLower = seqan::createVector<ScoreVec>(std::numeric_limits<ScoreType>::lowest());
            bestUpper = seqan::createVector<ScoreVec>(std::numeric_limits<ScoreType>::max());
            currentLower = seqan::createVector<ScoreVec>(std::numeric_limits<ScoreType>::lowest());
            currentUpper = seqan::createVector<ScoreVec>(std::numeric_limits<ScoreType>::max());
            nondecreasing = seqan::createVector<UIntVec>(0u);
            stepFactor = seqan::createVector<UIntVec>(static_cast<uint32_t>(stepSizeFactor * factor2int));
            remainingIterations = seqan::createVector<UIntVec>(numIterations);
        }
    } BoundInfo;
    std::vector<BoundInfo> bounds;


public:
    explicit SubgradientSolverMulti(InputStorage const & _store, Parameters & _params)
        : store(_store), params(_params), inputPairs(longer_seq{store})
    {
        // Add the sequence index pairs to the set of alignments, longer sequence first.
        for (size_t idxA = 0ul; idxA < store.size() - 1ul; ++idxA)
            for (size_t idxB = idxA + 1ul; idxB < store.size(); ++idxB)
                if (seqan::length(store[idxA].sequence) >= seqan::length(store[idxB].sequence))
                    inputPairs.emplace(idxA, idxB);
                else
                    inputPairs.emplace(idxB, idxA);
    }

    void solve(lara::OutputTCoffeeLibrary & results)
    {
        _LOG(1, "3) Solve " << inputPairs.size() << " structural alignments..." << std::endl);
        if (inputPairs.empty())
            return;

        size_t const simd_len = seqan::LENGTH<ScoreVec>::VALUE;
        static_assert(simd_len == seqan::LENGTH<UIntVec>::VALUE, "Incompatible vector size!");

        // Determine number of parallel alignments.
        Clock::time_point timeInit = Clock::now();
        size_t const num_parallel = std::min(simd_len * params.threads, inputPairs.size());
        size_t const num_threads = (num_parallel - 1) / simd_len + 1;

        // We iterate over all pairs of input sequences, starting with the longest.
        auto iter = inputPairs.cbegin(); // iter -> pair of sequence indices

        // Initialise the alignments.
        std::vector<std::pair<seqan::StringSet<GappedSeq>, seqan::StringSet<GappedSeq>>> alignments(num_threads);

        // Integer sequence from 0 until length of longest seq -1
        seqan::String<unsigned> integerSeq;
        seqan::resize(integerSeq, seqan::length(store[iter->first].sequence));
        std::iota(begin(integerSeq), end(integerSeq), 0u);

        // Store the integer sequences.
        using PrefixType = seqan::Prefix<seqan::String<unsigned>>::Type;
        seqan::StringSet<seqan::String<unsigned>> seq1;
        seqan::StringSet<seqan::String<unsigned>> seq2;
        seqan::reserve(seq1, num_parallel);
        seqan::reserve(seq2, num_parallel);

        // Initialise the scores.
        std::vector<RnaScoreType> scores(num_threads);
        size_t const max_2nd_length = seqan::length(store[iter->second].sequence);
        auto const go = static_cast<ScoreType>(params.rnaScore.data_gap_open * factor2int);
        auto const ge = static_cast<ScoreType>(params.rnaScore.data_gap_extend * factor2int);

        // Initialise the solvers.
        solvers.reserve(num_parallel);
        bounds.resize(num_threads, BoundInfo(params.stepSizeFactor, params.numIterations));

        for (iter = inputPairs.cbegin(); solvers.size() < num_parallel; ++iter)
        {
            size_t const aliIdx = solvers.size() / simd_len;
            size_t const seqIdx = solvers.size() % simd_len;
            auto const len = std::make_pair(length(store[iter->first].sequence), length(store[iter->second].sequence));

            // Once for each chunk of size simd_len.
            if (seqIdx == 0)
            {
                // Initialise the alignments.
                seqan::reserve(alignments[aliIdx].first, simd_len);
                seqan::reserve(alignments[aliIdx].second, simd_len);

                // Initialise the scores.
                scores[aliIdx].init(len.first, std::min(max_2nd_length, len.first), go, ge);
                _LOG(2, "     Resize matrix: " << len.first << "*" << std::min(max_2nd_length, len.first) << std::endl);
            }

            // Fill the alignments.
            appendValue(seq1, PrefixType(integerSeq, len.first));
            appendValue(seq2, PrefixType(integerSeq, len.second));
            appendValue(alignments[aliIdx].first, GappedSeq(seqan::back(seq1)));
            appendValue(alignments[aliIdx].second, GappedSeq(seqan::back(seq2)));

            // Fill the solvers.
            solvers.emplace_back(*iter, store, params, &(scores[aliIdx]), seqIdx);
        }
        SEQAN_ASSERT_EQ(num_parallel, solvers.size());
        SEQAN_ASSERT_EQ(num_parallel, seqan::length(seq1));
        SEQAN_ASSERT_EQ(num_parallel, seqan::length(seq2));
        SEQAN_ASSERT_EQ(num_threads, seqan::length(alignments));
        _LOG(1, "   * set up initial " << num_parallel << " structural alignments -> " << timeDiff(timeInit) << "ms"
                << std::endl);

        // fill the last alignment up to reach simd_len
        for (size_t idx = seqan::length(alignments.back().first); idx < simd_len; ++idx)
        {
            appendValue(alignments.back().first, GappedSeq(seqan::back(seq1)));
            appendValue(alignments.back().second, GappedSeq(seqan::back(seq2)));
        }

        // Definitions for SIMD alignment
        using SimdT = seqan::Score<ScoreVec,
                                   seqan::ScoreSimdWrapper<seqan::Score<ScoreType, seqan::PositionSpecificScoreSimd>>>;
        using ConfigT = seqan::AlignConfig2<seqan::DPGlobal,
                                            seqan::DPBandConfig<seqan::BandOff>,
                                            typename seqan::SubstituteAlignConfig_<seqan::AlignConfig<>>::Type>;
        using GapModelT = typename seqan::SubstituteAlgoTag_<seqan::Gotoh>::Type;
        UIntVec const nullvec = seqan::createVector<UIntVec>(0);
        UIntVec const ones = seqan::createVector<UIntVec>(1);
        UIntVec const twos = seqan::createVector<UIntVec>(2);
        UIntVec const maxiter = seqan::createVector<UIntVec>(params.maxNondecrIterations);

        Clock::duration durationAlign{};
        Clock::duration durationMatching{};
        Clock::duration durationUpdate{};
        Clock::duration durationSerial{};
        Clock::time_point timeIter = Clock::now();

        // in parallel for each (SIMD) alignment
        #pragma omp parallel for num_threads(params.threads)
        for (size_t aliIdx = 0ul; aliIdx < num_threads; ++aliIdx)
        {
            Clock::duration durationThreadAlign{};
            Clock::duration durationThreadMatching{};
            Clock::duration durationThreadUpdate{};
            Clock::time_point timeThreadSerial = Clock::now();
            auto const interval = std::make_pair(aliIdx * simd_len, std::min((aliIdx + 1) * simd_len, num_parallel));
            size_t num_at_work = interval.second - interval.first;
            std::vector<bool> at_work(num_at_work, true);
            scores[aliIdx].updateLongestSeq(seq1, seq2, interval);

            seqan::StringSet<seqan::String<seqan::TraceSegment_<size_t, size_t>>> trace;
            seqan::resize(trace, simd_len, seqan::Exact());

            // loop the thread until there is no more work to do
            while (num_at_work > 0ul)
            {
                BoundInfo & bound = bounds[aliIdx];

                // Performs the structural alignment. Returns the dual value (upper bound, solution of relaxed problem).
                Clock::time_point timeCurrent = Clock::now();

                seqan::String<ScoreType> res = seqan::globalAlignment(alignments[aliIdx].first,
                                                                      alignments[aliIdx].second,
                                                                      scores[aliIdx]);
                for (size_t idx = 0ul; idx < simd_len; ++idx)
                    bound.currentUpper[idx] = res[idx];

//                seqan::StringSet<std::remove_const_t<typename seqan::Source<typename seqan::Value<seqan::StringSet<GappedSeq>>::Type>::Type>, seqan::Dependent<> > depSetH;
//                seqan::StringSet<std::remove_const_t<typename seqan::Source<typename seqan::Value<seqan::StringSet<GappedSeq>>::Type>::Type>, seqan::Dependent<> > depSetV;
//                reserve(depSetH, simd_len);
//                reserve(depSetV, simd_len);
//                for (size_t idx = 0ul; idx < simd_len; ++idx)
//                {
//                    seqan::appendValue(depSetH, seqan::source(alignments[aliIdx].first[idx]));
//                    seqan::appendValue(depSetV, seqan::source(alignments[aliIdx].second[idx]));
//                }
//
//                seqan::_prepareAndRunSimdAlignment(bound.currentUpper,
//                                                   trace,
//                                                   depSetH,
//                                                   depSetV,
//                                                   SimdT{scores[aliIdx]},
//                                                   ConfigT(),
//                                                   GapModelT());
//
//                for (size_t idx = 0; idx < simd_len; ++idx)
//                    _adaptTraceSegmentsTo(alignments[aliIdx].first[idx], alignments[aliIdx].second[idx], trace[idx]);
//
//                std::cerr << alignments[aliIdx].first[0] << "\n" << alignments[aliIdx].second[0] << "\n\n";

                durationThreadAlign += Clock::now() - timeCurrent;

                // Evaluate each alignment result and adapt multipliers.
                for (size_t idx = interval.first; idx < interval.second; ++idx)
                {
                    size_t const seqIdx = idx % simd_len;
                    if (at_work[seqIdx])
                    {
                        SubgradientSolver & ss = solvers[idx];
                        timeCurrent = Clock::now();
                        float const res = ss.lagrange.valid_solution(ss.subgradient,
                                                                     ss.subgradientIndices,
                                                                     std::make_pair(alignments[aliIdx].first[seqIdx],
                                                                                    alignments[aliIdx].second[seqIdx]),
                                                                     params.matching,
                                                                     params.rnaScore);
                        bound.currentLower[seqIdx] = static_cast<ScoreType>(res * factor2int);
                        durationThreadMatching += Clock::now() - timeCurrent;
                    }
                }

                auto cmp = seqan::cmpGt(bound.bestUpper, bound.currentUpper);
                bound.bestUpper = seqan::blend(bound.bestUpper, bound.currentUpper, cmp);
                bound.nondecreasing = seqan::blend(bound.nondecreasing, nullvec, cmp);

                    // compare upper and lower bound
//                    if (ss.currentUpperBound < ss.bestUpperBound)
//                    {
//                        ss.bestUpperBound = ss.currentUpperBound;
//                        ss.nondecreasingRounds = 0ul;
//                    }

                cmp = seqan::cmpGt(bound.currentLower, bound.bestLower);
                bound.bestLower = seqan::blend(bound.bestLower, bound.currentLower, cmp);
                bound.nondecreasing = seqan::blend(bound.nondecreasing, nullvec, cmp);

//                    if (ss.currentLowerBound > ss.bestLowerBound)
//                    {
//                        ss.bestLowerBound = ss.currentLowerBound;
//                        ss.nondecreasingRounds = 0ul;
//
//                    }

                bound.nondecreasing = bound.nondecreasing + ones;
                auto mask = seqan::cmpGt(bound.nondecreasing, maxiter);
                bound.stepFactor = seqan::blend(bound.stepFactor, bound.stepFactor / twos, mask);
                bound.nondecreasing = seqan::blend(bound.nondecreasing, nullvec, mask);

//                    if (ss.nondecreasingRounds++ >= params.maxNondecrIterations)
//                    {
//                        ss.stepSizeFactor /= 2.0f;
//                        ss.nondecreasingRounds = 0;
//                    }

                UIntVec stepSize = bound.stepFactor * (bound.bestUpper - bound.bestLower);
                bound.remainingIterations -= ones;

                for (size_t idx = interval.first; idx < interval.second; ++idx)
                {
                    size_t const seqIdx = idx % simd_len;
                    if (!at_work[seqIdx])
                        continue;

                    SubgradientSolver & ss = solvers[idx];
//                    float stepSize = ss.stepSizeFactor * (ss.bestUpperBound - ss.bestLowerBound) /
//                                     ss.subgradientIndices.size();

                    for (size_t si : ss.subgradientIndices)
                    {
                        ss.dual[si] -= static_cast<float>(stepSize[seqIdx] / factor2int) * ss.subgradient[si] /
                                       ss.subgradientIndices.size();
                        ss.subgradient[si] = 0.0f;
                    }
//                    --ss.remainingIterations;

                    SEQAN_ASSERT_MSG(!ss.subgradientIndices.empty() ||
                                     (bound.currentUpper[seqIdx] - bound.currentLower[seqIdx]) / factor2int < 0.1f,
                                     (std::string{"The bounds differ, although there are no subgradients. "} +
                                         "Problem in aligning sequences " +
                                         seqan::toCString(store[ss.sequenceIndices.first].name) + " and " +
                                         seqan::toCString(store[ss.sequenceIndices.second].name)).c_str());
                    SEQAN_ASSERT_GT_MSG(bound.bestUpper[seqIdx] + static_cast<ScoreType>(params.epsilon * factor2int),
                                        bound.bestLower[seqIdx],
                                        (std::string{"The lower boundary exceeds the upper boundary. "} +
                                            "Problem in aligning sequences " +
                                            seqan::toCString(store[ss.sequenceIndices.first].name) + " and " +
                                            seqan::toCString(store[ss.sequenceIndices.second].name)).c_str());

                    // The alignment is finished.
                    if ((bound.bestUpper[seqIdx] - bound.bestLower[seqIdx]) / factor2int < params.epsilon || bound.remainingIterations[seqIdx] == 0u)
                    {
                        PosPair currentSeqIdx{};
                        #pragma omp critical (finished_alignment)
                        {
                            // write results
                            results.addAlignment(ss.lagrange, ss.sequenceIndices, params);
                            _LOG(2, "     Thread " << aliIdx << "." << seqIdx << " finished alignment "
                                    << ss.sequenceIndices.first << "/" << ss.sequenceIndices.second << std::endl);

                            if (iter == inputPairs.cend())
                            {
                                at_work[seqIdx] = false;
                                --num_at_work;
                            }
                            else
                            {
                                currentSeqIdx = *iter;
                                ++iter;
                            }
                        } // end critical region

                        if (at_work[seqIdx])
                        {
                            // Reset scores.
                            scores[aliIdx].reset(seqIdx);

                            // Set new sequences.
                            seq1[idx] = PrefixType(integerSeq, length(store[currentSeqIdx.first].sequence));
                            seq2[idx] = PrefixType(integerSeq, length(store[currentSeqIdx.second].sequence));
                            alignments[aliIdx].first[seqIdx] = GappedSeq(seq1[idx]);
                            alignments[aliIdx].second[seqIdx] = GappedSeq(seq2[idx]);

                            // Set new score matrix.
                            solvers[idx] = SubgradientSolver(currentSeqIdx, store, params, &(scores[aliIdx]), seqIdx);
                            scores[aliIdx].updateLongestSeq(seq1, seq2, interval);
                        }
                    }
                    else
                    {
                        timeCurrent = Clock::now();
                        ss.lagrange.updateScores(ss.dual, ss.subgradientIndices, params.rnaScore);
                        durationThreadUpdate += Clock::now() - timeCurrent;
                    }
                }
            }

            #pragma omp critical (update_time)
            {
                durationAlign += durationThreadAlign;
                durationMatching += durationThreadMatching;
                durationUpdate += durationThreadUpdate;
                durationSerial += Clock::now() - timeThreadSerial;
            }
        } // end parallel for

        auto durationToSeconds = [] (Clock::duration duration)
            { return std::chrono::duration_cast<std::chrono::seconds>(duration).count(); };

        _LOG(1, "   * parallel iterations -> " << timeDiff<std::chrono::seconds>(timeIter) << "s" << std::endl
                << "     (serial: " << durationToSeconds(durationSerial)
                << "s, ali: " << durationToSeconds(durationAlign)
                << "s, match: " << durationToSeconds(durationMatching)
                << "s, update: " << durationToSeconds(durationUpdate) << "s)" << std::endl);
    }
};
#else

#endif

} // namespace lara
