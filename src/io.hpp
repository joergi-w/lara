// ===========================================================================
//                LaRA: Lagrangian Relaxed structural Alignment
// ===========================================================================
// Copyright (c) 2016-2018, Jörg Winkler, Freie Universität Berlin
// Copyright (c) 2016-2018, Gianvito Urgese, Politecnico di Torino
// Copyright (c) 2006-2018, Knut Reinert, Freie Universität Berlin
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

/*!\file io.hpp
 * \brief This file contains LaRA's file handling, i.e. reading input files and writing output.
 */

#include <iostream>
#include <ostream>
#include <sstream>

#include <seqan/file.h>
#include <seqan/graph_types.h>
#include <seqan/rna_io.h>
#include <seqan/seq_io.h>
#include <seqan/sequence.h>

#include "parameters.hpp"

extern "C" {
    #include <ViennaRNA/utils.h>
    #include <ViennaRNA/fold_vars.h>
    #include <ViennaRNA/fold.h>
    #include <ViennaRNA/part_func.h>
}

namespace lara
{

class InputStorage : public std::pair<seqan::RnaStructContents, seqan::RnaStructContents>
{
public:
    explicit InputStorage(Parameters const & params)
    {
        readRnaFile(first, params.inFile);
        readRnaFile(second, params.inFileRef);
        _VV(params, "Successfully read " << first.records.size() + second.records.size() << " records.");

        if (params.dotplotFile.size() == first.records.size() + second.records.size())
        {
            // Load base pair probabilities from dot plot file.
            unsigned fileIdx;
            for (fileIdx = 0u; fileIdx < first.records.size(); ++fileIdx)
                extractBppFromDotplot(first.records[fileIdx], params.dotplotFile[fileIdx]);
            for (seqan::RnaRecord & record : second.records)
                extractBppFromDotplot(record, params.dotplotFile[fileIdx++]);
            _VV(params, "Successfully extracted base pair probabilities from given dotplot files.");
        }
        else
        {
            // If not present, compute the weighted interaction edges using ViennaRNA functions.
            bool const logScoring = params.structureScoring == ScoringMode::LOGARITHMIC;
            bool usedVienna = false;
            for (seqan::RnaRecord & record : first.records)
                computeStructure(record, usedVienna, logScoring);
            for (seqan::RnaRecord & record : second.records)
                computeStructure(record, usedVienna, logScoring);
            if (usedVienna)
                _VV(params, "Computed missing base pair probabilities with ViennaRNA library.");
        }
    }

private:
    void readRnaFile(seqan::RnaStructContents & fileContent, seqan::CharString filename)
    {
        using namespace seqan;

        if (empty(filename))
            return;

        RnaStructFileIn rnaStructFile;
        if (open(rnaStructFile, toCString(filename), OPEN_RDONLY))
        {
            readRecords(fileContent, rnaStructFile, std::numeric_limits<unsigned>::max());
            close(rnaStructFile);
        }
        else
        {
            // Read the file.
            SeqFileIn seqFileIn(toCString(filename));
            StringSet<CharString>  ids;
            StringSet<IupacString> seqs;
            StringSet<CharString>  quals;
            readRecords(ids, seqs, quals, seqFileIn);
            close(seqFileIn);

            // Fill the data structures: identifier and sequence.
            resize(fileContent.records, length(ids));
            SEQAN_ASSERT_EQ(length(ids), length(seqs));
            for (typename Size<StringSet<CharString>>::Type idx = 0u; idx < length(ids); ++idx)
            {
                fileContent.records[idx].name     = ids[idx];
                fileContent.records[idx].sequence = convert<Rna5String>(seqs[idx]);
            }
            // For FastQ files: add quality annotation.
            if (length(quals) == length(ids))
            {
                for (typename Size<StringSet<CharString>>::Type idx = 0u; idx < length(ids); ++idx)
                    fileContent.records[idx].quality = quals[idx];
            }
        }
    }

    void extractBppFromDotplot(seqan::RnaRecord & rnaRecord, std::string const & dotplotFile)
    {
        using namespace seqan;

        double const minProb = 0.003; // taken from LISA > Lara

        // add vertices to graph
        RnaStructureGraph bppMatrGraph;
        for (size_t idx = 0u; idx < length(rnaRecord.sequence); ++idx)
            addVertex(bppMatrGraph.inter);

        // open dotplot file and read lines
        std::ifstream file(dotplotFile);
        std::string   line;
        while (std::getline(file, line))
        {
            if (line.find("ubox") == std::string::npos)
                continue;

            std::istringstream iss(line);
            unsigned           iPos, jPos;
            double             prob;
            if (iss >> iPos >> jPos >> prob) // read values from line
            {   // create edges for graph
                SEQAN_ASSERT(iPos > 0 && iPos <= length(rnaRecord.sequence));
                SEQAN_ASSERT(jPos > 0 && jPos <= length(rnaRecord.sequence));
                // convert indices from range 1..length to 0..length-1
                if (prob * prob > minProb) // dot plot contains sqrt(prob)
                    addEdge(bppMatrGraph.inter, iPos - 1, jPos - 1, log(prob * prob / minProb));
            }
        }
        bppMatrGraph.specs = CharString("ViennaRNA dot plot from file " + std::string(dotplotFile));
        append(rnaRecord.bppMatrGraphs, bppMatrGraph);
    }

    void computeStructure(seqan::RnaRecord & rnaRecord, bool & usedVienna, bool logStructureScoring)
    {
        if (!seqan::empty(rnaRecord.bppMatrGraphs))
            return;

        usedVienna = true;
        size_t const length = seqan::length(rnaRecord.sequence);
        seqan::String<char, seqan::CStyle> sequence{rnaRecord.sequence};

        // Compute the partition function and base pair probabilities with ViennaRNA.
        seqan::RnaStructureGraph bppMatrGraph;
        init_pf_fold(static_cast<int>(length));
        bppMatrGraph.energy = pf_fold(seqan::toCString(sequence), nullptr);
        bppMatrGraph.specs = seqan::CharString{"ViennaRNA pf_fold"};

        for (size_t idx = 0u; idx < length; ++idx)
            seqan::addVertex(bppMatrGraph.inter);

        double const  minProb = 0.003; // taken from LISA > Lara
        for (size_t i = 0u; i < length; ++i)
        {
            for (size_t j = i + 1u; j < length; ++j)
            {
                if (logStructureScoring)
                {
                    if (pr[iindx[i + 1] - (j + 1)] > minProb)
                        seqan::addEdge(bppMatrGraph.inter, i, j, log(pr[iindx[i + 1] - (j + 1)] / minProb));
                }
                else
                {
                    if (pr[iindx[i + 1] - (j + 1)] > 0.0)
                        seqan::addEdge(bppMatrGraph.inter, i, j, pr[iindx[i + 1] - (j + 1)]);
                }
            }
        }
        seqan::append(rnaRecord.bppMatrGraphs, bppMatrGraph);

        // Compute the fixed structure with ViennaRNA.
        auto * structure = new char[length + 1];
        initialize_fold(static_cast<int>(length));
        float energy = fold(seqan::toCString(sequence), structure);
        seqan::bracket2graph(rnaRecord.fixedGraphs, seqan::CharString{structure}); // appends the graph
        seqan::back(rnaRecord.fixedGraphs).energy = energy;
        seqan::back(rnaRecord.fixedGraphs).specs = seqan::CharString{"ViennaRNA fold"};
        delete[] structure;
    }
};

std::ostream & operator<<(std::ostream & stream, InputStorage const & store)
{
    using namespace seqan;
    if (length(store.first.records) != 0)
    {
        for (RnaRecord const & rec : store.first.records)
        {
            writeRecord(stream, rec, DotBracket());
        }
    }
    if (length(store.second.records) != 0)
    {
        for (RnaRecord const & rec : store.second.records)
            writeRecord(stream, rec, DotBracket());
    }
    return stream;
}

} // namespace lara

