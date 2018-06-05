//===- Schedule.cpp - Calculate an optimized schedule ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass generates an entirely new schedule tree from the data dependences
// and iteration domains. The new schedule tree is computed in two steps:
//
// 1) The isl scheduling optimizer is run
//
// The isl scheduling optimizer creates a new schedule tree that maximizes
// parallelism and tileability and minimizes data-dependence distances. The
// algorithm used is a modified version of the ``Pluto'' algorithm:
//
//   U. Bondhugula, A. Hartono, J. Ramanujam, and P. Sadayappan.
//   A Practical Automatic Polyhedral Parallelizer and Locality Optimizer.
//   In Proceedings of the 2008 ACM SIGPLAN Conference On Programming Language
//   Design and Implementation, PLDI ’08, pages 101–113. ACM, 2008.
//
// 2) A set of post-scheduling transformations is applied on the schedule tree.
//
// These optimizations include:
//
//  - Tiling of the innermost tilable bands
//  - Prevectorization - The choice of a possible outer loop that is strip-mined
//                       to the innermost level to enable inner-loop
//                       vectorization.
//  - Some optimizations for spatial locality are also planned.
//
// For a detailed description of the schedule tree itself please see section 6
// of:
//
// Polyhedral AST generation is more than scanning polyhedra
// Tobias Grosser, Sven Verdoolaege, Albert Cohen
// ACM Transactions on Programming Languages and Systems (TOPLAS),
// 37(4), July 2015
// http://www.grosser.es/#pub-polyhedral-AST-generation
//
// This publication also contains a detailed discussion of the different options
// for polyhedral loop unrolling, full/partial tile separation and other uses
// of the schedule tree.
//
//===----------------------------------------------------------------------===//

#include "polly/ScheduleOptimizer.h"
#include "polly/CodeGen/CodeGeneration.h"
#include "polly/DependenceInfo.h"
#include "polly/LinkAllPasses.h"
#include "polly/Options.h"
#include "polly/ScopInfo.h"
#include "polly/ScopPass.h"
#include "polly/Simplify.h"
#include "polly/Support/GICHelper.h"
#include "polly/Support/ISLOStream.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "isl/constraint.h"
#include "isl/ctx.h"
#include "isl/map.h"
#include "isl/options.h"
#include "isl/printer.h"
#include "isl/schedule.h"
#include "isl/schedule_node.h"
#include "isl/space.h"
#include "isl/union_map.h"
#include "isl/union_set.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

using namespace llvm;
using namespace polly;

#define DEBUG_TYPE "polly-opt-isl"

static cl::opt<std::string>
    OptimizeDeps("polly-opt-optimize-only",
                 cl::desc("Only a certain kind of dependences (all/raw)"),
                 cl::Hidden, cl::init("all"), cl::ZeroOrMore,
                 cl::cat(PollyCategory));

static cl::opt<std::string>
    SimplifyDeps("polly-opt-simplify-deps",
                 cl::desc("Dependences should be simplified (yes/no)"),
                 cl::Hidden, cl::init("yes"), cl::ZeroOrMore,
                 cl::cat(PollyCategory));

static cl::opt<int> MaxConstantTerm(
    "polly-opt-max-constant-term",
    cl::desc("The maximal constant term allowed (-1 is unlimited)"), cl::Hidden,
    cl::init(20), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> MaxCoefficient(
    "polly-opt-max-coefficient",
    cl::desc("The maximal coefficient allowed (-1 is unlimited)"), cl::Hidden,
    cl::init(20), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<std::string> FusionStrategy(
    "polly-opt-fusion", cl::desc("The fusion strategy to choose (min/max)"),
    cl::Hidden, cl::init("min"), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<std::string>
    MaximizeBandDepth("polly-opt-maximize-bands",
                      cl::desc("Maximize the band depth (yes/no)"), cl::Hidden,
                      cl::init("yes"), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<std::string> OuterCoincidence(
    "polly-opt-outer-coincidence",
    cl::desc("Try to construct schedules where the outer member of each band "
             "satisfies the coincidence constraints (yes/no)"),
    cl::Hidden, cl::init("no"), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> PrevectorWidth(
    "polly-prevect-width",
    cl::desc(
        "The number of loop iterations to strip-mine for pre-vectorization"),
    cl::Hidden, cl::init(4), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<bool> FirstLevelTiling("polly-tiling",
                                      cl::desc("Enable loop tiling"),
                                      cl::init(true), cl::ZeroOrMore,
                                      cl::cat(PollyCategory));

static cl::opt<int> LatencyVectorFma(
    "polly-target-latency-vector-fma",
    cl::desc("The minimal number of cycles between issuing two "
             "dependent consecutive vector fused multiply-add "
             "instructions."),
    cl::Hidden, cl::init(8), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> ThroughputVectorFma(
    "polly-target-throughput-vector-fma",
    cl::desc("A throughput of the processor floating-point arithmetic units "
             "expressed in the number of vector fused multiply-add "
             "instructions per clock cycle."),
    cl::Hidden, cl::init(1), cl::ZeroOrMore, cl::cat(PollyCategory));

// This option, along with --polly-target-2nd-cache-level-associativity,
// --polly-target-1st-cache-level-size, and --polly-target-2st-cache-level-size
// represent the parameters of the target cache, which do not have typical
// values that can be used by default. However, to apply the pattern matching
// optimizations, we use the values of the parameters of Intel Core i7-3820
// SandyBridge in case the parameters are not specified or not provided by the
// TargetTransformInfo.
static cl::opt<int> FirstCacheLevelAssociativity(
    "polly-target-1st-cache-level-associativity",
    cl::desc("The associativity of the first cache level."), cl::Hidden,
    cl::init(-1), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> FirstCacheLevelDefaultAssociativity(
    "polly-target-1st-cache-level-default-associativity",
    cl::desc("The default associativity of the first cache level"
             " (if not enough were provided by the TargetTransformInfo)."),
    cl::Hidden, cl::init(8), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> SecondCacheLevelAssociativity(
    "polly-target-2nd-cache-level-associativity",
    cl::desc("The associativity of the second cache level."), cl::Hidden,
    cl::init(-1), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> SecondCacheLevelDefaultAssociativity(
    "polly-target-2nd-cache-level-default-associativity",
    cl::desc("The default associativity of the second cache level"
             " (if not enough were provided by the TargetTransformInfo)."),
    cl::Hidden, cl::init(8), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> FirstCacheLevelSize(
    "polly-target-1st-cache-level-size",
    cl::desc("The size of the first cache level specified in bytes."),
    cl::Hidden, cl::init(-1), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> FirstCacheLevelDefaultSize(
    "polly-target-1st-cache-level-default-size",
    cl::desc("The default size of the first cache level specified in bytes"
             " (if not enough were provided by the TargetTransformInfo)."),
    cl::Hidden, cl::init(32768), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> SecondCacheLevelSize(
    "polly-target-2nd-cache-level-size",
    cl::desc("The size of the second level specified in bytes."), cl::Hidden,
    cl::init(-1), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> SecondCacheLevelDefaultSize(
    "polly-target-2nd-cache-level-default-size",
    cl::desc("The default size of the second cache level specified in bytes"
             " (if not enough were provided by the TargetTransformInfo)."),
    cl::Hidden, cl::init(262144), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> VectorRegisterBitwidth(
    "polly-target-vector-register-bitwidth",
    cl::desc("The size in bits of a vector register (if not set, this "
             "information is taken from LLVM's target information."),
    cl::Hidden, cl::init(-1), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> FirstLevelDefaultTileSize(
    "polly-default-tile-size",
    cl::desc("The default tile size (if not enough were provided by"
             " --polly-tile-sizes)"),
    cl::Hidden, cl::init(32), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::list<int>
    FirstLevelTileSizes("polly-tile-sizes",
                        cl::desc("A tile size for each loop dimension, filled "
                                 "with --polly-default-tile-size"),
                        cl::Hidden, cl::ZeroOrMore, cl::CommaSeparated,
                        cl::cat(PollyCategory));

static cl::opt<bool>
    SecondLevelTiling("polly-2nd-level-tiling",
                      cl::desc("Enable a 2nd level loop of loop tiling"),
                      cl::init(false), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> SecondLevelDefaultTileSize(
    "polly-2nd-level-default-tile-size",
    cl::desc("The default 2nd-level tile size (if not enough were provided by"
             " --polly-2nd-level-tile-sizes)"),
    cl::Hidden, cl::init(16), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::list<int>
    SecondLevelTileSizes("polly-2nd-level-tile-sizes",
                         cl::desc("A tile size for each loop dimension, filled "
                                  "with --polly-default-tile-size"),
                         cl::Hidden, cl::ZeroOrMore, cl::CommaSeparated,
                         cl::cat(PollyCategory));

static cl::opt<bool> RegisterTiling("polly-register-tiling",
                                    cl::desc("Enable register tiling"),
                                    cl::init(false), cl::ZeroOrMore,
                                    cl::cat(PollyCategory));

static cl::opt<int> RegisterDefaultTileSize(
    "polly-register-tiling-default-tile-size",
    cl::desc("The default register tile size (if not enough were provided by"
             " --polly-register-tile-sizes)"),
    cl::Hidden, cl::init(2), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> PollyPatternMatchingNcQuotient(
    "polly-pattern-matching-nc-quotient",
    cl::desc("Quotient that is obtained by dividing Nc, the parameter of the"
             "macro-kernel, by Nr, the parameter of the micro-kernel"),
    cl::Hidden, cl::init(256), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::list<int>
    RegisterTileSizes("polly-register-tile-sizes",
                      cl::desc("A tile size for each loop dimension, filled "
                               "with --polly-register-tile-size"),
                      cl::Hidden, cl::ZeroOrMore, cl::CommaSeparated,
                      cl::cat(PollyCategory));

static cl::opt<bool>
    PMBasedOpts("polly-pattern-matching-based-opts",
                cl::desc("Perform optimizations based on pattern matching"),
                cl::init(true), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<bool> OptimizedScops(
    "polly-optimized-scops",
    cl::desc("Polly - Dump polyhedral description of Scops optimized with "
             "the isl scheduling optimizer and the set of post-scheduling "
             "transformations is applied on the schedule tree"),
    cl::init(false), cl::ZeroOrMore, cl::cat(PollyCategory));

STATISTIC(ScopsProcessed, "Number of scops processed");
STATISTIC(ScopsRescheduled, "Number of scops rescheduled");
STATISTIC(ScopsOptimized, "Number of scops optimized");

STATISTIC(NumAffineLoopsOptimized, "Number of affine loops optimized");
STATISTIC(NumBoxedLoopsOptimized, "Number of boxed loops optimized");

#define THREE_STATISTICS(VARNAME, DESC)                                        \
  static Statistic VARNAME[3] = {                                              \
      {DEBUG_TYPE, #VARNAME "0", DESC " (original)", {0}, {false}},            \
      {DEBUG_TYPE, #VARNAME "1", DESC " (after scheduler)", {0}, {false}},     \
      {DEBUG_TYPE, #VARNAME "2", DESC " (after optimizer)", {0}, {false}}}

THREE_STATISTICS(NumBands, "Number of bands");
THREE_STATISTICS(NumBandMembers, "Number of band members");
THREE_STATISTICS(NumCoincident, "Number of coincident band members");
THREE_STATISTICS(NumPermutable, "Number of permutable bands");
THREE_STATISTICS(NumFilters, "Number of filter nodes");
THREE_STATISTICS(NumExtension, "Number of extension nodes");

STATISTIC(FirstLevelTileOpts, "Number of first level tiling applied");
STATISTIC(SecondLevelTileOpts, "Number of second level tiling applied");
STATISTIC(RegisterTileOpts, "Number of register tiling applied");
STATISTIC(PrevectOpts, "Number of strip-mining for prevectorization applied");
STATISTIC(MatMulOpts,
          "Number of matrix multiplication patterns detected and optimized");

/// Create an isl::union_set, which describes the isolate option based on
/// IsolateDomain.
///
/// @param IsolateDomain An isl::set whose @p OutDimsNum last dimensions should
///                      belong to the current band node.
/// @param OutDimsNum    A number of dimensions that should belong to
///                      the current band node.
static isl::union_set getIsolateOptions(isl::set IsolateDomain,
                                        unsigned OutDimsNum) {
  unsigned Dims = IsolateDomain.dim(isl::dim::set);
  assert(OutDimsNum <= Dims &&
         "The isl::set IsolateDomain is used to describe the range of schedule "
         "dimensions values, which should be isolated. Consequently, the "
         "number of its dimensions should be greater than or equal to the "
         "number of the schedule dimensions.");
  isl::map IsolateRelation = isl::map::from_domain(IsolateDomain);
  IsolateRelation = IsolateRelation.move_dims(isl::dim::out, 0, isl::dim::in,
                                              Dims - OutDimsNum, OutDimsNum);
  isl::set IsolateOption = IsolateRelation.wrap();
  isl::id Id = isl::id::alloc(IsolateOption.get_ctx(), "isolate", nullptr);
  IsolateOption = IsolateOption.set_tuple_id(Id);
  return isl::union_set(IsolateOption);
}

namespace {
/// Create an isl::union_set, which describes the specified option for the
/// dimension of the current node.
///
/// @param Ctx    An isl::ctx, which is used to create the isl::union_set.
/// @param Option The name of the option.
isl::union_set getDimOptions(isl::ctx Ctx, const char *Option) {
  isl::space Space(Ctx, 0, 1);
  auto DimOption = isl::set::universe(Space);
  auto Id = isl::id::alloc(Ctx, Option, nullptr);
  DimOption = DimOption.set_tuple_id(Id);
  return isl::union_set(DimOption);
}
} // namespace

/// Create an isl::union_set, which describes the option of the form
/// [isolate[] -> unroll[x]].
///
/// @param Ctx An isl::ctx, which is used to create the isl::union_set.
static isl::union_set getUnrollIsolatedSetOptions(isl::ctx Ctx) {
  isl::space Space = isl::space(Ctx, 0, 0, 1);
  isl::map UnrollIsolatedSetOption = isl::map::universe(Space);
  isl::id DimInId = isl::id::alloc(Ctx, "isolate", nullptr);
  isl::id DimOutId = isl::id::alloc(Ctx, "unroll", nullptr);
  UnrollIsolatedSetOption =
      UnrollIsolatedSetOption.set_tuple_id(isl::dim::in, DimInId);
  UnrollIsolatedSetOption =
      UnrollIsolatedSetOption.set_tuple_id(isl::dim::out, DimOutId);
  return UnrollIsolatedSetOption.wrap();
}

/// Make the last dimension of Set to take values from 0 to VectorWidth - 1.
///
/// @param Set         A set, which should be modified.
/// @param VectorWidth A parameter, which determines the constraint.
static isl::set addExtentConstraints(isl::set Set, int VectorWidth) {
  unsigned Dims = Set.dim(isl::dim::set);
  isl::space Space = Set.get_space();
  isl::local_space LocalSpace = isl::local_space(Space);
  isl::constraint ExtConstr = isl::constraint::alloc_inequality(LocalSpace);
  ExtConstr = ExtConstr.set_constant_si(0);
  ExtConstr = ExtConstr.set_coefficient_si(isl::dim::set, Dims - 1, 1);
  Set = Set.add_constraint(ExtConstr);
  ExtConstr = isl::constraint::alloc_inequality(LocalSpace);
  ExtConstr = ExtConstr.set_constant_si(VectorWidth - 1);
  ExtConstr = ExtConstr.set_coefficient_si(isl::dim::set, Dims - 1, -1);
  return Set.add_constraint(ExtConstr);
}

isl::set getPartialTilePrefixes(isl::set ScheduleRange, int VectorWidth) {
  unsigned Dims = ScheduleRange.dim(isl::dim::set);
  isl::set LoopPrefixes =
      ScheduleRange.drop_constraints_involving_dims(isl::dim::set, Dims - 1, 1);
  auto ExtentPrefixes = addExtentConstraints(LoopPrefixes, VectorWidth);
  isl::set BadPrefixes = ExtentPrefixes.subtract(ScheduleRange);
  BadPrefixes = BadPrefixes.project_out(isl::dim::set, Dims - 1, 1);
  LoopPrefixes = LoopPrefixes.project_out(isl::dim::set, Dims - 1, 1);
  return LoopPrefixes.subtract(BadPrefixes);
}

isl::schedule_node
ScheduleTreeOptimizer::isolateFullPartialTiles(isl::schedule_node Node,
                                               int VectorWidth) {
  assert(isl_schedule_node_get_type(Node.get()) == isl_schedule_node_band);
  Node = Node.child(0).child(0);
  isl::union_map SchedRelUMap = Node.get_prefix_schedule_relation();
  isl::map ScheduleRelation = isl::map::from_union_map(SchedRelUMap);
  isl::set ScheduleRange = ScheduleRelation.range();
  isl::set IsolateDomain = getPartialTilePrefixes(ScheduleRange, VectorWidth);
  auto AtomicOption = getDimOptions(IsolateDomain.get_ctx(), "atomic");
  isl::union_set IsolateOption = getIsolateOptions(IsolateDomain, 1);
  Node = Node.parent().parent();
  isl::union_set Options = IsolateOption.unite(AtomicOption);
  Node = Node.band_set_ast_build_options(Options);
  return Node;
}

isl::schedule_node ScheduleTreeOptimizer::prevectSchedBand(
    isl::schedule_node Node, unsigned DimToVectorize, int VectorWidth) {
  assert(isl_schedule_node_get_type(Node.get()) == isl_schedule_node_band);

  auto Space = isl::manage(isl_schedule_node_band_get_space(Node.get()));
  auto ScheduleDimensions = Space.dim(isl::dim::set);
  assert(DimToVectorize < ScheduleDimensions);

  if (DimToVectorize > 0) {
    Node = isl::manage(
        isl_schedule_node_band_split(Node.release(), DimToVectorize));
    Node = Node.child(0);
  }
  if (DimToVectorize < ScheduleDimensions - 1)
    Node = isl::manage(isl_schedule_node_band_split(Node.release(), 1));
  Space = isl::manage(isl_schedule_node_band_get_space(Node.get()));
  auto Sizes = isl::multi_val::zero(Space);
  Sizes = Sizes.set_val(0, isl::val(Node.get_ctx(), VectorWidth));
  Node =
      isl::manage(isl_schedule_node_band_tile(Node.release(), Sizes.release()));
  Node = isolateFullPartialTiles(Node, VectorWidth);
  Node = Node.child(0);
  // Make sure the "trivially vectorizable loop" is not unrolled. Otherwise,
  // we will have troubles to match it in the backend.
  Node = Node.band_set_ast_build_options(
      isl::union_set(Node.get_ctx(), "{ unroll[x]: 1 = 0 }"));
  Node = isl::manage(isl_schedule_node_band_sink(Node.release()));
  Node = Node.child(0);
  if (isl_schedule_node_get_type(Node.get()) == isl_schedule_node_leaf)
    Node = Node.parent();
  auto LoopMarker = isl::id::alloc(Node.get_ctx(), "SIMD", nullptr);
  PrevectOpts++;
  return Node.insert_mark(LoopMarker);
}

isl::schedule_node ScheduleTreeOptimizer::tileNode(isl::schedule_node Node,
                                                   const char *Identifier,
                                                   ArrayRef<int> TileSizes,
                                                   int DefaultTileSize) {
  auto Space = isl::manage(isl_schedule_node_band_get_space(Node.get()));
  auto Dims = Space.dim(isl::dim::set);
  auto Sizes = isl::multi_val::zero(Space);
  std::string IdentifierString(Identifier);
  for (unsigned i = 0; i < Dims; i++) {
    auto tileSize = i < TileSizes.size() ? TileSizes[i] : DefaultTileSize;
    Sizes = Sizes.set_val(i, isl::val(Node.get_ctx(), tileSize));
  }
  auto TileLoopMarkerStr = IdentifierString + " - Tiles";
  auto TileLoopMarker =
      isl::id::alloc(Node.get_ctx(), TileLoopMarkerStr, nullptr);
  Node = Node.insert_mark(TileLoopMarker);
  Node = Node.child(0);
  Node =
      isl::manage(isl_schedule_node_band_tile(Node.release(), Sizes.release()));
  Node = Node.child(0);
  auto PointLoopMarkerStr = IdentifierString + " - Points";
  auto PointLoopMarker =
      isl::id::alloc(Node.get_ctx(), PointLoopMarkerStr, nullptr);
  Node = Node.insert_mark(PointLoopMarker);
  return Node.child(0);
}

isl::schedule_node ScheduleTreeOptimizer::applyRegisterTiling(
    isl::schedule_node Node, ArrayRef<int> TileSizes, int DefaultTileSize) {
  Node = tileNode(Node, "Register tiling", TileSizes, DefaultTileSize);
  auto Ctx = Node.get_ctx();
  return Node.band_set_ast_build_options(isl::union_set(Ctx, "{unroll[x]}"));
}

static bool isSimpleInnermostBand(const isl::schedule_node &Node) {
  assert(isl_schedule_node_get_type(Node.get()) == isl_schedule_node_band);
  assert(isl_schedule_node_n_children(Node.get()) == 1);

  auto ChildType = isl_schedule_node_get_type(Node.child(0).get());

  if (ChildType == isl_schedule_node_leaf)
    return true;

  if (ChildType != isl_schedule_node_sequence)
    return false;

  auto Sequence = Node.child(0);

  for (int c = 0, nc = isl_schedule_node_n_children(Sequence.get()); c < nc;
       ++c) {
    auto Child = Sequence.child(c);
    if (isl_schedule_node_get_type(Child.get()) != isl_schedule_node_filter)
      return false;
    if (isl_schedule_node_get_type(Child.child(0).get()) !=
        isl_schedule_node_leaf)
      return false;
  }
  return true;
}

bool ScheduleTreeOptimizer::isTileableBandNode(isl::schedule_node Node) {
  if (isl_schedule_node_get_type(Node.get()) != isl_schedule_node_band)
    return false;

  if (isl_schedule_node_n_children(Node.get()) != 1)
    return false;

  if (!isl_schedule_node_band_get_permutable(Node.get()))
    return false;

  auto Space = isl::manage(isl_schedule_node_band_get_space(Node.get()));
  auto Dims = Space.dim(isl::dim::set);

  if (Dims <= 1)
    return false;

  return isSimpleInnermostBand(Node);
}

__isl_give isl::schedule_node
ScheduleTreeOptimizer::standardBandOpts(isl::schedule_node Node, void *User) {
  if (FirstLevelTiling) {
    Node = tileNode(Node, "1st level tiling", FirstLevelTileSizes,
                    FirstLevelDefaultTileSize);
    FirstLevelTileOpts++;
  }

  if (SecondLevelTiling) {
    Node = tileNode(Node, "2nd level tiling", SecondLevelTileSizes,
                    SecondLevelDefaultTileSize);
    SecondLevelTileOpts++;
  }

  if (RegisterTiling) {
    Node =
        applyRegisterTiling(Node, RegisterTileSizes, RegisterDefaultTileSize);
    RegisterTileOpts++;
  }

  if (PollyVectorizerChoice == VECTORIZER_NONE)
    return Node;

  auto Space = isl::manage(isl_schedule_node_band_get_space(Node.get()));
  auto Dims = Space.dim(isl::dim::set);

  for (int i = Dims - 1; i >= 0; i--)
    if (Node.band_member_get_coincident(i)) {
      Node = prevectSchedBand(Node, i, PrevectorWidth);
      break;
    }

  return Node;
}

/// Permute the two dimensions of the isl map.
///
/// Permute @p DstPos and @p SrcPos dimensions of the isl map @p Map that
/// have type @p DimType.
///
/// @param Map     The isl map to be modified.
/// @param DimType The type of the dimensions.
/// @param DstPos  The first dimension.
/// @param SrcPos  The second dimension.
/// @return        The modified map.
isl::map permuteDimensions(isl::map Map, isl::dim DimType, unsigned DstPos,
                           unsigned SrcPos) {
  assert(DstPos < Map.dim(DimType) && SrcPos < Map.dim(DimType));
  if (DstPos == SrcPos)
    return Map;
  isl::id DimId;
  if (Map.has_tuple_id(DimType))
    DimId = Map.get_tuple_id(DimType);
  auto FreeDim = DimType == isl::dim::in ? isl::dim::out : isl::dim::in;
  isl::id FreeDimId;
  if (Map.has_tuple_id(FreeDim))
    FreeDimId = Map.get_tuple_id(FreeDim);
  auto MaxDim = std::max(DstPos, SrcPos);
  auto MinDim = std::min(DstPos, SrcPos);
  Map = Map.move_dims(FreeDim, 0, DimType, MaxDim, 1);
  Map = Map.move_dims(FreeDim, 0, DimType, MinDim, 1);
  Map = Map.move_dims(DimType, MinDim, FreeDim, 1, 1);
  Map = Map.move_dims(DimType, MaxDim, FreeDim, 0, 1);
  if (DimId)
    Map = Map.set_tuple_id(DimType, DimId);
  if (FreeDimId)
    Map = Map.set_tuple_id(FreeDim, FreeDimId);
  return Map;
}

/// Check the form of the access relation.
///
/// Check that the access relation @p AccMap has the form M[i][j], where i
/// is a @p FirstPos and j is a @p SecondPos.
///
/// @param AccMap    The access relation to be checked.
/// @param FirstPos  The index of the input dimension that is mapped to
///                  the first output dimension.
/// @param SecondPos The index of the input dimension that is mapped to the
///                  second output dimension.
/// @return          True in case @p AccMap has the expected form and false,
///                  otherwise.
static bool isMatMulOperandAcc(isl::set Domain, isl::map AccMap, int &FirstPos,
                               int &SecondPos) {
  isl::space Space = AccMap.get_space();
  isl::map Universe = isl::map::universe(Space);

  if (Space.dim(isl::dim::out) != 2)
    return false;

  // MatMul has the form:
  // for (i = 0; i < N; i++)
  //   for (j = 0; j < M; j++)
  //     for (k = 0; k < P; k++)
  //       C[i, j] += A[i, k] * B[k, j]
  //
  // Permutation of three outer loops: 3! = 6 possibilities.
  int FirstDims[] = {0, 0, 1, 1, 2, 2};
  int SecondDims[] = {1, 2, 2, 0, 0, 1};
  for (int i = 0; i < 6; i += 1) {
    auto PossibleMatMul =
        Universe.equate(isl::dim::in, FirstDims[i], isl::dim::out, 0)
            .equate(isl::dim::in, SecondDims[i], isl::dim::out, 1);

    AccMap = AccMap.intersect_domain(Domain);
    PossibleMatMul = PossibleMatMul.intersect_domain(Domain);

    // If AccMap spans entire domain (Non-partial write),
    // compute FirstPos and SecondPos.
    // If AccMap != PossibleMatMul here (the two maps have been gisted at
    // this point), it means that the writes are not complete, or in other
    // words, it is a Partial write and Partial writes must be rejected.
    if (AccMap.is_equal(PossibleMatMul)) {
      if (FirstPos != -1 && FirstPos != FirstDims[i])
        continue;
      FirstPos = FirstDims[i];
      if (SecondPos != -1 && SecondPos != SecondDims[i])
        continue;
      SecondPos = SecondDims[i];
      return true;
    }
  }

  return false;
}

/// Does the memory access represent a non-scalar operand of the matrix
/// multiplication.
///
/// Check that the memory access @p MemAccess is the read access to a non-scalar
/// operand of the matrix multiplication or its result.
///
/// @param MemAccess The memory access to be checked.
/// @param MMI       Parameters of the matrix multiplication operands.
/// @return          True in case the memory access represents the read access
///                  to a non-scalar operand of the matrix multiplication and
///                  false, otherwise.
static bool isMatMulNonScalarReadAccess(MemoryAccess *MemAccess,
                                        MatMulInfoTy &MMI) {
  if (!MemAccess->isLatestArrayKind() || !MemAccess->isRead())
    return false;
  auto AccMap = MemAccess->getLatestAccessRelation();
  isl::set StmtDomain = MemAccess->getStatement()->getDomain();
  if (isMatMulOperandAcc(StmtDomain, AccMap, MMI.i, MMI.j) && !MMI.ReadFromC) {
    MMI.ReadFromC = MemAccess;
    return true;
  }
  if (isMatMulOperandAcc(StmtDomain, AccMap, MMI.i, MMI.k) && !MMI.A) {
    MMI.A = MemAccess;
    return true;
  }
  if (isMatMulOperandAcc(StmtDomain, AccMap, MMI.k, MMI.j) && !MMI.B) {
    MMI.B = MemAccess;
    return true;
  }
  DEBUG(dbgs() << "exit"
               << "\n");
  return false;
}

/// Check accesses to operands of the matrix multiplication.
///
/// Check that accesses of the SCoP statement, which corresponds to
/// the partial schedule @p PartialSchedule, are scalar in terms of loops
/// containing the matrix multiplication, in case they do not represent
/// accesses to the non-scalar operands of the matrix multiplication or
/// its result.
///
/// @param  PartialSchedule The partial schedule of the SCoP statement.
/// @param  MMI             Parameters of the matrix multiplication operands.
/// @return                 True in case the corresponding SCoP statement
///                         represents matrix multiplication and false,
///                         otherwise.
static bool containsOnlyMatrMultAcc(isl::map PartialSchedule,
                                    MatMulInfoTy &MMI) {
  auto InputDimId = PartialSchedule.get_tuple_id(isl::dim::in);
  auto *Stmt = static_cast<ScopStmt *>(InputDimId.get_user());
  unsigned OutDimNum = PartialSchedule.dim(isl::dim::out);
  assert(OutDimNum > 2 && "In case of the matrix multiplication the loop nest "
                          "and, consequently, the corresponding scheduling "
                          "functions have at least three dimensions.");
  auto MapI =
      permuteDimensions(PartialSchedule, isl::dim::out, MMI.i, OutDimNum - 1);
  auto MapJ =
      permuteDimensions(PartialSchedule, isl::dim::out, MMI.j, OutDimNum - 1);
  auto MapK =
      permuteDimensions(PartialSchedule, isl::dim::out, MMI.k, OutDimNum - 1);

  // DEBUG(dbgs() << "MApi" << MapI << "\n");
  // DEBUG(dbgs() << "Mapj" << MapJ << "\n");
  // DEBUG(dbgs() << "Mapk" << MapK << "\n");

  auto Accesses = getAccessesInOrder(*Stmt);
  for (auto *MemA = Accesses.begin(); MemA != Accesses.end() - 1; MemA++) {
    auto *MemAccessPtr = *MemA;
    // DEBUG(dbgs() << MemAccessPtr->isStrideZero(MapI) << "\n");
    if (MemAccessPtr->isLatestArrayKind() && MemAccessPtr != MMI.WriteToC &&
        !isMatMulNonScalarReadAccess(MemAccessPtr, MMI) &&
        !(MemAccessPtr->isStrideZero(MapI)) &&
        MemAccessPtr->isStrideZero(MapJ) && MemAccessPtr->isStrideZero(MapK))
      return false;
  }
  return true;
}

/// Check for dependencies corresponding to the matrix multiplication.
///
/// Check that there is only true dependence of the form
/// S(..., k, ...) -> S(..., k + 1, …), where S is the SCoP statement
/// represented by @p Schedule and k is @p Pos. Such a dependence corresponds
/// to the dependency produced by the matrix multiplication.
///
/// @param  Schedule The schedule of the SCoP statement.
/// @param  D The SCoP dependencies.
/// @param  Pos The parameter to describe an acceptable true dependence.
///             In case it has a negative value, try to determine its
///             acceptable value.
/// @return True in case dependencies correspond to the matrix multiplication
///         and false, otherwise.
static bool containsOnlyMatMulDep(isl::map Schedule, const Dependences *D,
                                  int &Pos) {
  auto Dep = isl::manage(D->getDependences(Dependences::TYPE_RAW));
  auto Red = isl::manage(D->getDependences(Dependences::TYPE_RED));
  if (Red)
    Dep = Dep.unite(Red);
  auto DomainSpace = Schedule.get_space().domain();
  auto Space = DomainSpace.map_from_domain_and_range(DomainSpace);
  auto Deltas = Dep.extract_map(Space).deltas();
  int DeltasDimNum = Deltas.dim(isl::dim::set);
  for (int i = 0; i < DeltasDimNum; i++) {
    auto Val = Deltas.plain_get_val_if_fixed(isl::dim::set, i);
    Pos = Pos < 0 && Val.is_one() ? i : Pos;
    if (Val.is_nan() || !(Val.is_zero() || (i == Pos && Val.is_one())))
      return false;
  }
  if (DeltasDimNum == 0 || Pos < 0)
    return false;
  return true;
}

/// Check if the SCoP statement could probably be optimized with analytical
/// modeling.
///
/// containsMatrMult tries to determine whether the following conditions
/// are true:
/// 1. The last memory access modeling an array, MA1, represents writing to
///    memory and has the form S(..., i1, ..., i2, ...) -> M(i1, i2) or
///    S(..., i2, ..., i1, ...) -> M(i1, i2), where S is the SCoP statement
///    under consideration.
/// 2. There is only one loop-carried true dependency, and it has the
///    form S(..., i3, ...) -> S(..., i3 + 1, ...), and there are no
///    loop-carried or anti dependencies.
/// 3. SCoP contains three access relations, MA2, MA3, and MA4 that represent
///    reading from memory and have the form S(..., i3, ...) -> M(i1, i3),
///    S(..., i3, ...) -> M(i3, i2), S(...) -> M(i1, i2), respectively,
///    and all memory accesses of the SCoP that are different from MA1, MA2,
///    MA3, and MA4 have stride 0, if the innermost loop is exchanged with any
///    of loops i1, i2 and i3.
///
/// @param PartialSchedule The PartialSchedule that contains a SCoP statement
///        to check.
/// @D     The SCoP dependencies.
/// @MMI   Parameters of the matrix multiplication operands.
static bool containsMatrMult(isl::map PartialSchedule, const Dependences *D,
                             MatMulInfoTy &MMI) {
  auto InputDimsId = PartialSchedule.get_tuple_id(isl::dim::in);
  auto *Stmt = static_cast<ScopStmt *>(InputDimsId.get_user());
  if (Stmt->size() <= 1)
    return false;

  auto Accesses = getAccessesInOrder(*Stmt);
  for (auto *MemA = Accesses.end() - 1; MemA != Accesses.begin(); MemA--) {
    auto *MemAccessPtr = *MemA;
    if (!MemAccessPtr->isLatestArrayKind())
      continue;
    if (!MemAccessPtr->isWrite())
      return false;
    auto AccMap = MemAccessPtr->getLatestAccessRelation();
    if (!isMatMulOperandAcc(Stmt->getDomain(), AccMap, MMI.i, MMI.j))
      return false;
    MMI.WriteToC = MemAccessPtr;
    break;
  }

  if (!containsOnlyMatMulDep(PartialSchedule, D, MMI.k))
    return false;

  if (!MMI.WriteToC || !containsOnlyMatrMultAcc(PartialSchedule, MMI))
    return false;

  if (!MMI.A || !MMI.B || !MMI.ReadFromC)
    return false;
  return true;
}

/// Permute two dimensions of the band node.
///
/// Permute FirstDim and SecondDim dimensions of the Node.
///
/// @param Node The band node to be modified.
/// @param FirstDim The first dimension to be permuted.
/// @param SecondDim The second dimension to be permuted.
static isl::schedule_node permuteBandNodeDimensions(isl::schedule_node Node,
                                                    unsigned FirstDim,
                                                    unsigned SecondDim) {
  assert(isl_schedule_node_get_type(Node.get()) == isl_schedule_node_band &&
         isl_schedule_node_band_n_member(Node.get()) >
             std::max(FirstDim, SecondDim));
  auto PartialSchedule =
      isl::manage(isl_schedule_node_band_get_partial_schedule(Node.get()));
  auto PartialScheduleFirstDim = PartialSchedule.get_union_pw_aff(FirstDim);
  auto PartialScheduleSecondDim = PartialSchedule.get_union_pw_aff(SecondDim);
  PartialSchedule =
      PartialSchedule.set_union_pw_aff(SecondDim, PartialScheduleFirstDim);
  PartialSchedule =
      PartialSchedule.set_union_pw_aff(FirstDim, PartialScheduleSecondDim);
  Node = isl::manage(isl_schedule_node_delete(Node.release()));
  return Node.insert_partial_schedule(PartialSchedule);
}

isl::schedule_node ScheduleTreeOptimizer::createMicroKernel(
    isl::schedule_node Node, MicroKernelParamsTy MicroKernelParams) {
  Node = applyRegisterTiling(Node, {MicroKernelParams.Mr, MicroKernelParams.Nr},
                             1);
  Node = Node.parent().parent();
  return permuteBandNodeDimensions(Node, 0, 1).child(0).child(0);
}

isl::schedule_node ScheduleTreeOptimizer::createMacroKernel(
    isl::schedule_node Node, MacroKernelParamsTy MacroKernelParams) {
  assert(isl_schedule_node_get_type(Node.get()) == isl_schedule_node_band);
  if (MacroKernelParams.Mc == 1 && MacroKernelParams.Nc == 1 &&
      MacroKernelParams.Kc == 1)
    return Node;
  int DimOutNum = isl_schedule_node_band_n_member(Node.get());
  std::vector<int> TileSizes(DimOutNum, 1);
  TileSizes[DimOutNum - 3] = MacroKernelParams.Mc;
  TileSizes[DimOutNum - 2] = MacroKernelParams.Nc;
  TileSizes[DimOutNum - 1] = MacroKernelParams.Kc;
  Node = tileNode(Node, "1st level tiling", TileSizes, 1);
  Node = Node.parent().parent();
  Node = permuteBandNodeDimensions(Node, DimOutNum - 2, DimOutNum - 1);
  Node = permuteBandNodeDimensions(Node, DimOutNum - 3, DimOutNum - 1);
  return Node.child(0).child(0);
}

/// Get the size of the widest type of the matrix multiplication operands
/// in bytes, including alignment padding.
///
/// @param MMI Parameters of the matrix multiplication operands.
/// @return The size of the widest type of the matrix multiplication operands
///         in bytes, including alignment padding.
static uint64_t getMatMulAlignTypeSize(MatMulInfoTy MMI) {
  auto *S = MMI.A->getStatement()->getParent();
  auto &DL = S->getFunction().getParent()->getDataLayout();
  auto ElementSizeA = DL.getTypeAllocSize(MMI.A->getElementType());
  auto ElementSizeB = DL.getTypeAllocSize(MMI.B->getElementType());
  auto ElementSizeC = DL.getTypeAllocSize(MMI.WriteToC->getElementType());
  return std::max({ElementSizeA, ElementSizeB, ElementSizeC});
}

/// Get the size of the widest type of the matrix multiplication operands
/// in bits.
///
/// @param MMI Parameters of the matrix multiplication operands.
/// @return The size of the widest type of the matrix multiplication operands
///         in bits.
static uint64_t getMatMulTypeSize(MatMulInfoTy MMI) {
  auto *S = MMI.A->getStatement()->getParent();
  auto &DL = S->getFunction().getParent()->getDataLayout();
  auto ElementSizeA = DL.getTypeSizeInBits(MMI.A->getElementType());
  auto ElementSizeB = DL.getTypeSizeInBits(MMI.B->getElementType());
  auto ElementSizeC = DL.getTypeSizeInBits(MMI.WriteToC->getElementType());
  return std::max({ElementSizeA, ElementSizeB, ElementSizeC});
}

/// Get parameters of the BLIS micro kernel.
///
/// We choose the Mr and Nr parameters of the micro kernel to be large enough
/// such that no stalls caused by the combination of latencies and dependencies
/// are introduced during the updates of the resulting matrix of the matrix
/// multiplication. However, they should also be as small as possible to
/// release more registers for entries of multiplied matrices.
///
/// @param TTI Target Transform Info.
/// @param MMI Parameters of the matrix multiplication operands.
/// @return The structure of type MicroKernelParamsTy.
/// @see MicroKernelParamsTy
static struct MicroKernelParamsTy
getMicroKernelParams(const TargetTransformInfo *TTI, MatMulInfoTy MMI) {
  assert(TTI && "The target transform info should be provided.");

  // Nvec - Number of double-precision floating-point numbers that can be hold
  // by a vector register. Use 2 by default.
  long RegisterBitwidth = VectorRegisterBitwidth;

  if (RegisterBitwidth == -1)
    RegisterBitwidth = TTI->getRegisterBitWidth(true);
  auto ElementSize = getMatMulTypeSize(MMI);
  assert(ElementSize > 0 && "The element size of the matrix multiplication "
                            "operands should be greater than zero.");
  auto Nvec = RegisterBitwidth / ElementSize;
  if (Nvec == 0)
    Nvec = 2;
  int Nr =
      ceil(sqrt(Nvec * LatencyVectorFma * ThroughputVectorFma) / Nvec) * Nvec;
  int Mr = ceil(Nvec * LatencyVectorFma * ThroughputVectorFma / Nr);
  return {Mr, Nr};
}

namespace {
/// Determine parameters of the target cache.
///
/// @param TTI Target Transform Info.
void getTargetCacheParameters(const llvm::TargetTransformInfo *TTI) {
  auto L1DCache = llvm::TargetTransformInfo::CacheLevel::L1D;
  auto L2DCache = llvm::TargetTransformInfo::CacheLevel::L2D;
  if (FirstCacheLevelSize == -1) {
    if (TTI->getCacheSize(L1DCache).hasValue())
      FirstCacheLevelSize = TTI->getCacheSize(L1DCache).getValue();
    else
      FirstCacheLevelSize = static_cast<int>(FirstCacheLevelDefaultSize);
  }
  if (SecondCacheLevelSize == -1) {
    if (TTI->getCacheSize(L2DCache).hasValue())
      SecondCacheLevelSize = TTI->getCacheSize(L2DCache).getValue();
    else
      SecondCacheLevelSize = static_cast<int>(SecondCacheLevelDefaultSize);
  }
  if (FirstCacheLevelAssociativity == -1) {
    if (TTI->getCacheAssociativity(L1DCache).hasValue())
      FirstCacheLevelAssociativity =
          TTI->getCacheAssociativity(L1DCache).getValue();
    else
      FirstCacheLevelAssociativity =
          static_cast<int>(FirstCacheLevelDefaultAssociativity);
  }
  if (SecondCacheLevelAssociativity == -1) {
    if (TTI->getCacheAssociativity(L2DCache).hasValue())
      SecondCacheLevelAssociativity =
          TTI->getCacheAssociativity(L2DCache).getValue();
    else
      SecondCacheLevelAssociativity =
          static_cast<int>(SecondCacheLevelDefaultAssociativity);
  }
}
} // namespace

/// Get parameters of the BLIS macro kernel.
///
/// During the computation of matrix multiplication, blocks of partitioned
/// matrices are mapped to different layers of the memory hierarchy.
/// To optimize data reuse, blocks should be ideally kept in cache between
/// iterations. Since parameters of the macro kernel determine sizes of these
/// blocks, there are upper and lower bounds on these parameters.
///
/// @param TTI Target Transform Info.
/// @param MicroKernelParams Parameters of the micro-kernel
///                          to be taken into account.
/// @param MMI Parameters of the matrix multiplication operands.
/// @return The structure of type MacroKernelParamsTy.
/// @see MacroKernelParamsTy
/// @see MicroKernelParamsTy
static struct MacroKernelParamsTy
getMacroKernelParams(const llvm::TargetTransformInfo *TTI,
                     const MicroKernelParamsTy &MicroKernelParams,
                     MatMulInfoTy MMI) {
  getTargetCacheParameters(TTI);
  // According to www.cs.utexas.edu/users/flame/pubs/TOMS-BLIS-Analytical.pdf,
  // it requires information about the first two levels of a cache to determine
  // all the parameters of a macro-kernel. It also checks that an associativity
  // degree of a cache level is greater than two. Otherwise, another algorithm
  // for determination of the parameters should be used.
  if (!(MicroKernelParams.Mr > 0 && MicroKernelParams.Nr > 0 &&
        FirstCacheLevelSize > 0 && SecondCacheLevelSize > 0 &&
        FirstCacheLevelAssociativity > 2 && SecondCacheLevelAssociativity > 2))
    return {1, 1, 1};
  // The quotient should be greater than zero.
  if (PollyPatternMatchingNcQuotient <= 0)
    return {1, 1, 1};
  int Car = floor(
      (FirstCacheLevelAssociativity - 1) /
      (1 + static_cast<double>(MicroKernelParams.Nr) / MicroKernelParams.Mr));

  // Car can be computed to be zero since it is floor to int.
  // On Mac OS, division by 0 does not raise a signal. This causes negative
  // tile sizes to be computed. Prevent division by Cac==0 by early returning
  // if this happens.
  if (Car == 0)
    return {1, 1, 1};

  auto ElementSize = getMatMulAlignTypeSize(MMI);
  assert(ElementSize > 0 && "The element size of the matrix multiplication "
                            "operands should be greater than zero.");
  int Kc = (Car * FirstCacheLevelSize) /
           (MicroKernelParams.Mr * FirstCacheLevelAssociativity * ElementSize);
  double Cac =
      static_cast<double>(Kc * ElementSize * SecondCacheLevelAssociativity) /
      SecondCacheLevelSize;
  int Mc = floor((SecondCacheLevelAssociativity - 2) / Cac);
  int Nc = PollyPatternMatchingNcQuotient * MicroKernelParams.Nr;

  assert(Mc > 0 && Nc > 0 && Kc > 0 &&
         "Matrix block sizes should be  greater than zero");
  return {Mc, Nc, Kc};
}

/// Create an access relation that is specific to
///        the matrix multiplication pattern.
///
/// Create an access relation of the following form:
/// [O0, O1, O2, O3, O4, O5, O6, O7, O8] -> [OI, O5, OJ]
/// where I is @p FirstDim, J is @p SecondDim.
///
/// It can be used, for example, to create relations that helps to consequently
/// access elements of operands of a matrix multiplication after creation of
/// the BLIS micro and macro kernels.
///
/// @see ScheduleTreeOptimizer::createMicroKernel
/// @see ScheduleTreeOptimizer::createMacroKernel
///
/// Subsequently, the described access relation is applied to the range of
/// @p MapOldIndVar, that is used to map original induction variables to
/// the ones, which are produced by schedule transformations. It helps to
/// define relations using a new space and, at the same time, keep them
/// in the original one.
///
/// @param MapOldIndVar The relation, which maps original induction variables
///                     to the ones, which are produced by schedule
///                     transformations.
/// @param FirstDim, SecondDim The input dimensions that are used to define
///        the specified access relation.
/// @return The specified access relation.
isl::map getMatMulAccRel(isl::map MapOldIndVar, unsigned FirstDim,
                         unsigned SecondDim) {
  auto AccessRelSpace = isl::space(MapOldIndVar.get_ctx(), 0, 9, 3);
  auto AccessRel = isl::map::universe(AccessRelSpace);
  AccessRel = AccessRel.equate(isl::dim::in, FirstDim, isl::dim::out, 0);
  AccessRel = AccessRel.equate(isl::dim::in, 5, isl::dim::out, 1);
  AccessRel = AccessRel.equate(isl::dim::in, SecondDim, isl::dim::out, 2);
  return MapOldIndVar.apply_range(AccessRel);
}

isl::schedule_node createExtensionNode(isl::schedule_node Node,
                                       isl::map ExtensionMap) {
  auto Extension = isl::union_map(ExtensionMap);
  auto NewNode = isl::schedule_node::from_extension(Extension);
  return Node.graft_before(NewNode);
}

/// Apply the packing transformation.
///
/// The packing transformation can be described as a data-layout
/// transformation that requires to introduce a new array, copy data
/// to the array, and change memory access locations to reference the array.
/// It can be used to ensure that elements of the new array are read in-stride
/// access, aligned to cache lines boundaries, and preloaded into certain cache
/// levels.
///
/// As an example let us consider the packing of the array A that would help
/// to read its elements with in-stride access. An access to the array A
/// is represented by an access relation that has the form
/// S[i, j, k] -> A[i, k]. The scheduling function of the SCoP statement S has
/// the form S[i,j, k] -> [floor((j mod Nc) / Nr), floor((i mod Mc) / Mr),
/// k mod Kc, j mod Nr, i mod Mr].
///
/// To ensure that elements of the array A are read in-stride access, we add
/// a new array Packed_A[Mc/Mr][Kc][Mr] to the SCoP, using
/// Scop::createScopArrayInfo, change the access relation
/// S[i, j, k] -> A[i, k] to
/// S[i, j, k] -> Packed_A[floor((i mod Mc) / Mr), k mod Kc, i mod Mr], using
/// MemoryAccess::setNewAccessRelation, and copy the data to the array, using
/// the copy statement created by Scop::addScopStmt.
///
/// @param Node The schedule node to be optimized.
/// @param MapOldIndVar The relation, which maps original induction variables
///                     to the ones, which are produced by schedule
///                     transformations.
/// @param MicroParams, MacroParams Parameters of the BLIS kernel
///                                 to be taken into account.
/// @param MMI Parameters of the matrix multiplication operands.
/// @return The optimized schedule node.
static isl::schedule_node
optimizeDataLayoutMatrMulPattern(isl::schedule_node Node, isl::map MapOldIndVar,
                                 MicroKernelParamsTy MicroParams,
                                 MacroKernelParamsTy MacroParams,
                                 MatMulInfoTy &MMI) {
  auto InputDimsId = MapOldIndVar.get_tuple_id(isl::dim::in);
  auto *Stmt = static_cast<ScopStmt *>(InputDimsId.get_user());

  // Create a copy statement that corresponds to the memory access to the
  // matrix B, the second operand of the matrix multiplication.
  Node = Node.parent().parent().parent().parent().parent().parent();
  Node = isl::manage(isl_schedule_node_band_split(Node.release(), 2)).child(0);
  auto AccRel = getMatMulAccRel(MapOldIndVar, 3, 7);
  unsigned FirstDimSize = MacroParams.Nc / MicroParams.Nr;
  unsigned SecondDimSize = MacroParams.Kc;
  unsigned ThirdDimSize = MicroParams.Nr;
  auto *SAI = Stmt->getParent()->createScopArrayInfo(
      MMI.B->getElementType(), "Packed_B",
      {FirstDimSize, SecondDimSize, ThirdDimSize});
  AccRel = AccRel.set_tuple_id(isl::dim::out, SAI->getBasePtrId());
  auto OldAcc = MMI.B->getLatestAccessRelation();
  MMI.B->setNewAccessRelation(AccRel);
  auto ExtMap = MapOldIndVar.project_out(isl::dim::out, 2,
                                         MapOldIndVar.dim(isl::dim::out) - 2);
  ExtMap = ExtMap.reverse();
  ExtMap = ExtMap.fix_si(isl::dim::out, MMI.i, 0);
  auto Domain = Stmt->getDomain();

  // Restrict the domains of the copy statements to only execute when also its
  // originating statement is executed.
  auto DomainId = Domain.get_tuple_id();
  auto *NewStmt = Stmt->getParent()->addScopStmt(
      OldAcc, MMI.B->getLatestAccessRelation(), Domain);
  ExtMap = ExtMap.set_tuple_id(isl::dim::out, DomainId);
  ExtMap = ExtMap.intersect_range(Domain);
  ExtMap = ExtMap.set_tuple_id(isl::dim::out, NewStmt->getDomainId());
  Node = createExtensionNode(Node, ExtMap);

  // Create a copy statement that corresponds to the memory access
  // to the matrix A, the first operand of the matrix multiplication.
  Node = Node.child(0);
  AccRel = getMatMulAccRel(MapOldIndVar, 4, 6);
  FirstDimSize = MacroParams.Mc / MicroParams.Mr;
  ThirdDimSize = MicroParams.Mr;
  SAI = Stmt->getParent()->createScopArrayInfo(
      MMI.A->getElementType(), "Packed_A",
      {FirstDimSize, SecondDimSize, ThirdDimSize});
  AccRel = AccRel.set_tuple_id(isl::dim::out, SAI->getBasePtrId());
  OldAcc = MMI.A->getLatestAccessRelation();
  MMI.A->setNewAccessRelation(AccRel);
  ExtMap = MapOldIndVar.project_out(isl::dim::out, 3,
                                    MapOldIndVar.dim(isl::dim::out) - 3);
  ExtMap = ExtMap.reverse();
  ExtMap = ExtMap.fix_si(isl::dim::out, MMI.j, 0);
  NewStmt = Stmt->getParent()->addScopStmt(
      OldAcc, MMI.A->getLatestAccessRelation(), Domain);

  // Restrict the domains of the copy statements to only execute when also its
  // originating statement is executed.
  ExtMap = ExtMap.set_tuple_id(isl::dim::out, DomainId);
  ExtMap = ExtMap.intersect_range(Domain);
  ExtMap = ExtMap.set_tuple_id(isl::dim::out, NewStmt->getDomainId());
  Node = createExtensionNode(Node, ExtMap);
  return Node.child(0).child(0).child(0).child(0).child(0);
}

/// Get a relation mapping induction variables produced by schedule
/// transformations to the original ones.
///
/// @param Node The schedule node produced as the result of creation
///        of the BLIS kernels.
/// @param MicroKernelParams, MacroKernelParams Parameters of the BLIS kernel
///                                             to be taken into account.
/// @return  The relation mapping original induction variables to the ones
///          produced by schedule transformation.
/// @see ScheduleTreeOptimizer::createMicroKernel
/// @see ScheduleTreeOptimizer::createMacroKernel
/// @see getMacroKernelParams
isl::map
getInductionVariablesSubstitution(isl::schedule_node Node,
                                  MicroKernelParamsTy MicroKernelParams,
                                  MacroKernelParamsTy MacroKernelParams) {
  auto Child = Node.child(0);
  auto UnMapOldIndVar = Child.get_prefix_schedule_union_map();
  auto MapOldIndVar = isl::map::from_union_map(UnMapOldIndVar);
  if (MapOldIndVar.dim(isl::dim::out) > 9)
    return MapOldIndVar.project_out(isl::dim::out, 0,
                                    MapOldIndVar.dim(isl::dim::out) - 9);
  return MapOldIndVar;
}

/// Isolate a set of partial tile prefixes and unroll the isolated part.
///
/// The set should ensure that it contains only partial tile prefixes that have
/// exactly Mr x Nr iterations of the two innermost loops produced by
/// the optimization of the matrix multiplication. Mr and Nr are parameters of
/// the micro-kernel.
///
/// In case of parametric bounds, this helps to auto-vectorize the unrolled
/// innermost loops, using the SLP vectorizer.
///
/// @param Node              The schedule node to be modified.
/// @param MicroKernelParams Parameters of the micro-kernel
///                          to be taken into account.
/// @return The modified isl_schedule_node.
static isl::schedule_node
isolateAndUnrollMatMulInnerLoops(isl::schedule_node Node,
                                 struct MicroKernelParamsTy MicroKernelParams) {
  isl::schedule_node Child = Node.get_child(0);
  isl::union_map UnMapOldIndVar = Child.get_prefix_schedule_relation();
  isl::set Prefix = isl::map::from_union_map(UnMapOldIndVar).range();
  unsigned Dims = Prefix.dim(isl::dim::set);
  Prefix = Prefix.project_out(isl::dim::set, Dims - 1, 1);
  Prefix = getPartialTilePrefixes(Prefix, MicroKernelParams.Nr);
  Prefix = getPartialTilePrefixes(Prefix, MicroKernelParams.Mr);

  isl::union_set IsolateOption =
      getIsolateOptions(Prefix.add_dims(isl::dim::set, 3), 3);
  isl::ctx Ctx = Node.get_ctx();
  auto Options = IsolateOption.unite(getDimOptions(Ctx, "unroll"));
  Options = Options.unite(getUnrollIsolatedSetOptions(Ctx));
  Node = Node.band_set_ast_build_options(Options);
  Node = Node.parent().parent().parent();
  IsolateOption = getIsolateOptions(Prefix, 3);
  Options = IsolateOption.unite(getDimOptions(Ctx, "separate"));
  Node = Node.band_set_ast_build_options(Options);
  Node = Node.child(0).child(0).child(0);
  return Node;
}

/// Mark @p BasePtr with "Inter iteration alias-free" mark node.
///
/// @param Node The child of the mark node to be inserted.
/// @param BasePtr The pointer to be marked.
/// @return The modified isl_schedule_node.
static isl::schedule_node markInterIterationAliasFree(isl::schedule_node Node,
                                                      Value *BasePtr) {
  if (!BasePtr)
    return Node;

  auto Id =
      isl::id::alloc(Node.get_ctx(), "Inter iteration alias-free", BasePtr);
  return Node.insert_mark(Id).child(0);
}

/// Insert "Loop Vectorizer Disabled" mark node.
///
/// @param Node The child of the mark node to be inserted.
/// @return The modified isl_schedule_node.
static isl::schedule_node markLoopVectorizerDisabled(isl::schedule_node Node) {
  auto Id = isl::id::alloc(Node.get_ctx(), "Loop Vectorizer Disabled", nullptr);
  return Node.insert_mark(Id).child(0);
}

/// Restore the initial ordering of dimensions of the band node
///
/// In case the band node represents all the dimensions of the iteration
/// domain, recreate the band node to restore the initial ordering of the
/// dimensions.
///
/// @param Node The band node to be modified.
/// @return The modified schedule node.
static isl::schedule_node
getBandNodeWithOriginDimOrder(isl::schedule_node Node) {
  assert(isl_schedule_node_get_type(Node.get()) == isl_schedule_node_band);
  if (isl_schedule_node_get_type(Node.child(0).get()) != isl_schedule_node_leaf)
    return Node;
  auto Domain = Node.get_universe_domain();
  assert(isl_union_set_n_set(Domain.get()) == 1);
  if (Node.get_schedule_depth() != 0 ||
      (isl::set(Domain).dim(isl::dim::set) !=
       isl_schedule_node_band_n_member(Node.get())))
    return Node;
  Node = isl::manage(isl_schedule_node_delete(Node.copy()));
  auto PartialSchedulePwAff = Domain.identity_union_pw_multi_aff();
  auto PartialScheduleMultiPwAff =
      isl::multi_union_pw_aff(PartialSchedulePwAff);
  PartialScheduleMultiPwAff =
      PartialScheduleMultiPwAff.reset_tuple_id(isl::dim::set);
  return Node.insert_partial_schedule(PartialScheduleMultiPwAff);
}

isl::schedule_node
ScheduleTreeOptimizer::optimizeMatMulPattern(isl::schedule_node Node,
                                             const TargetTransformInfo *TTI,
                                             MatMulInfoTy &MMI) {
  assert(TTI && "The target transform info should be provided.");
  Node = markInterIterationAliasFree(
      Node, MMI.WriteToC->getLatestScopArrayInfo()->getBasePtr());
  int DimOutNum = isl_schedule_node_band_n_member(Node.get());
  assert(DimOutNum > 2 && "In case of the matrix multiplication the loop nest "
                          "and, consequently, the corresponding scheduling "
                          "functions have at least three dimensions.");
  Node = getBandNodeWithOriginDimOrder(Node);
  Node = permuteBandNodeDimensions(Node, MMI.i, DimOutNum - 3);
  int NewJ = MMI.j == DimOutNum - 3 ? MMI.i : MMI.j;
  int NewK = MMI.k == DimOutNum - 3 ? MMI.i : MMI.k;
  Node = permuteBandNodeDimensions(Node, NewJ, DimOutNum - 2);
  NewK = NewK == DimOutNum - 2 ? NewJ : NewK;
  Node = permuteBandNodeDimensions(Node, NewK, DimOutNum - 1);
  auto MicroKernelParams = getMicroKernelParams(TTI, MMI);
  auto MacroKernelParams = getMacroKernelParams(TTI, MicroKernelParams, MMI);
  Node = createMacroKernel(Node, MacroKernelParams);
  Node = createMicroKernel(Node, MicroKernelParams);
  if (MacroKernelParams.Mc == 1 || MacroKernelParams.Nc == 1 ||
      MacroKernelParams.Kc == 1)
    return Node;
  auto MapOldIndVar = getInductionVariablesSubstitution(Node, MicroKernelParams,
                                                        MacroKernelParams);
  if (!MapOldIndVar)
    return Node;
  Node = markLoopVectorizerDisabled(Node.parent()).child(0);
  Node = isolateAndUnrollMatMulInnerLoops(Node, MicroKernelParams);
  return optimizeDataLayoutMatrMulPattern(Node, MapOldIndVar, MicroKernelParams,
                                          MacroKernelParams, MMI);
}

bool ScheduleTreeOptimizer::isMatrMultPattern(isl::schedule_node Node,
                                              const Dependences *D,
                                              MatMulInfoTy &MMI) {
  auto PartialSchedule = isl::manage(
      isl_schedule_node_band_get_partial_schedule_union_map(Node.get()));
  Node = Node.child(0);
  auto LeafType = isl_schedule_node_get_type(Node.get());
  Node = Node.parent();
  if (LeafType != isl_schedule_node_leaf ||
      isl_schedule_node_band_n_member(Node.get()) < 3 ||
      Node.get_schedule_depth() != 0 ||
      isl_union_map_n_map(PartialSchedule.get()) != 1)
    return false;
  auto NewPartialSchedule = isl::map::from_union_map(PartialSchedule);
  if (containsMatrMult(NewPartialSchedule, D, MMI))
    return true;
  return false;
}

/*
// collect domain for the access.
static isl::set collectDomain(isl::set InvalidDomain, isl::set DomainStmt) {
  isl::set EmptySet = isl::set::empty(InvalidDomain.get_space());
  DEBUG(dbgs() << "DomainStmt :=" << DomainStmt << "\n");
  isl::set d = DomainStmt.project_out(isl::dim::param, 0,
DomainStmt.dim(isl::dim::param)); DEBUG(dbgs() << "d :=" << d << "\n");
  //DEBUG(dbgs() << "InvalidDom :=" << InvalidDomain << "\n");
  d.foreach_basic_set([&EmptySet](isl::basic_set BSet) -> isl::stat {
    EmptySet = EmptySet.unite(isl::set(BSet));
    return isl::stat::ok;
  });
  return EmptySet;
}
*/

static void getDomain(MemoryAccess *MemAcc, isl::set &Ext) {

  auto ArrayInfo = MemAcc->getLatestScopArrayInfo();
  unsigned NumDims = ArrayInfo->getNumberOfDimensions();

  if (NumDims == 0)
    Ext = isl::set::universe(ArrayInfo->getSpace());

  Scop *S = ArrayInfo->getScop();
  isl::union_map Accesses = S->getAccesses();
  isl::union_map sched = ArrayInfo->getScop()->getSchedule();
  // DEBUG(dbgs() << "Sched :=" << sched << "\n");
  // DEBUG(dbgs() << "Accesses :=" << Accesses << "\n");
  Accesses = Accesses.apply_domain(sched);
  isl::union_set AccessRange = Accesses.range();
  // DEBUG(dbgs() << "Accesses Range :=" << AccessRange <<"\n");
  // simplify range.
  AccessRange = AccessRange.coalesce();
  AccessRange = AccessRange.detect_equalities();
  AccessRange = AccessRange.coalesce();

  if (AccessRange.is_empty())
    Ext = isl::set::empty(ArrayInfo->getSpace());

  isl::set AccessSet = AccessRange.extract_set(ArrayInfo->getSpace());
  // DEBUG(dbgs() << "AccessSet in Domain :=" << AccessSet << "\n");
  isl::local_space LS = isl::local_space(ArrayInfo->getSpace());
  isl::set Extent = isl::set::universe(ArrayInfo->getSpace());

  for (unsigned i = 0; i < NumDims; ++i) {
    isl::pw_aff Val = isl::aff::var_on_domain(LS, isl::dim::set, i);
    // DEBUG(dbgs() << "val :=" << Val << "\n");
    isl::pw_aff OuterMin = AccessSet.dim_min(i);
    // DEBUG(dbgs() << "OuterMin :=" << OuterMin << "\n");
    isl::pw_aff OuterMax = AccessSet.dim_max(i);
    // DEBUG(dbgs() << "OuterMax :=" << OuterMax << "\n");
    OuterMin = OuterMin.add_dims(isl::dim::in, Val.dim(isl::dim::in));
    // DEBUG(dbgs() << "OuterMin :=" << OuterMin << "\n");
    OuterMax = OuterMax.add_dims(isl::dim::in, Val.dim(isl::dim::in));
    // DEBUG(dbgs() << "OuterMax :=" << OuterMax << "\n");
    OuterMin = OuterMin.set_tuple_id(isl::dim::in, ArrayInfo->getBasePtrId());
    // DEBUG(dbgs() << "OuterMin :=" << OuterMin << "\n");
    OuterMax = OuterMax.set_tuple_id(isl::dim::in, ArrayInfo->getBasePtrId());
    // DEBUG(dbgs() << "OuterMax :=" << OuterMax << "\n");
    Extent = Extent.intersect(OuterMin.le_set(Val));
    // DEBUG(dbgs() << "Extent :=" << Extent << "\n");
    Extent = Extent.intersect(OuterMax.ge_set(Val));
    // DEBUG(dbgs() << "Extent :=" << Extent << "\n");
    Ext = Extent;
    // TODO: from here you can compute the total
    // number of accessed element. look at pw_aff_sub
  }
  /*
      for (unsigned i = 1; i < NumDims; ++i)
        Extent = Extent.lower_bound_si(isl::dim::set, i, 0);
      DEBUG(dbgs() << "Extent :=" << Extent << "\n");

      for (unsigned i = 0; i < NumDims; ++i) {
        isl::pw_aff PwAff = ArrayInfo->getDimensionSizePw(i);
        DEBUG(dbgs() << "PwAff :=" << PwAff << "\n");

        // isl_pw_aff can be NULL for zero dimension. Only in the case of a
        // Fortran array will we have a legitimate dimension.
        if (PwAff.is_null()) {
          assert(i == 0 && "invalid dimension isl_pw_aff for nonzero
     dimension"); continue;
        }

        isl::pw_aff Val = isl::aff::var_on_domain(
          isl::local_space(ArrayInfo->getSpace()), isl::dim::set, i);
        DEBUG(dbgs() << "Val :=" << Val << "\n");
        PwAff = PwAff.add_dims(isl::dim::in, Val.dim(isl::dim::in));
        DEBUG(dbgs() << "PwAff :=" << PwAff << "\n");
        PwAff = PwAff.set_tuple_id(isl::dim::in,
     Val.get_tuple_id(isl::dim::in)); DEBUG(dbgs() << "PwAff :=" << PwAff <<
     "\n"); isl::set Set = PwAff.gt_set(Val); DEBUG(dbgs() << "Set :=" << Set <<
     "\n"); Ext = Set.intersect(Extent);
      }
  */
}

/*
static void computeArrayBounds(std::vector<isl::pw_aff> &B, isl::set &E,
                               ScopArrayInfo *A) {
  unsigned Dims = A->getNumberOfDimensions();
  //DEBUG(dbgs() << "Dimm :=" << Dims << "\n");
  if (Dims > 0) {
    if (E.is_empty()) {
      isl::local_space LS = isl::local_space(E.get_space());
      isl::pw_aff Zero = isl::pw_aff(isl::aff(LS));
      B.push_back(Zero);
      //DEBUG(dbgs() << "Zero" << "\n");
    } else {
      //DEBUG(dbgs() << "else" <<"\n");
      isl::set Dom = E;
      //DEBUG(dbgs() << "Dom :=" << Dom << "\n");
      isl::pw_aff Bound = Dom.dim_max(0);
      //DEBUG(dbgs() << "Bound :=" << Bound << "\n");
      Dom = Bound.domain();
      //DEBUG(dbgs() << "Dom := "<< Dom << "\n");
      isl::local_space LS = isl::local_space(Dom.get_space());
      isl::aff One = isl::aff(LS);
      //DEBUG(dbgs() << "One :=" << One << "\n");
      One = One.add_constant_si(1);
      //DEBUG(dbgs() << "One :=" << One << "\n");
      Bound = Bound.add(isl::pw_aff::alloc(Dom,One));
      Bound = Bound.gist(A->getScop()->getContext());
      //DEBUG(dbgs() << "Bound :=" << Bound <<"\n");
      B.push_back(Bound);
    }
  }
}
*/

/// Derive the extent of an array.
///
/// The extent of an array is the set of elements that are within the
/// accessed array. For the inner dimensions, the extent constraints are
/// 0 and the size of the corresponding array dimension. For the first
/// (outermost) dimension, the extent constraints are the minimal and maximal
/// subscript value for the first dimension.
///
/// @param Array The array to derive the extent for.
///
/// @returns An isl_set describing the extent of the array.
isl::set getExtent(MemoryAccess *MemAcc) {
  auto Array = MemAcc->getLatestScopArrayInfo();
  unsigned NumDims = Array->getNumberOfDimensions();

  if (Array->getNumberOfDimensions() == 0)
    return isl::set::universe(Array->getSpace());
  Scop *S = Array->getScop();
  isl::union_map Accesses = S->getAccesses(const_cast<ScopArrayInfo *>(Array));
  isl::union_set AccessUSet = Accesses.range();
  AccessUSet = AccessUSet.coalesce();
  AccessUSet = AccessUSet.detect_equalities();
  AccessUSet = AccessUSet.coalesce();

  if (AccessUSet.is_empty())
    return isl::set::empty(Array->getSpace());

  isl::set AccessSet = AccessUSet.extract_set(Array->getSpace());

  isl::local_space LS = isl::local_space(Array->getSpace());

  isl::pw_aff Val = isl::aff::var_on_domain(LS, isl::dim::set, 0);
  isl::pw_aff OuterMin = AccessSet.dim_min(0);
  isl::pw_aff OuterMax = AccessSet.dim_max(0);
  OuterMin = OuterMin.add_dims(isl::dim::in, Val.dim(isl::dim::in));
  OuterMax = OuterMax.add_dims(isl::dim::in, Val.dim(isl::dim::in));
  OuterMin = OuterMin.set_tuple_id(isl::dim::in, Array->getBasePtrId());
  OuterMax = OuterMax.set_tuple_id(isl::dim::in, Array->getBasePtrId());

  isl::set Extent = isl::set::universe(Array->getSpace());

  Extent = Extent.intersect(OuterMin.le_set(Val));
  Extent = Extent.intersect(OuterMax.ge_set(Val));

  for (unsigned i = 1; i < NumDims; ++i)
    Extent = Extent.lower_bound_si(isl::dim::set, i, 0);

  for (unsigned i = 0; i < NumDims; ++i) {
    isl::pw_aff PwAff = Array->getDimensionSizePw(i);

    // isl_pw_aff can be NULL for zero dimension. Only in the case of a
    // Fortran array will we have a legitimate dimension.
    if (PwAff.is_null()) {
      assert(i == 0 && "invalid dimension isl_pw_aff for nonzero dimension");
      continue;
    }

    isl::pw_aff Val = isl::aff::var_on_domain(
        isl::local_space(Array->getSpace()), isl::dim::set, i);
    PwAff = PwAff.add_dims(isl::dim::in, Val.dim(isl::dim::in));
    PwAff = PwAff.set_tuple_id(isl::dim::in, Val.get_tuple_id(isl::dim::in));
    isl::set Set = PwAff.gt_set(Val);
    Extent = Set.intersect(Extent);
  }

  return Extent;
}
/*
/// Align all the `PwAffs` such that they have the same parameter dimensions.
///
/// We loop over all `pw_aff` and align all of their spaces together to
/// create a common space for all the `pw_aff`. This common space is the
/// `AlignSpace`. We then align all the `pw_aff` to this space. We start
/// with the given `SeedSpace`.
/// @param PwAffs    The list of piecewise affine functions we want to align.
///                  This is an rvalue reference because the entire vector is
///                  used up by the end of the operation.
/// @param SeedSpace The space to start the alignment process with.
/// @returns         A std::pair, whose first element is the aligned space,
///                  whose second element is the vector of aligned piecewise
///                  affines.

static std::pair<isl::space, std::vector<isl::pw_aff>>
alignPwAffs(const std::vector<isl::pw_aff> &PwAffs, isl::space SeedSpace) {
  assert(SeedSpace && "Invalid seed space given.");
  isl::space AlignSpace = SeedSpace;
  for (isl::pw_aff PwAff : PwAffs) {
    isl::space PwAffSpace = PwAff.get_domain_space();
    AlignSpace = AlignSpace.align_params(PwAffSpace);
  }
  std::vector<isl::pw_aff> AdjustedPwAffs;
  for (unsigned i = 0; i < PwAffs.size(); ++i) {
    isl::pw_aff Adjusted = PwAffs[i];
    assert(Adjusted && "Invalid pw_aff given");
    Adjusted = Adjusted.align_params(AlignSpace);
    AdjustedPwAffs.push_back(Adjusted);
  }
  return std::make_pair(AlignSpace, AdjustedPwAffs);
}

static std::pair<__isl_give isl_space *, std::vector<__isl_give isl_pw_aff *>>
alignPwAffs(const std::vector<__isl_take isl_pw_aff *> &&PwAffs,
            __isl_take isl_space *SeedSpace) {
  assert(SeedSpace && "Invalid seed space given.");

  isl_space *AlignSpace = SeedSpace;
  for (isl_pw_aff *PwAff : PwAffs) {
    isl_space *PwAffSpace = isl_pw_aff_get_domain_space(PwAff);
    AlignSpace = isl_space_align_params(AlignSpace, PwAffSpace);
  }
  std::vector<isl_pw_aff *> AdjustedPwAffs;

  for (unsigned i = 0; i < PwAffs.size(); i++) {
    isl_pw_aff *Adjusted = PwAffs[i];
    assert(Adjusted && "Invalid pw_aff given.");
    Adjusted = isl_pw_aff_align_params(Adjusted, isl_space_copy(AlignSpace));
    AdjustedPwAffs.push_back(Adjusted);
  }
  return std::make_pair(AlignSpace, AdjustedPwAffs);
}

/// Construct an `isl_pw_aff_list` from a vector of `isl_pw_aff`
/// @param PwAffs The list of piecewise affine functions to create an
///               `isl_pw_aff_list` from. We expect an rvalue ref because
///               all the isl_pw_aff are used up by this function.
///
/// @returns  The `isl_pw_aff_list`.
__isl_give isl_pw_aff_list *
createPwAffList(isl_ctx *Context,
                const std::vector<__isl_take isl_pw_aff *> &&PwAffs) {
  isl_pw_aff_list *List = isl_pw_aff_list_alloc(Context, PwAffs.size());

  for (unsigned i = 0; i < PwAffs.size(); i++) {
    List = isl_pw_aff_list_insert(List, i, PwAffs[i]);
  }
  return List;
}

/// Derive the bounds of an array.
///
/// For the first dimension we derive the bound of the array from the extent
/// of this dimension. For inner dimensions we obtain their size directly from
/// ScopArrayInfo.
///
/// @param PPCGArray The array to compute bounds for.
/// @param Array The polly array from which to take the information.
void computeBounds(std::vector<isl::pw_aff> Bounds, isl::set &Extent,
                   ScopArrayInfo *Array) {
  unsigned Dims = Array->getNumberOfDimensions();

  if (Extent.is_empty()) {
    isl::set Dom = Extent;
    isl::local_space LS = isl::local_space(Dom.get_space());
    isl::pw_aff Zero = isl::pw_aff(isl::aff(LS));
    Bounds.push_back(Zero);
  } else {
    isl::set Dom = Extent;
    Dom = Dom.project_out(isl::dim::set, 1, Dims - 1);
    isl::pw_aff Bound = Dom.dim_max(0);
    Dom = Bound.domain();
    isl::local_space LS = isl::local_space(Dom.get_space());
    isl::aff One = isl::aff(LS);
    One = One.add_constant_si(1);
    Bound = Bound.add(isl::pw_aff::alloc(Dom, One));
    Bound = Bound.gist(Array->getScop()->getContext());
    Bounds.push_back(Bound);
  }

  for (unsigned i = 1; i < Dims; ++i) {
    isl::pw_aff Bound = Array->getDimensionSizePw(i);
    isl::space LS = Bound.get_domain_space();
    isl::multi_aff Aff = isl::multi_aff::zero(LS);
    Bound = Bound.pullback(Aff);
    Bounds.push_back(Bound);
  }

  isl_space *SeedAlignSpace = Array->getScop()->getParamSpace().release();
  SeedAlignSpace = isl_space_add_dims(SeedAlignSpace, isl_dim_set, 1);

  isl_space *AlignSpace = nullptr;
  std::vector<isl_pw_aff *> AlignedBounds;
  std::tie(AlignSpace, AlignedBounds) =
      alignPwAffs(std::move(isl::manage(Bounds)), SeedAlignSpace);

  assert(AlignSpace && "alignPwAffs did not initialise AlignSpace");

  isl_pw_aff_list *BoundsList =
      createPwAffList(S->getIslCtx().get(), std::move(AlignedBounds));

  isl_space *BoundsSpace = isl_set_get_space(PPCGArray.extent);
  BoundsSpace = isl_space_align_params(BoundsSpace, AlignSpace);

  assert(BoundsSpace && "Unable to access space of array.");
  assert(BoundsList && "Unable to access list of bounds.");

  // isl::space BoundsSpace = Extent.get_space();
  // BoundsSpace = BoundsSpace.align_params(AlignSpace);
  // assert(BoundsSpace && "Unable to access space of array.");
  // assert(BoundsList && "Unable to access list of bounds.");
  // isl::multi_pw_aff dummy =
  // isl::multi_pw_aff::from_pw_aff_list(BoundsSpace, BoundsList);
}
*/

/// At the moment I am not able to quantify the step.
/// so we push a '1' if we detect a step along one loop dimension, 0 otherwise.

static void getStep(MemoryAccess *MemAcc, isl::map &Sched,
                    /*unsigned LoopLevel,*/ std::vector<bool> &HasStride) {

  unsigned DimNumOut = Sched.dim(isl::dim::out);
  unsigned StrideDetected = false;
  for (unsigned uu = 0; uu < DimNumOut; uu++) {
    StrideDetected = false;
    // the loop for which you want to compute the step
    // should be the innermost one.
    isl::map NewSched =
        permuteDimensions(Sched, isl::dim::out, uu, DimNumOut - 1);
    isl::set Deltas = MemAcc->getStride(NewSched);
    isl::pw_multi_aff MultAff = isl::pw_multi_aff::from_set(Deltas);
    if (MultAff.n_piece() == 0)
      continue;
    if (MultAff.n_piece() > 1)
      continue;
    DEBUG(dbgs() << MultAff << "\n");
    for (unsigned u = 0; u < MultAff.dim(isl::dim::out); ++u) {
      isl::pw_aff PwAff = MultAff.get_pw_aff(u);
      PwAff.foreach_piece([&](isl::set S, isl::aff Aff) -> isl::stat {
        isl::val Val = Aff.get_constant_val();
        if (!Val.is_zero()) {
          DEBUG(dbgs() << "Stride on dimension :" << uu << "\n");
          HasStride.push_back(true);
          StrideDetected = true;
        }
        return isl::stat::ok;
      });
    }
    if (!StrideDetected)
      HasStride.push_back(false);
  }
  DEBUG(dbgs() << "****"
               << "\n");
}

/// find relative index order + compute element accessed.
/// Three cases:
/// 1. the access is dependent on the outer loops
/// 2. the access is dependent on the inner loops
/// 3. the access is not dependent on any loops

static void getElementAccessed(MemoryAccess *MemAcc, isl::map &Sched,
                               isl::set &AccessSet,
                               std::vector<isl::val> &ElementAccessed,
                               std::vector<unsigned> &LoopOrder) {

  // DEBUG(dbgs() << "Schedule" << Sched << "\n");
  // DEBUG(dbgs() << "Domain" << MemAcc->getStatement()->getDomain() << "\n");
  // DEBUG(dbgs() << "AccesSet" << AccessSet << "\n");
  isl::map AccMap = MemAcc->getLatestAccessRelation();
  // isl:: pw_multi_aff MultiAff = MemAcc->applyScheduleToAccessRelation(Sched);
  // DEBUG(dbgs() << "MultiAff" << MultiAff << "\n");
  AccMap = AccMap.intersect_domain(MemAcc->getStatement()->getDomain());
  isl::pw_multi_aff MultiAff = isl::pw_multi_aff::from_map(AccMap);
  // DEBUG(dbgs() << "MultiAff" << isl::pw_multi_aff::from_map(AccMap) << "\n");
  unsigned IndexAccessMap = 0;
  // DEBUG(dbgs() << "AccMap" << AccMap << "\n");
  // DEBUG(dbgs() << "AccMapDim" << AccMap.dim(isl::dim::out) << "\n");
  // MultiAff = MultiAff.drop_dims(isl::dim::in, 0, 2);
  // DEBUG(dbgs() << "MultiAff" << MultiAff << "\n");
  for (unsigned u = 0; u < AccMap.dim(isl::dim::out); ++u) {
    isl::pw_aff PwAff = MultiAff.get_pw_aff(u);
    PwAff.foreach_piece([&](isl::set S, isl::aff Aff) -> isl::stat {
      bool Relate = false;
      // DEBUG(dbgs() << "AFF" << Aff << "\n");
      // DEBUG(dbgs() << PwAff.domain() << "\n");
      for (unsigned LoopDim = 0; LoopDim < AccMap.dim(isl::dim::in);
           ++LoopDim) {
        isl::val V = Aff.get_coefficient_val(isl::dim::in, LoopDim);
        if (!V.is_zero()) {
          isl::pw_aff OuterMin, OuterMax;
          if (LoopDim == 0) {
            // DEBUG(dbgs() << "dimension relates to outer loops" << "\n");
            isl::local_space Ls =
                isl::local_space(AccessSet.get_space().params());
            isl::aff One = isl::aff(Ls);
            One = One.add_constant_si(1);
            isl::val Val = One.get_constant_val();
            ElementAccessed.push_back(Val);
          } else {
            // DEBUG(dbgs() << "dimension relates to inner loops" << "\n");
            OuterMin = AccessSet.dim_min(IndexAccessMap);
            OuterMax = AccessSet.dim_max(IndexAccessMap);
            OuterMax = OuterMax.sub(OuterMin);
            // DEBUG(dbgs() << "ELEM" << OuterMax << "\n");
            // DEBUG(dbgs() << "LOOPDIM" << LoopDim << "\n");
            /// create isl::val from isl::pw_aff
            assert(OuterMax.n_piece() == 1);
            OuterMax.foreach_piece([&](isl::set S, isl::aff Aff) -> isl::stat {
              isl::val Val = Aff.get_constant_val();
              ElementAccessed.push_back(Val);
              return isl::stat::ok;
            });
          }
          LoopOrder.push_back(LoopDim);
          Relate = true;
          IndexAccessMap++;
        }
        // The dimension does not relate.
        else if (V.is_zero()) {
          if (!Relate && LoopDim == AccMap.dim(isl::dim::in) - 1) {
            // DEBUG(dbgs() << "found dimension that does not relate to any loop
            // index\n");
            isl::local_space Ls =
                isl::local_space(AccessSet.get_space().params());
            isl::aff One = isl::aff(Ls);
            One = One.add_constant_si(1);
            isl::val Val = One.get_constant_val();
            ElementAccessed.push_back(Val);
          }
        }
      }
      return isl::stat::ok;
    });
  }
  // DEBUG(dbgs() << "****" << "\n");
}

/*
static void getElementAccessed(MemoryAccess *MemAcc, isl::map &Sched,
        isl::set &AccessSet, std::vector<isl::val>& ElementAccessed,
        std::vector<unsigned>& LoopOrder) {

    //DEBUG(dbgs() << "&&&&&&&&&&&&&&&" << "\n");
    //const SCEV *S = MemAcc->getSubscript(0);
    //S->dump();
    //auto Array = MemAcc->getLatestScopArrayInfo();
    //S = Array->getDimensionSize(1);
    //S->dump();
    //DEBUG(dbgs() << Array->getElemSizeInBytes() << "\n");
    //DEBUG(dbgs() << "&&&&&&&&&&&&&&&" << "\n");
    //isl::union_map Accesses = p->getStatement()->getParent()->getAccesses();
    //Accesses = Accesses.apply_domain(S);
    //isl::union_set AccessRange = Accesses.range();
    //AccessRange = AccessRange.coalesce();
    //AccessRange = AccessRange.detect_equalities();
    //AccessRange = AccessRange.coalesce();
    //DEBUG(dbgs() << "AccessRange@" << AccessRange << "\n");
    //DEBUG(dbgs() << "ScheduleSPace@" << S.get_space() << "\n");
    //isl::set AccessSet = AccessRange.extract_set(S.get_space().range());
    //DEBUG(dbgs() << "AccessSet@" << AccessSet << "\n");
    //TODO: can we extract the Dom in a better way
    //isl::union_set Dom = MemAcc->getStatement()->getParent()->getDomains();
    //DEBUG(dbgs() << "Dom@" << Dom << "\n");


    //DEBUG(dbgs() << "New?" << MemAcc->hasNewAccessRelation() << "\n");
    isl::map AccMap = MemAcc->getLatestAccessRelation();
    //DEBUG(dbgs() << "AccessMapP " << AccMap << "\n");
    isl::pw_multi_aff MultAff = MemAcc->applyScheduleToAccessRelation(Sched);
    //DEBUG(dbgs() << "Size =" << AccMap.dim(isl::dim::in) << "\n");
    DEBUG(dbgs() << "MultAff@" << MultAff << "\n");
    for(unsigned u=0; u<AccMap.dim(isl::dim::out); ++u){
      isl::pw_aff PwAff = MultAff.get_pw_aff(u);
      DEBUG(dbgs() << "PwAff@" << PwAff << "\n");
      PwAff.foreach_piece([&](isl::set S, isl::aff Aff) -> isl::stat {
        //DEBUG(dbgs() << "Aff@" << Aff <<"\n");
        for(unsigned LoopDim=0; LoopDim < AccMap.dim(isl::dim::in); ++LoopDim) {
          unsigned IndexAccMap = 0;
          //DEBUG(dbgs() << "LoopDim" << LoopDim << "\n");
          //DEBUG(dbgs() << "dim Map" << AccMap.dim(isl::dim::in)-1 << "\n");
          isl::val V = Aff.get_coefficient_val(isl::dim::in,LoopDim);
          //DEBUG(dbgs() << "VV" <<V.to_str() << "\n");
          if(!V.is_zero() && LoopDim != 0) {
            isl::pw_aff OuterMin, OuterMax;
            OuterMin = AccessSet.dim_min(IndexAccMap);
            OuterMax = AccessSet.dim_max(IndexAccMap);
            OuterMax = OuterMax.sub(OuterMin);
            /// if we match the dimension we also
            /// record the (relative) loop order.
            LoopOrder.push_back(LoopDim);

            assert(OuterMax.n_piece() == 1);
            OuterMax.foreach_piece([&](isl::set S, isl::aff Aff) -> isl::stat {
              isl::val Val = Aff.get_constant_val();
              ElementAccessed.push_back(Val);
              //DEBUG(dbgs() << "Constant@ := " <<
Aff.get_constant_val().to_str() << "\n"); return isl::stat::ok;
            });
            IndexAccMap++;
          }
          /// In this case the schedule dimension relates
          /// to a dimension of the memory location. However,
          /// we are classifying w.r.t this loop as a consequence
          /// only a single element is accessed.
          if(!V.is_zero() && LoopDim == 0) {
            DEBUG(dbgs() << "OuterMost Loop => " << LoopDim << "\n");
            isl::pw_aff OuterMin = AccessSet.dim_min(IndexAccMap);
            DEBUG(dbgs() << "OuterMin:= " << OuterMin << "\n");
            isl::pw_aff OuterMax = AccessSet.dim_max(IndexAccMap);
            DEBUG(dbgs() << "OuterMax:= " << OuterMax << "\n");
            isl::local_space Ls =
isl::local_space(AccessSet.get_space().params()); isl::aff One = isl::aff(Ls);
            One = One.add_constant_si(1);
            isl::pw_aff OnePw = isl::pw_aff(One);
            /// if we match the dimension we also record the loop order.
            LoopOrder.push_back(LoopDim);

            assert(OnePw.n_piece() == 1);
            OnePw.foreach_piece([&](isl::set S, isl::aff Aff) -> isl::stat {
              isl::val Val = Aff.get_constant_val();
              ElementAccessed.push_back(Val);
              return isl::stat::ok;
            });
            IndexAccMap++;
          }
          /// In case the schedule dimension does not relate
          /// to a dimension of the memory location the number of
          /// elements accessed is equal to zero.

          if(V.is_zero()) {
          }


        }
        return isl::stat::ok;
      });
    }
}
*/

static bool accessSingleElement(std::vector<isl::val> &E) {
  bool isElement = true;
  std::vector<isl::val>::iterator it = E.begin();
  while (it != E.end()) {
    if (!it->is_one())
      isElement = false;
    it++;
  }
  return isElement;
}

static void extractSpecies(std::vector<Reference> &R /*, int LoopLevel*/) {
  assert(!R.empty() && "Structure is empty");
  std::vector<Reference>::iterator it = R.begin();
  while (it != R.end()) {
    if (!it->HasStride[0] && it->AccessType == 1) {
      // DEBUG(dbgs() << it->Step[0].is_zero() << "\n");
      it->Type = Reference::FULL;
      // DEBUG(dbgs() << "[ " << it->Name << " ]" << " FULL " << "\n");
    } else if (!it->HasStride[0] && it->AccessType == 2) {
      // DEBUG(dbgs() << it->Step[0].is_zero() << "\n");
      it->Type = Reference::SHARED;
      // DEBUG(dbgs() << "[ " << it->Name << " ]" << " SHARED " << "\n");
    }

    else if (accessSingleElement(it->ElementAccessed)) {
      it->Type = Reference::SINGLE_ELEMENT;
      // DEBUG(dbgs() << "[ " << it->Name << " ]" << " ELEMENT " << "\n");
    }

    else {
      it->Type = Reference::CHUNK;
      // DEBUG(dbgs() << "[ " << it->Name << " ]" << " CHUNK " << "\n");
    }
    it++;
  }
}

static std::string getTypeAsString(int Index) {
  switch (Index) {
  case 0:
    return "SINGLE_ELEMENT";
  case 1:
    return "CHUNK";
  case 2:
    return "NEIGHBOURHOOD";
  case 3:
    return "FULL";
  case 4:
    return "SHARED";
  default:
    return "ND";
  }
}

static void printStructure(std::vector<Reference> &R, const char *StmtBaseName) {
  assert(!R.empty() && "Structure is empty");
  // DEBUG(dbgs() << R.size() << "\n");
  DEBUG(dbgs() << "Stmt Name := " << StmtBaseName << "\n");
  std::vector<Reference>::iterator it = R.begin();
  while (it != R.end()) {
    if (it->AccessType == 2)
      DEBUG(dbgs() << "[" << it->Name << " " << getTypeAsString(it->Type) << "]"
                   << " -> ");
    it++;
  }
  it = R.begin();
  while (it != R.end()) {
    if (it->AccessType != 2)
      DEBUG(dbgs() << "[" << it->Name << " " << getTypeAsString(it->Type)
                   << "]");
    it++;
  }
  DEBUG(dbgs() << "\n");
  while (!R.empty()) {
    Reference RR = R.back();
    DEBUG(dbgs() << "AccessType@: " << RR.AccessType << "\n");
    DEBUG(dbgs() << "Name@: " << RR.Name << "\n");
    DEBUG(dbgs() << "Domain@: " << RR.Domain << "\n");
    DEBUG(dbgs() << "ElementAccessed for each outer loop iteration :=");
    while (!RR.ElementAccessed.empty()) {
      isl::val Val = RR.ElementAccessed.front();
      DEBUG(dbgs() << "[" << Val.to_str() << "]");
      RR.ElementAccessed.erase(RR.ElementAccessed.begin());
    }
    DEBUG(dbgs() << "\n");
    DEBUG(dbgs() << "StrideOnDimension :=");
    while (!RR.HasStride.empty()) {
      bool Val = RR.HasStride.front();
      DEBUG(dbgs() << "[" << Val << "]");
      RR.HasStride.erase(RR.HasStride.begin());
    }
    DEBUG(dbgs() << "\n");
    DEBUG(dbgs() << "Partial loop order :=");
    while (!RR.LoopOrder.empty()) {
      DEBUG(dbgs() << "[" << RR.LoopOrder.front() << "]");
      RR.LoopOrder.erase(RR.LoopOrder.begin());
    }
    DEBUG(dbgs() << "\n");
    R.pop_back();
  }
}

// GSOC
static void classifyReference(ScopStmt &Stmt,
                              std::vector<Reference> &References) {

  DEBUG(dbgs() << "@@@@@@@@@@@@@@@@@@@@@@@@@@\n");
  Stmt.dump();
  Scop *S = Stmt.getParent();
  S->getSchedule();
  DEBUG(dbgs() << "Global Schedule := " << S->getSchedule() << "\n");
  isl::map StmtSchedule = Stmt.getSchedule();
  DEBUG(dbgs() << "Stmt Domain :=" << StmtSchedule << "\n");
  isl::set Domain = Stmt.getDomain();
  DEBUG(dbgs() << "Domain" << Domain << "\n");
  DEBUG(dbgs() << StmtSchedule.intersect_domain(Domain).coalesce() << "\n");
  DEBUG(dbgs() << "Base Name: " << Stmt.getBaseName() << "\n");
  // DEBUG(dbgs() << StmtSchedule.range() << "\n");
  // isl::set setRange = StmtSchedule.range();
  // setRange.foreach_basic_set([&](isl::basic_set BSet) -> isl::stat {
  // DEBUG(dbgs() << "Bset" << BSet << "\n");
  // return isl::stat::ok;
  //});
  DEBUG(dbgs() << "@@@@@@@@@@@@@@@@@@@@@@@@@@\n");

  // try to find loop order.
  /*
    Loop *l = Stmt.getSurroundingLoop();
    l->dumpVerbose();
    Scop *S = Stmt.getParent();
    isl::union_map Schedule = Schedule.empty(S->getParamSpace());
    int CurrDim = Stmt.getParent()->getRelativeLoopDepth(l);
    DEBUG(dbgs() << "Relative Loop Depth: " << CurrDim << "\n");
    for( auto &SS : *S) {
      if(l->contains(SS.getSurroundingLoop())) {
        unsigned int MaxDim = SS.getNumIterators();
        DEBUG(dbgs() << "MaxDim :=" << MaxDim << "\n");
        isl::map ScheduleMap = SS.getSchedule();
        ScheduleMap = ScheduleMap.project_out(isl::dim::out, CurrDim+1,
    MaxDim-CurrDim-1); ScheduleMap = ScheduleMap.set_tuple_id(isl::dim::in,
    SS.getDomainId()); Schedule = Schedule.unite(isl::union_map(ScheduleMap));
      }
    }
    Schedule = Schedule.coalesce();
    DEBUG(dbgs() << "Schedule :=" << Schedule << "\n");
  */

  auto Accesses = getAccessesInOrder(Stmt);

  /// Used to populate References.
  int Index = 0;

  for (auto *MemA = Accesses.begin(); MemA != Accesses.end(); MemA++) {

    auto *MemAccessPtr = *MemA;

    if (MemAccessPtr->isMemoryIntrinsic()) {
      DEBUG(dbgs() << "intrinsic access (memcpy, memset, memmove) "
                   << "\n");
      continue;
    }

    /*
        enum MemoryKind MK = MemAccessPtr->getOriginalKind();
        if(MK == MemoryKind::Array) {
          DEBUG(dbgs() << "Array kind\n");
          DEBUG(dbgs() << MemAccessPtr->getOriginalBaseAddr() << "\n");
        }
        if(MK == MemoryKind::Value) {
          DEBUG(dbgs() << "Value kind\n");
          DEBUG(dbgs() << MemAccessPtr->getOriginalBaseAddr() << "\n");
        }
        if(MK == MemoryKind::PHI) {
          DEBUG(dbgs() << "PHI kind\n");
          DEBUG(dbgs() << MemAccessPtr->getOriginalBaseAddr() << "\n");
          auto PHI = cast<PHINode>(MemAccessPtr->getAccessInstruction());
          PHI->dump();
        }
        if(MK == MemoryKind::ExitPHI) {
          DEBUG(dbgs() << "Exit kind\n");
          DEBUG(dbgs() << MemAccessPtr->getOriginalBaseAddr() << "\n");
        }
    */

    enum MemoryKind MK = MemAccessPtr->getOriginalKind();
    if (MK != MemoryKind::Array)
      continue;

    /// 5-tuples classification.

    References.push_back(Reference());

    /// get access type. required by the classification.
    References[Index].AccessType = MemAccessPtr->getType();
    // DEBUG(dbgs() << "AccessType@ :=" << References[Index].AccessType <<
    // "\n");
    /// get array name. required by the classification.
    References[Index].Name = MemAccessPtr->getLatestScopArrayInfo()->getName();
    // DEBUG(dbgs() << "Name@ :=" << References[Index].Name << "\n");

    // unsigned LoopLevel = 0;
    getDomain(MemAccessPtr, References[Index].Domain);
    isl::map ScheduleMap = Stmt.getSchedule();
    getElementAccessed(MemAccessPtr, ScheduleMap, /*LoopLevel,*/
                       References[Index].Domain,
                       References[Index].ElementAccessed,
                       References[Index].LoopOrder);
    getStep(MemAccessPtr, ScheduleMap,
            /*LoopLevel,*/ References[Index].HasStride);
    Index++;
  }
}

static bool checkForStreamingAccesses(const std::vector<Reference> &References) {

  for(auto it = References.begin(); it != References.end(); it++) {
    if(it->Type != Reference::CHUNK && it->Type != Reference::SINGLE_ELEMENT)
      return false;
  }
 
  assert(References.size() != 1);
 
  for(std::size_t i = 1; i < References.size(); ++i) {
    if(!std::equal(References[0].LoopOrder.begin(),References[0].LoopOrder.end(),
             References[i].LoopOrder.begin()))
      return false;
  }
  
  return true;

}

/// Is a streaming kernel?
/// 1. The skeleton should look like: CHUNK -> CHUNK or ELEMENT -> ELEMENT.
/// 2. The partial index order should be the same.
/// 3. TODO:It shoudl have a stride in the innermost dimension.
/// 4. TODO: Dependencies?

static bool isStreaming(isl::schedule_node Node, const std::vector<Skeleton> &SK) {
  isl::union_map PartialSchedule = isl::manage(
    isl_schedule_node_band_get_partial_schedule_union_map(Node.get()));
  if(PartialSchedule.n_map() != 1)
    return false;
  isl::map NewPartialSchedule = isl::map::from_union_map(PartialSchedule);
  isl::id InputDimsId = NewPartialSchedule.get_tuple_id(isl::dim::in);
  ScopStmt *Stmt = static_cast<ScopStmt *>(InputDimsId.get_user());
  bool Streaming = false;
  std::vector<Skeleton>::const_iterator it = SK.begin();
  for(; it != SK.end(); it++) {
    if(strcmp(it->StmtBaseName, Stmt->getBaseName()) != 0)
      continue;
    if(checkForStreamingAccesses(it->References))
      Streaming = true;
  } 
  return Streaming;
}


__isl_give isl_schedule_node *
ScheduleTreeOptimizer::optimizeBand(__isl_take isl_schedule_node *Node,
                                    void *User) {

  // classifyBand(isl::manage_copy(Node));

  if (!isTileableBandNode(isl::manage_copy(Node)))
    return Node;

  const OptimizerAdditionalInfoTy *OAI =
    static_cast<const OptimizerAdditionalInfoTy *>(User);

  if(isStreaming(isl::manage_copy(Node), OAI->Skeletons)) {
    DEBUG(dbgs() << "Streaming\n");
    return Node;
  }

  MatMulInfoTy MMI;
  if (PMBasedOpts && User &&
      isMatrMultPattern(isl::manage_copy(Node), OAI->D, MMI)) {
    LLVM_DEBUG(dbgs() << "The matrix multiplication pattern was detected\n");
    MatMulOpts++;
    return optimizeMatMulPattern(isl::manage(Node), OAI->TTI, MMI).release();
  }

  return standardBandOpts(isl::manage(Node), User).release();
}

isl::schedule
ScheduleTreeOptimizer::optimizeSchedule(isl::schedule Schedule,
                                        const OptimizerAdditionalInfoTy *OAI) {
  auto Root = Schedule.get_root();
  Root = optimizeScheduleNode(Root, OAI);
  return Root.get_schedule();
}

isl::schedule_node ScheduleTreeOptimizer::optimizeScheduleNode(
    isl::schedule_node Node, const OptimizerAdditionalInfoTy *OAI) {
  Node = isl::manage(isl_schedule_node_map_descendant_bottom_up(
      Node.release(), optimizeBand,
      const_cast<void *>(static_cast<const void *>(OAI))));
  return Node;
}

bool ScheduleTreeOptimizer::isProfitableSchedule(Scop &S,
                                                 isl::schedule NewSchedule) {
  // To understand if the schedule has been optimized we check if the schedule
  // has changed at all.
  // TODO: We can improve this by tracking if any necessarily beneficial
  // transformations have been performed. This can e.g. be tiling, loop
  // interchange, or ...) We can track this either at the place where the
  // transformation has been performed or, in case of automatic ILP based
  // optimizations, by comparing (yet to be defined) performance metrics
  // before/after the scheduling optimizer
  // (e.g., #stride-one accesses)
  if (S.containsExtensionNode(NewSchedule))
    return true;
  auto NewScheduleMap = NewSchedule.get_map();
  auto OldSchedule = S.getSchedule();
  assert(OldSchedule && "Only IslScheduleOptimizer can insert extension nodes "
                        "that make Scop::getSchedule() return nullptr.");
  bool changed = !OldSchedule.is_equal(NewScheduleMap);
  return changed;
}

namespace {

class IslScheduleOptimizer : public ScopPass {
public:
  static char ID;

  explicit IslScheduleOptimizer() : ScopPass(ID) {}

  ~IslScheduleOptimizer() override { isl_schedule_free(LastSchedule); }

  /// Optimize the schedule of the SCoP @p S.
  bool runOnScop(Scop &S) override;

  /// Print the new schedule for the SCoP @p S.
  void printScop(raw_ostream &OS, Scop &S) const override;

  /// Register all analyses and transformation required.
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  /// Release the internal memory.
  void releaseMemory() override {
    isl_schedule_free(LastSchedule);
    LastSchedule = nullptr;
  }

private:
  isl_schedule *LastSchedule = nullptr;
};
} // namespace

char IslScheduleOptimizer::ID = 0;

/// isOnlyRead?
/*
static bool containOnlyReads(ScopStmt &Stmt) {

  auto Accesses = getAccessesInOrder(Stmt);
  bool OnlyReads = true;
  for( auto *MemA = Accesses.begin(); MemA != Accesses.end(); MemA++) {
    auto *MemAccessPtr = *MemA;
    if(!MemAccessPtr->isRead())
      OnlyReads = false;
  }

  return OnlyReads;
}
*/

/// Check Stmt profittability.
/// TODO; Skip initialization Stmt.
/// Counting the number of memory references/instructions
/// executed is a good metrics?
static bool isProfitableStmt(ScopStmt &Stmt) {

  if (Stmt.isEmpty()) {
    DEBUG(dbgs() << "Profitability: Empty Stmt"
                 << "\n");
    return false;
  }

  /// TODO: find a better profitability check.
  /// maybe it is better to count the number of instructions (i.e. load/store).
  if (getAccessesInOrder(Stmt).size() <= 2) {
    DEBUG(dbgs() << "Profitability: Few Mem. references"
                 << "\n");
    return false;
  }

  DEBUG(dbgs() << "Profitability: Stmt is profitable"
               << "\n");
  return true;
}

/// Collect the statments for the classification.
static void collectStmtForClassification(Scop &S, std::vector<Skeleton> &SK) {
  int Index = 0;
  for (auto &Stmt : S) {
    if (isProfitableStmt(Stmt)) {
      SK.push_back(Skeleton());
      SK[Index].StmtBaseName = Stmt.getBaseName(); 
      classifyReference(Stmt, SK[Index].References);
      extractSpecies(SK[Index].References);
      //printStructure(SK[Index].References, SK[Index].StmtBaseName);
      Index++;
    } else {
      DEBUG(dbgs() << "Skip Stmt"
                   << "\n");
    }
  }
}

/*
/// Classify the statment.
static void walkScheduleTreeForClassification(isl::schedule Schedule) {
  isl::schedule_node Root = Schedule.get_root();
  if(!Root)
    return;
  isl_schedule_node_foreach_descendant_top_down(
    Root.get(),
    [](__isl_keep isl_schedule_node *nodeptr, void *user) -> isl_bool {
      isl::schedule_node Node = isl::manage_copy(nodeptr);

      auto FilterSequence = isl_schedule_node_get_type(Node.get());
      if(FilterSequence == isl_schedule_node_sequence)
        DEBUG(dbgs() << isl_schedule_node_n_children(Node.get()) << "\n");


      auto BandNode = isl_schedule_node_get_type(Node.get());
      if(BandNode == isl_schedule_node_band) {
        DEBUG(dbgs() << "classify" << "\n");

        classifyReference(Node);
        //extractSpecies(References);
        printStructure(References);
        assert(References.empty());
        //return isl_bool_true;
      }

      else {
        DEBUG(dbgs() << "Continue due to Other Node\n");
        //return isl_bool_true;
      }

     return isl_bool_true;
  },
  nullptr);
}
*/

/// Collect statistics for the schedule tree.
///
/// @param Schedule The schedule tree to analyze. If not a schedule tree it is
/// ignored.
/// @param Version  The version of the schedule tree that is analyzed.
///                 0 for the original schedule tree before any transformation.
///                 1 for the schedule tree after isl's rescheduling.
///                 2 for the schedule tree after optimizations are applied
///                 (tiling, pattern matching)
static void walkScheduleTreeForStatistics(isl::schedule Schedule, int Version) {
  auto Root = Schedule.get_root();
  if (!Root)
    return;

  isl_schedule_node_foreach_descendant_top_down(
      Root.get(),
      [](__isl_keep isl_schedule_node *nodeptr, void *user) -> isl_bool {
        isl::schedule_node Node = isl::manage_copy(nodeptr);
        int Version = *static_cast<int *>(user);

        switch (isl_schedule_node_get_type(Node.get())) {
        case isl_schedule_node_band: {
          NumBands[Version]++;
          if (isl_schedule_node_band_get_permutable(Node.get()) ==
              isl_bool_true)
            NumPermutable[Version]++;

          int CountMembers = isl_schedule_node_band_n_member(Node.get());
          NumBandMembers[Version] += CountMembers;
          for (int i = 0; i < CountMembers; i += 1) {
            if (Node.band_member_get_coincident(i))
              NumCoincident[Version]++;
          }
          break;
        }

        case isl_schedule_node_filter:
          NumFilters[Version]++;
          break;

        case isl_schedule_node_extension:
          NumExtension[Version]++;
          break;

        default:
          break;
        }

        return isl_bool_true;
      },
      &Version);
}

bool IslScheduleOptimizer::runOnScop(Scop &S) {
  // Skip SCoPs in case they're already optimised by PPCGCodeGeneration
  if (S.isToBeSkipped())
    return false;

  // Skip empty SCoPs but still allow code generation as it will delete the
  // loops present but not needed.
  if (S.getSize() == 0) {
    S.markAsOptimized();
    return false;
  }

  const Dependences &D =
      getAnalysis<DependenceInfo>().getDependences(Dependences::AL_Statement);

  if (D.getSharedIslCtx() != S.getSharedIslCtx()) {
    LLVM_DEBUG(dbgs() << "DependenceInfo for another SCoP/isl_ctx\n");
    return false;
  }

  if (!D.hasValidDependences())
    return false;

  isl_schedule_free(LastSchedule);
  LastSchedule = nullptr;

  // Build input data.
  int ValidityKinds =
      Dependences::TYPE_RAW | Dependences::TYPE_WAR | Dependences::TYPE_WAW;
  int ProximityKinds;

  if (OptimizeDeps == "all")
    ProximityKinds =
        Dependences::TYPE_RAW | Dependences::TYPE_WAR | Dependences::TYPE_WAW;
  else if (OptimizeDeps == "raw")
    ProximityKinds = Dependences::TYPE_RAW;
  else {
    errs() << "Do not know how to optimize for '" << OptimizeDeps << "'"
           << " Falling back to optimizing all dependences.\n";
    ProximityKinds =
        Dependences::TYPE_RAW | Dependences::TYPE_WAR | Dependences::TYPE_WAW;
  }

  isl::union_set Domain = S.getDomains();

  if (!Domain)
    return false;

  ScopsProcessed++;
  walkScheduleTreeForStatistics(S.getScheduleTree(), 0);

  isl::union_map Validity = isl::manage(D.getDependences(ValidityKinds));
  isl::union_map Proximity = isl::manage(D.getDependences(ProximityKinds));

  // Simplify the dependences by removing the constraints introduced by the
  // domains. This can speed up the scheduling time significantly, as large
  // constant coefficients will be removed from the dependences. The
  // introduction of some additional dependences reduces the possible
  // transformations, but in most cases, such transformation do not seem to be
  // interesting anyway. In some cases this option may stop the scheduler to
  // find any schedule.
  if (SimplifyDeps == "yes") {
    Validity = Validity.gist_domain(Domain);
    Validity = Validity.gist_range(Domain);
    Proximity = Proximity.gist_domain(Domain);
    Proximity = Proximity.gist_range(Domain);
  } else if (SimplifyDeps != "no") {
    errs() << "warning: Option -polly-opt-simplify-deps should either be 'yes' "
              "or 'no'. Falling back to default: 'yes'\n";
  }

  LLVM_DEBUG(dbgs() << "\n\nCompute schedule from: ");
  LLVM_DEBUG(dbgs() << "Domain := " << Domain << ";\n");
  LLVM_DEBUG(dbgs() << "Proximity := " << Proximity << ";\n");
  LLVM_DEBUG(dbgs() << "Validity := " << Validity << ";\n");

  unsigned IslSerializeSCCs;

  if (FusionStrategy == "max") {
    IslSerializeSCCs = 0;
  } else if (FusionStrategy == "min") {
    IslSerializeSCCs = 1;
  } else {
    errs() << "warning: Unknown fusion strategy. Falling back to maximal "
              "fusion.\n";
    IslSerializeSCCs = 0;
  }

  int IslMaximizeBands;

  if (MaximizeBandDepth == "yes") {
    IslMaximizeBands = 1;
  } else if (MaximizeBandDepth == "no") {
    IslMaximizeBands = 0;
  } else {
    errs() << "warning: Option -polly-opt-maximize-bands should either be 'yes'"
              " or 'no'. Falling back to default: 'yes'\n";
    IslMaximizeBands = 1;
  }

  int IslOuterCoincidence;

  if (OuterCoincidence == "yes") {
    IslOuterCoincidence = 1;
  } else if (OuterCoincidence == "no") {
    IslOuterCoincidence = 0;
  } else {
    errs() << "warning: Option -polly-opt-outer-coincidence should either be "
              "'yes' or 'no'. Falling back to default: 'no'\n";
    IslOuterCoincidence = 0;
  }

  isl_ctx *Ctx = S.getIslCtx().get();

  isl_options_set_schedule_outer_coincidence(Ctx, IslOuterCoincidence);
  isl_options_set_schedule_serialize_sccs(Ctx, IslSerializeSCCs);
  isl_options_set_schedule_maximize_band_depth(Ctx, IslMaximizeBands);
  isl_options_set_schedule_max_constant_term(Ctx, MaxConstantTerm);
  isl_options_set_schedule_max_coefficient(Ctx, MaxCoefficient);
  isl_options_set_tile_scale_tile_loops(Ctx, 0);

  auto OnErrorStatus = isl_options_get_on_error(Ctx);
  isl_options_set_on_error(Ctx, ISL_ON_ERROR_CONTINUE);

  auto SC = isl::schedule_constraints::on_domain(Domain);
  SC = SC.set_proximity(Proximity);
  SC = SC.set_validity(Validity);
  SC = SC.set_coincidence(Validity);
  auto Schedule = SC.compute_schedule();
  isl_options_set_on_error(Ctx, OnErrorStatus);

  walkScheduleTreeForStatistics(Schedule, 1);

  // In cases the scheduler is not able to optimize the code, we just do not
  // touch the schedule.
  if (!Schedule)
    return false;

  ScopsRescheduled++;

  LLVM_DEBUG({
    auto *P = isl_printer_to_str(Ctx);
    P = isl_printer_set_yaml_style(P, ISL_YAML_STYLE_BLOCK);
    P = isl_printer_print_schedule(P, Schedule.get());
    auto *str = isl_printer_get_str(P);
    dbgs() << "NewScheduleTree: \n" << str << "\n";
    free(str);
    isl_printer_free(P);
  });

  /*GSOC*/
  //Skeletons SK = {std::vector<Skeleton>(),const_cast<Dependences *>(&D)};
  //collectStmtForClassification(S, SK);
  //isl::union_map DWAR = D.getDependences(Dependences::TYPE_WAW);
  // DEBUG(dbgs() << DWAR << "\n");
  // walkScheduleTreeForClassification(Schedule);
  // DEBUG(dbgs() << "END walkScheduleTreForClassificaition\n");

  Function &F = S.getFunction();
  auto *TTI = &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
  std::vector<Skeleton> Skeletons = std::vector<Skeleton>();
  collectStmtForClassification(S, Skeletons);
 
  const OptimizerAdditionalInfoTy OAI = {TTI, const_cast<Dependences *>(&D), Skeletons};
  auto NewSchedule = ScheduleTreeOptimizer::optimizeSchedule(Schedule, &OAI);
  walkScheduleTreeForStatistics(NewSchedule, 2);

  for(std::size_t i = 0; i != Skeletons.size(); ++i)
    printStructure(Skeletons[i].References, Skeletons[i].StmtBaseName);

  if (!ScheduleTreeOptimizer::isProfitableSchedule(S, NewSchedule))
    return false;

  auto ScopStats = S.getStatistics();
  ScopsOptimized++;
  NumAffineLoopsOptimized += ScopStats.NumAffineLoops;
  NumBoxedLoopsOptimized += ScopStats.NumBoxedLoops;

  S.setScheduleTree(NewSchedule);
  S.markAsOptimized();

  if (OptimizedScops)
    errs() << S;

  return false;
}

void IslScheduleOptimizer::printScop(raw_ostream &OS, Scop &) const {
  isl_printer *p;
  char *ScheduleStr;

  OS << "Calculated schedule:\n";

  if (!LastSchedule) {
    OS << "n/a\n";
    return;
  }

  p = isl_printer_to_str(isl_schedule_get_ctx(LastSchedule));
  p = isl_printer_print_schedule(p, LastSchedule);
  ScheduleStr = isl_printer_get_str(p);
  isl_printer_free(p);

  OS << ScheduleStr << "\n";
}

void IslScheduleOptimizer::getAnalysisUsage(AnalysisUsage &AU) const {
  ScopPass::getAnalysisUsage(AU);
  AU.addRequired<DependenceInfo>();
  AU.addRequired<TargetTransformInfoWrapperPass>();

  AU.addPreserved<DependenceInfo>();
}

Pass *polly::createIslScheduleOptimizerPass() {
  return new IslScheduleOptimizer();
}

INITIALIZE_PASS_BEGIN(IslScheduleOptimizer, "polly-opt-isl",
                      "Polly - Optimize schedule of SCoP", false, false);
INITIALIZE_PASS_DEPENDENCY(DependenceInfo);
INITIALIZE_PASS_DEPENDENCY(ScopInfoRegionPass);
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass);
INITIALIZE_PASS_END(IslScheduleOptimizer, "polly-opt-isl",
                    "Polly - Optimize schedule of SCoP", false, false)
