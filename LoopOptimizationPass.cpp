#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
    struct LoopOptimizationPass : public FunctionPass {
        static char ID;
        LoopOptimizationPass() : FunctionPass(ID) {}
        
        bool runOnFunction(Function &F) override {
            auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
            auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
            
            bool Changed = false;
            
            for (auto *L : LI) {
                Changed |= optimizeLoop(L, SE);
            }
            
            return Changed;
        }
        
        bool optimizeLoop(Loop *L, ScalarEvolution &SE) {
            bool Changed = false;
            
            if (!L->isLoopSimplifyForm()) return false;
            
            auto TripCount = SE.getSmallConstantTripCount(L);
            if (!TripCount) return false;
            
            errs() << "Optimizing loop with trip count: " << *TripCount << "\n";
            
            if (*TripCount <= 16) {
                UnrollLoopOptions ULO;
                ULO.Count = *TripCount;
                ULO.Force = true;
                ULO.AllowExpensiveTripCount = true;
                
                if (UnrollLoop(L, ULO, &LI, &SE, nullptr, nullptr, nullptr, nullptr)) {
                    Changed = true;
                    errs() << "  - Unrolled loop completely\n";
                }
            }
            
            for (auto *SubL : L->getSubLoops()) {
                Changed |= optimizeLoop(SubL, SE);
            }
            
            if (L->isPerfectlyNested()) {
                auto OuterLoop = L;
                auto InnerLoop = *L->begin();
                
                if (shouldInterchangeLoops(OuterLoop, InnerLoop, SE)) {
                    Changed |= interchangeLoops(OuterLoop, InnerLoop, SE);
                    errs() << "  - Interchanged loops\n";
                }
            }
            
            return Changed;
        }
        
        bool shouldInterchangeLoops(Loop *Outer, Loop *Inner, ScalarEvolution &SE) {
            auto OuterMemOps = getMemoryOpsInLoop(Outer);
            bool hasStride1InInner = false;
            
            for (auto *I : OuterMemOps) {
                if (auto *LI = dyn_cast<LoadInst>(I)) {
                    if (auto *GEP = dyn_cast<GetElementPtrInst>(LI->getPointerOperand())) {
                        if (auto *Idx = dyn_cast<Instruction>(GEP->getOperand(1))) {
                            auto *SCEV = SE.getSCEV(Idx);
                            if (auto *AddRec = dyn_cast<SCEVAddRecExpr>(SCEV)) {
                                if (AddRec->getLoop() == Outer) {
                                    if (auto *Const = dyn_cast<SCEVConstant>(AddRec->getStepOperand())) {
                                        if (Const->getAPInt() == 1) {
                                            hasStride1InInner = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            return hasStride1InInner;
        }
        
        bool interchangeLoops(Loop *Outer, Loop *Inner, ScalarEvolution &SE) {
            return true;
        }
        
        SmallVector<Instruction*, 16> getMemoryOpsInLoop(Loop *L) {
            SmallVector<Instruction*, 16> MemOps;
            
            for (auto *BB : L->getBlocks()) {
                for (auto &I : *BB) {
                    if (isa<LoadInst>(I) || isa<StoreInst>(I)) {
                        MemOps.push_back(&I);
                    }
                }
            }
            
            return MemOps;
        }
        
        void getAnalysisUsage(AnalysisUsage &AU) const override {
            AU.addRequired<LoopInfoWrapperPass>();
            AU.addRequired<ScalarEvolutionWrapperPass>();
            AU.addRequired<DominatorTreeWrapperPass>();
            AU.setPreservesAll();
        }
    };
}

char LoopOptimizationPass::ID = 0;

static RegisterPass<LoopOptimizationPass> X(
    "loop-opt", "Custom Loop Optimization Pass",
    false, false
);

static void registerLoopOptimizationPass(const PassManagerBuilder &,
                                         legacy::PassManagerBase &PM) {
    PM.add(new LoopOptimizationPass());
}

static RegisterStandardPasses
    RegisterLoopOptimizationPass(PassManagerBuilder::EP_EarlyAsPossible,
                                registerLoopOptimizationPass);
