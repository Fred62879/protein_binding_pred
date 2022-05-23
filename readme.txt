# Phase 1
I tried only the '9A_100sup_epoch64' model for dMaSIF.

         F1    AUC
IDR     0.32   0.67
dMaSIF  0.24   0.63

roc graph in result/phase1


# Phase 2
I tried all four models
0 - 9A_100sup_epoch64
1 - 9A_100sup_epoch64
2 - 12A_0.7res_150sup_epoch59
3 - 12A_100sup_epoch71

F1 score and auc can be found in result/phase2/{f1/auc}_{experiment_name}.npy
These arrays are n by 2, where each entry (e.g. ['1H8B-A', '0.47']) records the
pdb name and the corresponding metric score (f1 or auc).

The corresponding roc graph can be found in result/phase2/roc_{experiment_name}

NOTE, I seem to only get 76 out of the 84 pdb files successfully processed.
Since most of them can be predicted without issues, I decide to submit my
work at this point and debugging afterwards. Sorry for this rush work.

# Phase 3
