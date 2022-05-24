# Overview
This is the repo for Zhuoting Xie's application assignment to the Deep
leanrning assistant role in Prof. Jörg Gsponer's lab.

# Method
The major operation I am doing is to convert from dMaSIF predictions of
bfactors for each point in the sampled point cloud to the classification
of each corresponding residue as either interface(1) / non-interface (0) /
unknown(-1).

To do this, I first gather k nearest points to each atom in the given
structure with distances below a given threshold. Then, I take the mean value
of bfactors for all these points as the estimated bfactor of the atom.
After comparing each atom bfactor with a threshold bfactor, I can classify
each atom in the structure. Note that atom with no satisfying nearest points
has bfactor set to -1 to indicate that the model makes no prediction for
this particular atom.

Then, to classify each residue in the structure, I check the classification
of all atoms this residuce contains. With a given ratio, it more than ratio%
of the atoms are interface, the residuce is classified as interface and vice
versa. Note that if a we don't have classifications for any atom contained
in a residue, then this residue is classified as unknown.

To compare residue classification with GT, I convert GT atom classification
to residue classification. Here, as long as at least one atom is interface,
the whole residue is classified as interface.

Finally, to calculate metrics, I ignore unknown residues (-1) in our prediction
and only calculate scores using the left residues.

# About code
This repo is not fully tested as I don't have GPU on my laptop. However,
I compiled almost exactly the same code on [colab](https://colab.research.google.com/drive/1kmgT0Y98yafi2nvr84UxMMcfddHQNOod), which should work without any issues.

# Results
## Phase 1
I tried only the '9A_100sup_epoch64' model for dMaSIF.

|        |  F1   |  AUC |
|--------|:-----:|-----:|
|  IDR   |  0.32 | 0.67 |
| dMaSIF |  0.24 | 0.64 |

roc graph in `./result/phase1`


## Phase 2
I tried all four models
0 - 9A_100sup_epoch64
1 - 9A_100sup_epoch64
2 - 12A_0.7res_150sup_epoch59
3 - 12A_100sup_epoch71

F1 score and auc can be found in `./result/phase2/{f1/auc}_{experiment_name}.npy`
These arrays are n by 2, where each entry (e.g. ['1H8B-A', '0.47']) records the
pdb name and the corresponding metric score (f1 or auc).

The corresponding roc graph can be found in `./result/phase2/roc_{experiment_name}`

NOTE: I seem to only get 76 out of the 84 pdb files successfully processed.
Since most of them can be predicted without issues, I decide to submit my
work at this point and debugging afterwards. Sorry for this rush work.

## Phase 3
### Experiment Analysis
Technically, IDR gives both better F1 score and AUC than dMaSIF. However,
this may be due to an inappropriate selection of thresholds to convert
from dMaSIF predicted point bfactors to residue classifications from my code.

Moreover, based on the theoertical basis of IDR, if the protein we are
studying (5VAY-A) is in MORF binding mode, which is what IDR is targeted
towards identifying, then IDR will also show some advantages.

### Potential Improvements
From computational perspective, for IDR it's probably not ideal to train two
separate classifiers, one for Rim and one for core. As features that helps
classify residue as Rim may provide information for core as well. Thus, we
may also try multi-class classification, i.e. train a single classifier that
classify residue as either rim, core, or non-interface.

Regarding dMaSIF, one key issue is that it only focuses on the protein surface,
while abstracting the core residues. However, in case of certain protein binding
mode such as the MORF, parts of the binding resiudes may also fold itself and
present as core residues. Actually, one observation for phase 1 experiment with
5VAY-A is that bfactors for atoms in binding regions are even smaller than
those in other regions. This is likely the result of the corresponding
residues being MORF and are thus treated as irrelevant by dMaSIF.

As a result, we can possibly gain some improvements via extending the model to
inner space of the protein as well which however may not be computationally
feasible. While another possibility is to train something similar to the IDR
directly on the the primary amino acid sequence (atoms) instead of the 3D
tertiary protein structure so that we won't miss interface atoms that are
not directly on the sufrace of the protein. Then when we run dMaSIF, we can
apply the score calculated for each atom as a correction score which gives
some further constraints on how atoms should be classified.

Overall, these improevement ideas are only based on a brief skim over the
paper. It's very likely that I miss some information which conveys the
superiority of the actual chosen strategies.

### Other Possibilities
Finally, regarding other possibilities, we can probably try implicit
representations (NeRF) which is another popular 3D geometry model. To the
best of my knowledge, NeRF have not ever been applied on protein modeling,
but it presents some unique advantages.

For example, a NeRF once trained, enables continuous querying of values of
intereset (e.g. bfactor) within the coordinate system. This basically enables
bypassing the conversion step from point cloud to atom which requies careful
tunning of thresholding values.

Moreover, if we are also able to sample a reasonably large but computationally
feasible amount of points within the protein (e.g. for MORF residues), the
network may be able to learn for these inner residues as well.

To get this idea more solid, I need to read more about protein modeling though.