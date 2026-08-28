cd /Users/demichel/WORK/NEURAL_NETWORKS/PAINNLT/MLCG_Framework_v2
LR="/Users/demichel/WORK/NEURAL_NETWORKS/PAINNLT/MLCG_Framework_v2/tutorials/tel22/long_run"
AA_TOPOLOGY="$LR/md.gro" \
AA_TRAJECTORY="$LR/md_whole.trr" \
TRAINER="$PWD/training/build/train_painn" \
PYRESSO="$PWD/espresso/build/pypresso" \
PIPELINE_TEST_RUN_DIR="$PWD/tutorials/tel22/diagnostics/smoke/antiparallel_long_40ep" \
DEVICE=auto \
bash tutorials/tel22/diagnostics/scripts/07_test_antiparallel_pipeline_40ep.sh
