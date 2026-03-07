set -euo pipefail

INPUT_DIR="/data/nfs/lhj/DenseMatching/MegaDepth"
PAIRS_500="./data/megadepth_500.txt"
PAIRS_MR="./data/megadepth_mr.txt"
OUT_BASE="outputs/eval_500_g"
COMMON_ARGS=(--input_dir "$INPUT_DIR" --input_pairs "$PAIRS_500" --output_dir "$OUT_BASE" --resize 1024 --save --overlaper samatcher --mask_filter --mask_filter_thresh 0.4)

run_eval() {
  python3 evaluation.py "${COMMON_ARGS[@]}" "$@"
}

run_eval --matcher loftr --direct
run_eval --matcher NN --extractor d2net-ss
run_eval --matcher NN --extractor landmark
run_eval --matcher NN --extractor disk-desc
run_eval --matcher NN --extractor r2d2-desc
run_eval --matcher NN --extractor context-desc
run_eval --matcher NN --extractor superpoint_aachen
run_eval --matcher superglue_disk --extractor disk-desc
run_eval --matcher superglue_outdoor --extractor superpoint_aachen

python3 evaluation.py --input_dir "$INPUT_DIR" --input_pairs "$PAIRS_500" --output_dir outputs/eval_v --matcher superglue_outdoor --extractor superpoint_aachen --resize 1024 --save --overlaper samatcher --mask_filter --mask_filter_thresh 0.4 --viz
python3 evaluation.py --input_dir "$INPUT_DIR" --input_pairs "$PAIRS_MR" --output_dir outputs/eval_mr --matcher superglue_outdoor --extractor superpoint_aachen --resize 1024 --save --overlaper samatcher --mask_filter --mask_filter_thresh 0.4 --viz_box_mask_constraint