echo 'SIFT'
# python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./data/MegaDepth_Val_Scales_sampled_3000.txt --output_dir outputs/eval/SIFT --matcher NN --extractor landmark --resize -1 --save
python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./data/MegaDepth_Val_Scales_sampled_3000.txt --output_dir outputs/eval/SIFT --matcher NN --extractor landmark --resize 1024 --save --overlaper samatcher

# python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/MegaDepth_Val_Scales_sampled_3000.txt --output_dir outputs/LoFTR --matcher loftr --direct --resize -1 --save
# python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/MegaDepth_Val_Scales_sampled_3000.txt --output_dir outputs/LoFTR --matcher loftr --direct --resize 640 --save --overlaper detmatcher

#echo 'with DetMatcher'
# python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/megadepth_22 --matcher loftr --direct --resize 640 --save --overlaper detmatcher
# python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/megadepth_22 --matcher NN --extractor d2net-ss  --resize 640 --save --overlaper detmatcher
# python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/megadepth_22 --matcher NN --extractor context-desc  --resize 640 --save --overlaper detmatcher
# python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/megadepth_22 --matcher NN --extractor aslfeat-desc  --resize 640 --save --overlaper detmatcher
# python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/megadepth_2 --matcher NN --extractor superpoint_aachen  --resize 640 --save --overlaper detmatcher
# python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/megadepth_2 --matcher NN --extractor disk-desc  --resize 640 --save --overlaper detmatcher
# python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/megadepth_2 --matcher NN --extractor r2d2-desc  --resize 640 --save --overlaper detmatcher
# python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/megadepth_2 --matcher superglue_disk --extractor disk-desc  --resize 640 --save --overlaper detmatcher
# python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/megadepth_22 --matcher superglue_outdoor --extractor superpoint_aachen  --resize 640 --save --overlaper detmatcher
