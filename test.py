from src.build_samatcher import build_samatcher
# Baseline SAM2.1
# checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
# model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Ours HQ-SAM 2
checkpoint = 'checkpoints/sam2.1_hq_hiera_large.pt'
model_cfg = 'configs/sam2.1/sam2.1_hiera_l_sammatcher.yaml'
preserved_layers_file_path='configs/sam_2.1_hiera_preserved_args.txt'

model = build_samatcher(model_cfg, checkpoint, preserved_layers_file_path)

for name, module in model.named_children():
    print(name)
    