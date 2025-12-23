import argparse
import glob
import os

import torch

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("pretrained_model_path", default=None, type=str)
    args = parser.parse_args()

    pretrained_model_path = args.pretrained_model_path
    total_pretrained_model_path = os.path.join(pretrained_model_path, "model.pt")
    saved_transformer3d_path = os.path.join(pretrained_model_path, "transformer.pt")
    saved_portrait_encoder_path = os.path.join(pretrained_model_path, "portrait_encoder.pt")
    checkpoint_files = glob.glob(os.path.join(pretrained_model_path, "*.bin"))
    print(checkpoint_files)
    state_dict = {}
    for checkpoint_file in checkpoint_files:
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        for key, value in checkpoint.items():
            if key in state_dict:
                print(key)
                print(1 / 0)
                state_dict[key] = torch.cat([state_dict[key], value], dim=0)
            else:
                state_dict[key] = value
    torch.save(state_dict, total_pretrained_model_path)
    del checkpoint
    checkpoint = torch.load(total_pretrained_model_path, map_location='cpu')
    for key, value in checkpoint.items():
        print(key)
    print("--------------------------------")
    transformer3d = {}
    portrait_encoder = {}
    for key, value in checkpoint.items():
        if key.startswith("transformer3d"):
            new_key = key[len("transformer3d."):]
            transformer3d[new_key] = value
        elif key.startswith("portrait_encoder"):
            new_key = key[len("portrait_encoder."):]
            portrait_encoder[new_key] = value
    torch.save(transformer3d, saved_transformer3d_path)
    torch.save(portrait_encoder, saved_portrait_encoder_path)
    print("------------------------------------------")
    for key, value in transformer3d.items():
        print(key)
    print("------------------------------------------")
    for key, value in portrait_encoder.items():
        print(key)


# python bin_convert_pt.py --pretrained_model_path="/path/FlashPortrait/output_14B_dir/checkpoint-x-fp32-infer"
