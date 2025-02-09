<div align="center">

# Lumina-Video

**Official repository for Lumina-Video, a preliminary tryout of the Lumina series for Video Generation**

</div>


<p align="center">
 <img src="assets/architecture.png" width="90%"/>
 <br>
</p>

<h2 id="custom-gallery"> 📽️ Gallery</h2>

### Text to video results

<table border="0" style="width: 100%; text-align: center; margin-top: 1px;">
  <tr>
    <td><video src="https://github.com/user-attachments/assets/10cf854f-9f9b-4820-a0d1-f5bbf6189620" width="100%" controls autoplay loop muted></video></td>
    <td><video src="https://github.com/user-attachments/assets/cd3e140e-b3b5-4465-9565-7c2dd727353c" width="100%" controls autoplay loop muted></video></td>
    <td><video src="https://github.com/user-attachments/assets/197c2274-85ac-417d-b0a5-45c7057af586" width="100%" controls autoplay loop muted></video></td>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/21b0b287-899d-4925-b9d4-2d2fa9dffff3" width="100%" controls autoplay loop muted></video></td>
    <td><video src="https://github.com/user-attachments/assets/92d2dd40-0ba0-46e0-a12b-b0b1ec9e9032" width="100%" controls autoplay loop muted></video></td>
    <td><video src="https://github.com/user-attachments/assets/758d45cb-017a-487b-b8e7-7e97b035d910" width="100%" controls autoplay loop muted></video></td>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/f927d829-bc35-47be-8cbe-823cc7227363" width="100%" controls autoplay loop muted></video></td>
    <td><video src="https://github.com/user-attachments/assets/bb2935af-5109-4255-b206-c582353f915d" width="100%" controls autoplay loop muted></video></td>
    <td><video src="https://github.com/user-attachments/assets/c0375a16-4b1a-40c4-92ca-cd43ce3a82c6" width="100%" controls autoplay loop muted></video></td>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/8ab35b36-ae8a-4ec1-9d37-8d1eb1fb0bcf" width="100%" controls autoplay loop muted></video></td>
    <td><video src="https://github.com/user-attachments/assets/077179a9-79f5-4912-8673-6032b3e5f6e9" width="100%" controls autoplay loop muted></video></td>
    <td><video src="https://github.com/user-attachments/assets/239825c1-2fe8-41b7-9ba4-6ccdd0debea1" width="100%" controls autoplay loop muted></video></td>
  </tr>
</table>

### Text to video+audio results

<table border="0" style="width: 100%; text-align: left; margin-top: 15px; border-collapse: collapse;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/3f66b06e-0c85-474e-9b0d-b2e2b4989464" width="100%" controls autoplay loop muted></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/b279f157-a84d-47cc-8b52-ce9faf0eb9c2" width="100%" controls autoplay loop muted></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/11a71a38-c2ea-4b50-bbbb-766b322626ed" width="100%" controls autoplay loop muted></video>
      </td>
  </tr>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/9f7e38b7-9de8-46b2-bf33-4aff64708fdf" width="100%" controls autoplay loop muted></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/3f3c55f9-c075-40ed-94d2-b7d64240019b" width="100%" controls autoplay loop muted></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/91ab3778-5191-4a8e-aee9-648b6acf362c" width="100%" controls autoplay loop muted></video>
      </td>
  </tr>
</table>

## 📰 News

<!-- - **[2025-02-09] 🎉🎉🎉 [Training codes and documents](./lumina_mgpt/TRAIN.md) are released! 🎉🎉🎉** -->

- **[2025-02-09] 🎉🎉🎉 Lumina-Video is released! 🎉🎉🎉**

## ⚙️ Installation

See [INSTALL.md](./INSTALL.md) for detailed instructions.

## 🤗 Checkpoints

**T2V models**

| resolution | fps  | max frames | Huggingface                                                  |
| ---------- | ---- | ---------- | ------------------------------------------------------------ |
| 960        | 24   | 96         | [Alpha-VLLM/Lumina-Video-f24R960](https://huggingface.co/Alpha-VLLM/Lumina-Video-f24R960) |

## ⛽ Inference

### Preparations

Download the checkpoints before continue. You can use the following code to download the checkpoints to the `./ckpts` directory

```
huggingface-cli download --resume-download Alpha-VLLM/Lumina-Video-f24R960 --local-dir ./ckpts/f24R960
```

### Inference

You can quickly run video generation using the command below:


```bash
# Example for generatingan video with 4s duration, fps=24, resolution=1248x704
python -u generate.py \
    --ckpt ./ckpts/f24R960 \
    --resolution 1248x704 \
    --fps 24 \
    --frames 96 \
    --prompt "your prompt here" \
    --neg_prompt "" \
    --sample_config f24F96R960  # set to "f24F96R960-MultiScale" for efficient multi-scale inference
```

#### QAs

**Q1**: Why using the 1248x704 resolution?

**A1**: The resolution is originally expected to be 1280x720. However, to ensure compatibility with the largest patch size
(smallest scale), both the width and height must be divisible by 32. As a result, the resolution is adjusted to
1248x704.

**Q2**: Does the model support flexible aspect ratio?

**A2**: Yes, you can use the following code for checking all usable resolutions

```Python
# Python
from imgproc import generate_crop_size_list

target_size = 960
patch_size = 32
max_num_patches = (target_size // patch_size) ** 2
crop_size_list = generate_crop_size_list(max_num_patches, patch_size)

print(crop_size_list)
```

## Training

### Preparations

Before starting the training process, two preparation steps are required to optimize training efficiency and enable motion conditioning:

1. **Pre-extract and cache VAE latents for video data**: This significantly enhances training speed.
2. **Compute motion scores for videos**: These are used for micro-conditioning input during training.

#### Pre-Extract VAE Latents

The code for pre-extracting and caching VAE latents can be found in the [./tools/pre_extract](tools/pre_extract) directory. For an example of how to run this, refer to the [run.sh](tools/pre_extract/scripts/run.sh) script.

#### Compute Motion Score

We use UniMatch to estimate optical flow, with the average optical flow serving as the motion score. This code is primarily derived from [Open-Sora](https://github.com/hpcaitech/Open-Sora/tree/main/tools/scoring/optical_flow), and we'd like to thank them for their excellent work!

The code for computing motion scores is available in the [./tools/unimatch](tools/unimatch) directory. To see how to run it, refer to the [run.sh](tools/unimatch/scripts/run.sh) script.

### Training

Once the data has been prepared, you're ready to start training! For an example, you can refer to the [training directory](train_exps/f8F32R256), which demonstrates how to train with:

- **FPS**: 8
- **Duration**: 4 seconds
- **Resolution**: widthxheight≈256x256
- **Training Techniques**: Image-text joint training and multi-scale training applied together.




## 📑 Open-source Plan

- [X] Inference code
- [X] Training code
