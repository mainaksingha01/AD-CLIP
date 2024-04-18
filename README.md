# AD-CLIP: Adapting Domains in Prompt Space Using CLIP
Official repository of AD-CLIP, which is focused on domain adaptation using *prompt learning* by adapting pre-trained vision-language models (VLM) like CLIP.

## **ICCVw 2023**

[![paper](https://img.shields.io/badge/Conference-Paper-blue)](https://openaccess.thecvf.com/content/ICCV2023W/OODCV/papers/Singha_AD-CLIP_Adapting_Domains_in_Prompt_Space_Using_CLIP_ICCVW_2023_paper.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-brightgreen)](https://arxiv.org/pdf/2308.05659.pdf)

## Abstract
![teaser](https://github.com/mainaksingha01/AD-CLIP/blob/master/images/teaser2.png)

Although deep learning models have shown impressive performance on supervised learning tasks, they often struggle to generalize well when the training (source) and test (target) domains differ. Unsupervised domain adaptation (DA) has emerged as a popular solution to this problem. However, current DA techniques rely on visual backbones, which may lack semantic richness. Despite the potential of large-scale vision-language foundation models like CLIP, their effectiveness for DA has yet to be fully explored. To address this gap, we introduce AD-CLIP, a domain-agnostic prompt learning strategy for CLIP that aims to solve the DA problem in the prompt space. We leverage the frozen vision backbone of CLIP to extract both image style (domain) and content information, which we apply to learn prompt tokens. Our prompts are designed to be domain-invariant and class-generalizable, by conditioning prompt learning on image style and content features simultaneously. We use standard supervised contrastive learning in the source domain, while proposing an entropy minimization strategy to align domains in the embedding space given the target domain data. We also consider a scenario where only target domain samples are available during testing, without any source domain data, and propose a cross-domain style mapping network to hallucinate domain-agnostic tokens. Our extensive experiments on three benchmark DA datasets demonstrate the effectiveness of AD-CLIP compared to existing literature.

## Architecture

![architecture](https://github.com/mainaksingha01/AD-CLIP/blob/master/images/architecture.png)

Also clone the [dassl](https://github.com/KaiyangZhou/Dassl.pytorch/tree/master/dassl) files.

## Bibtex

Please cite the paper if you use our work . Thanks.

```
@inproceedings{singha2023ad,
  title={Ad-clip: Adapting domains in prompt space using clip},
  author={Singha, Mainak and Pal, Harsh and Jha, Ankit and Banerjee, Biplab},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={4355--4364},
  year={2023}
}
```
