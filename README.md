# Generative-model-unlearning-survey
[![](https://img.shields.io/badge/📑-Survey_Paper-blue)](https://arxiv.org/abs/2507.19894)
[![Awesome](https://awesome.re/badge.svg)](https://github.com/caxLee/Generative-model-unlearning-survey)
![](https://img.shields.io/github/last-commit/caxLee/Generative-model-unlearning-survey?color=green)
![](https://img.shields.io/badge/PRs-Welcome-red)
![](https://img.shields.io/github/stars/caxLee/Generative-model-unlearning-survey?color=yellow)
![](https://img.shields.io/github/forks/caxLee/Generative-model-unlearning-survey?color=lightblue)


A collection of papers and resources about  Generative Model Unlearning (**GenMU**).

With the rapid advancement of generative models, associated privacy concerns have attracted growing attention. To address this,
researchers have begun adapting machine unlearning techniques from traditional classification models to generative settings. Although
notable progress has been made in this area, a unified framework for systematically organizing and integrating existing work is still
lacking. The substantial differences among current studies in terms of unlearning objectives and evaluation protocols hinder the
objective and fair comparison of various approaches. While some studies focus on specific types of generative models, they often
overlook the commonalities and systematic characteristics inherent in Generative Model Unlearning (GenMU). To bridge this gap,
we provide a comprehensive review of current research on GenMU and propose a unified analytical framework for categorizing
unlearning objectives, methodological strategies, and evaluation metrics. In addition, we explore the connections between GenMU and
related techniques, including model editing, reinforcement learning from human feedback, and controllable generation. We further
highlight the potential practical value of unlearning techniques in real-world applications. Finally, we identify key challenges and
outline future research directions aimed at laying a solid foundation for further advancements in this field.

<p align="center">
<img src="fig/framework.png" alt="Framework" />
</p>

## News
🤗 We're actively working on this project, and your interest is greatly appreciated! To keep up with the latest developments, please consider hit the **STAR** 🌟 and **WATCH** for updates.
* Our survey paper: [A Survey on Generative Model Unlearning: Fundamentals, Taxonomy, Evaluation, and Future Direction]((https://arxiv.org/abs/2507.19894) is public.

## Overview



We hope this repository proves valuable to your research or practice in the field of self-supervised learning for recommendation systems. If you find it helpful, please consider citing our work:
```bibtex
@misc{feng2025surveygenerativemodelunlearning,
      title={A Survey on Generative Model Unlearning: Fundamentals, Taxonomy, Evaluation, and Future Direction}, 
      author={Xiaohua Feng and Jiaming Zhang and Fengyuan Yu and Chengye Wang and Li Zhang and Kaixiang Li and Yuyuan Li and Chaochao Chen and Jianwei Yin},
      year={2025},
      eprint={2507.19894},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2507.19894}, 
}
```

## Table of Contents
- [Generative-model-unlearning-survey](#generative-model-unlearning-survey)
  - [News](#news)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
 

## Related 

## 🌐 Point-wise Unlearning
### Text_demo（Unofficial version, currently being updated, please understand）
- ('202)  [[paper]()]
- ('202)  [[paper]()]
- ('202)  [[paper]()]
- ('202)  [[paper]()]
- ('202)  [[paper]()]
- ('202)  [[paper]()]
- ('202)  [[paper]()]
- ('202)  [[paper]()]
- ('202)  [[paper]()]
- ('202)  [[paper]()]
- ('202)  [[paper]()]
- ('202)  [[paper]()]
- ('202)  [[paper]()]
- ('202)  [[paper]()]
- ('202)  [[paper]()]
- ('202)  [[paper]()]
- ('202)  [[paper]()]
- ('202)  [[paper]()]
### Image
- ('202)  [[paper]()]
- ('202)  [[paper]()]
- ('202)  [[paper]()]
- ('202)  [[paper]()]
- ('202)  [[paper]()]
- ('202)  [[paper]()]
### Audio
- ('202)  [[paper]()]
- ('202)  [[paper]()]
- ('202)  [[paper]()]
- ('202)  [[paper]()]
- ('202)  [[paper]()]
- ('202)  [[paper]()]
### Multimodal
- ('202)  [[paper]()]
- ('202)  [[paper]()]
- ('202)  [[paper]()]
- ('202)  [[paper]()]
- ('202)  [[paper]()]
- ('202)  [[paper]()]

## 🌐 Concept-wise Unlearning
### Text
- (arxiv'2023) Who’s Harry Potter? Approximate Unlearning in LLMs. [[paper](https://arxiv.org/abs/2310.02238)]
- (EMNLP'2024) Revisiting Who’s Harry Potter: Towards Targeted Unlearning from a Causal Intervention Perspective. [[paper]- (https://arxiv.org/abs/2407.16997)]
- (ICASSP'2025) Multi-Objective Large Language Model Unlearning. [[paper](https://arxiv.org/abs/2412.20412)]
- (ACL'2023) The CRINGE Loss: Learning what language not to model. [[paper](https://arxiv.org/abs/2211.05826)]
- (NeurIPS'2024) Large language model unlearning. [[paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/be52acf6bccf4a8c0a90fe2f5cfcead3-Abstract-Conference.html)]
- (ACL'2023) Unlearning bias in language models by partitioning gradients. [[paper](https://aclanthology.org/2023.findings-acl.375/)]
- (ICML'2025) Tool Unlearning for Tool-Augmented LLMs. [[paper](https://arxiv.org/abs/2502.01083)]
- (NeurIPS'2022) Quark: Controllable text generation with reinforced unlearning. [[paper](https://arxiv.org/abs/2205.13636)]
- (ACL'2025) Opt-Out: Investigating Entity-Level Unlearning for Large Language Models via Optimal Transport. [[paper](https://arxiv.org/abs/2406.12329)]
- (ICLR'2023) Editing models with task arithmetic. [[paper](https://arxiv.org/abs/2212.04089)]
- (NAACL'2024) Ethos: Rectifying Language Models in Orthogonal Parameter Space. [[paper](https://arxiv.org/abs/2403.08994)]
- (ACL'2024) Making Harmful Behaviors Unlearnable for Large Language Models. [[paper](https://arxiv.org/abs/2311.02105)]
- (AAAI'2024) Separate the wheat from the chaff: Model deficiency unlearning via parameter-efficient module operation. [[paper](https://arxiv.org/abs/2308.08090)]
- (ACL'2024) Towards Safer Large Language Models through Machine Unlearning. [[paper](https://arxiv.org/abs/2402.10058)]
- (EMNLP'2024) To Forget or Not? Towards Practical Knowledge Unlearning for Large Language Models. [[paper](https://arxiv.org/abs/2407.01920)]
- (NAACL'2025) Unlearn efficient removal of knowledge in large language models. [[paper](https://arxiv.org/abs/2408.04140)]
- (NeurIPS'2024) Applying Sparse Autoencoders to Unlearn Knowledge in Language Models. [[paper](https://arxiv.org/abs/2410.19278)]
- (arxiv'2024) When Machine Unlearning Meets Retrieval-Augmented Generation (RAG): Keep Secret or Forget Knowledge? [[paper](https://arxiv.org/abs/2410.15267)]
- (ACL'2021) DExperts: Decoding-Time Controlled Text Generation with Experts and Anti-Experts. [[paper](https://arxiv.org/abs/2105.03023)]
### Image
- (SaTML'2023) Data redaction from pre-trained gans. [[paper](https://arxiv.org/abs/2206.14389)]
- (TDSC'2025) Generative Adversarial Networks Unlearning. [[paper](https://arxiv.org/abs/2308.09881)]
- (ICML'2023) Gradient Surgery for One-shot Unlearning on Generative Model. [[paper](https://arxiv.org/abs/2307.04550)]
- (AAAI'2024) Feature unlearning for pre-trained GANs and VAEs. [[paper](https://arxiv.org/abs/2303.05699)]
- (WACV'2024) Taming Normalizing Flows. [[paper](https://arxiv.org/abs/2211.16488)]
- (ICCV'2023) Erasing Concepts from Diffusion Models. [[paper](https://openaccess.thecvf.com/content/ICCV2023/html/Gandikota_Erasing_Concepts_from_Diffusion_Models_ICCV_2023_paper.html)]
- (CVPR'2025) Ace: Anti-editing concept erasure in text-to-image models. [[paper](https://arxiv.org/abs/2501.01633)]
- (ICCV'2023) Ablating Concepts in Text-to-Image Diffusion Models. [[paper](https://arxiv.org/abs/2303.13516)]
- (ACM MM'2023) Degeneration-Tuning: Using Scrambled Grid shield Unwanted Concepts from Stable Diffusion. [[paper](https://arxiv.org/abs/2308.02552)]
- (ACM CCS'2024) Safegen: Mitigating sexually explicit content generation in text-to-image models. [[paper](https://arxiv.org/abs/2404.06666)]
- (ICML'2023) Towards safe self-distillation of internet-scale text-to-image diffusion models. [[paper](https://arxiv.org/abs/2307.05977)]
- (ICLR'2024) SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency in Both Image Classification and Generation. [[paper](https://arxiv.org/abs/2310.12508)]
- (ECCV'204) Safeguard Text-to-Image Diffusion Models with Human Feedback Inversion. [[paper](https://arxiv.org/abs/2407.21032)]
- (CVPR'2024) One-dimensional adapter to rule them all: Concepts diffusion models and erasing applications. [[paper](https://arxiv.org/abs/2312.16145)]
- (CVPR'2025) Efficient fine-tuning and concept suppression for pruned diffusion models. [[paper](https://arxiv.org/abs/2412.15341)]
- (ECCV'2024) Receler: Reliable concept erasing of text-to-image diffusion models via lightweight erasers. [[paper](https://arxiv.org/abs/2311.17717)]
- (ECCV'2024) Race: Robust adversarial concept erasure for secure text-to-image diffusion model. [[paper](https://arxiv.org/abs/2405.16341)]
- (NeurIPS'2024) Defensive unlearning with adversarial training for robust concept erasure in diffusion models. [[paper](https://arxiv.org/abs/2405.15234)]
- (NeurIPS'2024) Erasing Undesirable Concepts in Diffusion Models with Adversarial Preservation. [[paper](https://arxiv.org/abs/2410.15618)]
- (ICLR'2025) Fantastic Targets for Concept Erasure in Diffusion Models and Where To Find Them. [[paper](https://arxiv.org/abs/2501.18950)]
- (CVPR'2025) Stereo: A two-stage framework for adversarially robust concept erasing from text-to-image diffusion models. [[paper](https://arxiv.org/abs/2408.16807)]
- (CVPR'2024) Forget-me-not: Learning to forget in text-to-image diffusion models. [[paper](https://arxiv.org/abs/2303.17591)]
- (CVPR'2024) Mace: Mass concept erasure in diffusion models. [[paper](https://arxiv.org/abs/2403.06135)]
- (ICLR'2025) Hiding and Recovering Knowledge in Text-to-Image Diffusion Models via Learnable Prompts. [[paper](https://arxiv.org/abs/2403.12326)]
- (CVPR'2025) Localized concept erasure for text-to-image diffusion models using training-free gated low-rank adaptation. [[paper](https://arxiv.org/abs/2503.12356)]
- (NeurIPS'2023) Selective amnesia: a continual learning approach to forgetting in deep generative models. [[paper](https://arxiv.org/abs/2305.10120)]
- (ICCV'2025) MUNBa: Machine Unlearning via Nash Bargaining. [[paper](https://arxiv.org/abs/2411.15537)]
- (AAAI'2025) DuMo: Dual Encoder Modulation Network for Precise Concept Erasure. [[paper](https://arxiv.org/abs/2501.01125)]
- (ICLR'2025) Controllable Unlearning for Image-to-Image Generative Models via 𝑒𝑝𝑠𝑖𝑙𝑜𝑛-Constrained Optimization. [[paper](https://arxiv.org/abs/2408.01689)]
- (ICLR'2024) Machine Unlearning for Image-to-Image Generative Models. [[paper](https://arxiv.org/abs/2402.00351)]
- (NeurIPS'2024) Direct unlearning optimization for robust and safe text-to-image models. [[paper](https://arxiv.org/abs/2407.21035)]
- (CVPR'2024) Diffusion model alignment using direct preference optimization. [[paper](https://openaccess.thecvf.com/content/CVPR2024/html/Wallace_Diffusion_Model_Alignment_Using_Direct_Preference_Optimization_CVPR_2024_paper.html)]
- (WACV'2024) Unified concept editing in diffusion models. [[paper](https://arxiv.org/abs/2308.14761)]
- (ECCV'2024) Reliable and efficient concept erasure of text-to-image diffusion models. [[paper](https://arxiv.org/abs/2407.12383)]
- (ICML'2024) On Mechanistic Knowledge Localization in Text-to-Image Generative Models. [[paper](https://arxiv.org/abs/2405.01008)]
- (ICLR'2024) Localizing and editing knowledge in text-to-image generative models. [[paper](https://arxiv.org/abs/2310.13730)]
- (ICLR'2025) ConceptPrune: Concept Editing in Diffusion Models via Skilled Neuron Pruning. [[paper](https://arxiv.org/abs/2405.19237)]
- (ICLR'2024) Get What You Want, Not What You Don’t: Image Content Suppression for Text-to-Image Diffusion Models. [[paper](https://arxiv.org/abs/2402.05375)]
- (NAACL'2024) Universal Prompt Optimizer for Safe Text-to-Image Generation. [[paper](https://arxiv.org/abs/2402.10882)]
- (ECCV'2024) Understanding the Impact of Negative Prompts: When and HowDoTheyTake Effect? [[paper](https://link.springer.com/chapter/10.1007/978-3-031-73024-5_12)]
- (CVPR'2023) Safe latent diffusion: Mitigating inappropriate degeneration in diffusion models. [[paper](https://arxiv.org/abs/2211.05105)]
- (NeurIPS'2023) Sega: Instructing text-to-image models using semantic guidance. [[paper](https://arxiv.org/abs/2301.12247)]
- (ECCV'2024) Prompt sliders for fine-grained control, editing and erasing of concepts in diffusion models. [[paper](https://arxiv.org/abs/2409.16535)]
- (CVPR'2025) Detect-and-Guide: Self-regulation of Diffusion Models for Safe Text-to-Image Generation via Guideline Token Optimization. [[paper](https://arxiv.org/abs/2503.15197)]
- (ICLR'2024) Get What You Want, Not What You Don’t: Image Content Suppression for Text-to-Image Diffusion Models. [[paper](https://arxiv.org/abs/2402.05375)]
- (ICLR'2025) SAeUron: Interpretable Concept Unlearning in Diffusion Models with Sparse Autoencoders. [[paper](https://arxiv.org/abs/2501.18052)]
- (CVPR'2025) Precise, fast, and low-cost concept erasure in value space: Orthogonal complement matters. [[paper](https://arxiv.org/abs/2412.06143)]
- (NeurIPS'2023) Concept algebra for (score-based) text-controlled generative models. [[paper](https://arxiv.org/abs/2302.03693)]
- (TAI'2025) FAST: Feature Aware Similarity Thresholding for weak unlearning in black-box generative models. [[paper](https://arxiv.org/abs/2312.14895)]
### Audio

### Multimodal
- (NeurIPS'2024) Single image unlearning: Efficient machine unlearning in multimodal large language models. [[paper](https://arxiv.org/abs/2405.12523)]
- (ACL'2025) MMUnlearner: Reformulating Multimodal Machine Unlearning in the Era of Multimodal Large Language Models. [[paper](https://arxiv.org/abs/2502.11051)]
## Contributing
If you have come across relevant resources, feel free to submit a pull request.
```
- (Journal/Confernce'20XX) **paper_name** [[paper](link)]
```



## Acknowledgements
The design of our README.md is inspired by [Awesome-SSLRec-Papers](https://github.com/HKUDS/Awesome-SSLRec-Papers), thanks to their works!
