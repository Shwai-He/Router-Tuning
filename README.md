## Router-Tuning: A Simple and Effective Approach for Enabling Dynamic-Depth in Transformers

**[Shwai He](https://shwai-he.github.io/), Tao Ge, Guoheng Sun, [Bowei Tian](https://bowei.netlify.app/#about), Xiaoyang Wang, [Ang Li](https://www.ang-li.com/), Dong Yu**

## Introduction

Traditional transformer models allocate a fixed amount of computational resources to every input token, leading to inefficient and unnecessary computation. To address this inefficiency, [Mixture of Depths (MoD)](https://arxiv.org/abs/2404.02258) was introduced, dynamically adjusting computational depth by skipping less important layers. While promising, current MoD approaches face two significant challenges:

1. **High Training Costs**: Existing methods require training the entire model alongside routers, which determine which layers to skip, resulting in substantial computational overhead.
2. **Risk of Performance Degradation**: Bypassing important layers can lead to a drop in model performance.

To overcome these challenges, we introduce [**Router-Tuning**](https://arxiv.org/abs/2410.13184), a method that fine-tunes only the router on a small dataset, drastically reducing the training costs. Additionally, we propose **Mindskip** (Attention with Dynamic Depths), which preserves model performance while significantly enhancing computational and memory efficiency. 

Our approach delivers competitive results, achieving up to **21% speedup** with only a **0.2% performance drop**, demonstrating its effectiveness in balancing efficiency and performance.


## News

- **Oct 2024**: Published preprint on [arXiv](https://arxiv.org/abs/2410.13184) along with the related codebase.

## Quick Start

#### Installation

```bash
conda create -n router-tuning python=3.10
conda activate router-tuning

git clone https://github.com/CASE-Lab-UMD/Router-Tuning

cd ./Router-Tuning
pip install -e .
```


## Train

```bash
sh /scripts/finetune_mndskip.sh
```

## Evaluation 

The evaluation code is based on [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). To fully reproduce our results, please use [this version](https://github.com/s1ghhh/lm-evaluation-harness). It samples few-shot based on the index of the samples, avoiding the issue of result variation with the number of processes during data parallel inference.

## Citation

```latex
@misc{he2024matters,
      title={What Matters in Transformers? Not All Attention is Needed}, 
      author={Shwai He and Guoheng Sun and Zheyu Shen and Ang Li},
      year={2024},
      eprint={2406.15786},
      archivePrefix={arXiv},
      primaryClass={id='cs.LG' full_name='Machine Learning' is_active=True alt_name=None in_archive='cs' is_general=False description='Papers on all aspects of machine learning research (supervised, unsupervised, reinforcement learning, bandit problems, and so on) including also robustness, explanation, fairness, and methodology. cs.LG is also an appropriate primary category for applications of machine learning methods.'}
}
```


## Contact Us

If you have any questions, please contact:

- Shwai He: shwaihe@umd.edu