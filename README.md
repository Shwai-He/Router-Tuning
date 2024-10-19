## Router-Tuning: A Simple and Effective Approach for Enabling Dynamic-Depth in Transformers

**[Shwai He](https://shwai-he.github.io/), Tao Ge, Guoheng Sun, Bowei Tian, Xiaoyang Wang, [Ang Li](https://www.ang-li.com/), Dong Yu**

## Introduction

## News

- **Oct 2024**: Published preprint on [arXiv](https://arxiv.org/abs/2410.13184) along with the related codebase.

## Quick Start

#### Installation


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