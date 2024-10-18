## Router-Tuning: A Simple and Effective Approach for Enabling Dynamic-Depth in Transformers

## Train

```bash
sh /scripts/finetune_mndskip.sh
```

## Evaluation 

The evaluation code is based on [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). To fully reproduce our results, please use [this version](https://github.com/s1ghhh/lm-evaluation-harness). It samples few-shot based on the index of the samples, avoiding the issue of result variation with the number of processes during data parallel inference.
