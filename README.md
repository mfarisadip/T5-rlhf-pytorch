## T5 + GAN + RLHF - Pytorch (wip)
Implementation of RLHF (Reinforcement Learning with Human Feedback) and GAN (Generative Adversarial Network) on top of the T5 architecture.

## References
Our implementation uses code from the following repositories:
- [X Transformers](https://github.com/lucidrains/x-transformers.git) for full-attention transformer

## Todo
- [ ] Reward Model - T5 with a scalar head

## Citations
```bibtex
@inproceedings{Singhal2022LargeLM,
    title   = {Large Language Models Encode Clinical Knowledge},
    author  = {Karan Singhal and Shekoofeh Azizi and Tao Tu and Said Mahdavi and Jason Lee Kai Wei and Hyung Won Chung and Nathan Scales and Ajay Kumar Tanwani and Heather J. Cole-Lewis and Stephen J. Pfohl and P A Payne and Martin G. Seneviratne and Paul Gamble and Chris Kelly and Nathaneal Scharli and Aakanksha Chowdhery and P. D. Mansfield and Blaise Ag{\"u}era y Arcas and Dale R. Webster and Greg S. Corrado and Y. Matias and Katherine Hui-Ling Chou and Juraj Gottweis and Nenad Toma{\vs}ev and Yun Liu and Alvin Rajkomar and Jo{\"e}lle K. Barral and Christopher Semturs and Alan Karthikesalingam and Vivek Natarajan},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2212.13138}
}
```

```bibtex
@article{ingraham2022illuminating,
  title={Illuminating protein space with a programmable generative model},
  author={Ingraham, John and Baranov, Max and Costello, Zak and Frappier, Vincent and Ismail, Ahmed and Tie, Shan and Wang, Wujie and Xue, Vincent and Obermeyer, Fritz and Beam, Andrew and others},
  journal={bioRxiv},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}
```