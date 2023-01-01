## T5 + GAN + RLHF - Pytorch (wip)
Implementation of RLHF (Reinforcement Learning with Human Feedback) and GAN (Generative Adversarial Network) on top of the T5 architecture.

## References
Our implementation uses code from the following repositories:
- [X Transformers](https://github.com/lucidrains/x-transformers.git) for full-attention transformer

## Todo
- [x] Reward Model - T5 with a scalar head

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

```bibtex
@article{Stiennon2020LearningTS,
    title   = {Learning to summarize from human feedback},
    author  = {Nisan Stiennon and Long Ouyang and Jeff Wu and Daniel M. Ziegler and Ryan J. Lowe and Chelsea Voss and Alec Radford and Dario Amodei and Paul Christiano},
    journal = {ArXiv},
    year    = {2020},
    volume  = {abs/2009.01325}
}
```
```bibtex
@inproceedings{Chowdhery2022PaLMSL,
    title   = {PaLM: Scaling Language Modeling with Pathways},
    author  = {Aakanksha Chowdhery and Sharan Narang and Jacob Devlin and Maarten Bosma and Gaurav Mishra and Adam Roberts and Paul Barham and Hyung Won Chung and Charles Sutton and Sebastian Gehrmann and Parker Schuh and Kensen Shi and Sasha Tsvyashchenko and Joshua Maynez and Abhishek Rao and Parker Barnes and Yi Tay and Noam M. Shazeer and Vinodkumar Prabhakaran and Emily Reif and Nan Du and Benton C. Hutchinson and Reiner Pope and James Bradbury and Jacob Austin and Michael Isard and Guy Gur-Ari and Pengcheng Yin and Toju Duke and Anselm Levskaya and Sanjay Ghemawat and Sunipa Dev and Henryk Michalewski and Xavier Garc{\'i}a and Vedant Misra and Kevin Robinson and Liam Fedus and Denny Zhou and Daphne Ippolito and David Luan and Hyeontaek Lim and Barret Zoph and Alexander Spiridonov and Ryan Sepassi and David Dohan and Shivani Agrawal and Mark Omernick and Andrew M. Dai and Thanumalayan Sankaranarayana Pillai and Marie Pellat and Aitor Lewkowycz and Erica Oliveira Moreira and Rewon Child and Oleksandr Polozov and Katherine Lee and Zongwei Zhou and Xuezhi Wang and Brennan Saeta and Mark Diaz and Orhan Firat and Michele Catasta and Jason Wei and Kathleen S. Meier-Hellstern and Douglas Eck and Jeff Dean and Slav Petrov and Noah Fiedel},
    year    = {2022}
}
```

```bibtex
@techreport{48643,
title	= {Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer},
author	= {Adam Roberts and Colin Raffel and Katherine Lee and Michael Matena and Noam Shazeer and Peter J. Liu and Sharan Narang and Wei Li and Yanqi Zhou},
year	= {2019},
institution	= {Google}
}
```

