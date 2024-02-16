<div align="center">
<h1>MEL: Efficient Multi-Task Evolutionary Learning for High-Dimensional Feature Selection</h1>

[**Xubin Wang**](https://github.com/wangxb96)<sup>12</sup> · **Haojiong Shangguan**<sup>1</sup> · **Fengyi Huang**<sup>1</sup> · **Shangrui Wu**<sup>1</sup> · [**Weijia Jia**](https://scholar.google.com/citations?user=jtvFB20AAAAJ&hl=zh-CN&oi=ao)<sup>23*</sup>


<sup>1</sup>Hong Kong Baptist University · <sup>2</sup>Beijing Normal University · <sup>3</sup>BNU-HKBU United International College  

<sup>*</sup>corresponding authors

[**Paper**](https://www.wangxubin.site/Paper/MEL_TKDE.pdf) · [**Code**](https://github.com/wangxb96/MEL)

</div>

# Contents 
- [Overview](#Overview)
- [Framework](#Framework)
- [Baselines](#Baselines)
- [Dependencies](#Dependencies)
- [Instructions](#Instructions)
- [Results](#Results)
- [Contact](#Contact)


## Overview
Feature selection is a crucial step in data mining to enhance model performance by reducing data dimensionality. However, the increasing dimensionality of collected data exacerbates the challenge known as the "curse of dimensionality", where computation grows exponentially with the number of dimensions. To tackle this issue, evolutionary computational (EC) approaches have gained popularity due to their simplicity and applicability. Unfortunately, the diverse designs of EC methods result in varying abilities to handle different data, often underutilizing and not sharing information effectively. In this paper, we propose a novel approach called PSO-based Multi-task Evolutionary Learning (MEL) that leverages multi-task learning to address these challenges. By incorporating information sharing between different feature selection tasks, MEL achieves enhanced learning ability and efficiency. We evaluate the effectiveness of MEL through extensive experiments on 22 high-dimensional datasets. Comparing against 24 EC approaches, our method exhibits strong competitiveness.

## Framework
![model](https://github.com/wangxb96/MEL/blob/main/Figures/framework.png)
A schematic diagram of the proposed MEL method. The parent population is divided into two subpopulations: Sub1 learns the feature importance during evolution, and its search is affected by Sub2 best. Sub2 also learns the importance of features during evolution, and searches for the optimal feature subset based on the results learned from Sub1 and Sub2. In particular, features with higher weights have a higher probability of being selected.

## Baselines
![baselines](https://github.com/wangxb96/MEL/blob/main/Figures/metaheuristic.png)
To illustrate the performance of our method, 18 evolutionary computation algorithms were employed. Specifically, we selected four **swarm-based methods** (inspired by mutual behavior of swarm creatures), four **nature-inspired methods** (inspired by natural system), two **evolutionary algorithms** (inspired by natural selection concepts), four **bio-stimulated methods** (inspired by the foraging and hunting behavior in the wild) and four **physics-based methods** (inspired by physical rules). These meta-heuristic optimization methods basically represent the most classical, representative, and widely used methods in the field. 

Additionally, we have incorporated six evolutionary methods for feature selection published in recent four years ([SaWDE](https://github.com/wangxb96/SaWDE), [FWPSO](https://github.com/wangxb96/FWPSO), [VGS-MOEA](https://github.com/BIMK/VGS-MOEA), [MTPSO](https://github.com/SZU-AdvTech-2022/304-Evolutionary-Multitasking-for-Feature-Selection-in-High-Dimensional-Classification-via-Particle-), [PSO-EMT](https://github.com/SZU-AdvTech-2022/271-An-Evolutionary-Multitasking-based-Feature-Selection-Method-For-High-dimensional-Classification) and [DENCA](https://github.com/ehancer06/DENCA). These methods represent the cutting-edge research in the field and offer promising solutions to the challenges at hand. By incorporating these state-of-the-art evolutionary algorithms, we aim to enhance the effectiveness and robustness of our proposed framework.

## Dependencies
- This project was developed with **MATLAB 2019b** and **MATLAB 2023b**. Early versions of MATLAB may have incompatibilities.

## Instructions
- [MEL.m](https://github.com/wangxb96/MEL/blob/main/MEL_Methods/MEL.m) (This is the main file of the proposed model)
  - You can replace your data in the **Problem**. For example:
    - Problem = {'The_name_of_your_own_data'};
  - How to load your own data?
    ```
      traindata = load(['C:\Users\c\Desktop\MEL\data\',p_name]);
      traindata = getfield(traindata, p_name);
      data = traindata;
      feat = data(:,1:end-1); 
      label = data(:,end);
    ```
  - You can set the number of iterations of the whole experiment through **numRun**
  - The parameters of PSO algorithm can be replaced in:
    - opts.k = 3; % number of k in K-nearest neighbor
    - opts.N = 20; % number of solutions
    - opts.T = 100; % maximum number of iterations
    - opts.thres = 0.6; % threshold
      
To reproduce our experiments, you can run **MEL.m** ten times and take the average of the results.

## Results
We evaluate the effectiveness of our method using three metrics: **accuracy**, **feature subset size**, and **algorithm running time**. Extensive experiments on 12 high-dimensional genetic datasets showed that MEL can effectively improve classification accuracy while obtaining a small feature subset. MEL also showed highly competitive running time compared to 18 state-of-the-art meta-heuristic optimization algorithms and five recently published evolutionary feature selection methods. Additionally, we provided further experiments on a separate set of 10 larger sample size datasets, comparing MEL against five representative algorithms. The results demonstrated MEL's superior overall performance in classification metrics, validating its effectiveness on high-dimensional data with both few and many samples.

## Contact
wangxb19 at mails.jlu.edu.cn
