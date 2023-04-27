2021/7/27更新：原本的中文笔记可以在readme-legacy.md中找到，其主要写于2019-2020年。现在的英文版补充了一些文章，并去掉了极个别暴论。

2021/7/27 Update: The original Chinese notes can be found at readme-legacy.md; they are mainly written in 2019-2020. Current English version adds some papers, and remove several erroneous comments.

# 121 Useful Tips of the Day (updated 2023.3)

1. Vanilla A2C/PPO without reward shaping/prolonged episode/ exploration skills are actually hard to deal with mountain car, as the reward is too sparse.

2. It is important to do state/reward normalization for PPO to maintain numerical stability.  

3. DO NOT put any function that changes global variables in Pycharm's watch! (e.g. a function in Pycharm's watch which adds a global counter by 1 may cause the wrong value of the counter).

4. SCIP (not scipy!) is currently the best open-source optimization solver.

5. Don't use scipy in your research project as an optimization solver; use Gurobi instead. An academic license costs $0, yet Gurobi is ~250x faster than scipy (and also more numerically stable).

6. Normally, a good solver (e.g. Gurobi) will do some numerical tricks for actions that may cause singularity.

7. If you don't know what hyper-parameter to set, go find a previous work and inherit their params. This will help to convince the reviewers that your idea works.

8. Randomly initialized NN has a compressing effect (see Benjamin/Rechat's work), which means its output is probably a contract mapping (with possible shifts) with random inputs. This effect can be used in anomaly detection.

9. When dealing with a temporal sequence, use the first part (e.g. year 1-6 for a 10-year dataset) as the training set, then validation set, finally the test set.

10. Prediction models for time sequence (e.g. electricity/VM demand) usually underestimates, for there are systematic bias (e.g. peaks) in the dataset. On the other hand, underestimating the demands are usually more serious than overestimating in real life.

11. You can get Azure VM information from Kusto.

12. Exploration for RL matters, even for toy environment. The same environment with different default behavior for illegal actions (e.g. stay or randomly moving or giving large negative reward) causes huge performance gap for A2C. As for my own experience, the first two are better choices.

13. L1 loss fits better for data with **sparse** entries, and is more **robust against outliers**.

14. The goal of experimental parts in a paper is not stating "what we've done". It should be organized by "What we're going to validate" (e.g. Why do we design this experiment, and what is the conclusion).
 
15. The MIT book and the Boyd book are the two classical textbooks for convex optimization; strongly recommending the two books.

16. The difference of **\forall** and **for x \in X**: The former emphasizes "satisfaction of conditions", usually used in proofs of advanced mathematics; the latter is an enumeration. They are generally the same, but proper usage helps comprehension for readers.

17. A **sparse embedding** (e.g. holiday tag) with **small training set** is inherently infavorable over two-stage method and favors decision-focused method.

18.	Write papers! Only by writing papers can you be more rigorous in language for papers.

19.	*Constraint* is for decision variables' feasible domain. The relationship between problem parameters should not appear in the constraint part. 

20. tensor([0]) < 0.5 is **False**. Note the **round down of integer types of torch.tensor!**

21. To check the difference of two general distributions (e.g. When you are comparing the performance of two methods), mean and std are not enough. Try percentile and **Maximum Mean Discrepancy**!

22. Add axis label and title for debugging figures, as you may forget what you were plotting.

23. Do periodically save your **code** and model for an actively debugged program; preferably automatically doing so every time you run your code.

24. Add a L1/L2 regularization (e.g. ||f(\theta)-f(\hat theta)||, ||\theta-\hat theta|| where theta is the predicted param) is by essence Lipschitz regularization for target function.

25. Some ways to note current update for your research field:  

1) arxiv subscribing cs.AI cs.LG, plus manually searching the key word *proceedings of ICML、NeurIPS、ICLR、UAI、AISTATS, etc.*

2) reddit.com/r/MachineLearning

26. Put a demo one-line run script for cmd/shell in your project readme. The most common one will do.

27. Do note your notations for theoretical parts, and try your best to make it coherent for each of the theorem / both main paper and appendix.

28. Recurrent DDPG is unreliable and hard to tune. MADDPG/Recurrent MADDPG is even more painful. So do recurrent TD3; try to avoid recurrent policy if you want stable performance.

29. Programming dataset, e.g. GAMS, has a very large number of dimensions for decisions (e.g. >100k).

30. A noise of ~0.05 over a value 1 causes a SNR less than 15db, and by this aspect is not a small noise.

31. If you can tell a good story / establish a good framework, then the experimental part will be much easier as it only serves as a validation. Otherwise, your research will be an empirical one, which requires high demand on performance.

32.	General Multi-Objective problem may seem luring, but it is not trivial: pareto optimal means balance over multiple goals, yet such goals usually depends on the settings of real scenario.

33. "Add noise then discretization(e.g. rounding)" is more close to reality than "discretization then add noise".

34. Sometimes, if the experiment code is not working, you can fix some elements to debug. 

E.g. for off-policy 2-step RL, you can fix the first step and try to train the 2nd step; if the current picture training set is not working, you can pick one picture as the training set to see if it can overfit; if not, the code may be buggy.

However, such practice (the one datapoint method) may face the problem of **not having enough support for optimization surface**, so it is not a panecea.

35. Intuitively, the following situation will put decision-focused method at advantage over 2-stage method:

1) the optimization part, with surrogate, has a differentiable argmax and good generalization,

2) the prediction part has some outlier dimensions which has low weight on optimization quality.

36. If you find an unnecessary condition set in your experiment due to early decisions, If you have no time for re-runs, you can simply explain the condition in the appendix, and give a real-life example if necessary.

37. For a multi-dimensional decision vector in optimization, the influence of single/minority number of dimension may be overwhelmed.

38. 2-stage early stopping has an inherent logic of "doing prediction well first". Thus, it should be early stopping according to **prediction loss** instead of **optimization performance**.

39. Significance tests are usually conducted in traditional statistic works for hypotheses, especially where test set does not exist. 

40. Use **on-policy** methods for MARL, as stationarity is not preserved!  

41. When you are imitating someone else's code but failed, a useful debugging method is to take his code, and changing his code into yours function by function (instead of  changing yours onto his). You can try the differnet versions of code in parallel to quicker iterate.

42. Batchnorm is influenced by eval/train! By default, the running_stats is on. Then for training, the normalization is conducted with batch statistics; but for evaluation, the normalization is conducted with a fixed mean and variance estimate kept with a momentum of 0.1. This could have a VERY BIG influence if you ignore the difference.

43. You can try to feed feature^2 besides feature into MLP to get better expressivity, which works particularly well in fitting near-quadratic functions. 

44. torch implementations such as **logsumexp** are numerically stable, and should be used instead of self-implemented vanilla code.

45. Be patient when you are training a large network. For a classifier, the training loss may be not decreasing in a relatively long period at the beginning of the training (although the output is changing greatly), but the loss will decrease quicker in the later training process. 

46. One technique for serious outliers in a dataset is to clip the loss to a constant, e.g. minimize max(-log(y|x), 0.1); this effectively "rejects" the gradient from the outliers and upper bounds the loss.

47. Note: Pytorch passes address, so if you want to only pass value to a function, make sure that you use clone() function! (e.g. for normalizing flows) 

48. Do not trust "manual design" too much against randomization in deep learning. (e.g. permutations of channels in normalizing flows)

49. Note that torch.KLDivLoss(q.log(), p) = KL(p||q).

50. When you are tuning performance, try keep observing the curve for the first run if possible; this takes a little time, but it helps you to grab a sense of what is happening, and what epoch is the best. Also, try to run your code **through** before starting a long experiment (e.g. set epoch to 1 to see if the model can save correctly).

51. Use the following code to fix your pytorch random seeds, preferably at the beginning of main process:

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) \# when using multiple GPUs
    torch.cuda.manual_seed(seed)     
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True 
    \# torch.use_deterministic_algorithms(True) use with caution; this line of code changes many behavior of program. 
    torch.backends.cudnn.benchmark = False \# CUDNN will try different methods and use an optimal one if this is set to true. This could be harmful if your input size / architecture is changing.
    
Note: once the random seed is set anywhere in this process (regardless of which file it is in), the seed remain fixed (unless implicitly set by other libraries).

52. You should reduce the learning rate if you are using batchnorm. Batchnorm changes the landscape.

53. What is the difference between optimizing KL and reverse KL? Mode-seeking (reverse) and mode-covering (forward)! See https://www.tuananhle.co.uk/notes/reverse-forward-kl.html for a brief explanation.

54. You can use the following code to visualize your RL episode:

In your gym step: 
    img = env.render(mode='rgb_array')
    IMG.append(img)
and at the end of the episode, you write

 imageio.mimsave(name+'.mp4', IMG, fps=25)
 
55. If you need to change distribution in expectation in your derivation, try importance sampling. But as this introduces a possibly instable denominator, you may need surrogates to stabilize the whole thing.  

56. If you are encountering strange problems in python import (e.g. missing .so), there are a few possible things that you can do:

1) check if the import path is wrong. For example, is your python importing a package from /.local/lib/python3.x/site-packages instead of your conda directory? 

2) check both pip and conda envrionment, especially where there is a version mismatch between python xxx.\_\_version\_\_ and pip list / conda list. https://www.anaconda.com/blog/using-pip-in-a-conda-environment Actually, you may want to avoid using pip and conda together.
    
3) check your python, linux, and some system library version (e.g. MPI) under different settings. Sometimes the problem comes from version mismatch.

4) DO NOT try to copy-paste a file from another version into this version and simply change its name for a quick fix, unless you absolutely know what you are doing. While sometimes it can fix the problem, this creates an environment that is hard to reproduce and would be questioned by future readers of your papers.

57. There are many people write ML code, but many of them do not write the code well. Make sure you have seen others' code and wisely referred to them for your own code; you should not easily believe in one single reference, even if it has many stars.

58. Whenever you see a exp() function, ask yourself: can you substitute it with a log?

59. When using remote Python debugger with pycharm professional, you can close "attach to subprocess" option if you witness strange bugs. (At settings -> build, execution, deployment -> Python debugger)

60. When developing a new architecture for deep learning, you cannot simply considering throwing a dart randomly and hoping it can work. You should deviate from original design gradually, that is, stand on the shoulders of the giants.

61. You should never use gradient clipping along with weight decay. The gradient clipping take effect **Before** weight decay, thus greatly amplifying the weight decay factor and cause the training to be weird.

62. You should not copy models by iterating through parameters(), as parameters() does not contain parameters from the batchnorm layers. Use copy.deepcopy() or load_state_dict() instead.

63. If you are confronting strange problems in installing a package with "pip install -e .", take a look at the setup.py, especially if you cannot find any version. Sometimes, authors will use "git+git@xxxx" to fetch dependency posts. you should change it to git+https://xxxx ... as you are not collaborator / author of the repo.

64. If you cannot run "git submodule update --init --recursive", you should check .gitmodule file to see if there is problem, especially that mentioned in 63. After that, "run git submodule sync" and the problem should be fixed.

65. Even the smallest library/package version change can cause a big difference in a particular environment. For example, mpi4py 3.1.1 and 3.1.3, though seemingly no big difference in the update log, can decide whether a program is runnable or not.

66. Different version of GPU (e.g. Nvidia A6000 and RTX 2080Ti) and different computation platform (e.g. GPU and CPU) could lead to non-egligible value difference when doing matrix multiplication! See https://forums.developer.nvidia.com/t/cpu-and-gpu-floating-point-calculations-results-are-different/18175 for details.

67. If we decrease the number of steps in the diffusion model, for each sampled diffusion timestep t, on average, the product of \alpha, which is \bar{\alpha} will increase as there are less terms less than 1. As we are fitting epsilon, this leads to lower signal-noise-ratio for epsilon and higher MSEloss. Therefore, fewer number of steps requires higher beta. (https://arxiv.org/pdf/2006.11239.pdf)

68. Remember to save your powerpoint every time you made a slide, and add a timestamp to your experiment results.

69. You can use the following code to output all attributes in args from argparse:

for arg in vars(args): f.write(str(arg)+" "+str(getattr(args, arg))+"\n")

70. Use the same color for the same method throughout the paper of your work, and same notation for the same thing as well. 

71. Keep the citations nice, neat and simple for your papers. Just keep the platform of publication (e.g. ICML, Nature), paper name, year and author on it and don't put the pages / publishers etc.

72. Use grammarly to check the writing of your paper, but do not overly rely on it. It may not recognize terms in your field and make mistakes.

73. You can use a plain notation in Python for elementwise multiplication (Hadamard product), but need to state elementwise clearly when writing papers.

74. Parenthesis around sum symbols should be at least as large as the sum symbols.

75. Beware of any presence of your identification in the code of paper, including absolute path and platform username (e.g. for wandb)!

76. Figures and its captions should be self-contained, especially in appendix where the space is unlimited; put settings and brief conclusion there.

77. The most important hyperparam for PPO is #update epoch and update interval (# env steps).

78. Boil down your slides (in mind); nobody will parse a slide that is full of text.

79. BC is a good baseline if a complete trajectory presents, and if the initial position is of small variance. On discrete MDP, the best BC result is simply counting transitions and do random action if current state is never witnessed.

80. Continuing from 79, you should be pessimistic in offline RL / IL, so that you policy does not astray from what you have witnessed.

81. Wasserstein distance is a much "weaker" distance than f-divergences (e.g. KL divergence), which means in many scenario, the f-divergence method will give either a infinite value / being invalid, or being uncontinuous, or losing a gradient. Intuitively, this is because Wasserstein distance represents a much "weaker" norm in topology (see WGAN paper for details). Wasserstein-1 distance is also called earth mover's distance.

82. If you have a score model which estimates the gradient of log probability of some unknown distribution, you can sample from the distribution using **Langevin dynamics**. This is score matching method.

83. The notion of spaces:

A normed space is a special metric space, which means elements have the notion of "large / small" by norm. A complete normed space is called a **Banach space**; by "complete" it means the limit of a Cauchy sequence is still in the space (counterexample: rational number Q)

A Euclidean space is a finite-dimenisional linear space with an inner product. A **Hilbert space** is an expansion of Euclidean space; it means a complete inner product space, but can be infinite dimensional and not confined to real numbers.

84. By Mercer theorem, any semi-positive definite function can be a kernel function.

85. You should not put plotting plt and ax inside an object and make it a property of other object, especially one that is not a singleton (e.g. solver class for a ML solution). Remember that plt settings are global; object duplication will ruin your plotting.

86. Be wary of the subtle constraints on Lagrange multipliers when you try to derive the dual problem (without them the optimal value could be unbounded; e.g. when you derive dual for linear programming). You should be extra careful when you only apply Lagrange on part of the constraints; when the other part of constraints is a bounded closed set, the problem might be much harder to discover. 

87. Be very careful when you try to explain something with a toy example but change cases (e.g. when talking about something for continuous space, use discrete space as an example). 

88. In a rebuttal, write in a way that is considerate for the reviewers:

1) Answer the question clearly with a few words at the beginning of the problem;

2) Do not show them that you are lazy typing, but you are helping them (e.g. for brevity -> for readability);

3) If possible, do not force the reviewer to get back to the paper. List the points briefly besides reference to the paper. Similarly, avoid reference to the other reviewers;

4) Do not simply write "We will change this", but show them "how we will change this" (and in the case where you can update pdf, do it) and invite them for advice on further modification;

5) Reply after everything is ready, but immediately beyond that point.

89. Do not assume that the battle is over until the authors are not expected to say anything (e.g. reviewer-metareviewer discussion). For NeurIPS, finishing rebuttal period is only half-way there; there are still much work to do at author-reviewer discussion period.

90. For L-BFGS, increasing the history hyperparam will increase the stability of iteration. Also, you should use line search with 'strong wolfe' in pytorch.

L-BFGS needs optimizer.step(closure()) where closure() gives the loss function. It might be invoked multiple times in one timestep, sometimes with gradient and sometimes without. That's why you will sometimes get backward second time error if you do not put everything in the closure() function. Here are two examples: https://gist.github.com/tuelwer/0b52817e9b6251d940fd8e2921ec5e20#file-pytorch-lbfgs-example-py-L27; http://sagecal.sourceforge.net/pytorch/index.html.

91. Be very careful when you try to generate dataset with some tricks (e.g. manipulate the distribution so that the states are guaranteed to be covered) and handling the "default value" for corner case. They might lead to very counter-intuitive behavior if not considered properly.

92. Gurobi max (gp.max_) operator can only take constant and variable (that is, no expressions such as x+1) as of Sept. 2022.

93.  torch.nn.parameter.Parameter(a.double(), requires_grad=True) is correct; torch.nn.parameter.Parameter(a, requires_grad=True).double() is not.

94. Wasserstein-1 distance with hamming distance metric is total variation distance.

95. If you are doing stochastic gradient descent by sampling some pairs of variables (e.g. uniformly sampling (i,j) for x_i+x_j), you'd better sample each pair independently, instead of sampling two state uniformly and then select all pairs of chosen states. In the latter case, you cannot break the correlation between (i,j) and (i,*), (i,j) and (j, *), as they are always updated together.

96. While there is a rule of thumb that choices the learning rate, it really depends on your scale of the loss and batch size. Be open to rare learning rates when you finetune your algorithm. 

97. Remember to "git add" your new file when you are using git to do version control, especially writing script to auto-commit things. otherwise, you may find that your modifications are all untracked.

98. You can use gym.spaces.MultiBinary(n=10) for one-hot observation space.

99. torch.multinomial and torch.Categorical supports batch sampling, which means you only need to get a input of batchsize * n tensor and it will sample batchsize different groups of samples for you. You don't have to go over the whole array! And use F.one_hot(m, num_classes=n) if neccessary.

100. Some points on making slides: [Under Construction]

1) Text size should be unified across all slides and inside figure. Don't be lazy and just using existing figures; draw a nice figure that expands as your presentation progresses. And change text instead of font size to fit in spaces.

2) The shorter the presentation is, the more rigorous logic your slides must have because you don't have too much time to do "overviews" to remind people what you are discussing.  

3) You can align the elements on your slides by choosing items and selecting "align". 

4) You can make changing text w.r.t. time by animation with time delay. No need to make a video using video edit tools.

5) Use animations to avoid making your slides to be overwhelming. Let the slide be filled gradually as your speak progresses.

6) Always introduce math symbols first before using them, even the most common-sense ones in the subfield (e.g. state s in reinforcement learning). You should use as less symbols as possible in a short presentation.

7) Colors and shapes matter; they can be a strong indicator in the figure. Ask yourself why use this color for this color / this shape?  

101. self-made dataloader based on torch.randperm could be much faster than torch dataloader, especially if the data is stored in dict for each dataset. torch dataloader need to concatenate them every time and that can be very slow.

102. If you are trying to overfit behavior cloning on a small dataset to debug, remember to add variance lower bound (e.g. clip / tanh) to avoid spikes.

103. If you are training an action distribution on a closed set (e.g. in behavior cloning in gym environment), and you are using Gaussian / GMM / normalizing flow. One thing you could try to optimize log probability a lot is to use tanh to converge your output into a bounded one. And probability tractable will still be tractable.

104. Wasserstein distance in the Rubinstein-Kantorovich form assumes the underlying metric to be Euclidean, unless the definition of 1-Lipschitz is modified.

105. The sample complexity of Wasserstein distance is bad, but for MMD it is good. Sinkhorn Divergence stands between them, and have a corresponding sample complexity. They are all called integral probability methods.

106. Do not undo commit in github desktop unless you are absolutely certain! Undoing commit makes you lose all the progresses during this commit. 

107. You need to use clf() instead of cla() to remove the old colorbar in your last figure in matplotlib. However, after that you need ax = fig.add_subplot() to re-insert subfigures in order to draw anything more on the canvas. 

108. If you feel lost about why your method is not working while the baseline is, a way out is to implement your method inside the codebase of the baseline. In that way, you can make your method to be as similar to the baseline as possible, and to rule out the factors that does not matter one by one.

109. Be bold and aggressive when you first try to tune your algorithm; often it takes longer than expected to train / bolder choice of hyperparameter than your expecation to make your algorithm work.

110. Do read the experiment details of your baselines, and make sure of how they set up their experiment, especially what do they do to their dataset (e.g. merging). You do not want to waste time on settings that is unnecessarily harder / easier than prior work.

111. When you don't know where is the problem of your algorithm, go and check if your dataset has problems.

112. If you are working optimizations of f-divergences on a probability simplex, consider Fenchel conjugate; consider Donsker-Varadhan representation and https://people.lids.mit.edu/yp/homepage/data/LN_fdiv.pdf Thm 7.14. 

113. Continuing from 112: when considering relaxing optimization (e.g. use Lagrange multiplier to relax some constraints), relax as less constraint as possible (as long as you can solve it). Relax to probability simplex is better than relax to positivity constraint.

114. remember to set CUDA_LAUNCH_BLOCKING=1 whenever you meet a device-side assert triggered error.

115. If you don't know what parameter to tune, try to do the following two things:
1) check very closely on your direct baseline to see how they solve the problem;
2) retry factors excluded before last bug fix. Sometimes bug fixes will make factors behave very differently and you may overlook some crucial factors.

116. For RL evaluation, you should try to use deterministic action (mean as output) as stochastic ones are often with fairly high variance and cannot do well, especially in those environments requiring accurate actions.

117. If you need to send your computer to repair, make sure you have copied everything you need out of it. **Especially the private keys for the server.**

118. If you need to copy datasets to different folders on your server, consider soft links; this saves your disk space and frees you from copying everytime you change your dataset. 

119. If you were to build up a desktop, remember that do not throw the boxes until you have lighten up the machine. There might be some important information or some material (e.g. screws, cables) in the boxes.

120. When building up your desktop, remember to observe the minimal principle: use only as least as possible components to light up your mainboard first. Do not haste to install the extra memory / disk / graphics card.

121. Make sure to check the debugging light and code on your mainboard to figure out the problem.

122. When swapping an element in an array and its index in python, be very careful: a[a[0]], a[0] = a[0], a[a[0]] might not behave the expected way. A better choice is to use a, b = copy.deepcopy(b), copy.deepcopy(a), or use the tmp variable. 

# Useful Linux Debugging Commands

Checking CPU/cache config: lscpu

Checking GPU status (with nvidia): nvidia-smi

Checking memory usage: htop

Checking the disk space taken up by subfolders in the directory: cd some_directory; du -h --max-depth=1  

Checking free space: df -h

Checking the location of an installed package: import xxx; print(xxx.\_\_file\_\_)

Creating conda environment from old ones: conda create -n new_env --clone old_env

Removing conda environment: conda env remove -n old_env

Removing file from path that is older than a specific time: find model -type f -not -newermt "2022-12-03 11:11:11" -delete (if remove delete then show file)

* *The Linux Command Line*, William Shouts*

[Under Construction]

# Tensor Decomposition / Neural Network Compression

See "tensor.md" in the repo.

# Imitation Learning with Observation / Visual Imitation

See "IfO-VI.md" in the repo.

# Robotics 

See "robotics.md" in the repo. 

# Semi-Supervised Learning

* *Not All Unlabeled Data are Equal: Learning to Weight Data in Semi-supervised Learning* (2020) [TBD]

* *Unsupervised Data Augmentation for Consistency Training* (2019) [TBD]

* *Fixmatch: Simplifying semi-supervised learning with consistency and confidence*

# RL and LLM

[Under Construction]

## Text-based Games

## RL for LLM

## LLM for RL

# Causal Inference

Thanks the causal reading group @ MSRA for their valuable opinions on causal inference! For a more thorough and professional summary, see *https://github.com/fulifeng/Causal_Reading_Group*.

Causality-related work includes Judea Pearl's *causal graphical model*, physics-based *structural causal model* (SCM), and statistical *potential outcome framework*.

Below is a (rather random) collection of causal inference paper, as I am a beginner and have not read them all to classify them. For beginners, read *https://www.bradyneal.com/causal-inference-course*. For more papers, you can also read Prof. David Sontag's work. 

* *Understanding Simpson’s Paradox* 

(Copied from wikipedia) Simpson's paradox, which also goes by several other names, is a phenomenon in probability and statistics in which a trend appears in several groups of data but disappears or reverses when the groups are combined.

The paradox can be resolved when confounding variables and causal relations are appropriately addressed in the statistical modeling. Simpson's paradox has been used to illustrate the kind of misleading results that the misuse of statistics can generate.

This paper (tutorial?) by Judea Pearl consists of two parts; one is the history and introduction of Simpson's paradox, 

the other is the "solution" of the paradox, which includes three parts: 1) why it is suprising, 2) when will it happen and 3) how to get the correct result. Rigorously solving the third point requires *do-calculus*, which is an important notion in today's causal inference work.

* *https://amlab.science.uva.nl/meetings/causality-reading-club/*

The discussion group of Prof. Joris Mooij's.

* *https://causalai.net/r60.pdf* 

This paper introduces the three layers of Pearl Causal Hierarchy(association, intervention, counterfact) with both the perspective of logical-probabilistic and inferential-graphical.


More paper can be seen in causal.md in this repo.

Some miscellanous remarks:

1. Skeleton learning is a subfield of Bayesian network sturcture learning.

2. From the perspective of causality, identifying an edge is an output, while ruling out an edge is only a progress.

# Experimental Papers

https://github.com/clvrai/awesome-rl-envs has many RL testbeds.

* *An Analysis of Frame-skipping in Reinforcement Learning* (2021)

* *TRANSIENT NON-STATIONARITY AND GENERALISATION IN DEEP REINFORCEMENT LEARNING* (ICLR 21')

This paper proposes alternative hypothesis besides classic catastrophic forgetting: NN exhibit a memory effect in their learned representations which
can harm generalisation permanently if the data-distribution changed over the course of training (catastrophic memory? This is also mentioned by another paper "Understanding Catastrophic Forgetting and Remembering in Continual Learning with Optimal Relevance Mapping").

By the result of  this paper, it seems that the non-stationarity (use the same network for changing distribution as data stream flows) which is present in many deep RL algorithms might lead to impaired generalisation on held-out test environments. To mitigate this and improve generalisation to previously unseen states, the author proposes Iterated Relearning (ITER), which periodically distills old teacher network into new student network by imposing KL divergence on actor, MSEloss on critic, and policy gradient / TD loss for updating the actor & critic.

* *Learning Dynamics and Generalization in Reinforcement Learning* (ICML 22')
(Quotes from the paper)

NN trained with TD algorithms on dense reward tasks exhibit weaker generalization between states than randomly initialized networks and networks trained with policy gradient methods. On the good side, it is better for the agents' stability; on the bad side, it runs the risk of observational overfitting.

Non-smooth components of a predicted value function, while contributing smaller to MC error, contribute disproportionately to the TD error, providing in incentive to fit them early in training. 

Post-training distillation can improve robustness and generalization.

* *Rethinking the Implementation Tricks and Monotonicity Constraint in Cooperative Multi-Agent Reinforcement Learning* (ICML 22')

It is surprising that, among so many methods proposed, QMIX is still the most tried-and-true algorithm. The monotonicity matters as indicated by this paper.

* *Cliff Diving: Exploring Reward Surfaces in Reinforcement Learning Environments* (ICML 22') 

This paper investigates the reward landscape of RL agent on some classic gym environments and confirmed that there exists some "cliffs" in the reward landscape w.r.t. existing randomized-direction-gradient plotting methods (see another landscape paper in this collection). This paper also confirms that such cliffs have a negative impact on agent performance, which supports PPO/TRPO's motivations.

* *When should agents explore* (ICLR 22')

This paper conducts an extensive research into the exploration of the agents, including exploration in step (epsilon-greedy), intra-episode, episode (warmup) and experiment-level (intrinsic reward). It also investigates the best way to switch between exploration and exploitation mode, e.g., **value-promised discrepancy**, which is the difference between value k-steps ago and the discounted value for now plus actual discounted return (i.e. discounted bootstrapped return) during the k-steps. 

The paper is mainly based on Atari environment. The conclusion is twofold:

1) Intra-episodic exploration is a promising direction for future exploration policy.

2) the huge design space of the proposed family of methods cannot be
reduced to simplistic metrics, such as pX.Jointly using two bandits across factored dimensions is very adaptive, but can sometimes be harmful
when they decrease the signal-to-noise ratio in each other’s learning signal. Also, the choice of the uncertainty-based trigger should be informed by the switching modes.

* *Visualizing the Loss Landscape of Neural Nets* 

An illustration of how the art of network architecture refining matters. [TBD]

* *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour* This paper gives some interesting ideas on tuning the neural network training.

1) **Linear Scaling Rule**: When the minibatch size is multiplied by k, multiply the learning rate by k.

2) For large minibatches, the linear scaling rule breaks down when the network is changing rapidly, which commonly occurs in early stages of training. **This issue can be alleviated by a properly designed warmup**, namely, a strategy of using less aggressive learning rates at the start of training. (See paper "Deep residual learning
for image recognition".)

3) One thing worth noting for minibatch normalization is that changes in minibatch size change the underlying definition of the loss function being optimized. Therefore, different worker threads should not share the same normalization parameters. However, you should normalize the per-worker loss by total minibatch size kn, not per-worker size n.

4) Weight decay is actually the outcome of the gradient of an L2-regularization term in the loss function.  Scaling the cross-entropy loss is **not** equivalent to scaling the learning rate.

5) Use a single random shuffling of the training data (per epoch) that is divided amongst all k workers.

* *Neural MMO: A Massively Multiagent Game Environment for Training and Evaluating Intelligent Agents* (2019) A testbed for massive number of agents for MARL.

* *Continual World: A Robotic Benchmark For Continual Reinforcement Learning* (NeurIPS 21')

Continual learning (CL) — the ability to continuously learn, building on previously acquired knowledge — is a natural requirement for long-lived autonomous
reinforcement learning (RL) agents. 

* *Breaking the Deadly Triad with a Target Network* (ICML 21')

* *Deep Reinforcement Learning and the Deadly Triad* (2018)

**Deadly Triad** is a phenomenon in deep reinforcement learning where RL algorithms could diverge given the following three factors:

1) bootstrapping (using any sort of TD-learning, e.g. TD(0));

2) function approximator;

3) off-policy.

The problem that leads to the deadly triad is: 

a) With a function approximator, **V(s) or Q(s,a) from adjacent states become related**;

b) With bootstrapping, the function is sometimes trained with a wrongly estimated target (if there is no bootstrapping, then every update in expectation leads to ground truth)

c) With off-policy, the function is not update in correspondence to the visiting probability, thus adding **a false weight** on the Bellman equation, breaching the contract mapping property to converge; and since there is a function approximator, we cannot let each state to converge at a separate speed.


This paper studies the convergence behavior of the Q-value, and concluded that:

1) action value can commonly exhibit exponential initial growth, yet still subsequently recover to normal.

2) The instability in standard epsilon-greedy scheme can be reduced by bootstrapping on a separate network & reducing overestimation bias.

(Note: See "conservative Q-learning" for a recent(2020) solution; the overestimation bias is caused by the max operator. hint: E(max(x))>=max(E(x)))

3) Longer bootstrap length reduces the prevalence of instabilities.

4) Unrealisitic value estimation (positively) correlates with poor performance. For those cases where agents have wrong estimation yet good performance, the ranking of the values are roughly preserved.

* *Deep Reinforcement Learning that Matters* (2017)

This paper finishes many experiments yet concluded with a pessimism summary:

1) Hyperparameters are very important;

2) Variance across trials are large;

3) Even different realization of the same algorithm may vary greatly on a given environment.

* *Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO* (2020)

It is interesting that PPO without optimizations actually has poor performance and no edge over TRPO. Some important notes:

1) Common optimization of PPO includes value function clipping, reward scaling, orthogonal initialization & layer scaling, Adam LR annealing, observation normalization & clipping, tanh activation and gradient clipping.

2) It is difficult to attribute success to different aspects of policy gradient methods without careful analysis.

3) PPO and TRPO with code optimization can maintain a certain average KL-divergence (thus remain in the "trust region"), but PPO without code optimization may not.


* *Are Deep Policy Gradient Algorithms Truly Policy Gradient Algorithms?* (2018)

This paper questions DPG algorithms from principle and calls for a multi-perspective remark for RL algorithms (e.g. the ability to capture the structure of real-life problems).

The paper suggests that the normal sampling size, the gradient and the landscape of the value function are actually far different from ground truth, and the gradient varies greatly among multiple runs; to eliminate the difference would require ~1000x more samples.


* *Diagnosing Bottlenecks in Deep Q-learning Algorithms* (ICLR 19') 

A paper that analyses deep Q-learning in practice. The paper uses three variant of DQN which are inspiring: 

1) Exact FQI, which has no sampling error due to a complete traverse in the (s,a) pair tree.

2) Sampled FQI, where the Bellman error is approximated with MC.

3) Replay FQI, which uses a replay buffer.

Some conclusions of this paper:
 
1) Smaller architectures introduce significant bias in the learning process. This gap may be due to the fact that when the target is bootstrapped, we must be able to represent all Q-function along the path to the solution, and not just the final result. 

2) Higher sample count leads to improved learning speed and a better final solution, confirming our hypothesis that overfitting has a significant effect on the performance of Q-learning. 

3) Replay buffers and early stopping can be used to mitigate the effects of overfitting. nonstationarities in both distributions and target values, when isolated, do not cause significant stability issues. Instead, other factors such as sampling error and function approximation appear to have more significant effects on performance. 

The paper proposes a better sampling method：Adversarial Feature Matching.

* *Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning*

A benchmark for meta-RL and multi-task RL. There are 50 tasks of manipulating the robotic arms, such as opening the door or manipulating a ball; in meta-training scenario, 45 tasks are used for training and 5 for evaluation.

* *ACCELERATED METHODS FOR DEEP REINFORCEMENT LEARNING*

This paper proposes a framework for fast RL training with highly parallel sampling and huge batch sizes. In this framework, multiple simulators are run per core to mitigate synchronization losses and hides NN inference time;
I think this could be particularly useful for companies using RL such as Google, Deepmind, etc., but not so useful for researchers.

# Domain Transfer for RL

Domain Transfer has close relations with Sim2real, a subfield that is very useful for robotics. 

## Embodiment Difference

e.g. transfer human demonstration & skills to robot, or between robots of different kinetics.

* *Translating Robot Skills: Learning Unsupervised Skill Correspondences Across Robots* (ICML 22')

The authors use GMM to explicitly get the log probability of the skill distribution in the latent space (so why not normalizing flow)? They try to match the target distribution and the transformed source distribution by the encoder, by sampling one distribution (transformed source / target) and estimate logprob on the other distribution (target / transformed source) and then optimize both log likelihood. This somehow makes me think of something like JS divergence.

Current domain transfer in robotics & graphics methods are: paired data (need label for pairing!), morphological transfer by modularity in policy, state-based, motion retargetting, and unsupervised action correspondence.

* *REvolveR: Continuous Evolutionary Models for Robot-to-robot Policy Transfer* (ICML 22')

Continuous evolutionary models for robotic policy transfer!

Common kinematic tree. The matching is simply a weighted sum of kinetic parameters from both robots, and implemented by editing the robots' URDF / MJCF files in Mujoco. The weight parameter is the "robot evolution parameter".

The idea is as follows: you start from the original robot, where the parameter is 0 for source robot and 1 for target robot. Then, you randomly add some value (non-negative) to current evolutionary parameter, and try to optimize with the corresponding robot. You sample & train multiple robots with different "evolutionary progress", and then step forward the lower bound of "evolution" a little, until you reach "evolutionary progress" of 1, which is exactly the target robot.

I think this can be called some sort of domain randomization + curriculum learning; it is not so "evolutionary". To be "evolutionary", you should be selecting the best progress with the highest reward to step into, instead of using a fixed evolutionary progress step length.

One interesting thing is that it applies bigger reward for robots with larger evolution progress.

 * *Translating robot skills learning unsupervised skill correspondences across robots* (ICML 22')

## Domain Randomization

## System Identification



## Observation Adaptation

## Others

* *MAXIMUM A POSTERIORI POLICY OPTIMISATION* (MPO) (ICLR 18')

MPO assumes that there is a variable for "the event of success", and then use ELBO to get a objective quite similar to SAC: reward plus temperature parameter times some f-divergences. However, SAC uses entropy, while MPO uses "auxiliary distribution over trajectories" which turns the algorithm into the optimization of expecation of modified reward from the auxiliary distribution. Then a closed-form distribution w.r.t. a different set of Q function is assumed for q, which turns the problem into an optimization of the separate Q function and E-M optimization of the policy (essentially, maximimization of log likelihood with a KL regularization term with the auxiliarty distribution Q.)

* *V-MPO: ON-POLICY MAXIMUM A POSTERIORI POLICY OPTIMIZATION FOR DISCRETE AND CONTINUOUS CONTROL* (ICLR 20')

Introduce MPO to the on-policy setting, and relies on a learned state-value function instread of the state-action Q.

* *CUP: Critic-Guided Policy Reuse* (NeurIPS 22') TBD

* *An Analysis of Frame-skipping in Reinforcement Learning* (TBD)

Theoretical analysis of frame-skipping!

* *Off-Dynamics Reinforcement Learning: Training for Transfer with Domain Classifiers*

This paper tries to do domain transfer with a variational perspective. Intuitively, if we want the RL policy from original domain works well on target domain, then the "distribution of trajectories" should be similar. If we take the assumption of maximum entropy IRL (MEIRL; the probability is proportional to the exponentiated reward) and assume that the initial distribution is the same (which could be a fairly strong assumption actually), then we can factorize the probability of trajectories generated by the same policy on both domains. Minimizing this KL-divergence with MEIRL assumption brings us a modified RL objective. The author proves that the expected reward in the source and target domains differs by an upper bound.

* *Deepmdp: Learning continuous latent space models for representation learning* (ICML 19') 

This paper proposes to learn a representation of states that minimizes 1) the value of reward before and after mapping (L1 loss); 2) the transition probability between transition after mapping and mapping after transition under **some metrics**, which is carefully discussed in the paper. This paper theoreticaly analyzes many different losses, such as Wasserstein-1 / bisimulation, TV and MMD.

**Pinsker's inequality bounds total variation with the KL divergence.**

* *Scalable Methods for Computing State Similarity in Deterministic Markov Decision Processes* (AAAI 20')

This paper defines a equivalence relation ("bisimulation") between states where two states are equal iff: 1) for any action at the two states, the (instant) reward is the same; 2) if we consider all states to be partitioned into different classes, the transition probability from this two states to any class for any given action is the same. However, this equivalence is brittle, so we define a pseudo-metric d to quantify the difference between states. a **bisimulation metric** d is a metric between states which induces a equivalence relation between states s and t where d(s,t)=0 means equivalence.  

For any metric d, and state pair (s,t), we have a surrogate f(d)(s, t) =  max_a (|R(s, a) - R(t, a)| + gamma * W_1(d)(P(s, a), P(t, a))) where W_1 is the 1-Wasserstein distance (i.e. earth-moving distance on a simplex). This metric has a unique fixed point which is a bisimulation metric, so by iteratively applying F on a metric, we can get a bisimulation metric.

This paper's contribution is "scalable", which means it invents a method for on-policy **bisimulation**. And one disadvantage of bisimulation is that "actions with the same label may induce very different behaviors from different states, resulting in an improper behavioral comparison when using bisimulation." In order to solve this, the author relates bisimulation to a particular policy \pi, and define on-policy bisimulation relative to this particular policy. We can similarly define a operator F for iteration, and get the conclusion that for any pair of states, the difference between their V(alue) is bounded by the pi-bisimulation metric. This metric can be computed with sampled transitions.

The metric is evaluated on gridworld and Atari games.

* *Contrastive Behavioral Similarity Embeddings for Generalization in Reinforcement Learning* (ICLR 21')

The following work of the last paper.

This paper proposes a novel policy similarity metric (PSM) and uses SIMCLR (a contrastive learning method) to learn an embedding of policy difference for better generalizability. Instead of using reward as the first term of iteration operator, this paper uses the distance between action probability distributions. The theoretical bound thus is not related to value function, but to the distribution of trajectories. 

The embedding is trained by sampling pairs of training MDPs and updates a neural network to calculate metrics (?)

One important thing is that the paper mentioned that metric based on the reward is usually either too restrict (when the policies are the same but the obtained rewards are not) or permissive (when the policies are different but the rewards are the same) and has bad generalizability.  

* *Behavioral Priors and Dynamics Models: Improving Performance and Domain Transfer in Offline RL* (MABE)

1. MOPO: MOPO [21] is an uncertainty-based offline MBRL algorithm. MOPO uses MBPO
[31], an off-policy Dyna-style RL algorithm where a replay buffer is populated with synthetic
samples from a learned dynamics model and used to train an Soft Actor Critic (SAC) [38]
agent. MOPO build on MBPO by penalizing the reward experienced by an agent with a
penalty proportional to the prediction uncertainty of the dynamics model. MABE is also
built on top of MBPO and thus MOPO is the most directly competing baseline.

2. MOReL: MOReL [20] is also an uncertainty-based offline MBRL algorithm. The primary
difference between MOReL and MOPO is that MOReL uses an on-policy algorithm, TRPO
[64], as its backbone. Otherwise, MOPO and MOReL are similar - both penalize the reward
with a term proportional to the forward model uncertainty. The performance differences
between MOPO and MOReL on D4RL are mainly due to the performance of the backbone
algorithm, SAC and TRPO respectively. SAC outperforms TRPO on the mujoco Cheetah environment while TRPO outperforms TRPO in the Hopper environment, and these
differences are also evident in the offline RL results for MOPO and MOReL.

3. CQL: Conservative Q-Learning (CQL) [17] is a leading offline model-free baselines. CQL
learns Q-functions so that the expected value of a policy under the learned Q-function is a
lower-bound of the true policy value. CQL modifies the standard Bellman error with a term
that minimizes the Q-function under the policy distribution while maximizing it under the
offline data distribution. CQL does not leverage behavioral priors.

4. BRAC-v: BRAC-v is another leading model-free RL algorithm that utilizes behavioral
priors to learn a conservative policy. BRAC-v is the model-free algorithm most similar to
MABE. Like MABE, BRAC-v learns a behavioral prior by fitting a Gaussian distribution to
the offline data and regularizing a Gaussian evaluation policy with respect to the behavioral
data. Unlike MABE, BRAC-v does not weigh the behavioral prior with the advantage and
instead treats all data points equally regardless of the reward achieved.


* *Offline Reinforcement Learning with Pseudometric Learning* (ICML 21')

This works borrows the idea of pseudomeric to define a new lookup based bonus, which encourages the actor to stay close in terms of the defined pseudometric. Different from the online scalable paper above, this paper defines the metric to be between state-action pairs, and calculates the distance between current agent's (s,a) pairs to that in the offline dataset. The pseudometric is learned by a neural network. 

# Improving RL Efficiency

* *DARLA: Improving Zero-shot Transfer in Reinforcement Learning*

This paper tries to learn a RL policy with visual inputs that can be transferred onto another environment with different state space but the same action space with structurally similar transition and reward functions. To do this, the author assumes a "latent" (hidden) MDP where states from both domains roots from its state space, and tries to learn a beta-VAE that can recover the states on this state space. The author uses beta-VAE to do this, claiming that beta>1 can limit the expressivity of the encoder and forces better grasp of the essence of MDP. Instead of using normal MSE reconstruction loss, to grasp global semantics, the author uses another pretrained DAE and tries to compare the reconstruction loss in the DAE-abstracted latent space.

* *Towards Applicable Reinforcement Learning: Improving the Generalization and Sample Effciency with Policy Ensemble* (IJCAI 22')

This idea of ensemble is a bit popular recently it seems. Another paper in NeurIPS 22' also uses ensemble for policy prior https://arxiv.org/abs/2209.15205.

* *CCLF: A Contrastive-Curiosity-Driven Learning Framework for Sample-Efficient Reinforcement Learning* (IJCAI 22')

## Better Visual Representations

* *DRIBO: Robust Deep Reinforcement Learning via Multi-View Information Bottleneck* (ICML 22')

"train RL agents from pixels with this auxiliary objective to learn robust representations that can compress away task-irrelevant information and are predictive of
task-relevant dynamics". Multi-view is gained by random augmentation on the original visual observation. The objective is designed with mutual information.

* *Reinforcement Learning with Augmented Data* (RAD)

RAD takes a particular data augmentation within the same batch and feed the result into the RL agent's input. The result shows that this achieves SOTA result and even comparable to that with proprieceptive states. It seems that random cropping is the most effective to do data augmentation, as this forces the CNN to focus on the "robot body" manipulated by the agent. 

* *Image Augmentation is All You Need: Regularizing Deep Reinforcement Learning from Pixels* (DRQ)

The two works above are almost done in the same time; for DRQ, if we set the number of augmented samples to be 1, then it is RAD. DRQ claims that the Q-values should be the same before and after different data augmentations on states given by images, and thus it samples multiple data augmentations to "average the Q-values" among them.

* *Self-Predictive Dynamics for Generalization of Vision-based Reinforcement Learning* (IJCAI 22')

Another work of auxiliary task besides RL using data augmentation. The difference is that this work tries to add a forward dynamics to the system. There is a weak data augmentation and a strong data augmentation.

* *Don’t Touch What Matters: Task-Aware Lipschitz Data Augmentation for Visual Reinforcement Learning* (IJCAI 22')

## Model-based RL

* *Model-Ensemble Trust-Region Policy Optimization*

* *DREAM TO CONTROL: LEARNING BEHAVIORS BY LATENT IMAGINATION* (ICLR 20')

Dreamer has a complicated latent dynamics, including representation model, transition model and reward model.

### World Models

World models are a special type of model-based RL.

* *BLaDE: Robust Exploration via Diffusion Models* (NeurIPS 22')

BLaDE is a direct successor of BYOL-Explore, which is also a publication in NeurIPS 22'.

This paper focuses on two types of stochasticity: 1) noise from sensor noise and independent to the underlying dynamics; and 2) stochasticity at the action level due to imperfect actor / stochastic dynamics, e.g. sticky actions.

TV noise problem is difficult for vanilla count-based methods.

Ensemble methods are common to deal with stochasticity, but it is hard to train and scale.

* *BYOL-Explore: Exploration by Bootstrapped Prediction* (NeurIPS 22')



* *Dream to Control: Learning Behaviors by Latent Imagination* (Dreamer)

This paper increases RL data efficiency by constructing world models. The world model is estimated with trajectories gained from interaction with ground truth environment. policy and value function are then optimized with trajectories running on the "imagined" world model. Finally, the trained policy is used to interact with ground truth environment. The training of world model and interatciton with ground truth environment is done iteravtively. Unlike PlaNet which uses MPC, Dreamer is RL-based and can be plugged onto any model-free RL algorithm. The world model of Dreamer is different from PLaNet in that it encourages mutual information between model states and observations by predicting the states from the images, which is estimated via NCELoss. 

**NCELoss** is a common metric that turns multi-label classifier into a binary classifier. It breaks each data point (x,y) as a "positive" (real) pair and some noisy negative pairs, to *bypass the process of calculating the normalizing term at the denominator for the probability*, as it may be very expensive or even intractible.

**InfoNCELoss** is first presented in the paper *Representation Learning with Contrastive Predictive Coding*. It learns a function that is proportional to p(x|c)/p(x).

* *Learning Latent Dynamics for Planning from Pixels* (PLaNet)

This is also a work with world models to increase RL sampel efficiency, which is the previous work of Dreamer. The world model is parameterized "representation" p(s_t|s_t-1, a_t-1,o_t) and "transition" q(s_t|s_t-1,a_t-1); there can also be observation model q(o_t|s_t) and reward model q(r_t|s_t). An interesting point is that world models are not optimized with MSELoss; instead, they are optimized to increase the **variational lower bound** (ELBO) or **variational information bottleneck** (VIB). 

It uses MPC to solve the planning task. For every step it takes in the real environment, it looks into its transition/reward model for the best trajectory, and conduct the first step of that trajectory. Therefore, it needs its world model to run in the ground truth environment. 

## Auxiliary Tasks

* *CURL: Contrastive Unsupervised Representations for Reinforcement Learning*

CURL proposes a jointly trained auxiliary task to help the RL agent to learn a robust vision. The auxiliary task is a contrastive unsupervised learning, which uses InfoNCELoss. For each batch of transition, observations are data-augmented twice to form query and key observations, which are then encoded with the query and key encoders respectively. The queries are passed to the RL algorithm, while the query-key pairs are passed to the contrastive learning objective. The key encoder weights are the moving average (EMA) of the query weights.

The anchor and the positive observations are two different augmentations of the same image, while negatives come from other images.

Model-free RL algorithm often uses stack of frames instead of single frames as input to learn both spatial and temporal discriminative features.

* *Generalization in Reinforcement Learning by Soft Data Augmentation*

An auxiliary task to minimize the difference of output by extractor between input without augmentation and with augmentation. The "input without augmentation" part uses exponential moving average of the "with augmentation" part.

One interesting thing is that auxiliary task for RL is not a new thing; however, this paper still contributes something by looking at the realm of pixel input, which is not well-solved enough, with CV techniques. It seems that it is important to know *what is an interesting question to solve for today*; this is as important as finding a good architecture or algorithm.

* *https://www.borealisai.com/en/blog/tutorial-4-auxiliary-tasks-deep-reinforcement-learning/* A survey for early auxiliary tasks in reinforcement learning.

The auxiliary tasks can be roughly divided into the following types:

1. terminal prediction;
2. agent modeling (in MARL);
3. reward prediction;
4. CV prediction (depth prediction / data augmentation on images + contrastive learning)
5. intrinsic reward, usually for better exploration (e.g. maximizing pixel change such as UNREAL agent https://www.borealisai.com/en/blog/tutorial-4-auxiliary-tasks-deep-reinforcement-learning/)

One technique of auxiliary task is to only consider auxiliary task gradient when the dot product of such task and main objective is larger than 0 (i.e., not conflicting). 

# Contrastive Learning
Contrastive learning can be understood as learning a differentiable dictionary look-up task. Given a query q, set of keys K and a known partition of keys K = K+ \cup K\K+, contrastive learning aims to ensure that q matches K+ more than any keys in K\K+ (quote from CURL paper). q is called **anchor**, K is called **targets**, K+ is called **positive samples** and K\K+ is called **negative samples**.

* *A Theoretical Analysis of Contrastive Unsupervised Representation learning* 

This paper proposes a theortical framework that proves, if for a contrastive learning task, the downstream task is a supervised classification task with a linear classifier (y=Wx) and hinge/logistic loss, then it is guaranteed that downstream classification loss with optimal representation learned from contrastive learning is bounded by a linear term of contrastive learning loss with any representations, plus generalization loss that can be bounded with the number of classes with a high probability (1-\delta).

Tutorial: https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html


# Attention & Transformers

There are two tutorials that is suitable for layman (technology tree: RNN -> bidirectional RNN -> attention -> self-attention -> vanilla transformer (attention is all you need) -> variants of transformer)

https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html#self-attention

https://lilianweng.github.io/lil-log/2020/04/07/the-transformer-family.html


# Game Theory

Game theory is closely related to dynamic systems, where the iterative update is in coherent with a point moving in a field.  

For more information on regret matching and counterfactual regret minimization, see the AGT_survey file (arXiv: 2001.06487). This paper might(?) contain error w.r.t. LOLA. 

* *Multiagent Cooperation and Competition with Deep Reinforcement Learning* 

Shows the feasibility of using DQN for discovering cooperating/competiting strategy. Little novelty technologically from today's perspective; furthermore, value-based method has no theoretical guarantee for Nash equlibria. See the "solution concept" part.

Intrinsic motivation and automatic curricula via asymmetric self-play

## Differentiable Games

Differentiable Games are a special type of game where every player's reward function is known and fully determined by parameters \theta of policy; moreover, the reward function shall be differentiable w.r.t theta.

Such game is useful as it is closely related to GAN.

* *N-player Diffentiable Games*

* *Consensus Optimization*

* *Stable Opponent Shaping*

See "Stable Opponent Shaping in Differentiable Games.pptx". This is an improvement of LOLA.

* *Learning with Opponent Learning Awareness(LOLA)* 

See "Stable Opponent Shaping in Differentiable Games.pptx" for details. This paper models the opponent as a naive learner and predicts the update of the opponent.

However, such assumption will cause **arrogance** when the opponent is also a LOLA agent, which means the algorithm assumes that the opponent will comply to our update that benefits ourselves.

## Security Games

### Abstract Security Games

* *Improving learning and adaptation in security games by exploiting information asymmetry* (INFOCOM 15')

 It introduces tabular minmax Q-learning under an abstract security game with partial observation.
 
* *A survey of interdependent information security games*

* *Are security experts useful? Bayesian Nash equilibria for network security games with limited information*

An interesting conclusion for layman; this paper tells us not to install too much anti-virus softwares, quote, "expert users can be not only invaluable contirbutors, but also free-riders, defectors and narcissistic opportunists."

### Network Security (Games)

* *Defending Against Distributed Denial-of-Service Attacks with Max-min Fair Server-centric Router Throttles* (2005)

This paper uses a typical model for network attacks: The attacker controls some initial nodes (leaves) and launches attack on the server on the root.

The defender controls some of the routers in the tree, and its action space is drop rate to maximize the throughput of legal requests while ensuring safety.

Such setting is widely used in DDOS-related papers including those with MARL solutions.

* *Reinforcement Learning for Autonomous Defence in Software-Defined Networking* (2018)

This paper uses another typical model for network attacks: The attacker controls some initial nodes on a graph and launches attack on the server. 

Each turn, the attacker can infect (compromise) some of the neighbouring servers; the defender can break/reconstruct some of the links (edges), or move the data on the target server to another server.

* *Adversarial Reinforcement Learning for Observer Design in Autonomous Systems under Cyber Attacks* (2018)

This paper actually has less thing to do with "computer networks". This paper discusses the problem of defending an adversary that twists the input of a fixed RL agent by training a an agent that corrects the input.

Self-play is used here to train the adversarial and the observer in the same time. The paper uses classical pendulum as input and TRPO as the fixed RL agent.

###  Green Security Games

See RL-application **anti-poaching** part.

## Fictitious Play

Fictitious Play is a way to find Nash Equilibrium for games. Yet, its guarantee of convergence is greatly limited by the type of the game.

* *Games with multiple payoffs* (1975)

"Multiple" payoff is not multi-dimensional, but a unique weighted sum decided by players.

* *On the Convergence of Fictitious Play* (1998) 

For normal general-sum game, fictitious self-play is not converging, and will **often** not converge. It  almost never converges cyclically to a mixed strategy equilibrium in which both players use more than two pure strategies.

Shapley's example of nonconvergence is the norm rather than the exception.

Mixed strategy equilibria appear to be generally unstable with respect to cyclical fictitious play processes.

Actually, there is a conjecture by Hofbauer(1994): if CFP converges to a regular mixed strategy equilibrium, then the game is zero-sum.

* *On the Global Convergence of Stochastic Fictitious Play* (2002)

There are four type of games that can ensure global convergence:

1) games with an interior ESS (as a fully symmetric game, be optiaml in a mixed-strategy neighbourhood);

2) 0-sum games;

3) potential games, where everyone's motivation can be described by a potential (Note: a good design of potential function, possibly non-Nash solution concept ones, can make things easier; e.g. Alpharank);

4) supermodular games, where every participant's marginal profit by increasing their investion in strategy is increasing w.r.t. their opponent's investion.

* *Full-Width Extensive Form FSP*

Previous work studies self-play only for normal form. Theoretically, extensive form games can be trivially extended to normal form and applied with FSP, but as the possibility grows exponentially, this is a very low-efficiency method.

This paper proves that FSP can be applied on extensive form with **linear combination** for policy mixture.

* *Fictitious Self-Play in Extensive-Form Games* (2015)

* *Deep Reinforcement Learning from Self-Play in Imperfect-Information Games* (2016)

This paper proposes N(eural)FSP. NFSP is an extension of FSP where DQN instead of hand-crafted solution is used for giving optimal response oracle, and a supervised-learning network is used for modeling current opponent's behavior. 
 
* *Monte Carlo Neural Fictitious Self-Play: Approach to Approximate Nash Equilibrium of Imperfect-Information Games* (2019)

This paper uses asynchronous Monte-carlo method for self-play for games where all agents share the same parameters.

## Counterfactual Regret Minimization

Current research on counterfactual regret minimization (CFR, as of 2020) mainly focuses on 2-player 0-sum game.

"regret" means that the "reward brought by this action if this action **were** chosen instead of others.

CFR determines an iteration’s strategy by applying any of several regret minimization algorithms to each infoset (Littlestone & Warmuth, 1994; Chaudhuri et al., 2009). Typically, regret matching (RM) is used as the regret minimization algorithm within CFR due to RM’s simplicity and lack of parameters.

This somewhat relates to Pearl's 3-level hierarchy on AI and causal inference (e.g. the notion of "do" operator).

Counterfactual regret minimization is one of the popular direction and frontier w.r.t. RL + game theory (as of 2020). There are, however, not much *reliable toy* deep CFR code on Github. Try *https://github.com/lcskxj/Deep_CFR* maybe?

* *Using Counterfactual Regret Minimization to Create Competitive Multiplayer Poker Agents* (2010)

* *Regret Minimization in Games with Incomplete Information* (2007)

* *An Introduction to Counterfactual Regret Minimization* 

A nice tutorial for counterfactual regret minimization.

* *Regret Minimization in Non-Zero-Sum Games with Applications to Building Champion Multiplayer Computer Poker Agents* (2013) 

This work gives a regret bound for non-0-sum 2-player game with CFR [The original version of comment may be erronous], and proves that CFR works empirically.

* *Deep Counterfactual Regret Minimization* (2019) 

The goal of Deep CFR is to approximate the behavior of CFR without calculating and accumulating regrets at each infoset, by generalizing across similar infosets using function approximation via deep neural networks. One method to combat this is Monte Carlo CFR (MCCFR), in which only a portion of the game tree is traversed on each iteration (Lanctot et al.,2009). In MCCFR, a subset of nodes Qt in the game tree is traversed at each iteration, where Qt is sampled from some distribution.

* *Single Deep Counterfactual Regret Minimization* (2019)

* *Efficient Monte Carlo Counterfactual Regret Minimization in Games with Many Player Actions*

When you "translate" tabular classical algorithms to deep algorithm, you mainly deal with two problems:

1）How to use function approximator to properly estimate a large value table? 

2）How to improve sample efficiency?

## Regret Matching

Regret matching is the foundation of counterfactual regret minimization(CFR); (for most cases,) CFR is regret matching for extensive form game. 

Other algorithms may do, but regret matching is the most widely-used. See the AGT_survey file (arXiv: 2001.06487) for details.

* *A Simple Adaptive Procedure Leading to Correlated Equlibrium* (2000) 

## Solution concepts

Nash equilibrium, in essence, is continuous finding best response to the opponent and expecting to find a fixed point.

Nash equilibrium, though widely used, is sometimes limited, for its "max" operator has a nature of sudden changes. 

Choosing bounded rationality on the other hand, has more reason than simulating real-life behavior; it may lead to an iterative balance that is more **differentiable**.  

### Nash

* *Nash Q-Learning for General-Sum Stochastic Games* (2003) 

Simply change the optimal action in Bellman equation to Nash equilibrium. However, vanilla Nash-Q are very slow to converge (as far as my experience), and calculating Nash equilibrium itself is very costly (NP-hard; see Prof. Xiaotie Deng's work.)  

* *Actor-Critic Algorithms for Learning Nash Equilibria in N-player General-Sum Games* (2014)

Two desirable properties of any multi-agent learning algorithm are as follows: 

(a) Rationality: Learn to play optimally when other agents follow stationary strategies; and 

(b) Self-play convergence: Converge to a Nash equilibrium assuming all agents are using the same learning algorithm.

To bypass the conclusion of existing work (See the paper *Cyclic Equilibria in Markov Games*), the author "avoid this impossibility result by searching for both values and policies instead of just values, in our proposed algorithms".

(It is somewhat strange that this sentence is put in the footnote instead of main paper without explanation, as if the authors want to hide this point.)

* *Learning Nash Equilibrium for General-Sum Markov Games from Batch Data*

Markov Game（Stochastic Game）is a MDP "with rounds", which means action is purely based on current state. A partial observable counterpart of POMDP is POSG.

* *Markov games as a framework for multi-agent reinforcement learning*
 
Littman's classical papers, the pioneering work in MARL+game theory. 

* *Cyclic Equilibria in Markov Games*

 This paper proves that **no value-based method RL algorithm may calculate the static Nash equilibria of arbitrary general-sum game**.
 
 However, this paper proposes a new notion called **cyclic equilibria**: It satisfies the condition that no one can optimize its strategy with only his/her policy changes. 
 
 Yet, cyclic equilibria does not satisfies **folk theorem**; it cycles between a set of static strategies. (Consider rock-paper-scissors where you can only use pure strategy.)
 
 Many 2-player 2-action 2-state games cannot converge under value-based method; yet, they almost always achieve cyclic equlibria. 
 
 Note: **cyclic equilibrium is a correlated equilibrium**. Viewed from dynamic systems, cyclic equilibria is the **vortex** in the strategy optimization field. **This IS essentially Alpharank's MCC**.
 
*  *Actor-Critic Fictitious Play in Simultaneous Move Multistage Games* 

A variance of NFSP, use decentralized actor-critic to solve 2-player 0-sum game.

* *Robust Multi-Agent Reinforcement Learning via Minimax Deep Deterministic Policy Gradient* (AAAI 19’)

Its core idea is that during training, we force each agent to behave well even when its training opponents response in the worst way. There are two core ideas:

1) for robustness, we want *minimax* response;

2) to quickly calculate the inner *min* operatore, we use a linear function to approximate inner optimization problem and do a gradient descent (remember *Nash-DQN* for such approximation?)

There are, however, much hyperparameters for this algorithm.

### Other Novel 

* *Coco-Q: Learning in stochastic games with side payments* (2013) 

A novel solution concept where agents "pay" others reward to make the other sides cooperate. This gives insight to some cooperating MARL problems, such as reward assignment and communication.

* *MAAIRL* See RL-inverse part.

* *α-Rank: Multi-Agent Evaluation by Evolution*
 
See ALPHARANK.pptx. This paper first links game theory and dynamic system in my knowledge sturcture.

* *A Generalized Training Approach for Multi-agent Learning* PSRO; see ALPHARANK.pptx.

* *Multiagent Evaluation under Incomplete Information* A following work of alpharank under partial observable environment.

* *αα-Rank: Practically Scaling α-Rank through Stochastic Optimisation* A following work of alpharank by Huawei to improve the exponentially growing (w.r.t. # agents) random walk space.

## Deception

Deception under partly observable environment are essentially **Bayesian security game**. MADDPG also does experiment in deception.

* *Designing Deception in Adversarial Reinforcement Learning* 

Designing policy under traditional framework to deceive the opponents in a soccer game.

This paper brings up an interesting definition on "deception": Deception is a policy to lure your opponent to adapt a special policy, and thus control the game into a favorable subgame. 

* *Finding Friend and Foe in Multi-Agent Games* 

A paper using CFR with much prior knowledge to let the agents learn how to play Avalon; an interesting attempt.

* *Learning Existing Social Conventions via Observationally Augmented Self-Play* (AIES 19’)

This paper defines a partial order relationship to define the similarity of policy.

Quoting its abstract, "**conventions** can be viewed as **a choice of equilibrium** in a coordination game. We consider the problem of an agent learning a policy ... and then using this policy when it enters an existing group."

(That is to say, social convention is a special type of prior knowledge learned in pretraining which is the choice of action). This could be related to **ad-hoc cooperation**; that field is led by Prof. Peter Stone @ UTexas.  

* *Deceptive Reinforcement Learning Under Adversarial Manipulations on Cost Signals*

This is a paper where the attacker distorts cost signals. 

* *Deception in Social Learning: A Multi-Agent Reinforcement Learning Perspective* 

A survey. Quoting its abstract, "Social Learning is a new class of algorithms that enables agents to reshape the reward function of other agents with the goal of promoting cooperation and achieving higher global rewards in mixed-motive games." (Does this ring a bell w.r.t. Coco-Q learning & side payment?)


## Human-Involved

* *Human-AI Shared Control via Policy Dissection* (NeurIPS 22')

## Miscellanous


* *Planning for Sample Efficient Imitation Learning* (NeurIPS 22') TBD

* *Mastering Atari Games with Limited Data* (NeurIPS 21') TBD

* *CROSS-TRAJECTORY REPRESENTATION LEARNING FOR ZERO-SHOT GENERALIZATION IN RL* (ICLR 22') TBD

* *Exploring Simple Siamese Representation Learning* (2020) TBD

* *Keep doing what worked: Behavioral modelling priors for offline reinforcement learning* TBD.

* *Learning to Utilize Shaping Rewards: A New Approach of Reward Shaping* (2020) 

This paper proposes to get adequate reward shaping value by bi-level optimization; in the outer loop, the objective is the final reward, and in the inner loop, the objective is the shaped reward.

An important thing about this paper is that it summarizes the progress in reward shaping: 

PBRS, the vanilla reward shaping proposed by Andrew Ng; \phi(s)-\gamma * \phi(s')

PBA (advice), include action in the shaping function \gamma * \phi(s', a') - \phi(s, a)

D(dynamics)PBA, include a time dimension in the state that allows for dynamic reward shaping (or, "recurrent shaping policy").

Another thing worth noting is that this bi-level optimization method is actually quite common; e.g. prediction+optimization, not all labeled data are equal paper. You get an auxiliary factor that you don't know in the inner-level optimization and in the outside optimize the original objective. The hardest point here is to calculate the gradient of outer loop variables w.r.t. inner-loop optimal solution.

The author proposes two lines of methods: explicit mapping, which is to expand the state and let shaping value become part of the state; (incremental) meta-gradient learning, which is set the approximated gradient as a new variable and try to estimate it.

* *Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion* 

A classical paper that has 6000+ citations; it helps in understanding what "reconstruction" is, starting from a perspective of mutual information and manifold learning. PCA is essentially Affine encoder and decoder without any nonlinearity and a squared error loss. The corruption and learning with robustness process is to project the data points back on a low-dimensional manifold.

This paper is not interested in denoising itself, but rather investigating denoising as a training criterion for learning to extract useful features.

* *Deep Q-Learning for Nash Equilibria: Nash-DQN* (2019) 

This paper uses a local 1st/2nd Taylor expansion to analytically solvable optimal actions for DQN. However, value-based method RL, again, is not a good solution for finding Nash equilibrium.

* *Multi-Agent Systems: Algorithmic, Game-theoretic and Logical Foundations* (A classical book!)

A great book that includes multi-agent and algorithms, game theory, distributed reasoning, RL, auction and social choice. Strongly recommending this book!

The book mentions some contents on asynchronous DP, ABT search for multi-agent and so on. 

It also mentions the most basic notion in multi-player/sequential games, e.g. extensive form and normal form.

For sequential games, there exists the notion of **"unreliable threat"**, which means, when one party breaks the Nash equilibria, the other party's reward will be more damaged if it chooses to retaliate. Such threat of keeping Nash equilibrium is not credible, and thus such Nash equilibrium is unstable. This urges us to introduce subgame-perfect equilibrium (SPE) for games.

There is also the notion of **Folk theorem** (Copied from wikipedia): Any degree of cooperation, as long as it is feasible and individually rational (e.g. if everybody else is against one agent, the agent's reward will be worse than that of a cooperating agent), can be achieved for an infinite game with adequate discount factor.

If the players are patient enough and far-sighted (i.e. if the discount factor , then repeated interaction can result in virtually any average payoff in an SPE equilibrium.[3] "Virtually any" is here technically defined as "feasible" and "individually rational".

There are also some contents on anthropology, including *locutionary, illocutionary and perlocutionary act* (see *https://www.douban.com/note/663802620/*)；The four principles of communication (quality，quantity，politeness and relativity).

* *Balancing Two-Player Stochastic Games with Soft Q-Learning* (2018)

* *Off-Policy Deep Reinforcement Learning Without Exploration* (2019) An algorithm learning from offline batches. One problem of learning offline is that it has *extrapolation error*, which are caused by three issues: absent data, model bias and training mismatch.

# Inverse RL

* *Robust Inverse Reinforcement Learning under Transition Dynamics Mismatch* (NeurIPS 21')

[TBD]

## Early

* *Inverse Reinforcement Learning* (2000)

The first work of inverse reinforcement learning (IRL). It assumes policy optimality, and solves reward function with linear programming. 

This work first proposes that reward is determined by a linear combination of "features", which is inherited in future work.

* *Apprenticeship Learning via Inverse Reinforcement Learning* (2004)

The final goal of apprenticeship learning is not getting an identical reward function, but a policy that works equally well with the expert.

* *Bayes Inverse Reinforcement Learning* (2008)

This work does not require optimal demonstration to recover reward function; suboptimal ones will do.

## Maximum Entropy

Maximizing the entropy of the distribution over paths subject to the feature constraints from observed data implies that we maximize the likelihood of the observed data under the maximum entropy; See *https://zhuanlan.zhihu.com/p/87442805* for a more detailed introduction. 

See *https://zhuanlan.zhihu.com/p/107998641* for reference.

* *Maximum Entropy Inverse Reinforcement Learning* (2008)

This paper is the founding work of modern inverse reinforcement learning. This paper makes two assumptions:

1) Reward function is a linear combination of some certain features (inherited from Andrew Ng's work in 2000)

2) **The probability distribution of trajectories is proportional to e^r where r is the reward.**

However, I have recurred the experiments and found that the recovery of reward function is not as accurate as one may imagine.   

* *Modeling Interaction via the Principle of Maximum Causal Entropy* (2010)

* *Infinite Time Horizon Maximum Causal Entropy Inverse Reinforcement Learning*

* *Maximum Entropy Deep Inverse Reinforcement Learning* (NIPS2015 workshop) 

"Deep" is used in this work to extract and combine features instead of handcrafting features and only combining them linearly. 

* *Guided Cost Learning* (2015)

Before this work, a main effort of MEIRL is to *estimate Z* in the denominator of maximum entropy probability distribution (Ziebart's work uses DP; there are also other works such as Laplacian approximation, value function based and sample based approximation).

This work finds a sample distribution and uses importance sampling to estimate Z. It iteratively optimizes both the reward function and the sampling distribution.

The goal of sampling distribution is to minimize the *KL-divergence* between the sampling distribution and the term exp(-reward_\theta(traj))/Z.

* *Generative Adversarial Imitation Learning* (2016) 

A generalization of GAN on imitation learning (note: unlike AIRL, GAIL cannot retrive reward function).

* *LEARNING ROBUST REWARDS WITH ADVERSARIAL INVERSE REINFORCEMENT LEARNING* (ICLR 18')

See MAAIRL.pptx in the github for details. Compared to GAIL, AIRL has the ability to retrieve reward function.

* *Adversarial Imitation via Variational Inverse Reinforcement Learning* (ICLR 19') 

The most important notion of this work is *empowerment*. Empowerment is I(s',a|s) where I represents mutual information; it describes the possiblity of agent "influencing its future".

The researcher suggests that by adding this regularizer, we can prevent agent from overfitting onto demonstration policy.

* *A Connection Between Generative Adversarial Networks, Inverse Reinforcement Learning, and Energy-Based Models*

Highly recommending this paper. Intuitively, GAN trains a generator which generates data (which could be *sequences*) that a discriminator (classifier) cannot tell from a particular distribution, which, if substituting the "sequence" to "trajectory", is exactly what imitation learning / IRL does; 

and energy-based models (with Boltzmann distribution) assign a sample with an energy value that has the same form with MEIRL.

* *Discriminator-actor-critic: Addressing sample inefficiency and reward bias in adversarial imitation learning* (ICLR 19') [TBD]

* *Imitation learning via off-policy distribution matching* (ICLR 20') [TBD]

* *On computation and generalization of generative adversarial imitation learning* (ICLR 20') [TBD]

* *Generative adversarial imitation learning with neural networks: Global optimality and convergence rate* (2020)
 
## Multi-agent

* *MAGAIL：Multi-Agent Generative Adversarial Imitation Learning* (2018)

* *Multi-Agent Adversarial Inverse Reinforcement Learning* (ICML 19')

See MAAIRL.pptx in the github for details. This paper proposes a new solution concept called **logistic stochastic best response equilibrium** (LSBRE).

MAGAIL is MaxEnt RL + Nash Equilibrium; MAAIRL is MaxEnt RL + LSBRE.

* *Competitive Multi-agent Inverse Reinforcement Learning with Sub-optimal Demonstrations* (2018)

* *Asynchronous Multi-Agent Generative Adversarial Imitation Learning* (2019)

## DICE (DistrIbution Corrected Estimation)

This is the most popular direction in recent years (since 2019). This line of work tries to minimize the state / state-action occupancy between learner and expert distribution, and turn the problem into a convex optimization. Compared to prior works, this line of work does not require iterative update. 

* *Constrained Offline Policy Optimization* (ICML 22')

safe RL. RL with cost regularization; keep projecting reward-feasible policy into cost-feasible area. Use DICE algorithm to find such reward-feasible solution.

* *Imitation Learning via Off-Policy Distribution Matching* (ICLR 20') ValueDice

* *Optidice: Offline policy optimization via stationary distribution correction estimation.* (ICML 21')

* *Smodice: Versatile offline imitation learning via state occupancy matching* (2022)

SMODICE and DemoDICE/LOBSDICE are the state-of-the-art as of middle 2022. It optimizes an upper bound of state-occupancy KL divergence. 

"Despite its generality, naively optimizing the stateoccupancy matching objective would result in an actor-critic style IL algorithm akin to prior work due to the entangled nature of actor and critic learning, leading to erroneous value bootstrapping."

* *Softdice for imitation learning: Rethinking off-policy distribution matching* (2021)

ValueDICE is biased when estimated in minibatch (log E ...) and assume the MDP is ergodic; this work uses earth-mover's distance and use maximum entropy (least commitment like GAIL) in their derivations.

* *DemoDICE: Offline imitation learning with supplementary imperfect demonstrations* (2022)

Online.

* *RETHINKING VALUEDICE: DOES IT REALLY IMPROVE PERFORMANCE?* (ICLR 22' blog)

BC is good enough in many cases if a complete trajectory is presented.

If there were only expert 1 trajectory and in offline settings, BC optimal is ValueDICE optimal.

Weight decay for BC can prevent overfitting to some extent.

* *LOBSDICE: OFFLINE IMITATION LEARNING FROM OBSERVATION VIA STATIONARY DISTRIBUTION CORRECTION ESTIMATION* (2022)

Offline version of DemoDICE.

## Wasserstein

Wasserstein distance, compared to f-divergences such as KL, is a much "weaker" distance, which means intuitively, it will be more smooth and have better property in many cases, such as those where two distributions have no intersection.

People either consider primal form, where we need to find a better joint probability, or dual form, where we need to find a best f(x) that discriminates x drawn from two samplers. The dual form is easier because it provides a paradigm of iteratively updating reward and actor. 

* *PRIMAL WASSERSTEIN IMITATION LEARNING* (ICLR 21')

This paper proposes a "greedy matching" surrogate for Wasserstein distance, which aims to decouple Wasserstein distance from functions of current policy (?). It is also the only work in this field which works on the primal form.

* *Wasserstein Adversarial Imitation Learning*

Iteratively updating reward in dual form and actor.

* *State Alignment-based Imitation Learning* (ICLR 20')

Wasserstein + local VAE / inverse dynamic model for alignment.

* *Cross-Domain Imitation Learning via Optimal Transport* (ICLR 22')

Aiming to solve embodiment difference.

* *Wasserstein Distance guided Adversarial Imitation Learning with Reward Shape Exploration* (DDCLS 20') 

* *Curriculum Reinforcement Learning via Constrained Optimal Transport* (ICML 22')

# Application

RL is currently not very popular in deployment of production (as of 2020/2021), as the sample efficiency are low and it may have no significant 
edge over expert systems and traditional analytical models in real-life. 

## Feature Selection

* *Feature and Instance Joint Selection: A Reinforcement Learning Perspective* (IJCAI 22')

RL is used to select the most useful dimensions and samples in the dataset. There are two agents performing cooerpative MARL: the first agent is "instance agent", and the second agent is "feature agent". Consider a matrix where the row is instances and columns are features. For one step, instance agent move one sample down, and feature agent move one sample right. The agents can choose to "select" or "deselect" one dimension of feature / sample. A convolution is used to calculate the state and reward. (Warning: though the authors do not mention, I think the order should be randomized from time to time to avoid correlation on particular combination of feature and instance. Also you might not visit all combinations if you do not randomize.)

This is not a "canon" MARL as they train their agent with independent DQN.

Both RL agents are aided with **random forest/isolated forest** advisor, which could be an interesting idea to adopt. But it is only forced to take advice at the beginning of exploration to fill up the buffer - couldn't there be better choices such as PARROT-like algorithms?

## Economy

* *Welfare Maximization in Competitive Equilibrium: Reinforcement Learning for Markov Exchange Economy* (ICML 22') [TBD]

* *Pessimism meets VCG: Learning Dynamic Mechanism Design via Offline Reinforcement Learning* (ICML 22') [TBD]

## Tensor Decomposition

* *DECORE: Deep Compression with Reinforcement Learning* (CVPR 22')

* *Optimizing Tensor Network Contraction Using Reinforcement Learning* (ICML 22') [TBD]

* *A novel rank selection scheme in tensor ring decomposition based on reinforcement learning for deep neural networks* (ICASSP 20')

## Software Engineering

* *Assessing and Accelerating Coverage in Deep Reinforcement Learning* (2020)

## Power Systems

* *Exploring the Vulnerability of Deep Reinforcement Learning-based Emergency Control for Low Carbon Power Systems* (IJCAI 22')

* *Multi-Agent Reinforcement Learning for Active Voltage Control on Power Distribution Networks* (NeurIPS 21')

This paper mainly focuses on the formulation of the problem and have not much novelty on MARL; it simply try a lot of RL algorithms. However, the idea of why using MARL on this problem is interesting (quote from the paper):

The active voltage control problem has many interesting properties. 

(1) It is a combination of local and global problem, i.e., the voltage of each node is influenced by the powers (real and reactive) of all other nodes but the impact recedes with increasing distance between nodes.

(2) It is a constrained optimisation problem where the constraint is the voltage threshold and the objective is the total power loss, but there is no explicit relationship between the constraint and the control action. (hard to be formulated as a constrained optimization problem) 
(3) A distribution network has a radial topology involving a rich structure that can be taken as prior knowledge for control, but the node-branch parameters of the topology
may be significantly uncertain. (prior knowledge is of limit use)
(4) Voltage control has a relatively large tolerance and less severe consequences if the control fails to meet standard requirements. (lenient for occasional failure)

Di Cao, Weihao Hu, Junbo Zhao, Qi Huang, Zhe Chen, and Frede Blaabjerg's team has many works in this area, mainly published on IEEE transactions on smart grid / power systems. See citation [13-16] of this paper.

* *Reinforcement learning and its applications in modern power and energy systems: A review* (2020)



A large collection (~100) papers of RL for energy systems. Such papers usually publish on IEEE transactions instead of conferences.

## Recommender Systems

RL is used in recommending systems due to its inherent **interactive and dynamic** nature; however, the sparsity of data is the greatest pain. Both GAN and supervised-learning with importance sampling are developed to address the issue.

* *Environment reconstruction with hidden confounders for reinforcement learning based recommendation* (KDD 19') 

GAIL that adds a "confounder agent" representing the confounding variable in causal inference, which means it has three policies and thus two GANs to learn jointly.

* *Virtual-taobao: Virtualizing real-world online retail environment for reinforcement learning* (AAAI 19') 

A reward designed for more realisitic simulator, a multi-agent GAIL (there are also other papers for MAGAIL/MAAIRL), and a GAN that learns distributional output (instead of a particular value).

* *Generative Adversarial User Model for Reinforcement Learning Based Recommendation System* (ICML 19')

This paper propose a novel **model-based** RL (for higher sample efficiency) framework to imitate user behavior dynamics & learn reward function.

The generative model yields the optimal policy (distribution over actions) given particular state and action set. With careful choice of regularizer, we can derive a closed-form solution for the generative model.

(This could be related to **Fenchel-young-ization**?) Thus the distribution is determined by reward function (This form clearly relates with maximum entropy IRL.)

The adversarial training is a minimax process, where one generates reward that differentiate the actual action from those generated by the generative model (serves as discriminator/outer min), the other maximizes the policy return given true state (serve as generator/inner max).

Since (# candidate items) can be large, the author proposes cascading DQN. Cascading Q-network breaks the multiple-dimensional action one by one; when making decision for one dimension, we use the "optimal" Q-value for the actions that are already considered.

* *Deep Reinforcement Learning for List-wise Recommendations* (2017)

This paper defines the recommendation process as a sequential decision making process and uses actor-critic to implement list-wise recommendation. Not many new ideas from today's perspective.

* *Reinforcement Learning to Optimize Long-term User Engagement in Recommender Systems* (2019)

This paper improves the state design by considering the length of time for users staying on the page. 

The common state design (including this paper) is a sequence of {u, [(item_i, action_mask_i, action_i, other_info_i) for i in range(n)]}.

This paper uses a S-network for the simulation of environment. It uses an imporance-weighted loss from ground truth to mimic the real-world situation and provide samples.

Note that Q-network uses multiple LSTM, as one LSTM is not enough for extracting all information. 

* *Recommendations with Negative Feedback via Pairwise Deep Reinforcement Learning* (2018)
 
This paper considers an important prior knowledge in recommending systems, that positive feedbacks (likes) are far more fewer than negative (dislikes) ones; it embeds both clicks and skips into the state space.

The proposed DEERS framework not only minimizes TD-losses, but also maximizes the difference between Q-values of the chosen action and a proposed "competitor" a to ensure "rankings" between competing items.

* *Toward Simulating Environments in Reinforcement Learning Based Recommendations*

This paper also uses GAN to make up for the sparsity of data as that in the ICML 19' paper. Such method can be viewed as opponent modeling in MARL, as you imagine the customer being the "opponent" of your recommending system.

See *https://zhuanlan.zhihu.com/p/77332847* for reference and a more detailed explanation.

## Packet Switching

Packet switching is a decision-making process which can be solved by RL.

* *Neural Packet Classification*

The main solution for packet switching is decision tree based on IP ranges (i.e. high-dimensional squares). 

This paper uses RL to generate better decision tree; it sees each split of decision tree as an "action", and minimizes the loss weighted by time and space usage.

To better parallelize the training process, the author takes the construction of each node as one episode, effectively trainining a 1-step RL agent (but not contextual bandit since those 1-steps are correlated). 

* *A Deep Reinforcement Learning Perspective on Internet Congestion Control* (ICML 19')

This paper presents a test suite for RL-guided congestion control based on the OpenAI Gym interface.

For congestion control. actions are changes to sending rate and states are bounded histories of network statistics.

## Network Intrusion Detection

There are some works of RL on network intrusion by modeling the problem as a competitive MARL environment / **Bayesian security game**, but there is no technical improvement w.r.t. RL in this field. 

Network intrusion detection can be divided roughly into host-based (based on log files) and network based (based on flow & packet); it can also be partitioned into signature-based and anomaly-based.

* *An Overview of Flow-based and Packet-based Intrusion Detection Performance in High-Speed Networks*

* *PHY-layer Spoofing Detection with Reinforcement Learning in Wireless Networks*

## Anti-poaching

Anti-poaching process can be viewed as a **Green security game** that is solvable by RL. Prof. Fang Fei in CMU has conducted lots of research in this field.

* *Dual-Mandate Patrols: Multi-Armed Bandits for Green Security* (AAAI 21' best paper runner-up)

[Under construction]

* *Green Security Game with Community Engagement* (AAMAS 20')

This is a game-theoretic paper, but not a RL paper.

"Community" means that the defender can "recruit" informants, but the (possibly multiple) attacker may update the strategy accordingly. The informant-atacker social network forms a bipartite graph.

One point worth noting is that to solve such competitive multi-agent problem, the author analyzes **the parameterized fixed point** of the game, then convert it to **bi-level optimization** problem.

Such paradigm can be used in competitive MARL settings. 

* *Deep Reinforcement Learning for Green Security Games with Real-Time Information* (AAAI 19')

The author uses 7*7 grid for experiment; such size can serve as a sample for our experiment design. 

## Traffic Control

Traffic control is one of the most common applications for multi-agent RL.

* *Multi-Agent Reinforcement Learning for Traffic Signal Control through Universal Communication Method* (IJCAI 22')

* *Multi-agent Deep Reinforcement Learning for Large-Scale Traffic Signal Control* (2019)

Indepdendent RL agent though multi-agent environment; the state of neighbouring lights are fed into the network for communication. 

* *Integrating Independent and Centralized Multi-Agent Reinforcement Learning For Traffic Signal Network Optimization*

This paper makes an improvement from independent RL. 

It solves a set of problem with two assumptions:

1) **The credit assignment is clear** (i.e. local & global reward are apparent). This is a strong assumption in MARL;

2) no agent is particular; all agents are homogeneous.

It designs a global Q-value, yet every individual agent maximizes local reward; the gradient descent considers both global and local Q-value.

Then, the author imposes a regularizer to limit the value difference between any local Q to global Q to maintain a "balance" between traffic lights. 

## Public Health

RL could be used in public health for both government control and individual agent simulation. 

* *A Microscopic Epidemic Model and Pandemic Prediction Using Multi-Agent Reinforcement Learning*

Individual agents can be seen as agents optimizing its rewards. However, to actually use the model, it is hard to pin down the reward function in the first place. 

One point worth noting (though not RL-related) is that "microscopic" is actually not precise in this title; "agent-based" would be more accurate.

* *A brief review of synthetic population generation practices in agent-based social simulation*

* *Generating Realistic Synthetic Population Datasets*

The above two papers are not direct applications of RL, but related to agent-based epidemic modeling.

* *Reinforcement learning for optimization of covid-19 mitigation policies*

A work to apply RL on Covid-19 mitigation. Nothing new technologically.

## Resource Provisioning

Resource provisioning requires prediction for the resource and optimization with given situation; due to the high-dimensional essence of provisioning, it is widely solved by two-stage method or differentiable optimization neural layer. 

This category includes both resource distribution and mitigation strategy (i.e. handling VM error online).

* *A Hierarchical Framework of Cloud Resource Allocation and Power Management Using Deep Reinforcement Learning* (2017)

* *Energy-Efficient Virtual Machines Consolidation in Cloud Data Centers Using Reinforcement Learning*

* *Resource Management with Deep Reinforcement Learning* (2016)

A work by MIT's Hongzi Mao; one of the earliest and classic work about resource management.

The paper uses policy gradient method (NOT actor-critic since return can be calculated precisely instead of bootstrapping)

The state and step design is the most important point of this paper. Resource allocation in one time step can be very complicated, therefore the author models the process as filling "slots" in a square, and the execution one real timestep is modeled as a special action to move slots upward for one unit.


* *Predictive and Adaptive Failure Mitigation to Avert Production Cloud VM Interruptions* (2020)

Utilizing multi-arm bandit to optimize mitigation strategy. The core contribution of this work is the design of the MDP; there are many devils in the details when it comes to production line (e.g. data correlations; how to retrieve data efficiently as one may need long observation to determine the effect of an action?)

## Autonomous Driving

Yet, it could be long before RL-powered autonomous driving actually put in use.

[Under construction]

* *Reinforcement Learning for Autonomous Driving with Latent State Inference and Spatial-Temporal Relationships*

* *Driverless Car: Autonomous Driving Using Deep Reinforcement Learning in Urban Environment*

# RL with Novel Architectures

## LM/LLM

* *IS REINFORCEMENT LEARNING (NOT) FOR NATURAL LANGUAGE PROCESSING: BENCHMARKS, BASELINES, AND BUILDING BLOCKS FOR NATURAL LANGUAGE POLICY OPTIMIZATION * (ICLR 23')

* *On the Effect of Pre-training for Transformer in Different Modality on Offline Reinforcement Learning* (NeurIPS 22')

## Diffusion Models

* *Is Conditional Generative Modeling all you need for Decision Making?* (ICLR 23')

# Miscellanous

## Continuous Learning

* *Understanding Catastrophic Forgetting and Remembering in Continual Learning with Optimal Relevance Mapping*

A paper that talks about **Catastrophic forget/remebering**. However, the solution offered by the paper itself ("authentic" solution for continual learning) is not so helpful in my opinion. After all, what is the difference between "use different parts of NN" and "adding new layers for each task" besides notion? I can put every layer for every task there in the first place and freeze them unless working on the corresponding task.

## Transfer Learning / Multi-task Learning

* *RotoGrad: Gradient Homogenization in Multitask Learning* (ICLR 22')

There are usually two problems in multitask learning: graident magnitude conflict and direction conflict. The phenomenon that sharing parameters across tasks leads to worse result is called **negative transfer**.

Quote: "RotoGrad addresses the gradient magnitude discrepancies by re-weighting task gradients at each step of the learning, while encouraging learning those tasks that have converged the least thus far."

For gradient magnitude: equalizing gradient magnitudes amounts to finding weights that normalize and scale each gradient; the core idea is that each task should converge at a similar rate. Therefore, the authors set the sum of all tasks' weight to be 1, and each weight to be proportional to its current task's graident norm ratio, compared with the first gradient step of this task. This is a resemblance of Normalized Gradient Descent in multi-task learning. A good point of this method is that it is a hyper-parameter-free approach which does not need tuning.

For gradient direction: try to rotate the feature space to decrease conflicts; more specifically, find a rotation matrix to the last shared feature vector and apply to it. Such matrix is learned, and optimized by maximize the batch-wise cosine similarity of gradient. This rotation optimization, as "follower", forms a **stackelberg game** with the main optimization process in each step. For scalability, only a small subset of dimensions are rotated each step.

**Note that the rotation matrix SO(n) is a Lie group; The optimization of rotation matrix can be done by considering parameterization via exponential maps on the Lie algebra of SO(n).** See the paper *Cheap Orthogonal Constraints in Neural Networks: A Simple Parametrization of the Orthogonal and Unitary Group*.

* *A Comprehensive Survey on Transfer Learning* (2019)

homogeneous transfer learning v.s. heterogeneous one (differ in feature space);

Many transfer learning approaches absorb the technology of semi-supervised
learning;

Domain adaptation refers to the
process that adapting one or more source domains to transfer knowledge and improve the performance of the target
learner.

[Under Construction]

## Semi-supervised Learning 

* *Not All Unlabeled Data are Equal: Learning to Weight Data in Semi-supervised Learning*

## Meta Learning 

Survey:

https://arxiv.org/pdf/1810.03548.pdf

https://lilianweng.github.io/lil-log/2018/11/30/meta-learning.html 优质meta learning综述。

## Federated Learning

Survey: 

* *A survey on Federated Learning Systems: Vision, Hype and Reality for Data Privacy and Protection*

## Distance of Distribution

* *Wassenstein Reinforcement Learning* 

This paper gives many theoretical results about extending RL onto Wasserstein metric; however, it remains to see that if RL on Wasserstein is better than traditional RL.

* *Wasserstein GAN* An advantage of Wasserstein over KL-divergence is that it does not require the support interval/set to be exactly the same. KL goes to infinity if the intersection is 0, but Wasserstein will not.

* *Munchausen Reinforcement Learning* (2020) 

SOTA (as of ICML 22') on Atari games. It modifies a little on energy-based DQN (soft Q-learning) by adding a positive reward for entropy (note: SAC modified the value function!) By adding this term, Munchausen DQN implicitly does KL regularization between successive policies, and increases the action-gap by a quantifiable amount which also helps dealing with approximation errors.

**A softmax is the maximizer of the Legendre-Fenchel transform of the entropy.** https://en.wikipedia.org/wiki/Legendre_transformation



## Active Learning

* *Active Classification based on Value of Classifier*

* *Learning how to Active Learn: A Deep Reinforcement Learning Approach active learning* 

It was originally a method to improve learning efficiency by actively selecting unlabeled text and sending it to experts for labeling by the classifier. 

Originally, active learning was used for NLP. Here, it is modeled as a problem of RL selecting samples as the policy. And it's about learning the policy in one language first and then migrating to another language. First disrupt the corpus, and then there are two actions in the face of a sentence: accept or not accept. 

If accepted, update the current classifier. Note that they model the current classifier's state as a state, so it can be considered that the training is off-policy.

## Overfitting Prevention

* *Protecting against evaluation overfitting in empirical reinforcement learning* (AAAI 11')

* *Improved Empirical Methods in Reinforcement Learning Evaluation* (2015)

* *A Max-min Entropy Network for Reinforcement Learning* (NeurIPS 21') 

This paper digs into the problem of **saturation** of SAC exploration. Consider a pure exploration scenario where there is no reward; the SAC agent, ideally, should learn uniform action to maximize the entropy. However, Q-value difference made by randomized initialization will force the agent to visit the states with higher Q-value, and further increase the Q-value there by optimization. This makes the agent stay in a small part of the observation space. To solve this, the authors propose to segregate the Q-value for exploration and the Q-value for optimization. 

By reversing the sign of entropy in the soft Q-function update, the result drives the policy to visit states with low entropy, and increase the entropy there. This set of Q is separated from original reward of the MDP, and the final update equation adds Q for reward and Q for exploration together to form a new Q value. nryThe shortcoming of this paper is the relatively simple experiment which only takes a few gym environments.

* *Lenient DQN* 

Temperature and lenient parameter are the two important notions in Lenient DQN. 

Leniency was designed to prevent relative overgeneralization, which occurs when agents gravitate towards a robust but sub-optimal joint policy due to noise induced by the mutual influence of each agent’s exploration strategy on others’ learning updates. 

To some extent, competitive MARL is harder than cooperative marl, as competitive MARL needs robust but sub-optimal policy for better generalization, whereas in cooperative MARL overfitting to other opponents can be a good thing, and agents even need to reduce the noise by its teammates that causes sub-optimal behavior.

Temperature-based exploration auto-encoder is a method to deal with high-dimensional / continuous (s,a) pairs. The autoencoder, consisting of convolutional, dense, and transposed convolutional layers, can be trained using the states stored in the agent’s replay memory.

 It then serves as a pre-processing function д : S → R D , with a dense layer consisting of D neurons with a saturating activation function (e.g. a Sigmoid function) at the centre. 
 
 SimHash, a locality-sensitive hashing (LSH) function, can be applied to the rounded output of the dense layer to generate a hash-key ϕ for a state s. 

## Novel Architectures

* *HYPERBOLIC DEEP REINFORCEMENT LEARNING* (2022)

A special type of state embedding; in this embedding, the LCA of states stands between them(?).

### Diffusion Model in RL

* *Planning with Diffusion for Flexible Behavior Synthesis* (ICML 22')

This paper considers the RL as a trajectory generation task and thus apply diffusion model to the whole trajectory. For each denoising step, a parameterized Gaussian noise is added onto the trajectory.

A small reception field enforces the model to have local consistency during a single diffusion step.

The diffusion target is a 2-dimensional array, with one column per timestep. 

The method is tested on maze2D environment and block stacking environment - both are common for skill-based robotic papers such as SPiRL / SKiLD.

Comment: if the quality of generated trajectory is not good enough, we will not get a feasible policy (not even a suboptimal one) because some actions are literally impossible?

### Transformer in RL

https://zhuanlan.zhihu.com/p/389748472 

https://zhuanlan.zhihu.com/p/384458362

* *Decision Transformer: Reinforcement Learning via Sequence Modeling* (2021)

* *Reinforcement Learning as One Big Sequence Modeling Problem* (2021)

* *Distributional Hamilton-Jacobi-Bellman Equations for Continuous-Time Reinforcement Learning* (ICML 22')

Change of distributions over a continuous interval of time.

* *MMD GAN: Towards Deeper Understanding of Moment Matching Network* Measure Matching (NIPS 17')

* *Sub-Goal Trees – a Framework for Goal-Based Reinforcement Learning* (ICML 20') [TBD]

Redefining the framework of reinforcement learning!

* *Implicit Quantile Networks for Distributional Reinforcement Learning* (ICML 18')

In distributional RL, the **distribution** over returns is considered instead of the scalar value function that is its expectation. This is beneficial for theoretical analysis.


* *On Layer Normalization in the Transformer Architecture* (2020) https://arxiv.org/pdf/2002.04745.pdf 

Pre-LN is much easier to train than Post-LN! 

* *THE HIDDEN CONVEX OPTIMIZATION LANDSCAPE OF REGULARIZED TWO-LAYER RELU NETWORKS: AN EXACT CHARACTERIZATION OF OPTIMAL SOLUTIONS* (ICLR 22' oral)

**All globally optimal two-layer ReLU neural networks can be performed by solving a convex optimization program with cone constraints.**

Wang & Lin (2021) showed that with an explicit regularizer based on the scaled variation norm, overparametrization is generally harmless to two-layer ReLU networks.

The authors assume that the loss is a convex one (e.g. hinge) plus a L2 regularization.

* *The loss surface of deep and wide neural networks* (2017)

Nguyen & Hein (2017) showed that no spurious minima occur provided that one of the layer’s inner width exceeds n and under additional non-degeneracy conditions.

Almost all local minima are globally optimal, for a fully connected network with squared loss and analytic activation function given that the number of hidden units of one layer of the network is larger than the number of training points and the network structure from this layer on is pyramidal.

* *Harmless Overparametrization in Two-layer Neural Networks* (2021)


### Neural ODE
* *IMPLICIT NORMALIZING FLOWS* (ICLR 21')

* *Continuous-in-depth Neural Networks* (2020)

* *Neural Ordinary Differential Equations* (2018)

* *Imbedding Deep Neural Networks* (ICLR 22' spotlight)

This paper proposes a novel architecutre called InImNet which improves over neural ODE. (TBD)

An interesting direction nowadays is to combine neural network (especially resnet) and ODE, which is called neural ODE. Neural ODE sees network depth as the variable in the ODE, and the output of neural network is the value of the ODE system with depth being a particular integer. 

The potential advantages of neural ODE are:
1) stronger expressivity. If we view resnet as a discretized ODE, then it is less expressive than neural ODE. Actually, there are some functions that neural ODE can approximate easily, but resnet is hard to approximate.
2) More freedom for tuning. The depth of the neural network now can be any real numbers!
3) smaller storage space. The whole network can be described by a parameterized ODE, which could be much simpler.

* *Value Propagation Networks* (ICLR 19')

An improvement over VIN. An interesting observation is that *for 2-dimensional gridworld / games on a planar graph, value iteration can represented by a convolutional progress on GCN*.

* *Structured Control Nets for Deep Reinforcement Learning* 

The strategy is divided into two independent streams: linear control and non-linear control. Linear control is a simple matrix multiplication; nonlinear control is an MLP (but not limited to MLP, one of the experiments uses a central pattern generator). 

The simple addition of the two is the output of the final neural network. " Intuitively, nonlinear control is used for front view and global control, while linear control revolves around the stability of local dynamic variables other than global control.

See https://www.jianshu.com/p/4f5d663803ba for a Chinese note.

* *Safe and efficient off-policy reinforcement learning* (NIPS 16’) 

This article proposes an algorithm called Retrace (lambda). It can perform off-policy training efficiently and "safely", and its variance is small. This is an algorithm that can converge without **the GLIE premise**. GLIE (*Greedy in the Limit with Infinite Exploration*) is the combination of the following two conditions (see technion's note for reference: https://webee.technion.ac.il/shimkin/LCS11/ch7_exploration.pdf)

1.If a state is visited infinitely often, then each action in that state is chosen infinitely often (with probability 1).

2. In the limit (as t → ∞), the learning policy is greedy with respect to the learned Q-function (with probability 1).

* *the IQ of neural networks* (2017) An interesting paper that uses CNN to do IQ testing quizes. CNN is quite clever!

* *What can neural networks reason about?* (2019)

A great article, it provides insight into the theoretical basis behind the various structural modules we have designed in NN over the years, especially GNN. 

The article takes PAC-learning as the cornerstone, and proposes that if the neural network module can have a good alignment with the classic algorithm (that is, the sample complexity is high), then there will be good performance and generalization.

### Normalizing Flows

Normalizing flow is a generative model which transforms a simple probability distribution (e.g. Normal) to a complicated probability distribution. It can be used as a generative model, trained by maximizing likelihood of P(y | x, theta) where x can be either the old distribution itself or old distribution and a state (used in RL embedding). It can also be used for causal inference.

The core improvement on normalizing flow in recent years is combining efficient computation of the Jacobian matrix with better expressivity. Some major process of normalizing flows include realNVP, 1x1 convolution flow and autoregressive flows.

For a easy-to-read survey, see https://arxiv.org/pdf/1908.09257.pdf .

## Model Reduction

* *DR-RNN: A deep residual recurrent neural network for model reduction*

For model reduction of large-scale simulations (such as physical models, etc.), there are three main ideas. The first is a simplified model based on physical formulas (so it is heavily prior-based); 

the second is a purely fitted black box model (similar to the imitation learning in the expert-apprentice problem); the third is a low projection based model Rank model (ROM). 

The premise of the third idea is to assume that the entire model can be represented by a low-rank linear expression. To get the projection base, use Galerkin projection (Galerkin method). 

Several main algorithms are: Proper Orthogonal Decomposition; Krylov subspace methods; truncated balanced realization. The article proposes a model reduction based on RNN.

## Behavior Cloning

* *Integrating Behavior Cloning and Reinforcement Learning for Improved Performance in Dense and Sparse Reward Environments*

* *Accelerating Online Reinforcement Learning with Offline Datasets*

## Multimodel RL (e.g. Gaussian Mixture)

Many multimodel RL uses navigation / goal-based task with multiple goals to illustrate the multimodality learned by the model.

* *Reinforcement Learning with Deep Energy-Based Policies* SQL paper.
* *Distributional Deep Reinforcement Learning with a Mixture of Gaussians* It has a prior work of C51.
* *Learning a Multi-Modal Policy via Imitating Demonstrations with Mixed Behaviors* (NeurIPS 18') it uses LSTM to learn a discrete latent variable for each trajectory (every trajectory has a fixed discrete latent variable), and use a decoder to feed current state and this latent variable as input to recover current action.
* *Multimodal Policy Search using Overlapping Mixtures of Sparse Gaussian Process Prior* (ICRA 19')
* *Confidence-Based Policy Learning from Demonstration Using Gaussian Mixture Models* (AAMAS 07')

## Bayesian RL
Bayesian RL is usually used for multi-tasking, where it believes that some factor of the environment (it could be opponent in MARL, or type of environment) is controlled by a latent variable. However, Bayesian RL does not necessarily means multimodel RL.
* *Multi-Task Reinforcement Learning: A Hierarchical Bayesian Approach* (ICML 07') 

## Other RL

* *Learning to Control Self-Assembling Morphologies: A Study of Generalization via Modularity* (NeurIPS 19')

Use DGN for a MARL task where agents can assemble into a larger one and complete task. A very interesting task!

* *Distilling Policy Distillation* (ICLR 19')

Policy distillation is the transfer of knowledge from one policy to another, enabling the training of agents based on already trained policies / human examples.

Consider the general problem of extracting knowledge from a teacher policy, π, and transferring it to a different student policy, πθ, using trajectories.

many widely used methods do not correspond to valid gradient vector fields, and thus may be susceptible to non-convergent learning dynamics.

* *OFFLINE REINFORCEMENT LEARNING WITH IMPLICIT Q-LEARNING* (ICLR 22')

SARSA, but learn the optimal policy by only querying state-action pairs in the dataset.

The method uses expectile regression, a variant for quantile regression. They substitute the MSE error in Bellman residual with the expectile regression.

One thing worth noting is that the authors use another value function that approximates an expectile only w.r.t. the action distribution which is to avoid "lucky" sample where a large target value is acquired because the agent happens to have transitioned into a good state.

Comment: can we use this in general settings and substitute mean square error with expectile objective everywhere?

* *Distributional Reinforcement Learning with Quantile Regression* (AAAI 18')

See this paper for expectile / quantile regression in RL.

* *LM-Nav: Robotic Navigation with Large Pre-Trained Models of Language, Vision, and Action* (2022)

### Bandits

Langevin Monte Carlo for Contextual Bandits (ICML 22')

### Symbolic Planning Models

* *Leveraging Approximate Symbolic Models for Reinforcement Learning via Skill Diversity* (ICML 22') Symbolic model can be seen as an abstract and formal version of RL with NLP. However, the problem for symbolic model is that users may give partly correct / inaccurate demands; the robot must find useful guidances. This paper constructs a partial ordering between symbolic state sequences, and then learns a "minimum viable task representation" model that gives a symbolic planning model which allows for success in some trajectories. The model is then used to extract landmark, which is the partition of tasks, and then a low-level model is trained as skills for reaching from one landmark to the next. The high-level agent, metacontroller, is a standard Q-learning. This work only tests toy scenarios.

 
### RL as Sequence Modeling

* *Decision Transformer: Reinforcement Learning via Sequence Modeling* (2021)

Similar to the paper below; quote, "Unlike prior approaches to RL that fit value functions or compute policy gradients, Decision Transformer simply outputs the optimal actions by leveraging a causally masked Transformer". In the paper below, the algorithm deals with goal-based RL, imitation learning and offline RL, and **predicts everything including rewards and states** as the generated sequence, using beam-search to make decisions; this paper only produces action.

* *Offline Reinforcement Learning as One Big Sequence Modeling Problem* (NeurIPS 21' spotlight)

A very interesting idea by Sergey Levine's group, to see (s_1, a_1, r_1, s_2, a_2, r_2, ...) as a big sequence (dimensions are flattened to ensure it is still a 1-dimensional "sequence"), and use transformer & NLP techniques to try to learn sequence modeling problem. The paper aims to convert the quality of RL to the quality of sequence representation; however, it remains to be seen whether this idea is scalable, since the experiments are kind of too simple.

With a transformer, the authors uses **beam search** to solve imitation learning, goal-based RL and offline RL. For imitation learning, the goal is to maximize log likelihood of states; for goal-based RL, the trajectory can be decoded with probability of a state conditioned on past states; for offline RL, we can replace the log-prob of token predictions with the predicted reward signal, effectively replacing transition's logprob in beam search with logprob of optimality. To avoid myopic behavior, the transition is augmented.

Another interesting thing is that the authors mentioned that a specific pattern the transformer learned is to focus more on past action instead of states, which resembles action smoothing in some optimization algorithms. 



### PAMDP

* *Augmenting Reinforcement Learning with Behavior Primitives for Diverse Manipulation Tasks* (2021) 
This paper proposes MAPLE, a learning framework that augments standard reinforcement learning algorithms with a pre-defined library of behavior primitives. This falls under the established reinforcement learning framework of **Parameterized Action MDPs** (PAMDPs), in which the agent executes a parameterized primitive at each decision-making step. Note that the pre-defined library is **hardcoded**; the difficulty lies in which to choose and what parameter (e.g. moving location) to apply.

* *Accelerating Robotic Reinforcement Learning via Parameterized Action Primitives* (2021)

Similar to the last paper, this paper uses a predefined and hard-coded library, and try to choose actions from the experts.



* *Hysteretic Q-learning:an algorithm for decentralized reinforcement learning in cooperative multi-agent teams*  

Hysteretic Q-learning is a method of obtaining a team of cooperative agents through distributed training. It originated from game theory and mainly studied repeated two-player matrix games. In fact, there is nothing new in essence, just adjust the parameters when the modulation is good and bad, so that the rate of change when the q-value estimation becomes high and when it becomes low is different. Soft-update to increase stability is a common technique.

This results in an optimistic update function which puts more weight on positive experiences, which is shown to be beneficial in cooperative multi-agent settings.

* *Deep decentralized multi-task multi-agent reinforcement learning under partial observability* (ICLR 17') MT-MARL

* *Neural Logic Reinforcement Learning* (ICML 19') 

NLRL: A rare attempt today, aiming to combine symbolism and connectionism, it represents policy with 1st-order logic. Actually, there are attempts of representing states at the beginning of the 21st century; however, such method requires knowledge for agent about the logical form of state and reward (i.e., known dynamics), and thus is impractical.

The algorithm is developed based on prolog, and can be regarded as a binary neural network; the final experiments are also very simple, including cliff walking on small gridworld and putting bricks. The calculation of policy can be regarded as processing the input with a sequence of clauses. It is "explainable" (not *human-readable* though), and do not rely on any background knowledge.

* *Reinforcement Learning with Deep Energy-Based Policies* (ICML 17’) See https://zhuanlan.zhihu.com/p/44783057 for detailed explanation.

* *Multiagent Soft Q-Learning*

Relative overgeneralization occurs when a suboptimal Nash Equilibrium in the joint space of actions is preferred over an optimal Nash Equilibrium because each agent’s action in the suboptimal equilibrium is a better choice when matched with arbitrary actions from the collaborating agents.

* *Improving Stochastic Policy Gradients in Continuous Control with Deep Reinforcement Learning using the Beta Distribution* (ICLR 17')

When outputting continuous actions, we often assume that the probability distribution of actions is Gaussian. But not all scenarios are suitable for Gaussian distribution; this article explores the possibility of using beta distribution instead of Gaussian distribution. 

There is a problem with the Gaussian distribution: *it will have a bias when dealing with actions with boundaries*, because some actions that have been assigned probabilities are actually impossible to achieve (for example, the angle of the robot can only be (-30, 30) ), at this time, if a Gaussian distribution with a mean value of -29 is output, a considerable part of the probability is actually located in the "infeasible" area. 

In contrast, the support set of the beta distribution is bounded, so it is on the boundary Bias-free. (It should be said that the Gaussian distribution is very good for many symmetrical or nearly symmetrical real work; a bad work situation is the long-tailed distribution).

* *Learning and Planning in Complex Action Spaces* (arxiv.org)

https://arxiv.org/pdf/1705.05035.pdf

* *Meta Q-learning*

* *Relational Reinforcement Learning*

* *Relational Deep Reinforcement Learning* (2018) Current attempts on logic + RL only works on very simple toy experiments.

* *Diff-DAC: Distributed Actor-Critic for Average Multitask Deep Reinforcement Learning* 

* *Recurrent DDPG* RDPG比DDPG更难训练，更容易收敛到局部最优。但凡是带Recurrent的RL过程，其必须保存下整个trajectory用于训练（只保存每个transition的hidden state实验证明是低效的，且很难训练出来。）

### Asynchronous / Parallel

* *Asynchronous Methods for Deep Reinforcement Learning* 

* *A3C*

* *Recurrent Experience Replay in Distributed Reinforcement Learning* (R2D2)

* *IMPALA*

### Hierarchical

* *HIRO: HIerarchical Reinforcement learning with Off-policy correction*

* *Hierarchical Actor-Critic*

* *MCP: Learning Composable Hierarchical Control with Multiplicative Compositional Policies*

### Surveys / Tutorials

* *Abstraction for Deep Reinforcement Learning* (IJCAI 22') foundamental problem of reinforcement learning today.



* *https://yang-song.github.io/blog/2021/score/* An introduction to score matching model, Langevin dynamics and etc.

* *Deep Reinforcement Learning for Cyber Security* (2019) 

* *Deep Reinforcement Learning for Multi-Agent Systems: A Review of Challenges, Solutions and Applications* (2018) A method-based survey.

* *Multi-Agent Reinforcement Learning: A Report on Challenges and Approaches* (2018) Half a year earlier than the paper above (July vs. December). A problem-based survey.

* *Autonomous Agents Modelling Other Agents: A Comprehensive Survey and Open Problems* (2017) 

A thorough paper for agent modeling. Actually agent modeling makes no major progress until this year; it remains at feature extraction and response to statisitcal opponent models (including fictitious play) and rely on information of environment.

An intriguing idea is to see your opponent's **policy as an automaton**; however, calculating response and modeling of such automaton is NP-hard.

* *Is multiagent deep reinforcement learning the answer or the question? A brief survey* (2018) 

This is a well-written survey that can serve as a handbook. MARL until then can be divided into 4 directions: single agent RL algorithm under MA; communication protocols; cooperation and opponent modeling.

Training Competitive RL agents requires the prevention of overfitting between the agent and its opponents. Some common ideas include training a response to a mixture of policy, adding noises, and turn to game theory for help (e.g. self play / regret matching / (minimax/Nash/CoCo/...) Q-learning.

* *Tutorial on Variational Autoencoders* https://arxiv.org/pdf/1606.05908.pdf The origin of VAE.

* *A Survey of Learning in Multiagent Environments: Dealing with Non-Stationarity* (2017) 

Five categories (in increasing order of sophistication): ignore, forget, respond to target models, learn models, and theory of mind. 

* *Self-supervised Learning: Generative or Contrastive*

* *A Divergence Minimization Perspective on Imitation Learning Methods* (2019)

In this paper, the author proposes a f-max framework that unifies the two main components in imitation learning, which are behavior cloning and inverse RL. AIRL is an example of f-max, and f-max is a subset of GAIL. BC, AIRL and GAIL can all be unified into a minimization of some f-metric over the generated data and the expert dataset; it could be KL, reverse KL, JS or other things. One very interesting thing about this paper is that it proposes two hypothesis:

1) in MDP of usual interest, the reward function depends more on state than action;
2) Being mode-seeking is more beneficial than mode-covering, especially in the low-data regime.

* *Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems* (2020)

### Ad-hoc teammate

See Prof. Peter Stone's work. [Under construction]

### Evolutionary

* *Evolution-guided policy gradient in reinforcement learning* (NIPS 18')

* *Proximal Distilled Evolutionary Reinforcement Learning* (AAAI 20')

### Adversarial

* *Planning in the Presence of Cost Functions Controlled by an Adversary 03'*

Planning in a Markov Decision Process where the cost function is chosen by an adversary after we fix our policy.

This paper represents reward as a vector and rewrite Bellman Equation as EV+c>=0, where V is the visiting frequence of each state, E is the transition matrix of current policy, and c is the reward.

This paper uses the classical algorithm, *Double Oracle* to solve RL. It converges normal RL probelm to a matrix game with state visitation frequency times environmetn transition as payoff.

The double oracle requires an oracle best pure strategical response given any other side's mixed strategy. The algorithm will finally converge to minimal Nash equilibrium; but the response set is finite, and both sides can only choose strategy from a finite set. After choosing strategy, calculate the pure strategy response.

Then, put this response to the strategy set. Repeat the process until the size response set no longer grows.

More current work uses DRL to train the response as an oracle (See the *PSRO* paper).

* *Adversarial policies: Attacking Deep Reinforcement Learning*

* *A Study on Overfitting in Deep Reinforcement Learning* (2018)

noise injection methods used in several DRL works cannot robustly detect or alleviate overfitting; In particular, the same agents and learning algorithms could have drastically different test performance, even when all of them achieve optimal rewards during training. 

Soem current solutions include stochastic policy，random starts，sticky actions（repeat the last action with some probability）and frame skipping。

### Multi-agent: Emergent Communication

The classic model for MARL in emergent communication is Levis game, which consists of a speaker and a listener, where the listener must make choices according to a picture (or other input) that is only visible to the speaker. The speaker and the listener shares the reward if the choice is made correctly.

* *Emergent Communication at Scale* (ICLR 22' spotlight)

### Multi-agent: Credit Assignment

Note: this part is a bit out-of-date and needs update.

* *MAAC: Actor-Attention-Critic for Multi-Agent Reinforcement Learning*

* *DP(R)IQN* Improve on the basis of D(R)QN, and use an inference branch with softmax to take the opponent's policy into consideration.

* *COMA*

* *QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning*

* *QMIX*

QMIX is a decent, robust and tried-and-true baseline in cooperate MARL.

* *BiCNet: Multiagent Bidirectionally-Coordinated Nets Emergence of Human-level Coordination in Learning to Play StarCraft Combat Game*

 As the bi-directional recurrent structure could serve not only as a communication channel but also as a local memory saver, each individual agent is able to maintain its own internal states, as well as to share the information with its collaborators. computing the backward gradients by unfolding the network of length N (the number of controlled agents) and then applying backpropagation through time (BPTT) The gradients pass to both the individual Qi function and the policy function.
 
 * *MAVEN: Multi-Agent Variational Exploration* (2019)
Quote: "Single agent RL can avoid convergence to suboptimal policies using various strategies like increasing the exploration rate () or policy variance, ensuring optimality in the limit. However ... both theoretically and empirically, ... the same is not possible in decentralised MARL."

"The reliance of QMIX on epsilon-greedy action selection prevents it from engaging in committed exploration, in which a precise sequence of actions must be chosen in order to reach novel, interesting parts of the state space". 

* *Deconfounded Value Decomposition for Multi-Agent Reinforcement Learning* (DVD) (ICML 22')

### Partly Observable 

* *DRQN* 

Use LSTM for recurrent strategies. Recurrent strategies are generally harder to train and more prone to oscilliate.

* *DDRQN* 

A NN structure that solves communication problem in cooperative MARL. The first D represents "distributed" There are three major improvements in the paper:

1) take the agent's last action as the input of next step;

2) parameter sharing;

3) **do not use experience replay in non-stationarity environment**; use soft-update instread of duplicating between target net and eval net. 

* *RIAL&DIAL: Learning to communicate with deep multi-agent reinforcement learning*

The goal is to design an end-to-end learning protocol; RIAL is based on DRQN. DIAL's idea is to achieve *centralized training and decentralized execution* (a common paradigm in MARL) through the *continuous flow of gradients*.

DIAL allows real-valued messages to pass between agents during centralised learning, thereby treating communication actions as bottleneck connections between agents. As a result, gradients can be pushed through the communication channel, yielding a system that is end-to-end trainable even across agents. During decentralised execution, real-valued messages are discretised and mapped to the discrete set of communication actions allowed by the task.

* *DRPIQN* 

For POMDP，One training method is to maintain a network's "belief" in the hidden state (the other two common methods are actor-critic to give additional information to the critic during training, and LSTM to remember the history). In practice, an extra network branch is used to predict the opponent's actions.

* *RLaR: Concurrent Reinforcement Learning as a Rehearsal for Decentralized Planning Under Uncertainty* (AAMAS 13') 

RLaR is a method used to solve dec-POMDP. dec-POMDP is a special POMDP that requires all agents to share a global reward. **Dec-POMDP is NEXP-Complete.** 

RLAR is a method in which *all states are visible during training but not visible during execution*. 

It calls training a kind of "rehearsal". It is divided into two steps: the first step is to learn a policy in an environment with visible states; the second step is for the agent to build a predictive model through exploration, and according to the predictive model and the original policy, learning a working policy under partial observation.

* *Actor-Critic Policy Optimization in Partially Observable Multiagent Environments* (NIPS 18')

#### Action Decoder

* *Bayes Action Decoder*

* *SAD: Simplified Bayes Action Decoder*

* *VariBAD: A Very Good Method for Bayes-Adaptive Deep RL via Meta-Learning*

### Variance Reduction and Overestimation

* *Reward Estimation for Variance Reduction in Deep Reinforcement Learning* 

Techniques to reduce variance for DQN: average DQN and ensemble DQN. The former is to use the Q obtained by averaging the past several parameters to be updated in the Bellman equation, and the latter is to solve the problem of the former that causes the calculation cost to increase while the variance is reduced-while maintaining and updating k sets of parameters .

* *Issues in using function approximation for reinforcement learning* 

An very old article, answering the question why Q-value often overestimate; it is because the max operator in the Bellman equation will capture and amplify those Q value estimates that are too large (**E max_a Q(s, a)>= max_a EQ(s, a)**); 

This will lead to spending a lot of time exploring those (s, a) pairs that are actually bad. A key observation is that in the case of fixed reward and deterministic transfer, assuming that Q(s,a)-Q\*(s,a) is expected to be 0, the error caused by it is gamma\*(max_a Q-max_a Q\*), 

the expectation of this value is often greater than 0 (not strictly speaking, when all Qs are the same and the probability of falling on both sides of 0 is equal, the probability of the error being positive is 1-(1/2)^|a|; When Q is different but the probability of falling to both sides of 0 is equal, the contribution of those with a smaller Q* value will be smaller.) 

When the gamma is too high, it is easy to cause q-learning fail.

*Stochastic Variance Reduction for Deep Q-learning* (AAMAS 19’)

Uses SVRG in DQN.

### Safe RL

"Safe" is actually a complicated notion; it is closely related to constrainted RL, for safety concerns can be modeled as hard constraint. However, safe can also mean "not forgetting old knowledge", which is related to meta-learning / transfer learning. Or, safe can mean prevent triggering some "backdoor" policy.

See Garcia's *A Comprehensive Survey on Safe Reinforcement Learning* (2015) for a comprehensive survey on multiple definitions of "safe".

*Prof. Andrea Krause* at ETHZ is adept at this field.

* *Projection-based Constrainted Policy Optimization*

* *First order Constrained Optimization in Policy Space* An improvement of the paper above which does not require calculating 2nd-order derivation.

* *Robot Reinforcement Learning on the Constraint Manifold* (ICML 22')

* *Provable Defense against Backdoor Policies in Reinforcement Learning* (NeurIPS 22')

This work assumes the presence of "trigger", which is a amortizally bounded offset onto the observation such that the observation can be push into a "dangerous" subspace where dangerous policy hidden in this subspace is performed.

### State-Action Occupancy

* *A State-Distribution Matching Approach to Non-Episodic Reinforcement Learning* (ICML 22')

This work learns two sets of policy, one is called forward policy and the other is called backward policy. The forward policy optimizes the task reward; the backward policy tries to match the state occupancy (the ratio between expert and learner is estimated by an discriminator). One interesting finding of this paper is that sampling initial state from the expert state occupancy is much better than that from the initial state, which is why the backward policy is applied if the forward policy cannot achieve the goal within a certain number of steps. The backward and forward policy, thus, is updated iteratively, and every few steps, the discriminator is updated independently.

This somehow reminds me of iterative update in the Lagrange update of occupancy matching algorithm in imitation learning. 

The thought of this paper is very similar to the following paper: * *Go-Explore: a New Approach for Hard-Exploration Problems* (2019). In this paper, the agent is encouraged to "return to a promising state" that have given the agent intrinsic reward before exploring unknown states. The paper mentions a hypothetical problem in intrinsic-reward based exploration, where an agent tries to explore a labyrinth of two branches; it happens to jump from one branch to another before the end of the path is reached, and cannot return to the original state because the intrinsic reward is decreased with exploration. 

### Autonomous RL

* *AUTONOMOUS REINFORCEMENT LEARNING: FORMALISM AND BENCHMARKING* (ICLR 22')

Continual, non-episodic setting. Also the MDP need to be reversible (one connected component in MDP.)

Conventional RL algorithms substantially depreciate in performance when applied in non-episodic settings.

The author provides some benchmark for autonomous RL.

### Demonstration-Guided RL (and imitation learning)

* *Bridge Data: Boosting Generalization of Robotic Skills with Cross-Domain Datasets* (RSS 22')

TBD

* *Pre-Training for Robots: Offline RL Enables Learning New Tasks from a Handful of Trials* (2022)

TBD

* *Generalization with Lossy Affordances: Leveraging Broad Offline Data for Learning Visuomotor Tasks* (CoRL 22' Oral)

Imitation learning where the embedding of current state and embedding of "plans" (from start and goal image) serve as input to generate final policy; a novel imitation learning architecture. The planning in the embedding space is done by **affordance model**.



* *GNM: A General Navigation Model to Drive Any Robot* (2022)

It seems that it is just large visual dataset pretrain.

#### Affordance Model

A new model proposed by Sergey Levine in 2022.

* *Planning to practice: Efficient online fine-tuning by composing goals in latent space* (2022)

"planning over subgoals for a goal-conditioned policy"

* *What can i do here? learning new skills by imagining visual affordances* (ICRA 21')

learn skills with a small amount of online exploration on novel objects.

They use VQVAE as the embedding model. VQVAE is VAE with a codebook, using nearest neighbour. The gradient is stopped as nearest neighbour is not differentiable.

#### VAE-based skill abstration architecture

* *ASPiRe: Adaptive Skill Priors for Reinforcement Learning* (NeurIPS 22')

Bagging of SPiRL, unsurprising.

* *Skill-based Meta-Reinforcement Learning (SIMPL)* (ICLR 22' Spotlight)

Another following work of SPiRL, also done by Karl Pertsch's team in USC. They inherit the skill extraction module of SPiRL; the difference is that, after extracting skills with SPiRL, they try to learn a task encoder with meta-task information, and use both the output of this encoder as well as the "prior" in SPiRL to train a high-level policy (which outputs skill z) for a particular task with the policy decoder in SPiRL frozen; then, for the transfered task, they freeze both task-encoder and skill-decoder, and fine-tune high-level policy.

Step 1: SPiRL, skill encoder, skill decoder and prior that mimics skill encoder
Step 2: high-level policy with task-encoder and prior's input as the input and output z to skill decoder. The skill decoder is frozen.
Step 3: fine-tune high-level policy with task-encoder and skill-decoder frozen.

* *Demonstration-Guided Reinforcement Learning with Learned Skills* (SKILD)

The following work of SPiRL. It assumes that there are two sets of demonstrations: task-specific, which is the particular downstream's task's demonstration; and task-agnostic, which is related but not the particular downstream task's demonstration. It trains two set of priors using SPiRL and use a discriminator to combine them.

* *Hierarchical Few-Shot Imitation with Skill Transition Models* (2021)

This paper addresses the few-shot imitation learning problem by proposing a novel architecture called FIST. They first train a VAE to extract skills in the latent space, with LSTM as encoder and MLP as decoder; then they train a skill posterior that takes states in the first and the last step as input and minimizes the KL divergence between this prior and the encoder. (**Note this is a common architecture; the difference between this and SKILD is that they additionally take the last step's state as input in the posterior. But this is a core difference, as, quote, "conditioning on future will make it more informative."**) Also, they did not use two sets of models; they instead fine-tune the whole model learned on task-agnostic dataset with task-speicfic dataset.

In deploy time, this "state from future" is picked by a lookahead module where we find the closest state to the current state according to the distance metric. This distance metric is learned by optimizing an encoder using InfoNCE loss, such that states that are H steps in the future are close to the current state while all other states are further away.

* *Accelerating Reinforcement Learning with Learned Skill Priors* (CoRL 20')

The founding paper of this line of work, working on task-agnostic dataset alone.

#### Skill Diversity

* *Leveraging Approximate Symbolic Models for Reinforcement Learning via Skill Diversity* (ICML 22')



* *Policy Improvement via Imitation of Multiple Oracles* (NeurIPS 20')

* *Plan Your Target And Learn Your Skills: Transferable State-Only Imitation Learning via Decoupled Policy Optimization* (ICML 22')

Explicitly decouples the policy as a high-level state planner and an inverse dynamics model. This architecture is becoming popular recently; see FIST (ICLR 22') and CEIP (NeurIPS 22'). But its state planner is a parametric method, which tries to minimize the KL difference between the expert dataset and expected output. 

* *Internal Model from Observations for Reward Shaping* (2018)

Another work that uses a predictor for future state. Quite popular I suppose (also used in SAIL https://arxiv.org/pdf/1911.10947.pdf).

* *Discriminator-Weighted Offline Imitation Learning from Suboptimal Demonstrations* (ICML 22')

This paper tries to solve offline imitation learning problem where only a small task-specific near-optimal transition dataset from expert and a large task-agnostic (could be very) sub-optimal transition dataset. It proposes a generalized BC objective (NLL loss with some weight f(s,a)) that contains many prior work, such as the two works listed below (ICLR 21' / 2020). Different from prior work, this works trains a discriminator to get weights. To pick out the "good" transitions in the task-agnostic dataset, **Positive-unlabeled (PU) learning** is used. PU learning re-weights different losses for positive and unlabeled data to obtain an estimate of model loss on negative samples that is not directly available.

The policy and the discriminator are learned jointly. The discriminator takes logprob of current policy as extra input, as when the policy is optimal, the discriminator will receive additional learning signal. Now, this loss and the discriminator becomes a functional of policy \pi.

The policy pi, besides mimicking expert action, also tries to maximize the discriminator loss such that the discriminator is minimizing worst-case error and becomes more robust. This "maximizing" process comes from finding a function \pi s.t. the functional derivative of the functional mentioned above becomes 0. In this way, the proposede algorithm avoids mode collapse by transforming the max optimization problem to a new learning loss imposed on the policy.

 * *Behavioral cloning from noisy demonstrations* (ICLR 21') f(s,a)=\pi'(a|s), \pi is an old policy previously optimized with D_b.

* *Offline learning from demonstrations and unlabeled experience* (2020) f(s,a)=\[A^\pi(s,a) > 0\] eliminates samples that are thought to be worse than current policy.

* *Continuous Control with Action Quantization from Demonstrations* (ICML 22') 

This paper proposes AQuaDem, which learns a discretization of continuous action spaces which can be installed on RL with demonstrations, RL with play data and imitation learning. The agent learns a set of K actions for each state by minimizing a loss extended from BC loss, and only utilizes that set of actions; this can be understood in the perspective of Gaussian mixture. The multiple models are used for multimodality. A temperature controls the probability distribution; the lower the temperature is, the more likely the loss will only impose a single candidate action.

* *Learning latent plans from play* (CoRL 20') [TBD]

plan proposal & plan recognition

* *Align-RUDDER: Learning From Few Demonstrations by Reward Redistribution* (ICML 22')

Align-RUDDER is a improvement from RUDDER (see below), where the safe explorations and lessons replay buffer of RUDDER are replaced by demonstrations with high reward (which is a commonly used skill - yet is only with limited effectiveness), and the LSTM of RUDDER is subtituted by a profile model acquired from sequence alignment of the demonstrations. Align-RUDDER identifies (s,a)-pairs that are indicative for high reward, and redistribute reward to them, thus making the algorithm "almost greedy".

**Profile model** is commonly used in bioinformatics to score new sequences w.r.t. the aligned sequences; the conservation score indicates the degree of consensus reached in multiple subsequences and is used for reward redistribution. The intuition is: if every demonstration does the same thing here, then I should do the same thing. 

The reward redistribution using profile model can be done in 5 steps: 1) defining events in demonstrations (clusters of (s,a)-pairs) using successor representation which gives a similarity matrix based on how connected 2 states are given a policy. The clustering is done by affinity propagation. 

One possible setback of this method is that the consensus-based algorithm might not handle noisy environments. Also, affinity propagation could suffer the risk of creating a very large cluster. 

* *POSITIVE-UNLABELED REWARD LEARNING* (2019) 

PU learning is a way to do semi-supervised learning when there only exists positive and unlabeled data. The core insight of PU learning relies on an **accurate prior estimation**; if the proportion of positive and negative data is known, then we can assume that the average loss of unlabeled data is a weighted sum of average loss of positive and negative data. Thus, the loss of negative data (which is not present) can be substituted by the weighted sum of loss of unlabeled+positive data.

* *Learning from Demonstration: Provably Efficient Adversarial Policy Imitation with Linear Function Approximation* (ICML 22')

This paper studies GAIL in both online / offline settings with linear function approximation (both transition & reward are linear in the feature map), and propose some generative adversarial policy optimization theorem that has provable bounds.

* *Imitation Learning by Estimating Expertise of Demonstrators* (ICML 22')

The author proproses ILEED, in which an imitation learning framework that accounts for the varying levels of suboptimality in large offline datasets by leveraging information about demonstrator identities is designed. The identities is learned in an unsupervised manner. Expertise level of a demonstrator is calculated by the inner product of an embedding of state and some learned weight vector, normalized by sigmoid. The demonstrator's action distribution at a state can be described as a function of the expertise level; more specifically, it is interpolated between a theoratically optimal policy and random policy.

However, the optimal policy cannot be acquired directly; thus we need to learn best policy, state embedding in expertise level and weight in expertise level jointly.

One limitation of the theoretical part is that they assume all demonstrators explore all states with non-zero probability, which is not always the case in real-life applications.

* *DISCRIMINATOR-ACTOR-CRITIC: ADDRESSING SAMPLE INEFFICIENCY AND REWARD BIAS IN ADVERSARIAL IMITATION LEARNING* (2018)

* *An Imitation from Observation Approach to Transfer Learning with Dynamics Mismatch* (NeurIPS 21')

* *Wasserstein Adversarial Imitation Learning* (2019)

* *Primal Wasserstein Imitation Learning* (2020)

#### Imitation Learning Pretrain + RL finetune

* *Awac: Accelerating online reinforcement learning with offline datasets.* (2020)

* *Offline reinforcement learning with implicit q-learning*  (IQL) (2021)

* *Conservative q-learning for offline reinforcement learning* (CQL) (2020)

* *Jump Start Reinforcement Learning* (JSRL) (2022)

A reasonable starting policy does not by itself readily provide an initial **value function** of comparable performance. So how to jump-start **value-function** based method? One way to do this is to let the expert do most of the previous steps before success, and only let the exploration policy learn the last, short MDP. Then we gradually increase the MDP that the learner value-based algorithm needs to learn, until the RL agent learns a good policy for the complete task. It is some sort of curriculum learning, but with theoretical guarantees.

#### Theory on Imitation Learning

* *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* (AISTATS 11') [TBD]

* *A Reduction from Apprenticeship Learning to Classification* (NIPS 10') [TBD]

* *Error Bounds of Imitating Policies and Environments* (NeurIPS 20') [TBD]

* *Provable Representation Learning for Imitation with Contrastive Fourier Features* (NeurIPS 21') [TBD]

infinite horizon on why GAIL is better than BC (note: a paper in the same conference stated that BC is the minimax at finite horizon setting? Is there any contradictory between the two papers?)

* *The information Geometry of Unsupervised Reinforcement Learning* (2021)

**This work focuses on infinite horizon MDP.**

This paper studies unsupervised skill learning, which tries to learn a policy that maximizes the mutual information between state and some latent variable (expected difference of logprob between occupancy measure given and not given the latent vairable). This latent variable is the combination of policy and additional input. In this sense, skill learning is assigning different probabilities to different policies; most of the probabilities are 0. Those policies that are given non-zero probabilities are the "skills". This learning paradigm does not cover all possible optimal policies, but provides the best state distribution for learning policies to optimize an adversarially-chosen reward.

Geometry: if we view each occupancy measure on each state (0 to 1) as a dimension, then a policy is a point on the n-1 dimensional probability simplex. **For any convex combination of valid state occupancy masures, there exists a Markovian policy that has the state occupancy measure. Policy search happens inside a convec polytope.** For state-dependent reward functions, the final reward is the dot product between the state marginal distribution and the reward vector.

* *Toward the fundamental limits of imitation learning* (NeurIPS 20')

With offline dataset, for any learner algorithm, there exists a MDP instance where the regret is lower-bounded by min(H, |S|H^2/N), where H is the length of episode, |S| is the number of states, and N is the number of trajectories. On the other hand, behavior cloning can reach such level of suboptimality, which means in the worst case, all algorithms are as bad as behavior cloning.

The core of this proof is to convert possible loss into the discrepancy of the probability of visiting states, and subsequently invoking Ross's result (see DAGGER paper). For the states that are not visited, the best thing an expert could do is to act uniformly randomly.

Note: This paper does not take the similarity of states into account; it treats states as totally different, only with a label attached on each state.

* *Provably Breaking the Quadratic Error Compounding Barrier in Imitation Learning, Optimally* (2021) The extension of the work above. 

* *Error Bounds of Imitating Policies and Environments* (NeurIPS 20')
 
GAIL can achieve a linear dependency on the effective horizon while BC has a quadratic dependency. The effective horizon is 1 / (1 - \gamma).

**BC is minimax optimal in the offline setting, which implies no method is better than BC in the worst case.**

#### Others

* *Relay Policy Learning: Solving Long-Horizon Tasks via Imitation and Reinforcement learning* (CoRL 19')

Relay policy learning divide an episode into different episodes; then hierarchically, the short episodes are learned by imitation learning, and the long episode is learned by RL.

* *DemoDICE: Offline Imitation Learning with Supplementary Imperfect Demonstrations* (ICLR 22')

This paper assumes that we have two datasets, one is optimal and the other is of unknown degree of optimality. It tries to give a weighted sum to the minimization of KL divergence between current policy and optimal dataset & KL divergence between current policy and suboptimal dataset in behavioral cloning, with the Bellman flow constraint (see "tutorial" https://yuanz.web.illinois.edu/teaching/IE498fa19/lec_15.pdf for the definition of Bellman flow constraint).

The author then consider the dual problem of the objective, which is a bilevel optimization. The author proves that, with adequate derivation (including importance sampling to change the expectation and surrogate for numerical stability), the problem has a closed-form solution, from which we can extract the policy with the maximum entropy assumption.

**Occupancy measure of MDP** 

is the (stationary?) distribution of state-action pairs that an agent encounters when navigating the environment with its model.

They test their result on mujoco environment, such as hopper and cheetah. 
 
Note: this paper is somewhat questioned by a reader on openreview.

* *Reinforcement Learning with Sparse Rewards using Guidance from Offline Demonstration* (ICLR 22' spotlight)

This paper proposes LOGO, which is based on TRPO and limit the policy to be similar to demonstrations by constraints. It is surprising that this work does not mention either PARROT or SPiRL/SKiLD/FIST line of work. Since TRPO is not suitable for sparse reward (as the policy won't change much and it is hard to gain any meaningful reward signal), the author add a step after each policy update step with original reward. The new step minimizes the KL divergence between current policy and offline dataset on the offline dataset. To calculate the KL divergence, we try to estimate the ratio of \pi(s, a) for current policy and for offline dataset by fist training a discriminator, then use the trained discriminator's output as the ratio result, as the discriminator's form matches at the optimal point of the discriminator. **This technique is also known as distribution matching, which leads to many important progresses in imitation learning.**

The paper gives theoretical bounds on performance.

* *Learning to Weight Imperfect Demonstrations* 

This paper from ICML 21' gives weight for datapoints in GAIL. Through mathematical derivations, it gives an upper bound to the weighted objective which is equivalent to GAIL with another unknown policy that can be estimated by importance sampling.

* *Importance Weighted Transfer of Samples in Reinforcement Learning* ICML 18'

* *PARROT: DATA-DRIVEN BEHAVIORAL PRIORS FOR REINFORCEMENT LEARNING* See below "state-conditioned" section.

* *TRAIL: NEAR-OPTIMAL IMITATION LEARNING WITH SUBOPTIMAL DATA* (2021)

This paper solves the problem where we have an expert dataset and a sub-optimal dataset (traj with (s, a)-pairs) where the action is drawn **uniformly at random**. The authors propose that, even if the uniform task-agnostic dataset is highly unrelated with our current task, it serves as a way to build transition models of the world. Thus, the author propose to learn a factorized structure, one half for transition model (given current state S and current latent action space Z and map to a probability distribution of S), the other half for learning latent action space. The author gives a fancy result that their method can provably give a boost to RL process by utilizing an existing conclusion in behavior cloning and factorize their algorithm to multiple steps with essentially the same form of objective.

* *Provable Representation Learning for Imitation Learning via Bi-level Optimization* (ICLR 20')

This paper mentions demonstration with observation only. In such settings, one line of work is model-based; the other line of work is to minimize the difference between the state distributions and the expert policy under certain distributional metric. There are already several wprls that gives sample complexity bounds on meta-learning and multi-task learning.

### State-Conditioned 


The common point of the following three papers is: they train a high-level skill policy π(z|s) whose outputs get **decoded into executable actions using the pre-trained skill decoder**. You can say that RL outputs a *preliminary action*, or it is running on a different MDP which should be simpler, or view it from a hierarchical learning perspective, where z is a high-level action and a is a low-level one. The first paper is from the first perspective, while the latter two papers are from the second perspective.

* *PARROT: DATA-DRIVEN BEHAVIORAL PRIORS FOR REINFORCEMENT LEARNING*

This paper proposes a light-weight solution for RL pretraining and acquiring prior knowledge; "Light-weight" here means it only needs (s,a)-pairs of demonstrations in the past; it does not need either reward function or simulator of past environment (which is required by meta-RL). *Note that the demonstration must be near-optimal so as to learn anything without reward function.* The intuition of the "prior" knowledge is to reshape the MDP by altering action distribution with a function f which rules out useless random action and gives more weight to "useful" interactions with objects.

To learn such a function from demonstration, the author uses a **normalizing flow** with a Gaussian distribution as pre-transformed distribution and observation/state as input to maximize the likelyhood of demonstrated (s,a) pairs generated by the normalizing flow. A key property of normalizing flow is invertibility; with such property the RL agent can have full control over the action space (otherwise it can only control in a subspace of action space or has redundant and degenerated action space).

 There are already attempts in exploiting the property of MDP and ruling out "useless" high dimensions. Dina Katabi et al.'s work (Sample Efficient Reinforcement Learning via Low-Rank Matrix Estimation) rules out such dimensions by enforcing the low-rank property of the Q(s,a) matrix. This paper, though, does not explicitly models a simplified MDP; the simplified MDP is learned.

I think the proposed method is overall elegant and promising; however, there is only one experiment in the paper.

* *Accelerating reinforcement learning with learned skill priors* (CORL 20')

The idea of this work (SPiRL) is leveraged into the paper * *Demonstration-Guided Reinforcement Learning with Learned Skills*. 

This work defines defines **a skill as a sequence of H consecutive actions** where H is a hyperparameter. It uses the task-agnostic data to jointly learn 

(1) a generative model of skills p(a|z), that decodes latent skill embeddings z into executable action sequences a, and 

(2) a state-conditioned prior distribution p(z|s) over skill embeddings.


* *Demonstration-Guided Reinforcement Learning with Learned Skills*

This work is somewhat similar to PARROT, where the RL policy trained after deployment only outputs a "preliminary" action, and the trained prior outputs the final action. However, the problem setting is subtly different: PARROT only has task-agnostic demonstrations, while in this paper, we have both a large task-agnostic dataset and a small task-specific dataset.  

The author learns action prior as a convolutional network. 

This paper has a thorough and clear related work section, discussing imitation learning, demonstration-guided learning, online RL with offline datasets, and skill-based RL. 

Common **Behavior cloning** are either behavior-cloning(BC)-based or inverse-reinforcement-learning(IRL)-based. However, there is a question left unanswered by both the author of this paper and PARROT: while we know that BC and IRL are brittle, *why is action prior working better than imitation learning?* 

**Demonstration-guided RL** combines BC and RL; it has several forms, where BC is used for initialization, buffered trajectories or reward shaping. However, quote, "While these approaches improve the efficiency of RL, they treat each new task as an independent learning problem, i.e., attempt to learn policies without taking any prior experience into account."

**Meta-RL** (discussed by PARROT) requires simulator for each task in the task distribution, which is sometimes expensive.

**Online RL with offline dataset** has the cheapest source of augmenting data: task-agnostic datasets can be collected cheaply from a variety of
sources like autonomous exploration or human tele-operation, but it will lead to slower learning than demonstrations since the data is not specific to the downstream task.

**Skill-based RL** extracts reusable skills from task-agnostic datasets & learn new tasks by recombining them. In a sense, it's hierarchical (discussed by PARROT as **Hierarchical RL**): it divides the controller into high-level "strategy selector" and low-level "implementation of strategy". The strategy can either be learned interactively or by demonstration. However, it is hard to train (RL over RL policy!) and easy to collapse.

Past works includes VAE as action priors.

I think PARROT has better theoretical guarantee than this paper, as **normalizing flow is invertible** , which guarantees the agent full control over the action space. 


The prior learning phase of this paper is inspired from SPiRL. It has two parts: skill inference network and closed-loop skill policy. The skill inference network is state-conditioned, as contrary to SPiRL. The "skills" are randomly drawn and proposed into skill inference network with the output of embedding result z, and z with state s are fed into the low-level policy for a. The representation is optimized using variational inference.

the skill posterior will often provide incorrect guidance in states outside the demonstrations’ support. Therefore the principle of both utilizing task-agnostic and task-specific is as follows:

(1) follow the skill posterior within the support of the
demonstrations;

(2) follow the skill prior outside the demonstration support;

(3) encourage the policy to reach states within the demonstration support.

To determine whether a given state is within the support of the demonstration data, we propose to use a learned discriminator D(s) to answer this question. D(s) is a binary classifier that distinguishes demonstration and non-demonstration states and it is trained using samples from the demonstration and task-agnostic datasets, respectively.

### Agent Modeling for MARL

According to the survey *A Survey of Learning in Multiagent Environments: Dealing with Non-Stationarity* , there are five levels of opponent modeling: 1) ignore (with fixed algorithm), 2) update (adapt to the environment including opponent), 3) assume (e.g. minimax), 4) modeling (assume your opponent is naive and will not model you) and 5) theory of mind (your opponent may be also modeling you).  

#### Classical Modeling：Feature Engineering

* *Player Modeling in Civilization IV* a traditional feature-engineering work with handcrafted features. 

#### Policy Representation

##### Divergence-based

* *Learning Policy Representations in Multiagent Systems*

* *Modeling Others using Oneself in Multi-Agent Reinforcement Learning*

* *Opponent Modeling in Deep Reinforcement Learning*

* *Learning Actionable Representation with Goal-Conditioned Policies*

* *Learning Action Representations for Reinforcement Learning*

##### Encoding & Hidden State

* *Provably efficient RL with Rich Observations via Latent State Decoding*

#### Theory of Mind

* *Machine Theory of Mind*

* *Theory of Minds: Understanding Behavior in Groups Through Inverse Planning*

#### Society of Agents

* *Social Influence as Intrinsic Motivation for Multi-Agent Deep RL* 

Reward those actions that allow teammates to make different actions (giving information) under CFR. The author pointed out that if there is no such special reward, then it will fall into *babbling* awkward equilibrium (saying something useless that is taken care by neither agent) 

The article uses mutual information as a measure. In addition, the network trained for CFR actually also gives embedding to other agents. Another point is that this agent training is completely decentralized. In fact, I think the setting of firing beam is quite reasonable. The Folk Theorem guarantees that in repeated games of indefinite length, if everyone unites to prevent one person from living well, then a certain degree of cooperation can occur.

## Other

* *Mean-field MARL*  The limitation of mean-field MARL is that its application must satisfy the assumption that reward can be well-described by agent in the neighbouring field. 

* *https://zhuanlan.zhihu.com/p/146065711* A introduction of transformer and DETR.

* *Discrete Optimization: beyond REINFORCE* https://medium.com/mini-distill/discrete-optimization-beyond-reinforce-5ca171bebf17

* *Differentiable Top-k Operator with Optimal Transport*

* *LEARNING WHAT YOU CAN DO BEFORE DOING ANYTHING* (ICLR 19') 

Find a way to get a kind of embedding from the video. A more worthwhile idea is to use mutual information to measure the effect of one embedding and the other of two completely different expressions as an objective function.

* *A Structured Prediction Approach of Generalization in Cooperative Multi-Agent Reinforcement Learning*

* *Probability Functional Descent: A Unifying Perspective on GANs, Variational Inference, and Reinforcement Learning*

[Under Construction]

* *Variational information maximisation for intrinsically motivated reinforcement learning* (NIPS 15’)
 
In addition to proposing empowerment, an important reference point of this article is: if the function itself is difficult to optimize, try to derive a lower bound and then optimize its lower bound. 

For instance, in convex optimization, we sometimes optimize the envelope function and look for proximal mapping.

# Predict + Optimization / Differentiable Argmax

I have written a collection of comments on papers in this field, but I think the survey below does a much better job. See Harvard team (e.g. Dr. Bryan Wilder) and Prof. Amos's team for updates in this field!

* *End-to-End Constrained Optimization Learning: A Survey* https://arxiv.org/abs/2103.16378

For starters, you can read this paper: *Melding the Data-Decisions Pipeline: Decision-Focused Learning for Combinatorial Optimization* which contains a precise summary for the KKT-based method. This work is based on the classic OptNet paper *OptNet: Differentiable Optimization as a Layer in Neural Networks* (2017).

This website is also helpful: http://implicit-layers-tutorial.org/differentiable_optimization/

For linear objectives, there are some classic works such as SPO+ (*Smart "Predict, then Optimize"*).

There is a recent work on RL + (Predict + Optimization), which is NeurIPS 2021 spotlight: *Learning MDPs from Features: Predict-Then-Optimize for Sequential Decision Problems by Reinforcement Learning*. 

Also, there is another work that is not typical predict+optimization, but uses the idea of "end-to-end" learning to bridge the gap between prediction (of model in MBRL) and optimization (of final reward) in RL: *Value Gradient Weighted Model-based Reinforcement Learning*  (2021)  

Most of the works above are somewhat based on KKT conditions. For exact penalty method, see this paper: *A Surrogate Objective Framework for Prediction+Optimization with Soft Constraints*.

# Feature Extraction

A well-studied field in few-shot learning.


The two papers below both mentions the idea of "trainable weight of combination", which is in coherent with the paper *Not All Unlabeled Data are Equal*.

* *Selecting Relevant Features from a Multi-domain Representation for Few-shot Classification* (2020)

Note: I always feel curious about how "different" are the datasets will make the method of finding common feature extraction to fail. Is there any theoretical guidance?

selecting features from a fixed set is an easier problem than learning a feature transformation. This paper is pretty similar to the next one, where you first train one parametric network family to obtain multi-domain representations, then combine them with linear weights.

* *Learning a Universal Template for Few-shot Dataset Generalization* (2021)

This paper proposes a novel model, FLUTE, for few shot learning tasks. It aims to capture the universal template that represents multiple feature extractors. This architecture uses FiLM as its basis, and try to initialize a set of new FiLM parameters for each evaluation task when plugged into the universal template.

The "Universal Template" here is the ResNet-18 feature extractor. In short, you would like some parts of your NN to be fixed, which is the "universal template", while the other parts to be a "selector" on the basis of NN.

The new FiLM parameters comes from a convex combination of the training datasets' FiLM params, and the weight is the "compatibility", which is the output of a dataset classifier.

Step 1: Train the feature extractor jointly over all training datasets, as well as many different FiLM parameters.

Step 2: train a dataset classifier by randomly sampling a training dataset and try to minimize the cross entropy loss (as a supervised learning where THE dataset is a label). The classifier is fixed then.

Step 3: use the support set (i.e. training set of this subtask) to find the blending factor.

Step 4: with the trained extractor, use NCC to give the result. The probability of an example belonging to a class is proportional to the exponential of cosine similarity between the feature and the class centroid. Give argmax as result.

Note: NCC, **nearest centroid classifier**, is a classifier building upon an established feature representation. It computes the average of each sample's representation in a category, which is the centroid, and classify an object to a specific category with the nearest distance (e.g. cosine similarity).


 
<!--
(Idea: substituting FiLM by some flow?)
The discriminator is not simplified here somehow?
-->


* *FiLM: Visual Reasoning with a General Conditioning Layer*

FiLM is a structure proposed to solve the problem of visual reasoning. a single FiLM layer is a very simple structure: an affine combination of original input, where the affine coefficient is extracted from the NLP part. Intuitively, NLP part gets to directly alter the features of CNN with a good property (from their experiment) that add and sub operations can be taken in a word-embedding-like style. FiLM is a very simple yet effective architecture; it does not have much parameters and thus is cheap to train.

FiLM can be viewed as a generalization of Conditional Normalization (CN) methods. See this blog for a brief introduction of CN: https://blog.csdn.net/Arthur_Holmes/article/details/103934892

The name "Conditional" comes from the dependency of the coefficient of affine transformation on the naive batch normalization basis. It is first proposed, also, in a vision-reasoning paper (VQA: Visual Question Answering). Conditional normalization is also used across different categories, which is called categorical conditional batch normalization, as the different category of data is sometimes unsuitable to do normalization together.

# ML Theory

## Wasserstein distance

The limitation of Wasserstein distance is that the Wasserstein distance between two continuous distributions cannot be estimated by a polynomial number (w.r.t. number of dimensions) of samples; i.e. the empirical estimation is a very bad choice for Wasserstien distance between continuous distributions. 

* *Parametric Adversarial Divergences are Good Losses for Generative Modeling* (2017)

KL divergence, JS divergence,  TV and Chi-square can all be regarded as the f-divergence formulation; MMD is different. Interestingly, this paper claims that parametric Wasserstein distance (dual form with Lipschitz constraint) is better than non-parametric Wasserstein distance (e.g. Sinkhorn), which is overturned by later papers.
 
* *Generative model based on minimizing exact empirical Wasserstein distance* (2019)

A rejected paper by ICLR, of which the review is very helpful. This paper try to optimize Wasserstein distance by solving linear programming, and use plug-in estimators for continuous distributions (which is a bad choice).

* *Estimation of Wasserstein distances in the Spiked Transport Model* (2019) 

* *Faster Wasserstein Distance Estimation with the Sinkhorn Divergence* (2020)

A better (but still bad) empirical estimator of Wasserstein distance between continuous distributions; from exponential to dimension to exponential to dimension / 2. This estimator is unbiased; the plug-in estimator is biased.

* *Wasserstein GANs Work Because They Fail (to Approximate the Wasserstein Distance)*  (2021)

None of the WGAN variants successfully estimate the Wasserstein distance.

* *Kantorovich Strikes Back! Wasserstein GANs are not Optimal Transport?* (2022)

Similar to the last paper.

* *Fairness with Continuous Optimal Transport* (2021)

optimizing Wasserstein distance for continuous domain on the Hilbert space (which is essentially optimizing the coefficient of basis functions) with dual form. This is the mainstream method till now.

* *Online Sinkhorn: Optimal Transport distances from sample streams* (NeurIPS 20')

exponentially decay Sinkhorn algorithm.

* *Efficient Wasserstein and Sinkhorn Policy Optimization* (2021)

* *THE CRAMÉR DISTANCE AS A SOLUTION TO BIASED WASSERSTEIN GRADIENTS* (2018) Another "a little bit better" Wasserstein empirical estimator (unbiased).

* *Sinkhorn Distances: Lightspeed Computation of Optimal Transport* (2013)

Sinkhorn distance is an entropy-regularized Wasserstein distance, which can be quicker solved than Hungarian by block optimization. Wasserstein distance can be solved in either primal, semi-dual or dual form.

* *FAST SINKHORN I: AN O(N) ALGORITHM FOR THE WASSERSTEIN-1 METRIC* (2022)

* *Differential Properties of Sinkhorn Approximation for Learning with Wasserstein Distance* (2018)

* *Quantifying the Empirical Wasserstein Distance to a Set of Measures: Beating the Curse of Dimensionality* (2020)

Claim to beat the curse of dimensionality (?)

* *Exact rate of convergence of the expected W2 distance between the empirical and true Gaussian distribution* (2020)

* *An Efficient Earth Mover’s Distance Algorithm for Robust Histogram Comparison* (ECCV 06')

* *A Fast Proximal Point Method for Computing Exact Wasserstein Distance* (2018)

* *Stochastic Optimization for Large-scale Optimal Transport* (NIPS 16')

Optimizing semi-dual form on reproducing kernel Hilbert space (RKHS).

* *Curriculum Reinforcement Learning via Constrained Optimal Transport* (ICML 22')

People use KL divergence for interpolation between tasks as a curriculum in the past. This is problematic. The paper proves why and gives a solution that uses Wasserstein distance for interpolation between tasks.
