2021/7/27更新：原本的中文笔记可以在readme-legacy.md中找到，其主要写于2019-2020年。现在的英文版补充了一些文章，并去掉了极个别暴论。

2021/7/27 Update: The original Chinese notes can be found at readme-legacy.md; they are mainly written in 2019-2020. Current English version adds some papers, and remove several erroneous comments.

# 40 Useful Tips of the Day (updated 2021.7)

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

# Causal Inference

Thanks the causal reading group @ MSRA for their valuable opinions on causal inference! For a more thorough and professional summary, see *https://github.com/fulifeng/Causal_Reading_Group*.

Causality-related work includes Judea Pearl's *causal graphical model*, physics-based *structural causal model* (SCM), and statistical *potential outcome framework*.

Below is a collection of causal inference paper. For beginners, read *https://www.bradyneal.com/causal-inference-course*. For more papers, you can also read Prof. David Sontag's work. 

* *Understanding Simpson’s Paradox* 

(Copied from wikipedia) Simpson's paradox, which also goes by several other names, is a phenomenon in probability and statistics in which a trend appears in several groups of data but disappears or reverses when the groups are combined.

The paradox can be resolved when confounding variables and causal relations are appropriately addressed in the statistical modeling.[4][5] Simpson's paradox has been used to illustrate the kind of misleading results that the misuse of statistics can generate.

This paper (tutorial?) by Judea Pearl consists of two parts; one is the history and introduction of Simpson's paradox, 

the other is the "solution" of the paradox, which includes three parts: 1) why it is suprising, 2) when will it happen and 3) how to get the correct result. Rigorously solving the third point requires *do-calculus*, which is an important notion in today's causal inference work.

* *https://amlab.science.uva.nl/meetings/causality-reading-club/*

The discussion group of Prof. Joris Mooij's.

* *https://causalai.net/r60.pdf* 

This paper introduces the three layers of Pearl Causal Hierarchy(association, intervention, counterfact) with both the perspective of logical-probabilistic and inferential-graphical.

* *ICYMI: https://diviyan-kalainathan.github.io* A Causal discovery toolbox which can be referenced to.

* *WILDS: A Benchmark of in-the-Wild Distribution Shifts* A systematic distribution shift / domain generalization dataset.

* *Bühlmann : Invariance, Causality and Robustness* (projecteuclid.org) (2018 Neyman Lecture)

* *Treatment Effect Estimation Using Invariant Risk Minimization*

* *http://www.engineering.org.cn/ch/10.1016/j.eng.2019.08.016* A Chinese survey on causal inference.

* *https://qbnets.wordpress.com* Quantum Bayesian Networks.

* *Unsuitability of NOTEARS for Causal Graph Discovery*

* *Causality-based Feature Selection: Methods and Evaluations* 

A survey on causality-based feature selection; It also proposes CausalFS, an open-source package of causal feature selection and causal (Bayesian network) structure learning.

* *https://arxiv.org/abs/1906.02226* GraN-DAG (ICLR'20)

* *Masked Gradient-Based Causal Structure Learning* (2019)

* *A Graph Autoencoder Approach to Causal Structure Learning* (NeurIPS 19' workshop)

* *DAGs with No Curl: Efficient DAG Structure Learning* (2020)

* *From counterfactual simulation to causal judgement* https://escholarship.org/content/qt7kk1g3t8/qt7kk1g3t8.pdf

* *A polynomial-time algorithm for learning nonparametric causal graphs* polynomial-time nonparametric causal graphs (nonparametric & minimize residual variance; NeurIPS'20)

* *DiBS: Differentiable Bayesian Structure Learning* makes the non-differentiable score in score based method differentiable.

* *Causal Autoregressive Flows* (2021) An application of *Neural Autoregressive Flows* (2018).

* *D'ya like DAGs? A Survey on Structure Learning and Causal Discovery*

* *Towards Efficient Local Causal Structure Learning* (2021)

* *Efficient and Scalable Structure Learning for Bayesian Networks: Algorithms and Applications* (2020)

* *Ordering-Based Causal Discovery with Reinforcement Learning* (2021) order-basewd w/ RL.

Some miscellanous remarks:

1. Skeleton learning is a subfield of Bayesian network sturcture learning.

2. From the perspective of causality, identifying an edge is an output, while ruling out an edge is only a progress.

# Experimental Papers

* *Deep Reinforcement Learning and the Deadly Triad* (2018)

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

## Miscellanous

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

# Inverse RL

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

* *Maximum Entropy Deep Inverse Reinforcement Learning* (2010) 

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

## Multi-agent

* *MAGAIL：Multi-Agent Generative Adversarial Imitation Learning* (2018)

* *Multi-Agent Adversarial Inverse Reinforcement Learning* (ICML 19')

See MAAIRL.pptx in the github for details. This paper proposes a new solution concept called **logistic stochastic best response equilibrium** (LSBRE).

MAGAIL is MaxEnt RL + Nash Equilibrium; MAAIRL is MaxEnt RL + LSBRE.

* *Competitive Multi-agent Inverse Reinforcement Learning with Sub-optimal Demonstrations* (2018)

* *Asynchronous Multi-Agent Generative Adversarial Imitation Learning* (2019)

# Application

RL is currently not very popular in deployment of production (as of 2020/2021), as the sample efficiency are low and it may have no significant edge over expert systems and traditional analytical models in real-life. 

## Recommending Systems

RL is used in recommending systems due to its inherent **interactive and dynamic** nature; however, the sparsity of data is the greatest pain. Both GAN and supervised-learning with importance sampling are developed to address the issue.

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

# Miscellanous

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

## Active Learning

* *Active Classification based on Value of Classifier*

* *Learning how to Active Learn: A Deep Reinforcement Learning Approach active learning* 

It was originally a method to improve learning efficiency by actively selecting unlabeled text and sending it to experts for labeling by the classifier. 

Originally, active learning was used for NLP. Here, it is modeled as a problem of RL selecting samples as the policy. And it's about learning the policy in one language first and then migrating to another language. First disrupt the corpus, and then there are two actions in the face of a sentence: accept or not accept. 

If accepted, update the current classifier. Note that they model the current classifier's state as a state, so it can be considered that the training is off-policy.

## Overfitting Prevention

* *Protecting against evaluation overfitting in empirical reinforcement learning* (AAAI 11')

* *Improved Empirical Methods in Reinforcement Learning Evaluation* (2015)

* *Lenient DQN* 

Temperature and lenient parameter are the two important notions in Lenient DQN. 

Leniency was designed to prevent relative overgeneralization, which occurs when agents gravitate towards a robust but sub-optimal joint policy due to noise induced by the mutual influence of each agent’s exploration strategy on others’ learning updates. 

To some extent, competitive MARL is harder than cooperative marl, as competitive MARL needs robust but sub-optimal policy for better generalization, whereas in cooperative MARL overfitting to other opponents can be a good thing, and agents even need to reduce the noise by its teammates that causes sub-optimal behavior.

Temperature-based exploration auto-encoder is a method to deal with high-dimensional / continuous (s,a) pairs. The autoencoder, consisting of convolutional, dense, and transposed convolutional layers, can be trained using the states stored in the agent’s replay memory.

 It then serves as a pre-processing function д : S → R D , with a dense layer consisting of D neurons with a saturating activation function (e.g. a Sigmoid function) at the centre. 
 
 SimHash, a locality-sensitive hashing (LSH) function, can be applied to the rounded output of the dense layer to generate a hash-key ϕ for a state s. 

## Novel Architectures

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


## Model Reduction

* *DR-RNN: A deep residual recurrent neural network for model reduction*

For model reduction of large-scale simulations (such as physical models, etc.), there are three main ideas. The first is a simplified model based on physical formulas (so it is heavily prior-based); 

the second is a purely fitted black box model (similar to the imitation learning in the expert-apprentice problem); the third is a low projection based model Rank model (ROM). 

The premise of the third idea is to assume that the entire model can be represented by a low-rank linear expression. To get the projection base, use Galerkin projection (Galerkin method). 

Several main algorithms are: Proper Orthogonal Decomposition; Krylov subspace methods; truncated balanced realization. The article proposes a model reduction based on RNN.

## Behavior Cloning

* *Integrating Behavior Cloning and Reinforcement Learning for Improved Performance in Dense and Sparse Reward Environments*

* *Accelerating Online Reinforcement Learning with Offline Datasets*

## RL

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


### Multi-agent: Credit Assignment

Note: this part is a bit out-of-date and needs update.

* *MAAC: Actor-Attention-Critic for Multi-Agent Reinforcement Learning*

* *DP(R)IQN* Improve on the basis of D(R)QN, and use an inference branch with softmax to take the opponent's policy into consideration.

* *COMA*

* *QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning*

* *QMIX*

* *BiCNet: Multiagent Bidirectionally-Coordinated Nets Emergence of Human-level Coordination in Learning to Play StarCraft Combat Game*

 As the bi-directional recurrent structure could serve not only as a communication channel but also as a local memory saver, each individual agent is able to maintain its own internal states, as well as to share the information with its collaborators. computing the backward gradients by unfolding the network of length N (the number of controlled agents) and then applying backpropagation through time (BPTT) The gradients pass to both the individual Qi function and the policy function.

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

"Safe" is actually a complicated notion; it is closely related to constrainted RL, for safety concerns can be modeled as hard constraint. However, safe can also mean "not forgetting old knowledge", which is related to meta-learning / transfer learning.

See Garcia's *A Comprehensive Survey on Safe Reinforcement Learning* (2015) for a comprehensive survey on multiple definitions of "safe".

*Prof. Andrea Krause* at ETHZ is adept at this field.

* *Projection-based Constrainted Policy Optimization*

* *First order Constrained Optimization in Policy Space* An improvement of the paper above which does not require calculating 2nd-order derivation.

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
