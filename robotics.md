# DDPG

## Central Take-away

This paper proposes a model-free and off-policy actor-critic deep reinforcement learning framework which learns deterministic policies in high-dimensional continuous action spaces.

## Strength

1) The experiment part is convincing, as the author tests their environments on numerous environments (25+) with either pixel input or low-dimensional vector input, and also conducted ablation tests on the variants of the baseline NFQCA using some ideas from the DDPG. 2) The paper is well-written and the motivation of the paper is clear: the author first investigates the possiblity of introducing deep learning into DPG, pointing out the problem of previous methods, and then adopting ideas from DQN (target network) and techniques recently (batch normalization).

## Weaknesses

1) The exploration issue. The exploration process is a noise of Ornstein-Uhlenbeck process, which is close to the current action and fixed. That means the wrong local optimal policy will be strengthened if the initialization is bad. 2) The overestimation issue of Q still exists; and since the algorithm is actor-critic, the error of critic will propagate to actor's gradient, which will cause suboptimal or even failing behavior.

# DQN

## Strength

1) The idea of the paper is simple and elegant, which can be straightforwardly implemented. It is directly derived from tabular Q-learning (and thus the correspondence and motivation are clear), and each step of the pseudo code can be implemented in a few lines of code. 2) The experiment result is concrete and promising, which overall reaches human-level or above on many tasks. The author tests seven different Atari games, and most of them reaches state-of-the art level performance.

## Weaknesses

1) The computational cost required by the proposed method is very high. In this paper, the author uses 10M frames to train the agent, which is a very large number even for the seemingly simple tasks by human. 2) Unavoidable overestimation of Q. When solving the Bellman equation, the correct value of Q* should be the maximum value of expectations with different actions, but the algorithm uses the expectation of maximum value. This will cause the overestimation of Q. 3) The proposed method can deal with neither continuous action space nor high-dimensional discrete space, which are very common in real life. The continous case is impossible to solve by the method directly, and high-dimensional discrete case will be prohibitly expensive for DQN.

# PPO

## Central Take-away

This paper inherits some ideas from the previous work trust-region policy optimization (TRPO) and proposes an RL algorithm where the gradient descent objective is to maximize the same objective as TRPO but substitutes the hard trust region constraint to penalty terms and add some clipping on the loss to keep the policy in the trust region.

## Strength
1) The formulation is very simple, intuitive yet effective. Unlike TRPO's complicated formulation of constrained optimization, PPO manage to keep the policy in the trust region by clipping and penalty term method, which is easier and cheaper to compute. 2) The experiment result is very solid; the authors test their algorithms on ~50 Atari games and get good results.

## Weaknesses

1) There is no theoretical guarantee on how "close" the algorithm can keep the policy in the trust region. Actually, there are a bunch of tricks for PPO and they exert quite an important influence on PPO's performance; see 2020 paper "Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO". 2) Since the policy must be kept in the trust region and the algorithm is an on-policy one, much trajectories are required to train the algorithm. Therefore, PPO is not suitable in the environment where one epoch is extremely expensive, yet one trajectory is cheap (e.g. a multi-agent system with a massive number of agents).

# ME-TRPO

## Central Take-away

The paper proposes a novel framework for model-based reinforcement learning (MBRL), which changes the RL algorithm from vanilla policy gradient (VPG) to trust region policy optimization (TRPO) and uses model ensemble for RL.

## Strength

1. There is a good ablation analysis in the supplementary material; the results clearly show that without model ensembling, the overestimation issue will be so severe as to harm the performance, and the tricks such as sampling techniques are carefully examined. 2.The real-time complexity, which is listed in appendix C, looks promising for a RL algorithm which usually takes a long time to train. The results look promising for its potential use in MBRL.

## Weaknesses

1. Insufficient novelty. The paper put two existing tricks into two orthogonal parts of MBRL framework (which are prediction and policy optimization) together, which is not novel enough from my point of view. (Imagine another group of people select another set of environment where PPO works better, and publishes another paper called ME-PPO.) 2. In the experiment details, the authors mentioned that they collect 3000~6000 timesteps for each environment, and uses a batch size of 1000. This choice is unusual as the batch size is very large compared to the total training set size, which is often 128/256 vs. 100k/million level of steps. I would appreciate an ablation analysis on the batch size of the algorithm.

# PETS

## Central Take-away

This paper proposes a new model-based reinforcement learning (MBRL) framework where the RL environment is predicted by neural networks that encodes distributions instead of points, and the policy optimization is solved by a model predicitve control (MPC) algorithm.

## Strength

1. A novel and convincing perspective on the prediction error. The authors propose to divide prediction error into two parts: aleatoric and epistemic, which accounts for inherent noise and undersampling; the author designs a prediction model to solve them respectively in the paper. 2. The authors look over a variety of baselines in their experimental part, which contains both model-based, model-free and hybrid RL algorithm, and the experiment results show that the proposed method is superior.

## Weaknesses

1. The author uses probabilisitc neural network (PNN) to address both aleatoric and epistemic problems. While this address the proposed problems, this solution requires more space and time complexity. 2. The author did not compare their solution to environment modeling with generative models such as GAN or VAE.

# Deep Visual Foresight

## Central Take-away

This paper proposes a method that combines deep action-conditioned video prediction models with model-predictive control (MPC) that trains robotic arms to successfully manipulate items with unlabeled training data and no special requirement for camera, initialization or sensors, and generalize to new items in test time.

## Strength

1. The computational cost is cheap. The authors mention that, quote, "All of the online model computations, including replanning, are done using a standard desktop computer and a single, commercially-availble GPU." This is a good news for academic researchers with relatively weaker computational resources, so they can reproduce and improve the proposed method (and while collecting data on 10 robots (fig.4) are difficult for most researchers, the authors make the dataset public so as to mitigate the burden). 2. The writings of this paper is clear and easy to follow. For example, in the experiment section, the author first states the question to prove in the whole section, and then evaluate and analyze their method to support the positive result that the proposed method has required property. There is also many figures and examples in the paper that helps comprehension.

## Weaknesses

1. The episodes of the experiment is rather short. The authors mention in page 5 that the episode of length is 15 and the planning horizon is 3, which means that the agent only needs to make decision for 5 times for each episode, which is a rather small number. 2. There is not enough qualitative analysis of the robotic arms' behavior. There is only one table with one column of data in the whole paper, which is the metric mean and standard deviation. There is no qualitative analysis on the loss function of prediction (not necessarily equals to performance but showing such results help the readers to understand the strength and potential shortcomings of the method), nor ablation analysis on the choice of predictive model and comparison with more complex scripts / solution with pretrained prediction model. 3. Self-occlusion issues can be problematic when trying to apply the proposed algorithm to other scenarios, e.g. grabbing items, as grabbing often requires some part of the arm to be in the back side of the item so as to hold it firmly.

# DAgger

## Central Take-away

This paper aims to solve the sequential prediction problem (in particular, imitation learning) and proposes a iterative algorithm where a sequence of policies is trained with a gradually expanding dataset by invoking the expert on the trajectories of the last acquired policy. This algorithm, dataset aggregation (DAGGER), under specific assumptions, is theoretically guaranteed and empirically verified to find a policy with good performance.

## Strength

1. The proposed method has a solid theoretical basis, which is largely absent in today's deep learning era. The author proves that DAgger has a better asymptotic behavior O(T\gamma) instead of O(T^2\gamma) with rigid and clear language of proof, though with the assumption of the form of loss and the bound of C. 2. The idea and algorithm implementation of DAgger is simple, yet the intuitive behin d DAgger is intriguing. On one hand, as the author mentioned in the paper, DAgger can be viewed as a follow-the-leader algorithm in hindsight; on the other hand, DAgger teaches the learning policy to "recover from mistake", which is why it is more robust and less likely to fail than other imitation methods in games like Super Tux Kart.

## Weaknesses

1. The DAgger may encounter the problem that the expert is too "strong" for the learning policy. Consider a very badly initialized learner; the initial policy is so bad that the (state, action) pair collected by expert policy is full of "recovery from mistake" instead of actually good moves from the beginning, which may hinder the capability of DAgger. 2.One limitation of DAgger is that it assumes the expert exhibits (almost) optimal behavior and that the demonstrations are abundant; the expert is always there to help. This is often not the case in real-life applications - after all, unless the expert is human (in which case collecting trajectories can be arduous), then why can't I simply duplicate/tune an expert instead of learning a new one from scratch?

# GAIL

## Central Take-away

This paper proposes an adversarial training framework for inverse reinforcement learning (IRL) that extracts policy from data directly instead of finding reward function explicitly first. In this framework, a discriminator and a generator is updated recursively, where the former tries to classify the trajectory generated by the generator and expert data, and the latter optimizes with TRPO rule where the cost function is the confidence of discriminator.

## Strength

1. Thorough theoretical insight. It is intuitive enough to propose a method that uses GAN on expert data in RL and adding an entropy term to avoid overfitting even without theoretical analysis; however, the authors carefully examine the duality of RL and IRL problem, and point out in an abstract manner that IRL with a psi-regularization function seeks a policy with closest occupancy measure to the experts' measured by some convex function. The selection of psi becomes the key difference of existing algorithms and a new one. 2. The paper is well-organized and pretty easy and natural to follow. The authors first bring forth the problem formulation in section 2, and then prove that the essence of IRL is dual of occupancy measure matching while underlining the importance of psi; then, by listing the possible choice of psi, the authors review existing algorithms with different choice, analyzing their shortcomings, and finally proposes their own algorithm and empirically validate it.

## Weaknesses

1. The experiments section is not detailed enough. From the authors' report, we can only acquire the fact that GAIL works better than other method in final performance. However, there is no analysis on detailed behavior of algorithms, e.g. an illustration of how algorithms succeed or fail on a particular setting, which could reveal more insights on the difficulties of different algorithms that will meet practically. 2. Adversarial training like GAN will often meet problems such as mode collapse and distribution shift, which could also be a problem for GAIL.

# Self-supervised Grasping

## Central Take-away

This paper is a experimental study which provides a very large scale dataset for object grasping with robotic arms with negligible human intervention and trains a grasper agent with it. To determine the available angle range for the arm to grasp things (which is used as label), the authors formulate the problem as a 18-way classification problem, discretizing the angle and turn regression problem into a 18-way classification problem.

## Strength

1. The insight of using CNN to predict grasping angles. This paper points out that CNNs are much better at classification than regression, and uses discretization based on such observation to achieve better results - which is pretty intriguing and shows that the researchers are experienced in this area. 2. This paper compares with various baselines, some are heuristic and others are learning-based, which proves the superiority of the proposed multi-stage learning paradigm. Moreover, the author uses many pictures as illustrations for the behavior of robotic arms, which builds a strong intuition into the reader's mind. Also, one thing that I particularly appreciate is that the authors uses clutters instead of isolated items in the test set to check the robustness of their model. 3. The description of how the authors build the dataset is clearly written and filled with details.

## Weaknesses

1. There is neither reference nor ablation test to the important claim "CNN are better at classification" that guides the architecture of this paper. 2. According to the experimental results, without data aggregation, the linear SVM will perform better than the proposed learning deep network. I would appreciate more if the author does a more thorough ablation test and applies the data aggregation on SVM to see the results.

# Planning to Explore

## Central Take-away

This paper proposes an unsupervised framework for exploration in reinforcement learning; unlike previous method which calculates state novelty after experiencing them, this framework uses planning to seek out maximum expected future novelty by combining the latent dynamic model of PlaNet and policy opmization of Dreamer for exploration and then train the downstream RL agent in "imagination" (i.e. the learned model).

## Strength

1. This paper has a thorough analysis for the behavior of the proposed algorithm. In section 5, it looks into the model's ability to cover zero-shot problems, trajectory needed to catch up with supervised oracle, generalization issue and the important claim that serves as the motivation of this paper: retrospective novelty let the model unable to train on states that they had never experienced. 2. The related work part is well-written, as it looks into many related but different area around this problem and clearly describe the relationship between the proposed solution and them. It is designed for better exploration, but task-agnostic; it is self-supervised, but model-based; it is model-based control, but scalable as it integrates model training and exploration; it actively explores, but solves much harder task than its predecessors.

## Weaknesses

1. It seems that initialization of the ensemble models for latent disagreement is important. Consider an extreme situation where all the models are initialized the same, then the disagreement reward will always be 0 and the proposed method collapses. Similarly, performance could vary to initialization of the ensemble models, which means variance of performance caused by random seeds could be high. 2. Some of the details of the algorithm, especially the downstream RL task part, can be further elaborated to make the reader clearer. For example, the algorithm pseudocode mentions "distill ... for sequences of D", yet there is no existence of the word "distill" in the rest of the paper; it uses two sections to introduce the combine of dreamer and PlaNet, but only in the last sentence of figure 2 does the paper mention "replacing novelty reward with task reward."

# Rapid Motor Adaptation

## Central Take-away

This paper proposes a framework for training quadruple robots to walk on varying terrain in simulation with fast adaptation on the environment upon deployment. The robot is RL-based and learned from latent vector extracted by ground-truth dynamics in training time; it then estimates the latent vector using interactions with the real-world environment when deployed.

## Strength

1. The idea of estimating environment dynamics from past trajetory is very simple yet effective, and intuitively sound. Consider a situation where the dynamic is known to be linear and has N degrees of freedom, then we only need about N steps (where 100 steps is 1 second, as RL agent is 100Hz when deployed) to get N equations to recover the dynamic. Although in real world the dynamic is more complicated, it is reasonable to believe that sufficient information can be acquired by a limited number of steps. 2. The experiment part is well-written and has detailed analysis on the robot's behavior. There are two points that I particularly appreciate: one is that the author set the robot configuration in testing wider than that in training, which validates the robustness of the method even if different environment itself is a suffice challenge; the other is that the author uses z_1 and z_5 to demonstrate the changes of vector and conducts ablation test, which helps the reader to see that the adapter which estimates the latent vector indeed helps.

## Weaknesses

1. There are too many hyper-parameters for reward. The reward consists of 10 terms and each with a different coefficient, which means that even grid search when applying on a new robot is hardly feasible; even if the provided result works good on quadruple robots, it may need major revision for robots with six legs, which may limit the use of this algorithm. 2. Although the training time is relatively efficient as a RL algorithm, the sample efficiency could be improved. 80000 is an extraordinary large batch size, and 1.2B steps is also a very large number even for model-free RL algorithm.

# OptNet

## Central Take-away

This paper aims to provide a differentiable optimization process (so that it can be intergrated into a neural network) for quadratic programming by considering the KKT condition at optimal point, and differentiate on both sides to get a linear equation systems whose solution leads to the desired gradient of estimated optimal solution with respect to estimated predictive parameters.

## Strength

1) This paper is novel and even ground-breaking in the differentiable optimization area (there are many papers following this work, such as https://arxiv.org/abs/1809.05504), and the proposed solution that differentiates on both sides of the KKT equations is very simple yet effective. Moreover, although the idea itself is simple, the mathematical derivations are rigorous and clearly written. 2) The related work is well written, revealing the relationship of this paper and many related areas of literature, such as argmin differentiation, which is very similar to what the authors do but is limited in specific settings. Also, a good number of the listed literature are applications in areas such as computer vision instead of pure optimization/theoretical papers, which shows the authors have very wide knowledge in the related area.

## Weaknesses

1) A KKT-based method like this requires some conditions to hold, which may hinder the generalization of the method (from quadratic programming): to solve the KKT equation system, strong duality must hold for our objective; also, the problem must be at least twice differentiable, which means this method cannot even directly solve linear programming, which is among the most common objectives in the field of optimization. 2) KKT-based method like this is fairly slow in practice (even admitted by the authors), as we need to solve the equation system for every pass, and this cannot be done by blackbox solvers such as Gurobi. (The most recent python package for OptNet is cvxpylayers built on CVXPY, an open-source python package). In fact, there is one paper (https://arxiv.org/abs/2006.10815) that points out the problem and gives a solution to reduce the dimensions; yet OptNet is still somewhat slow.

# FuN

## Central Take-away

This paper proposes a hierarchical RL framework which mainly consists of two components: Manager and Worker, where the manager assigns a directional goal for the worker to optimize. Beside the framework itself, the paper also uses an important technique called dilated LSTM, which allows gradient to flow through large hops in time by using separate groups of sub-states that were used iteratively from step to step.

## Strength

1. In the experiment section, the authors clearly proved that the goal learned by the manager is indeed explainable and corresponds to important subgoals in a task and by illustration. For example, in Montezuma's revenge in Figure 2, the author shows that every peak of goal count for the manager corresponds to a critical meaningful step for the gamem such as climbing the ladder and go across the river, and in figure 3, there clearly is a subgoal to get the oxygen for the agent. This supports the motivation in section 3.1 to stop gradients from worker to manager's goal. 2. The design of FuN model, from dilated LSTM to reformulated MDP and separate state extractor, clearly shows the effort of the authors to "make manager see further" than the original MDP problem, which is very natural as it is exactly why hierarchical RL is designed in the first place. On top of that, the authors considered the balance of workload between manager and worker, and crafted the reward to prevent the degradation of either worker or manager.

## Weaknesses

1. There is no guarantee that s_t will be a contract or Lipschitz-continuous mapping (though neural network often satisfy this, there is no effort such as regularizer on s_t to encourage this), as to prevent a trivial worker or manager, FuN uses cosine similarity between s_t to weigh the difference of goals. That is to say, FuN assumes that in the latent space of s_t, similar s_t values actually represents similar state - but there should be something like regularizers to ensure this. 2. As the author written in the related work part, building hierarchical agents has been widely studied in RL. However, there is only one baseline which is the LSTM in the experiment. The author should add more baselines from the hierarchical RL literature. For ablation analysis, dilated LSTM without the manager-worker architecture should become a separate option for the baseline.

# Neural Topological SLAM

## Central Take-away

Inspired by human remembering the topological structure of surrounding environment, the authors proposed a framework for better navigation that maintains a structural graph to help choosing directions, where each node is represented with a 360-degree view. It uses a planner for finding the shortest path on the maintained graph at a high level, inherits a neural SLAM for moving within a "node", and try to localize current node (if failed, then a new node is added to the graph connecting to the last node).

## Strength

1) The idea of combining maintaining a graph globally is intriguing and effective. As the related work have mentioned, there are many methods which tries to learn a representation of space by mapping it onto a latent space; however, we know little about the property of such latent space and can only control it by tuning hyper-parameters. Instead, planning from a growing graph has a deep origin from the traditional robotics literature, is not only interpretable, but also tried-and-true. 2) The experiment part is well-written and easy to follow. With illustrations, the authors do a good job in introducing the behavior of the algorithm in deployment time as well as its structure. There is also ablation analysis which highlights the importance of the graph structure, the score function and the effect of noise. I particularly appreciate that the authors analyzes the difficulty of exploring the "stop" action for RL-based algorithm, which brings insight to my current research.

## Weaknesses

1) There are too many moving parts. According to the author, there is a graph localization part, a geometric explorable area prediction part, a semantic score prediction part and a relative pose prediction part - and they need to work together with planners. The number of moving parts introduces instability in training; there are risks of "collapses", e.g. the localization part is particularly weak so that there are too many nodes in the graph. 2) The edge weights between the nodes are not 100% aligned with the optimal control. The paper mentioned that the edge weights between the adjacent nodes are assigned by the pose difference, which is composed of angle and distance. This, however, assumes that the angle adjustment can be done by turning to the same direction. A counterexample will be a big pillar at the end of a thin wall, where the robot must first turn away from the wall to manuever around the pillar and then turn to the opposite direction. The robot cannot realize the extra cost spent in this turning.

# ReLMoGen

## Central Take-away

This paper proposes a framework called ReLMoGen to combine motion generator (MG) and reinforcement learning (RL) for better robot controlling; it uses RL to generate high-level subgoals and solve "where" to move, while using MG to more efficiently solve "how" to move. The author proposes two variants of ReLMoGen, RelMoGen-D and RelMoGen-R, which are respectively based on DQN and SAC.

## Strength

1) Though there are many papers that tries to combine RL and traditional planning algorithms (and often as this paper, RL is used to generate higher-level policy), I particularly appreciate the insight mentioned by the author that "MP excels at solving 'how' to move, while RL is better at solving 'where' to move". Currently, the field where RL becomes most successful is playing games, where the decision-making process is complicated and hard to find a good general heuristic solution; however, one of the biggest difficulties for RL in real life is that RL often needs much more data to perform similarly to heuristic experts or traditional algorithms. 2) The experiment section is solid and informative: the authors conduct a thorough analysis on their algorithm, including stress test to transfer the algorithm to a new robot and semantic analysis on the potential action space, and answering the three leading questions in the paper well. There are also illustrations on the robot's trajectory and its feature map, which greatly helps the reader to grasp the behavior pattern of this algorithm.

## Weaknesses

1) The authors argue that, quote, "the main advantage of ReLMoGen is that it explores efficiently while maintaining high 'subgoal success rates' thanks to its embedded motion generators, resulting in stable gradients during training". However, the experiment result from Appendix C.b) (fine-tuning with a new embodiment) shows that at the beginning of the training, the subgoal success rates are relatively low - even lower than 20%. This somehow contradicts with the authors argument. 2) in SGP-R, half of the action dimensions are actually useless (either base subgoal or arm subgoal) when the selection dimension is choosing the other part to move, which could cause instability to RL training.

# DexNet 2.0

## Central Take-away

This paper proposes a network architecture to train a CNN to predict the success rate of grasping items with single-view (2.5D) point cloud as input, and utilizes it to train a robotic arm that can reliably grasp things. To train such CNN, the authors build an analytical graphical model of grasping items and make the dataset, Dex-Net 2.0.

## Strength

1. The design of the synthetic dataset is convincing; it is carefully designed and analyzed with graphical models between related variables. The authors first well-define the objective to optimize, which is the cross entropy between an analytical success metric S determined by a grasp configuration u=(center p, angle \phi), then consider a joint distribution that can be factorized in the graphical model. 2. The related work is well written: the authors carefully discussed a wide range of works that are related to this work, including grasp planning with analytic methods, empirical methods and methods that learns in real-life interaction; the computer vision techniques including semantic segmentation and sim2real works are also considered.

## Weaknesses

1. Throughout the dataset, the friction is fixed to be 0.6. This is OK for training in one paper for a particular environment, but the authors mark their dataset as a major contribution, and fixing the friction (instead of giving a pair of different hyperparameter for each pair of robotic arm and items in the dataset) limits the use of such dataset, as even if we assume each item has the same friction property, we cannot simply apply this dataset on another robot as the new robot's arm has different friction property. At least I believe explicitly giving the friction parameter for each pair of arm and item and model both in making the dataset (instead of only one friction) is important. 2. The design of grasping metric does not take the width of robotic hand into consideration. As the rightmost subfigure of figure 8 suggests, one common failure pattern is collision with the object (and if the hand can magically penetrate the unwanted part, then the grasp is indeed robust). This limits the actual utility of the dataset.

# Re-grasping using Touch

## Central Take-away

This paper proposes a end-to-end approach to train an robotic arm for grasping and re-grasping items, by using a neural network with ResNet that takes both input and concatenate in the middle of the neural network.

## Strength

1. The paper is well-written and easy to follow, with many details that makes this paper looks solid; for example, the authors make a detailed description on the hardware setup, with explanations of how they are making the choice (e.g. the GelSight sensor provides a regular 2D grid image format data). 2. The authors do a good job at the related work section; in this section, they carefully examined the existing works on grasping, especially those with tactile sensors. 3. The idea of this paper is simple yet effective; this work fills in a blank hole of robotic grasping with multiple types of sensory data, which is very practical in real-world applications.

## Weaknesses

1. Although the process of producing the training set is interesting, the author did not publish their dataset for training, which could have been an important benchmark for the researchers to expand on. This dataset should be useful for robots of other brands, as the author mentioned "after collecting data with a few different gels, changing the gels did not seem to significantly affect performance anymore". 2. The idea somewhat lacks novelty; a neural network with multiple input is taken, and after several layers, the extracted features are concatenated and sent into another MLP. 3. It could be better if the author try more types of polytopes to grasp; most of the items that is used for grasping, especially in the easy set, is cylinder. The author could try some items with triangle-like "skewed" surfaces.

# The Bitter Lesson + A Better Lesson

## Central Take-away

In the bitter lesson, Rich Sutton argued that 70 years of development in AI field proves the failure of algorithms utilizing human prior knowledge for better computational efficiency when confronted with statistical approaches based on better computing infrastructure and massive scale of computation. In a better lesson, Rodney Brooks raises an objection to Sutton, arguing that deep learning today is far from perfect, and the idea that AI only needs large amount of computational resource will bring many problems for future AI development.

## Strength

The two points following are one strength for each blog, one for "bitter lesson" and another for "better lesson": 1. I understand Sutton's point of view as I myself have reinforcement learning as my major interested area of research. Reinforcement learning has seldom been a hot area of research before the deep learning era, as tabular learning can only solve toy examples. Then all of a sudden, reinforcement learning becomes one of the brightest stars in today's AI frontiers, simply because the advent of big data era empowers RL for meaningful scenarios such as Atari games, Go and even Starcraft. 2. I indeed agree with the argument "deep learning solves everything is simply shifting human wisdom to paving ways for neural networks" in "the better lesson". In the early years of deep learning, big companies hired many people for labeling raw data, which contradicts with the mission of AI to liberate human wisdom from solving task-specific problems. And also, we see that specifically designed neural networks are much better than networks with trivial architectures, the former of which also needs human wisdom.

## Weaknesses

The two points following are one weakness for each blog, one for "bitter lesson" and another for "better lesson": 1. The bitter lesson only considers academia; "more computational resource works for everything" is very wrong when it comes to industrial research. There are many, many dirty data and problem-specific patterns that could largely hinder the performance of neural network, and they can only be solved by human. For example, when you are classifying birds using clips of their sounds, you will not be able to properly erase the background noise and found the right filter if you have no idea what birds sound like. 2. As a comment suggests at the bottom of "the better lesson", it would be unfair to compare the ability of neural network with a human brain, for the latter has evolved for hundreds of millions of years and is already "tuned" very well at the beginning of learning process. One thing worth noting is that even without human prior knowledge in a particular domain, the neural network is still "evolving" useful structures, e.g. pretrained ResNet18 as feature extractor.

