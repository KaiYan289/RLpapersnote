# Imitation Learning from Observation (RL perspective), or Visual Imitation (robotic perspective)

[Under Construction]

1.Provable Representation Learning for Imitation Learning via Bi-level Optimization
The current paper proposes a bi-level optimization framework to formulate and analyze representation learning for imitation learning using multiple demonstrators
2.Recent Advances in Imitation Learning from Observation
3.Provably Efficient Imitation Learning from Observation Alone
Reduce Lfo into a sequence of minimax problems, one per time-step. Cannot share data across different time steps, not data efficient; only works for discrete actions; 
Still GAIL-based architecture
4.MobILE: Model-Based Imitation Learning From Observation Alone
Dynamics model learning by executing current policy online; design a bonus to incentivize exploration; do a imitation-exploration tradeoff surrogate to update policy. Inverse dynamics isn’t well defined except MDP dynamics is injective (no two actions could lead to the same next state from the current state)
Forward model is always unique and well-defined
With bounded loss, continuous action space
5.Sequential robot imitation learning from observations (cannot get pdf)
Invariant segment from visual observation sequenced together (skill?) motion2vec; image from same segment is together, from others are pushed away (contrastive learning)
6.Imitation from observation: Learning to imitate behaviors from raw video via context translation 
TRPO / guided policy search; context translation model for training an encoder
7.Learning invariant feature spaces to transfer skills with reinforcement learning
Learn shared feature space by means of proxy task  p(f(ss_p,r))=p(g(st_p,r))
Optimize \pi by directly mimicking the distribution over f(ss_p,r)
8.Reinforcement learning with videos: Combining offline observations with interaction.
RL get inverse dynamic model + human demo approx. imitation learning
9.Third-person imitation learning
Finally, the classes probabilities that were computed using this domain-agnostic feature vector are utilized as a cost signal in TRPO; which is subsequently utilized to train the novice policy to take expert-like actions and collect further rollouts.
GAIL-like state-matching-only approach
10.Playing hard exploration games by watching youtube
Temporal distance classification (how far two demonstration segments are apart?)
Cross-modal temporal distance classification (what are the important events?)
Use a learned auxiliary reward to guide exploration
11.Imitation learning from observations by minimizing inverse dynamics disagreement.
inverse dynamics disagreement minimization (the discrepancy between the inverse dynamics models of the expert and the agent)
Inverse dynamics prediction the inverse dynamics disagreement is defined as the KL divergence between the inverse dynamics models of the expert and the agent. Still GAIL-based but with a surrogate loss
the key challenge in LfO comes from the absence of action information, which prevents it from applying typical actioninvolved imitation learning approaches like behavior cloning [33, 3, 12, 31, 32] or apprenticeship learning [25, 1, 40]. Actually, action information can be implicitly encoded in the state transition (s, s0 ). 
12.Neural Task Graphs: Generalizing to Unseen Tasks from a Single Video Demonstration
https://rpl.cs.utexas.edu/talks/visual_imitation_learning_rss2020.pdf
One-of-a-kind, a symbolism work
Produce a graph, and then localize current node & 
13.A Geometric Perspective on Visual Imitation Learning (see following slides) (?)
14.Visual Imitation Learning for Robot Manipulation
MS thesis; learn a hand keypoint detection, feature, object detector
Goal-based reward; LQR (not deep RL!)
15.Cross-context Visual Imitation Learning from Demonstrations
16.Deep Visual Foresight for Planning Robot Motion
A predictor of future state + MPC
17.Visual Adversarial Imitation Learning using Variational Models
environment data collection, dynamics learning, adversarial policy learning
18.“Regression Planning Networks” NeurIPS 2019
Predict a sequence of intermediate goals conditioning on current observation, until the goal is reachable with a low-level controller
Divide-and-conquer thought
19.Imitating latent policies from observation
Latent policy and remapping; causal inference?
Given two consecutive observation, define a latent action z that cause the transition to occur (effect of policy and transition combined)
Retrieve latent policy; loss is the sum of next state MSE and causal model loss
20.Restored Action Generative Adversarial Imitation Learning from observation for robot manipulator (cannot get pdf) 
21.One-Shot Visual Imitation Learning via Meta-Learning
MAML-based
Concept2robot: Learning manipulation concepts from instructions and human demonstrations robotics+CV+NLP
22.State alignment-based imitation learning
Use some discriminator to discriminate random policy and expert policy; after the discriminator is trained and fixed, use RL to maximize discriminator loss by Wasserstein difference
Domain adaptive imitation learning (with state-action pairs of demonstration)
Aligning data by distribution match
23.Avid: Learning multi-stage tasks via pixel-level translation of human videos 
cycleGAN+stage learning cycleGAN aims to solve human-robot difference
MPC-CROSSENTROPYMETHOD to learn; state representation learning
24.Third-person visual imitation learning via decoupled hierarchical controller
a goal generator that predicts a goal visual state which is then used by the low-level controller as guidance to achieve a task (hierarchical)
25.Off-Policy Imitation Learning from Observations
f re-using cached transitions to improve sample-efficiency has been adopted by many RL algorithms  the LfO objective can be optimized by minimizing the RHS of Eq (4).
26.One-shot imitation from observing humans via domain-adaptive meta-learning.
Data from human are observation-only, data from robot is state-action pair (few)
Meta-learning-from-single-observation; need to train on various tasks
Predict action vs robot demo action
27.One-shot learning of multi-step tasks from observation via activity localization in auxiliary video
Meta-learning-from-single-observation; need to train on various tasks
Activity localization + reward function inference + RL
28.Extrapolating beyond suboptimal demonstrations via inverse reinforcement learning from observations
T-REX takes a sequence of ranked demonstrations and learns a reward function from these rankings that allows policy improvement over the demonstrator via reinforcement learning
T-REX has two steps: (1) reward inference and (2) policy optimization
Given the learned reward function rˆθ(s), T-REX then seeks to optimize a policy πˆ with better-than-demonstrator performance through reinforcement learning using rˆ
29.adversarial skill networks: Unsupervised robot skill learning from video. 
Adversarial Skill Networks yield a distance measure in skill embedding space which can be used as the reward signal for a reinforcement learning agent for multiple tasks.
Hierarchically decoupled imitation for morphological transfer. Somewhat related; embody AI embedding
30.State-only imitation learning for dexterous manipulation
inverse dynamics model.
31.Generative Adversarial Imitation from Observation GAIL with observation alone (GAIfO); earliest work of this kind
32.Unsupervised perceptual rewards for imitation learning
Design a rewards function: unsupervised discovery of intermediate steps, feature selection, real-time perceptual reward
Feature selection is pre-deep era method; manual design
Still needs some kinetshetic demonstration
33.XIRL: Cross-embodiment Inverse Reinforcement Learning
leverages temporal cycleconsistency constraints to learn deep visual embeddings that capture task progression from offline videos of demonstrations across multiple expert agents, each performing the same task differently due to embodiment differences. 
34.Time Contrastive Networks
Goal-based reward, mimicking third-person action; the point is state embedding, and mainly solve embody difference
35.Optiongan: Learning joint reward-policy options using generative adversarial inverse reinforcement learning
use GAIL, remove action, but use multiple discriminators and use a policy over options to combine them
36.Learning Generalizable Robotic Reward Functions from “In-The-Wild” Human Videos
Learn a similarity score as a reward between video clips, which can be used in VMPC
Massive task-agnostic dataset + small task-specific dataset
Video clip embedding: similarity score as RL function
Contrastive learning; with reward function, use VMPC (visual model predictive control)
In robotics, a good predictive function of future is suffice for planning (see MPC)  
37.Learning human behaviors from motion capture by adversarial imitation
Directly use GAIL but remove action; may be problematic
38.Behavioral cloning from observation
Learn inverse dynamics model from interaction
39.Zero-shot visual imitation
Forward consistency model; need stepping at arbitrary place?
40.Visual Imitation Made Easy
Label action by off-the-shelf structure from motion (Sfm); otherwise just a simple BC with data augmentation
41.Hybrid reinforcement learning with expert state sequences
Update a action inference model; see (s, a, s’) transition as a 3D tensor; pseudolabel the expert action and update agent policy based on hybrid loss (A2C loss + BC loss)
42.Generalizable Imitation Learning from Observation via Inferring Goal Proximity
Learn goal proximity and try to maximize it.
43.To follow or not to follow: Selective imitation learning from observations. (Reachability)
if the agent reaches the sub-goal (i.e. |o τ g − ot+1| < ), the meta policy will pick the next sub-goal. Otherwise, the low-level policy repeatedly generates actions until it reaches the sub-goal.
Perceptual reward functions (learned reward function) just normal reward function + DQN; no imitation learning
Learning grounded finite-state representations from unstructured demonstrations learn from unsegmented, possibly imcpmplete, and may originate from multiple tasks or skills demonstration
44.Unsupervised perceptual rewards for imitation learning (learned reward function)
Discovery of intermediate steps
Train a classifier for each step which combined to produce a single reward function per step prior to learning; the reward function is used by a real robot
45.OIL: Observational Imitation Learning
Follow leader in value function; need interaction with several suboptimal policies
46.The Surprising Effectiveness of Representation Learning for Visual Imitation
Visual embedding +Non-parametric: nearest neighbour matching + locally weighted regression
47.Combining self-supervised learning and imitation for vision-based rope manipulation
Explicit prior in CEIP; inverse dynamic model
Predict the action with current state and next state in the demonstration; train with adjacent demo frame
48.RIDM: Reinforced inverse dynamics modeling for learning from a single observed demonstration.
Explicit prior like our method on one trajectory
49.Learning Navigation Subroutines from Egocentric Videos
Learning inverse model from random interaction + pseudo-labeling egocentric expert videos
50.Semantic Visual Navigation by Watching YouTube Videos 
Pseudo-labeling inverse dynamic model
51.Imitation learning from video by leveraging proprioception
Plain AIRL without action but with multiple (two) actions

![image](https://user-images.githubusercontent.com/30424816/174714876-576c9872-a310-4656-a4d8-419765966f4b.png)
