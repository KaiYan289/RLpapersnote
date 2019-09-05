# RL Papers Note
这是一篇阅读文献的简记。

OpenAI spinning up：https://spinningup.openai.com/en/latest/index.html

## Surveys and Books
 * *Deep Reinforcement Learning for Cyber Security, 19'*
 一篇讲述DRL在网络安全中应用的综述。
 * *Deep Reinforcement Learning for Multi-Agent Systems: A Review of Challenges, Solutions and Applications,18'*
 一篇DRL在MARL中应用的文章。
 * *Multi-Agent Reinforcement Learning: A
Report on Challenges and Approaches, 18'* 7月份的文章，比上面那篇12月的要早一些。把MARL的几个分支都讲到了。应该说这一篇是基于问题导向的（上一篇则是基于算法导向的）。
 * *Autonomous Agents Modelling Other Agents:
A Comprehensive Survey and Open Problems, 17'*
 一篇非常全面地讲述agent modeling的文章。实际上直到这一年，agent modeling一直没有什么很大的进展，停留在提取特征和对对手过去行为做统计（fictitious learning也算），比较依赖于环境本身的信息。另一个比较新颖的思路是把对手的policy看成自动机；很不幸的是，这样不能建模非常复杂的对手，因为问题关于状态是非多项式可解的。
 * *multi-agent systems algorithmic game-theoretic and logical foundations*
 一本涵盖了多智能体与算法、博弈论、分布式推理、强化学习、拍卖机制、社会选择等交集的作品。
 前面提到了一些关于**异步DP，multi-agent的ABT搜索**等内容。
 这里面提到了一些多人博弈、时序博弈中基本的概念，比如**extensive form 和normal form**。对于时序博弈，存在一个“不可信威胁”概念，就是说如果整个Nash均衡，在第一步一方打破Nash均衡后，另一方采取反制措施会让自己的reward收到损失，那么这就是“不可信”的，所以说这样的Nash均衡是不稳定的。于是提出**子游戏精炼纳什均衡**。还有**颤抖手精炼纳什均衡**，大概就是指在假设一定犯错概率的情况下达到纳什均衡。另外还有一个有意思的**无名氏定理**：如果无限次重复进行的游戏具有合适的贴现因子，同时所有人针对一个人时，会给这个人带来额外的损失，那么agent之间是可以合作的。
 言内行为，言外行为和言后行为；交流四原则（quality，quantity，politeness和relativity）。
* *Is multiagent deep reinforcement learning the answer or the question? A brief survey 18'* 这篇文章和那篇18年12月的文章一样都是可以当成工具书使用的精良survey。作者将MARL当前的工作分成了四个方向：研究single Agent算法在MA环境下的反应；沟通协议；合作和对他人建模。
Competitive RL最怕的就是和对手之间产生过拟合；为此常见的方法包括训练一个对mixture of policy的对策以及加噪声。对于可以建模成博弈论的情况（Normal/Extensive form），还有一个方法就是self play。
另外这篇文章也提出了一个好的解决思路：Robust Multi-Agent Reinforcement Learning
via Minimax Deep Deterministic Policy Gradient

* *A Survey of Learning in Multiagent Environments: Dealing with Non-Stationarity 17'*
five categories (in
increasing order of sophistication): ignore, forget, respond to target models, learn models,
and theory of mind. 
**Policy Generating Function**
## Security Games
### Abstract Security Games
  * *Improving learning and adaptation in security games by exploiting information asymmetry, INFOCOM 15'*
  考虑了在抽象security game且状态部分可见的情况下如何利用tabular minmax Q-learning（Q-learning变种）去学习。
### Network Security (Games)
  * *Defending against distributed denial-of-service attacks with max-min fair server-centric router throttles*
  网络攻击的经典模型之一：攻击者控制了几个初始节点（叶子）。向目标服务器发动进攻，而防守者控制着到目标服务器（树根）上某些必经之路的路由器，需要调整丢包比例让保证安全的同时最大限度让合法请求通过。这个设定被用在之后大量的DDOS相关论文中，也包括一些用MARL解决该问题的文章。
  * *Reinforcement Learning for Autonomous Defence in Software-Defined Networking, 18'*
  网络攻击的经典模型之二：攻击者控制了几个初始节点，并且向目标服务器发动进攻，每回合感染一些与上一回合节点相邻的服务器。防守者可以选择断开某些服务器连接、重连某些服务器、转移目标服务器数据到其他指定位置。
  * *Adversarial Reinforcement Learning for Observer Design in Autonomous Systems under Cyber Attacks*
  固定control policy，用self play同时训练adversarial和观察者。注意到这里controller是fixed的；训练的可以说是一个“输入扭曲器”和“输入矫正器”。文章以经典的pendulum做实验。另外，为了保证self play能收敛，文章使用了TRPO。
### Green Security Games
  * *Deep Reinforcement Learning for Green Security Games with Real-Time Information, AAAI 19'*
 对Green Security Games这种特殊的安全游戏引入了一种DRL解法。Green Security Game是一个面对偷猎行为建模设计的游戏，在2D gridworld上进行。游戏分为两方，一个是偷猎者，另一个是巡逻者。偷猎者可以四处移动，或是放下偷猎工具，它每回合有一定概率爆炸，若爆炸则收获正reward（巡逻者收获负reward），并消失；巡逻者可以拆除偷猎工具，或者是抓到偷猎者以得到正reward（对应的，偷猎者收获负reward）。游戏是partial observation的。游戏在巡逻者抓住偷猎者且场上没有偷猎工具时结束。DRL本身似乎没有什么特别的。

## Ancient RL
### Distributed Cooperation
 * * Hysteretic Q-learning* an algorithm for decentralized reinforcement learning in cooperative multi-agent teams
Hysteretic Q-learning是一种通过分布式训练得到一队能够合作的agents的方法。它起源于博弈论，主要研究了重复双人矩阵游戏。其实本质没有什么新东西，只不过调了调变好和变坏时的参数，使得q-value估计变高和变低时变化的速率不同。soft-update增加稳定性这个稍有常识的人都会看出的吧。This results in an optimistic update function
which puts more weight on positive experiences, which is shown
to be beneficial in cooperative multi-agent settings.
* *Lenient learners in cooperative multiagent systems*
也是一个研究cooperative的情况。文章非常短，只有3页。本质上是把Q-value方程的迭代变成模拟退火。不过呢，需要指出的是虽然方法看起来很trivial，但是它应该是有内在的道理的：这篇文章和上面的文章一样，试图解决cooperative情况下agent不会合理利用自己队友的动作。"Lenient learners store temperature values that are associated with state-action pairs. Each time a state-action pair is visited
the respective temperature value is decayed, thereby decreasing the
amount of leniency that the agent applies when performing a policy
update for the state-action pair. The stored temperatures enable
the agents to gradually transition from optimists to average reward
learners for frequently encountered state-action pairs, allowing the
agents to outperform optimistic and maximum based learners in
environments with misleading stochastic rewards".

### Adversarial RL
* *Planning in the Presence of Cost Functions
Controlled by an Adversary 03'* planning in a
Markov Decision Process where the cost function is chosen by an adversary after we fix
our policy
找这篇文章看起来算是找对了。一上来举的一个例子（固定放置sensor，最小化暴露时间）就和我的一个设想不谋而合。
它把reward表示为一个向量。把Bellman方程表示为EV+c>=0，V是每个状态的造访频率，E是policy所对应生成的矩阵（（1/1-gamma）-1）转移矩阵，c是reward向量。
文章提出来一种叫Double Oracle的算法。它首先需要将一般的RL问题转化到基于state visitation frequency向量与环境相乘作为payoff的矩阵博弈，然后用来解决矩阵博弈问题。所谓double oracle，是指给定行玩家/列玩家任意一方的mixed strategy，都可以瞬间求出另一方的best pure strategic response。不过，response的集合一直是**有限大**的（虽然最后会收敛到minimax nash均衡点）。
首先假设一开始双方都只能在一个有限的集合内选策略。然后计算双方的一个对之前这些策略元素的最佳mixed strategy（概率分布）。接下来假装对方会按照这个mixed strategy行事，再计算最优pure strategic response。然后将response加入决策集，重复上述过程直到决策集大小不再增加即收敛。

## Classical DRL
 * *DQN*
  Q网络的拟合目标是用Q网络自己的早期版本（即target net）用Bellman方程作为结果。另外Experience Replay把时序过程中的步骤拆分出来作为训练集也是一个经典操作。
 * *Implicit Quantile Networks for Distributional Reinforcement Learning* DQN系到18年的SOTA，但是似乎不能解决POMDP。
 * *TRPO* Trust Region Policy Optimization 15' 
  思路：要保证策略梯度得到的结果单调不降----->只要满足advantage>=0---->需要近似计算所以不能离的太远----->对KL散度有限制---->用KL散度的二阶展开近似KL散度变成凸优化问题。
 * *PPO* Proximal Policy Optimization 17'
  简单实用的正则项动态系数调整法（系数动态调整+clip），加正则项的方法都可以借鉴它。
 * *TRPO/PPO for POMDP*
 * *DDPG* DDPG是一种难以训练的方法。虽然理论上说DDPG可以适用于gridworld这样的低维度动作环境中，但是实验表明其表现和收敛速度远不如DQN。DDPG依然算是一种面对连续/高维决策空间的无奈之举。
 * *TD3* TD3 is a direct successor of DDPG and improves it using three major tricks: clipped double Q-Learning, delayed policy update and target policy smoothing.
 * *AC*
  Actor-Critic从本质上说是Policy Iteration的升级版。
 * *A2C*
 advantage Actor-critic
 * *A3C*
 在之前的基础上加入了一个异步更新。
 * *SAC*
 * *ACER: SAMPLE EFFICIENT ACTOR-CRITIC WITH
EXPERIENCE REPLAY , ICLR 17'*
 ACER是A3C对于off-policy的改编版本。**Controlling the variance and stability of off-policy
estimators is notoriously hard. Importance sampling is one of the most popular approaches for off-policy learning**。
这里面有一个非常重要的地方：许多Actor-Critic算法的实现实际上使用了replay buffer。从理论上说，这是不对的（因为梯度这个东西涉及到policy采取各个动作进行转移的概率，但是off-policy情况下actor输出的概率不再是原来那个了，对应的critic的输入会有变化（我自己在实现replay buffer的时候可以说是瞎搞，比如要在转移里随机加噪声直接给对应方向+1再归一化），应该要加重要性采样加以修正；但是重要性采样方差太大，所以有了tree back-up和retrace（lambda），以及ACER），但是从实践上说，一般是可以work的——只限于1-step TD-learning。在n-step TD-learning时，由于转移与policy高度耦合，很快就会出现巨大的误差。（一般好像也没人用buffer去做n-step TD-learning）也有观点认为，1-step TD-learning仅仅是让程序额外学了一些转移。另外，off-policy情况下buffer里的policy原则上不应该和现在的policy偏移太大。

### Partly Observable RL
* *DRQN* 把第一个全连接层换成了LSTM，其他的和DQN 完 全 一 致。
* *DDRQN*是一个用来解决MARL中团队合作__交流__的网络结构。第一个D代表distributed。文章中提到了三个主要修改：第一个是把上一个agent的动作作为下一个时间步的输入；第二个是所有agent参数共享；第三个是**对于不稳定环境不使用experience replay**。使用soft更新（就是说加权更新target network参数而不是直接复制）。另外实验比较有意思，把帽子和开关的智力题建模成了MARL问题。
* *RIAL&DIAL: Learning to communicate with deep multi-agent reinforcement learning* 目标是设计一个end-to-end的学习协议。RIAL是建立在DRQN的基础上的。DIAL大概的想法似乎是通过梯度的串联实现中心化训练、非中心化执行。
DIAL allows real-valued messages to pass between agents during centralised learning, thereby treating communication
actions as bottleneck connections between agents. As a result, gradients can be pushed through the
communication channel, yielding a system that is end-to-end trainable even across agents. During
decentralised execution, real-valued messages are discretised and mapped to the discrete set of
communication actions allowed by the task. 
architecture
独立的Q-learning指agent互相都把对方当成环境的一部分。
the space of protocols is extremely high-dimensional. 
“during learning, agents can
backpropagate error derivatives through (noisy) communication channels”
color-digit MNIST
中心化训练与去中心化执行。
* *RDPG* RDPG比DDPG更难训练，更容易收敛到局部最优。但凡是带Recurrent的RL过程，其必须保存下整个trajectory用于训练（只保存每个transition的hidden state实验证明是低效的，且很难训练出来。）
* *DRPIQN* 面对POMDP，有一种训练方法是维护一个网络对隐藏状态的“信念”（另外两种常见的方法分别是actor-critic给critic训练时额外的信息，以及LSTM记住历史）。虽然“信念”听起来很贝叶斯，但是实际上就是额外拉出一个网络分支用来预测对手的动作。
* *BAD* 
* *RLaR: Concurrent Reinforcement Learning as a Rehearsal for
Decentralized Planning Under Uncertainty, AAMAS 13'* RLaR是一种用来解决dec-POMDP的方法。dec-POMDP是一种特殊的MARL，它要求所有的agent共享一个global reward。Dec-POMDP是NEXP-Complete的。 **RLAR是一种认为训练时全部状态可见、执行时不可见的方法，它把训练叫做一种“rehearsal”，即排练。**它分为两步：第一步是在完全状态下学到一个policy；第二步是agent通过探索去建立一个预测模型，根据预测模型和原policy学到新的不需要完全可见就可以work的policy。
* *Actor-Critic Policy Optimization in Partially Observable Multiagent Environments, NIPS 18'*
## MARL
 * *MADDPG*
   经典的Multi-agent算法。本质上说，是DDPG的扩展；它利用centralized training在训练时为critic网络给出了额外的信息，而actor则不直接利用这些信息；最后测试时只使用actor网络决策。另外它为了防止competitive情况下的overfit，训练了一堆平行的参数每次均匀随机选择。
 * *R-MADDPG*
   R-MADDPG是MADDPG的一个拓展，它在Actor和Critic（主要是Critic）上增加了一个循环结构，用以储存过去接收的信息，处理依赖于时序的任务。实验结果表明，在critic上安装循环结构效果显著，而在actor上安装循环结构几乎没有效果。（虽然极其难训就对了）
 * *COMA*
 * *DP(R)IQN*在D(R)QN的基础上改进，用一个带softmax的inference分支去将对手的policy纳入考虑。
 * *Learning with opponent-learning awareness*
### Communication
 * *BiCNet: Multiagent Bidirectionally-Coordinated Nets
Emergence of Human-level Coordination in Learning to Play StarCraft Combat Game*
As the bi-directional recurrent structure could serve not only as a communication channel but
also as a local memory saver, each individual agent is able
to maintain its own internal states, as well as to share the
information with its collaborators.
computing the backward gradients by unfolding the network
of length N (the number of controlled agents) and then applying backpropagation through time (BPTT)
The gradients pass to both the individual Qi function and the
policy function.
## Distance of Distribution
  * *Wassenstein Reinforcement Learning*
  不可无一不可有二的文章。作者对把RL推广到这一量度上做了很多非常用心的理论推导；但是其真的推广到这一领域能比传统RL的表现好多少是存疑的。
  
## Soft Q-learning
算法产生的模型探索更充分，探索到有用的子模式更多。
* *Reinforcement Learning with Deep Energy-Based Policies ICML17’* https://zhuanlan.zhihu.com/p/44783057 有详细解说。
* *Multiagent Soft Q-Learning*
Relative overgeneralization occurs when
a suboptimal Nash Equilibrium in the joint space of actions is preferred over an optimal Nash Equilibrium because
each agent’s action in the suboptimal equilibrium is a better
choice when matched with arbitrary actions from the collaborating agents.
* *Balancing Two-Player Stochastic Games with Soft Q-Learning*

## IRL
 * *Inverse Reinforcement Learning 00'*
  IRL开山之作，用线性规划解决问题。首次提出reward是由一些“feature”的未知系数的线性组合构成的这个概念，在后面直到deep maxent的一系列IRL工作中发挥了非常重要的作用。
 * *Apprenticeship Learning via Inverse Reinforcement Learning 04'*
  需要明确的一点是：IRL的终极目标不是得到和原来完全一样的reward，而是可以在原reward上“表现得和原来一样好的”policy。
 * *Bayes Inverse Reinforcement
 Learning 08'*
  不需要每次求“最优”Policy了，比较优的就可以。
 * *Maximum Entropy Inverse Reinforcement Learning 08'*
  两个假设：一个是reward是手动设计的一些特征的线性组合，另一个是认为轨迹的概率分布（这是一个重要概念！）出现的概率是和e^reward成正比。这一篇我复现过，实际效果嘛……emmm。
 * *Maximum Entropy Deep Inverse Reinforcement Learning 10'*
  “Deep”是用来解决上一篇中特征提取问题的。上一篇认为reward是手动设计的一些特征的线性组合，这里就变成了网络自动从地图里提取特征做任意的组合。
 * *Guided Cost Learning 15'*
   
 * *A Connection Between Generative Adversarial
Networks, Inverse Reinforcement Learning, and
Energy-Based Models*
  GAN，能量模型和GAIL是相通的。
  
 * *Generative Adversarial Imitation Learning 16'*
  它是GAN在强化学习（实际上是模仿学习）领域的推广。分为两个不断重复的阶段：其中一个阶段是固定Generator优化Discriminator；另一个阶段是固定Discriminator以分类结果的对数似然作为Reward去训练Generator。注意学习architecture要画清楚数据流！
## Behavior Cloning

## Agent Modeling
### Classical Modeling：Feature Engineering
* *Player Modeling in Civilization IV*
这个大概还是使用传统的特征工程，需要手动设计特征。Agent Modeling只有在近两年概率空间投影的policy representation才摆脱了传统方法。
### Divergence-based Policy Representation
* *Learning Policy Representations in Multiagent Systems*
* *Modeling Others using Oneself in Multi-Agent Reinforcement Learning*
* *Opponent Modeling in Deep Reinforcement Learning*
### Theory of Mind
* *Machine Theory of Mind*
* *Theory of Minds: Understanding Behavior in Groups Through Inverse Planning*
## Self Play

### Miscellanous
* *Multiagent Cooperation and Competition with Deep
Reinforcement Learning*
这篇文章好像就是调了一下DQN的reward然后说明可以合作/竞争。没有什么太大价值，应用也不存在的。
* *Intrinsic motivation and
automatic curricula via asymmetric self-play*
LOLA算法：这个算法似乎是把别人期望的梯度下降也考虑进去了。但是这个算法连OpenAI自己都说方差极大，不稳定，计算极为复杂，显然不适合嵌套到另一个算法的循环里。

### Evolutionary
* *Competitive coevolution through evolutionary complexification*
进化算法。

### Game theory
* *Deep Reinforcement Learning from Self-Play in Imperfect-Information Games 16'*
简介见下面Game theory一节。

### Monte-Carlo Based
* *Mastering the game of go without human knowledge*
（未完待续）
* *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm 17'*
这一篇和上一篇基本上没有太大区别，只是上一篇应用的一个扩展。
* *Emergent Complexity via Multi-Agent Competition ICLR 18'*
这篇文章的核心思想是competition introduces a natural learning curriculum。
一般来说避免overfit的方法包括随机抽取一个集合作为对手（同样的思路也用在了MADDPG里），给轨迹的熵加正则项（一般用在机器人控制里面，假设移动服从某种概率分布，如高斯分布）。
这篇文章用分布式PPO训练。总的来说很吃数据量（想来也是，它完全不采取centralize的方法进行训练。）。
这篇文章的一个小trick是它使用自己的过去作为sample，另一个是开始阶段手动设计的curriculum。


## Game Theory
### Fictitious Play
Fictitious Play是一种寻找双人博弈中Nash均衡的方法。
* *Full-Width Extensive Form FSP*
理论上说extensive form也可以直接暴力展开为normal form然后使用FSP，但是那样效率太低，因为可能的决策会以指数级别增长。这篇文章证明了FSP也可以直接被用在extensive form上并且还给出了policy mix起来的方法：线性组合。
* *Fictitious Self-Play in Extensive-Form Games 15'*
这里就提出用网络近似best response（？）
* *Deep Reinforcement Learning from Self-Play in Imperfect-Information Games 16'*
它使用了NFSP。NFSP是Fictitious Self Play和Neural Network的结合，其实就是应用DQN和Behavioral Cloning分别做计算最优对策和计算对手当前对策的工作。
* *Monte Carlo Neural Fictitious Self-Play:
Approach to Approximate Nash Equilibrium of
Imperfect-Information Games 19'*
使用Monte-Carlo/异步方法做self-play，提高对局质量。要求所有的agent共享同样的参数。

### Counterfactual
* *An Introduction to Counterfactual Regret Minimization*
* *Deep Counterfactual Regret Minimization ICML 19'*
The goal of Deep CFR is to approximate the behavior of CFR without calculating and accumulating regrets at each infoset, by generalizing across similar infosets using function approximation
via deep neural networks.
 One method to combat this is Monte Carlo CFR (MCCFR), in which only a portion of the game tree is traversed on each iteration (Lanctot et al.,2009). In MCCFR, a subset of nodes Qt in the game tree is traversed at each iteration, where Qt is sampled from some distribution. 
* *Single Deep Counterfactual Regret Minimization 19'*

### Nash Equilibrium
* *Nash Q-Learning for General-Sum Stochastic Games 03'*  就是把Q-learning的最优下一步Q换成下一步大家都按Nash均衡行事。
* *Actor-Critic Algorithms for Learning Nash Equilibria in N-player General-Sum Games*
two desirable properties of any multi-agent learning algorithm
are as follows:
(a) Rationality: Learn to play optimally when other agents follow stationary strategies; and
(b) Self-play convergence: Converge to a Nash equilibrium assuming all agents are using the same learning algorithm

## Reward Shaping
* *Policy Invariance Under Reward Transformations： Theory and Application to Reward Shaping, ICML 99'*
注意区分reward shaping和正则项。这二者都是改变reward函数，但reward shaping是不改变最优策略的；而正则项是会改变最优策略的。
reward shaping的优点在于完全不会改变最优策略，缺点在于其形式必须满足一个特殊要求：对于任何(s,a,s')的转移，函数都可以写成cf(s')-f(s)，其中c为一小于1的常数。[对于只有位置具有重要性的gridworld就很有用了]
正则项的优点在于可以加入任何东西，但它不能保证最优解。

## Deception
* *Designing Deception in Adversarial Reinforcement Learning*
在传统框架下设计policy让agent学会骗人。这里面提到了一个观点：欺骗是一种能够引诱对手进入某种特殊policy的技巧，有利于把对手拉进自己熟悉的子游戏并战胜之。（类似的MADDPG和18年OpenAI一篇分布式PPO的文章也做了和欺骗相关的multi-agent实验）

## Active Learning
* *Active Classification based on Value of Classifier*
* *Learning how to Active Learn: A Deep Reinforcement Learning Approach*
active learning本来是一种通过分类器主动将未标记文本选择并送给专家标记的方式提高学习效率的方法。本来是将active learning用于NLP，这里把它建模成一个RL选样本作为policy的问题。而且是先在一个语言上学习policy再迁移到另一个语言上。把语料库打乱，然后认为面对一个句子有两个action：接受或不接受。如果接受，则update当前的classifier。注意到他们把当前classifier的状态建模成了一个state，所以可以认为训练是off-policy的。


## Overfitting Prevention
* *Protecting against evaluation overfitting in empirical
reinforcement learning, AAAI 11'*
* *Improved Empirical Methods in Reinforcement Learning Evaluation, 15'*
* *A Unified Game-Theoretic Approach to
Multiagent Reinforcement Learning, 17'*
g. 
When the model is fully known and the setting is strictly adversarial with two players, there are policy
iteration methods based on regret minimization that scale very well when using domain-specific
abstractions
文章基于经典的Double Oracle算法，提出了Policy Space Response Oracle（PSRO）算法。它将policy视为一种pure strategy用来玩double oracle。区别只在于，计算best response使用的是DRL方法。我们认为这是一种比训练对mixture of policy更好的解决方案：因为它保证了收敛到minimax nash聚恒。
Here, we introduce a new solver we call projected
replicator dynamics (PRD)
* *Robust Multi-Agent Reinforcement Learning
via Minimax Deep Deterministic Policy Gradient AAAI 19’* Its core
idea is that during training, we force each agent to behave
well even when its training opponents response in the worst
way.
核心的想法有两个：第一，为了保证鲁棒性，希望对手每一步即使走出针对我们使我们表现最差的表现能够最好（即minimax）；
第二，为了快速计算内嵌的min，用一个线性的函数去拟合内部，然后做一次梯度下降。
不过超参是不是有点多啊。
* *Actor-Critic Algorithms for Learning Nash Equilibria in N-player General-Sum Games*
有纳什均衡理论背书的方法在generalization和鲁棒性上多少会好一点。


## Novel Architectures
* *Structured Control Nets for Deep Reinforcement Learning*
把策略分为两个独立的流：线性控制和非线性控制。线性控制就是一个简单的矩阵乘法；非线性控制是一个MLP（不过不局限于MLP，其中一个实验就使用了**中央模式生成器**）二者的简单相加就是最后神经网络的输出。“直观地，非线性控制用于前视角和全局控制，而线性控制围绕全局控制以外的局部动态变量的稳定”。中文教程见https://www.jianshu.com/p/4f5d663803ba

* *Safe and efficient off-policy reinforcement learning NIPS 16’*
这篇文章提出了一种叫Retrace(lambda)的算法。它可以高效、“安全”地进行off-policy训练，并且它的方差很小。这是一个不需要GLIE前提就可以收敛的算法。GLIE(Greedy in the Limit with Infinite Exploration)，直白的说是在有限的时间内进行无限可能的探索。具体表现为：所有已经经历的状态行为对（state-action pair）会被无限次探索；另外随着探索的无限延伸，贪婪算法中ϵ值趋向于０。

* *Hindsight Experience Replay (HER)*
HER是一个框架，可以和其他off-policy框架配合使用。

* *Lenient DQN*
Lenient DQN中有几个重要的概念：lenient参数和温度。Leniency was designed to prevent
relative overgeneralization, which occurs when agents gravitate
towards a robust but sub-optimal joint policy due to noise induced
by the mutual influence of each agent’s exploration strategy on
others’ learning updates. 反过来说，competitive的时候恰好需要这种robust but sub-optimal的policy才有generalization。所以可以说competitive MARL比cooperative MARL更难。
Temperature-based exploration（模拟退火）
Auto-encoder是一种对付高维/连续S-A pair的方法。
The autoencoder, consisting of convolutional, dense, and transposed convolutional layers, can be trained using the states stored in
the agent’s replay memory [30]. It then serves as a pre-processing
function д : S → R
D , with a dense layer consisting of D neurons
with a saturating activation function (e.g. a Sigmoid function) at
the centre. SimHash [9], a locality-sensitive hashing (LSH) function,
can be applied to the rounded output of the dense layer to generate a hash-key ϕ for a state s. This hash-key is computed using a
constant k × D matrix A with i.i.d. entries drawn from a standard
Gaussian distribution N(0, 1) as
ϕ(s) = sдn
Aд(s)∈ {−1, 1}k
where д(s) is the autoencoder pre-processing function, and k controls the granularity suc
其实可以反过来想：cooperative的目标是减少其他agent反复的noise带来的sub-optimal,而competitive恰恰要利用这种noise，要把它适当加大到一个可以避免过拟合的程度。
* *Deep decentralized multi-task multi-agent reinforcement learning under partial observability, ICLR 17'*
 MT-MARL， CERT

