# 一些实验心得
1.A2C/PPO很难处理mountain car（如果不加reward shaping或者延长episode），因为reward太稀疏了。

2.exploration是很重要的事情，同样一个环境让非法动作原地不动&随机到一个方向&给不给巨大reward，A2C的performance会有很大差距。一个神奇的情况是，在我自己的setting里地图边界出去时“保持不动”和中心“随机”都会让performance好很多。

3.对PPO来说做reward归一化是很重要的。

4.（按原文实现的）deep CFR一般不太能做步数很多（即树很深）的算法，因为深度增加导致需要遍历的状态数指数级扩大。德州扑克虽然信息集数量非常大，但是每局游戏的博弈树本身并不深。 

5.Recurrent DDPG的表现并不可靠，至少很难调。Recurrent MADDPG更是如此（MADDPG本身就不好调）。Recurrent TD3也是；能不用recurrent的情况尽量别用。

6.不要在pycharm调试的watch里放一些可能会改变全局变量的函数，这会改变程序的行为（比如有一个函数的作用是将全局变量+1，那么这样的函数就不要放在调试的观察里，否则会导致全局变量的值异常）

7.注意torch.tensor的类型。如果torch.tensor是整数类型的，拿它和常数比较的时候可能会把常数给round down掉！所以就会造成tensor(\[0\]) < 0.5为假这样的奇怪事件。

8.如果有的时候整个实验不work，不妨fix一些要素试一试。比如off-policy情况下两步动作训练不出来可以考虑先把第一步fix住看第二步是否能训练出来；图片当训练集不work就先拿一张图片当训练集看是否能收敛到过拟合，如果不是则说明代码有问题。

# Predict + Optimization Papers Note
TODO: 一个简单的预测优化文献小survey。
* *SPO+*
* *Melding the Decisions pipeline, AAAI 19'*
* *Optnet*
* *Direct Loss Minimization*

# RL Papers Note
这是一篇阅读文献的简记。注释仅供参考（从后来的观点看有些解释不太对；解释仅供参考）。

OpenAI spinning up：https://spinningup.openai.com/en/latest/index.html

## Sequential Decision Making
* *A Survey of Multi-Objective Sequential Decision-Making*


### Multi-arm Bandit
* *Multi-armed bandits with switching penalties*
* *Learning in A Changing World: Restless Multi-Armed Bandit with Unknown Dynamics*
* *Mortal Multi-Armed Bandits* 会消逝的摇杆

## Meta Learning Survey
https://arxiv.org/pdf/1810.03548.pdf
* *https://lilianweng.github.io/lil-log/2018/11/30/meta-learning.html* 优质meta learning综述。

## Surveys and Books
 * *Deep Reinforcement Learning for Cyber Security, 19'*
 一篇讲述DRL在网络安全中应用的综述。
 * *Deep Reinforcement Learning for Multi-Agent Systems: A Review of Challenges, Solutions and Applications,18'*
 一篇DRL在MARL中应用的文章。
 * *Multi-Agent Reinforcement Learning: A Report on Challenges and Approaches, 18'* 7月份的文章，比上面那篇12月的要早一些。把MARL的几个分支都讲到了。应该说这一篇是基于问题导向的（上一篇则是基于算法导向的）。
 * *Autonomous Agents Modelling Other Agents:
A Comprehensive Survey and Open Problems, 17'*
 一篇非常全面地讲述agent modeling的文章。实际上直到这一年，agent modeling一直没有什么很大的进展，停留在提取特征和对对手过去行为做统计（fictitious learning也算），比较依赖于环境本身的信息。另一个比较新颖的思路是把对手的policy看成自动机；很不幸的是，这样不能建模非常复杂的对手，因为问题关于状态是非多项式可解的。
 * *multi-agent systems algorithmic game-theoretic and logical foundations*
 一本涵盖了多智能体与算法、博弈论、分布式推理、强化学习、拍卖机制、社会选择等交集的作品。
 前面提到了一些关于**异步DP，multi-agent的ABT搜索**等内容。
 这里面提到了一些多人博弈、时序博弈中基本的概念，比如**extensive form 和normal form**。对于时序博弈，存在一个“不可信威胁”概念，就是说如果整个Nash均衡，在第一步一方打破Nash均衡后，另一方采取反制措施会让自己的reward收到损失，那么这就是“不可信”的，所以说这样的Nash均衡是不稳定的。于是提出**子游戏精炼纳什均衡**。还有**颤抖手精炼纳什均衡**，大概就是指在假设一定犯错概率的情况下达到纳什均衡。另外还有一个有意思的**无名氏定理**：如果无限次重复进行的游戏具有合适的贴现因子，同时所有人针对一个人时，会给这个人带来额外的损失，那么agent之间是可以合作的。
 还讲到了一些人类学内容，包括言内行为，言外行为和言后行为；交流四原则（quality，quantity，politeness和relativity）。
* *Is multiagent deep reinforcement learning the answer or the question? A brief survey 18'* 这篇文章和那篇18年12月的文章一样都是可以当成工具书使用的精良survey。作者将MARL当前的工作分成了四个方向：研究single Agent算法在MA环境下的反应；沟通协议；合作和对他人建模。
Competitive RL最怕的就是和对手之间产生过拟合；为此常见的方法包括训练一个对mixture of policy的对策以及加噪声。对于可以建模成博弈论的情况（Normal/Extensive form），还有一个方法就是self play。
另外这篇文章也提出了一个好的解决思路：Robust Multi-Agent Reinforcement Learning
via Minimax Deep Deterministic Policy Gradient

* *Tutorial on Variational Autoencoders*，https://arxiv.org/pdf/1606.05908.pdf

* *A Survey of Learning in Multiagent Environments: Dealing with Non-Stationarity 17'*
five categories (in
increasing order of sophistication): ignore, forget, respond to target models, learn models,
and theory of mind. 
**Policy Generating Function**

## Graph Neural Networks  
  * *GRAPH CONVOLUTIONAL REINFORCEMENT LEARNING， ICLR 20'* 学长的文章。

### Abstract Security Games
  * *Improving learning and adaptation in security games by exploiting information asymmetry, INFOCOM 15'*
  考虑了在抽象security game且状态部分可见的情况下如何利用tabular minmax Q-learning（Q-learning变种）去学习。
  * *A survey of interdependent information security games*

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
关于Green security games，CMU的Fang Fei在这方面做的工作是最多的。
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
* *Adversarial policies: Attacking Deep Reinforcement Learning*
* *A Study on Overfitting in Deep Reinforcement Learning 18'* noise injection methods used in several DRL works cannot robustly detect or alleviate overfitting; In particular, the same agents and learning algorithms could have drastically different test performance, even when all of them achieve optimal rewards during training. 现有的一些方法包括：stochastic policy，random starts，sticky actions（有一定概率复读上一轮动作）和frame skipping。

## Classical DRL
### Analysis
 * *Diagnosing Bottlenecks in Deep Q-learning Algorithms, ICLR 19'* 一篇比较全面地分析Q-learning算法在实际神经网络使用中情况的文章。除了很实用之外，文章使用的三种用于分析的策略（exact，sample和replay-FQI）也提供了一个非常好的insight。
 smaller architectures introduce significant bias in the learning process.This gap may be due to the fact that when the target is bootstrapped, we must be able to represent all Q-function along the path to the solution, and not just the final result.
 higher sample count leads to improved learning speed and a better final solution, confirming our hypothesis that overfitting has
a significant effect on the performance of Q-learning.
replay buffers and early stopping can be used to mitigate the effects of overfitting.
nonstationarities in both distributions and target values, when isolated, do not cause significant stability issues. Instead, other factors such as sampling error and function approximation appear to have more significant effects on performance. 
文章还提出了更好的sampling方法：Adversarial Feature Matching。
* *Near-Optimal Reinforcement Learning in Polynomial Time 02'* 很经典的一篇文章，出自大佬Singh之手。文章提出了The Explicit Explore or Exploit算法，它通过定义了一个“known state”的概念，然后将算法分为两个反复的阶段——balanced wandering（当来到“unknown状态”时进入，去尝试当前尝试次数最少的动作）和offline attempted exploration/exploitation（如果当前的已知集合足够好，则在其中exploit；否则尽快跳出这个已知状态集合。如果在过程中意外来到了未知状态，则立即回到balanced wandering）。“已知”的概念是每个动作都被探索一定的次数，这个“一定”是根据理论计算得到的。

### Algorithms
 * *DQN*
  Q网络的拟合目标是用Q网络自己的早期版本（即target net）用Bellman方程作为结果。另外Experience Replay把时序过程中的步骤拆分出来作为训练集也是一个经典操作。
 * *Implicit Quantile Networks for Distributional Reinforcement Learning* DQN系到18年的SOTA，但是似乎不能解决POMDP。
 * *TRPO* Trust Region Policy Optimization 15' 
  思路：要保证策略梯度得到的结果单调不降----->只要满足advantage>=0---->需要近似计算所以不能离的太远----->对KL散度有限制---->用KL散度的二阶展开近似KL散度变成凸优化问题。
 * *PPO* Proximal Policy Optimization 17'
  简单实用的正则项动态系数调整法（系数动态调整+clip），加正则项的方法都可以借鉴它。
 * *TRPO/PPO for POMDP*
 * *DDPG* DDPG是一种难以训练的方法。虽然理论上说DDPG可以适用于gridworld这样的低维度动作环境中，但是实验表明其表现和收敛速度远不如DQN。DDPG依然算是一种面对连续/高维决策空间的无奈之举。
 * *TD3* TD3 is a direct successor of DDPG and improves it using three major tricks: clipped double Q-Learning, delayed policy update and target policy smoothing. TD3本质上可以说是一种DDPG的变种，从算法上差异也不大。它主要解决的是Q-learning中由于无法知道value的准确值导致的estimation bias问题。
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

下面是一篇对这些经典算法的改进：
* *Improving Stochastic Policy Gradients in Continuous Control with Deep Reinforcement Learning using the Beta Distribution, ICLR 17'* 在输出连续动作时，我们常常假设动作的概率分布是高斯分布。但是并不是所有的场景都适合用高斯分布；这篇文章探索了使用beta分布去替代高斯分布的可能性。高斯分布有一个问题：它在处理具有边界的动作时会出现bias，因为有一些被分配了概率的动作实际上是不可能达到的（比如说，机器手的角度只能是(-30, 30)，这个时候假如输出一个均值-29的高斯分布，实际上就有相当一部分概率位于“不可行”的区域。相比之下，beta分布的支撑集是有界的，因此其在边界上是bias-free的。（应该说，高斯分布对很多对称或者近似对称的真实情况work的都很好；一种work的不好的情况是长尾分布）。
   
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
* *RLaR: Concurrent Reinforcement Learning as a Rehearsal for
Decentralized Planning Under Uncertainty, AAMAS 13'* RLaR是一种用来解决dec-POMDP的方法。dec-POMDP是一种特殊的MARL，它要求所有的agent共享一个global reward。Dec-POMDP是NEXP-Complete的。 **RLAR是一种认为训练时全部状态可见、执行时不可见的方法，它把训练叫做一种“rehearsal”，即排练。** 它分为两步：第一步是在完全状态下学到一个policy；第二步是agent通过探索去建立一个预测模型，根据预测模型和原policy学到新的不需要完全可见就可以work的policy。
* *Actor-Critic Policy Optimization in Partially Observable Multiagent Environments, NIPS 18'*
以下是三个按照时间先后排列的算法。
* *BAD: Bayes Action Decoder* 
* *SAD: Simplified Bayes Action Decoder*
* *VariBAD*

## MARL
 * *MADDPG*
   经典的Multi-agent算法。本质上说，是DDPG的扩展；它利用centralized training在训练时为critic网络给出了额外的信息，而actor则不直接利用这些信息；最后测试时只使用actor网络决策。另外它为了防止competitive情况下的overfit，训练了一堆平行的参数每次均匀随机选择。
 * *R-MADDPG*
   R-MADDPG是MADDPG的一个拓展，它在Actor和Critic（主要是Critic）上增加了一个循环结构，用以储存过去接收的信息，处理依赖于时序的任务。实验结果表明，在critic上安装循环结构效果显著，而在actor上安装循环结构几乎没有效果。（虽然极其难训就对了）
 * *COMA*
 * *MAAC: Actor-Attention-Critic for Multi-Agent Reinforcement Learning*
 * *DP(R)IQN*在D(R)QN的基础上改进，用一个带softmax的inference分支去将对手的policy纳入考虑。
 * *Learning with opponent-learning awareness*
 * *QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning*
 * *QMIX*
 * *Diff-DAC: Distributed Actor-Critic for Average Multitask Deep Reinforcement Learning* 其实或许这也是一种图卷积神经网络的思想？


### Hierarchical RL
 * *HIRO: HIerarchical Reinforcement learning with Off-policy correction*
 * *Hierarchical Actor-Critic*
 * *MCP: Learning Composable Hierarchical Control with Multiplicative Compositional Policies*
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
  * *Wasserstein GAN*
  Wasserstein对比KL的一大好处是不要求定义域完全重合（甚至交集为0也可以衡量两个分布之间的距离）。
## Soft Q-learning
算法产生的模型探索更充分，探索到有用的子模式更多。Soft Q-learning本来就是要解决exploration的问题，所以才在reward上加了一个正则项（注意它并不是神经网络的正则项，所以不一定要用到神经网络上，tabular也完全可能应用soft Q-learning）。感觉上，DQN似乎不太能解决这个问题；deterministic的决策原本就不利于做出探索。然而一个奇怪的现象是，按照我的实践经验，soft actor-critic似乎推广到离散环境时表现不好（soft Q-learning表现尚可）。
* *Reinforcement Learning with Deep Energy-Based Policies ICML17’* https://zhuanlan.zhihu.com/p/44783057 有详细解说。
* *Multiagent Soft Q-Learning*
Relative overgeneralization occurs when
a suboptimal Nash Equilibrium in the joint space of actions is preferred over an optimal Nash Equilibrium because
each agent’s action in the suboptimal equilibrium is a better
choice when matched with arbitrary actions from the collaborating agents.
* *Balancing Two-Player Stochastic Games with Soft Q-Learning 18’* 

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
   * *LEARNING ROBUST REWARDS WITH ADVERSARIAL
INVERSE REINFORCEMENT LEARNING, ICLR 18'*
 * *AIRL: LEARNING ROBUST REWARDS WITH ADVERSARIAL
INVERSE REINFORCEMENT LEARNING 17'*
  * *MAGAIL：Multi-Agent Generative Adversarial Imitation Learning 18'*
  * *Multi-Agent Adversarial Inverse Reinforcement Learning, ICML 19'* 见github里的MAAIRL.pptx。
  * *Adversarial Imitation via Variational Inverse Reinforcement Learning， ICLR 19'* 这里面提出了一个概念：empowerment。empowerment是I(s',a|s)，它表示的是有agent有多大的可能性“影响自己的未来”。研究者认为增加这一正则项有助于防止agent过拟合到专家的demonstration上。
  * *Asynchronous Multi-Agent Adversarial Inverse Reinforcement Learning*
  
## Behavior Cloning
* *Integrating Behavior Cloning and Reinforcement Learning for Improved Performance in Dense and Sparse Reward Environments*
* *Accelerating Online Reinforcement Learning with Offline Datasets*
## Agent Modeling
### Classical Modeling：Feature Engineering
* *Player Modeling in Civilization IV*
这个大概还是使用传统的特征工程，需要手动设计特征。Agent Modeling只有在近两年概率空间投影的policy representation才摆脱了传统方法。
## Policy Representation
### Divergence-based
* *Learning Policy Representations in Multiagent Systems*
* *Modeling Others using Oneself in Multi-Agent Reinforcement Learning*
* *Opponent Modeling in Deep Reinforcement Learning*
* *LEARNING ACTIONABLE REPRESENTATIONS WITH
GOAL-CONDITIONED POLICIES*
* *Learning Action Representations for Reinforcement Learning*
### Encoding & Hidden State
* *Provably efficient RL with Rich Observations via Latent State Decoding*

### Theory of Mind
* *Machine Theory of Mind*
* *Theory of Minds: Understanding Behavior in Groups Through Inverse Planning*

### Society of Agents
* *Social Influence as Intrinsic Motivation for Multi-Agent Deep RL* 奖励那些在CFR下能让队友做出不一样动作（给到信息）的动作。作者指出，如果没有这种特殊的奖励，那么就会陷入一种babbling的尴尬均衡。文章使用互信息来作为衡量指标。另外，为了CFR而训练的MOA网络其实也给出了对其他agent的embedding。还有一点，这个agent训练是完全decentralized的。其实firing beam的设定我感觉也挺有道理——无名氏定理保证了在不定长的repeated games中，如果所有人联合起来可以不让一个人好过，那么就能出现某种程度的合作。
* *Mean-field MARL* 是用“和它相关的附近的几个agent”考察一对一对的关系来降低维度。然而其限制在于，其应用环境必须满足reward能被和field内附近agent的互动很好地勾勒。

## Relational Reinforcement Learning
似乎是符号主义和连接主义的结合。
* *Relational Deep Reinforcement Learning, 18'* 别出心裁的定义。但是实验过于简单，实际推广的效果如何还存疑。

## Mathematical Background
* *Rethinking the effective sample size*

### Miscellanous
* *Allocation of Virtual Machines in Cloud Data Centers- A survey of problem models and optimization algorithms*
* *SEIR epidemic model with delay*（AI无关）
* *DR-RNN: A deep residual recurrent neural network for model reduction*
对于（如物理模型等）大规模模拟的简化，有三种主要思路。第一种是基于物理公式的简化模型（所以是heavily prior-based）；第二种是纯拟合的黑箱模型（类似于专家-学徒问题中的模仿学习）；第三种是基于投影的低秩模型（ROM）。第三种思路的前提是必须假设整个模型可以被低秩线性表出。得到投影基使用**Galerkin projection（伽辽金方法）**。几种主要算法是：Proper Orthogonal Decomposition; Krylov subspace methods; truncated balanced realization.文章提出了一种基于RNN的model reduction(?)
* *Meta Q-learning*
* *A survey on Federated Learning Systems: Vision, Hype and Reality for Data Privacy and Protection* 联邦学习的综述。
* *the IQ of neural networks* 一篇还算比较有趣的文章，用CNN来做智力测试题。
* *What can neural networks reason about?* 非常棒的文章，它为我们这些年来在NN方面设计的各种结构模块背后的理论依据提供了insight，特别是GNN。文章以PAC-learning作为基石，提出如果神经网络的模块能够和经典算法有好的alignment（即sample complexity高），那么就会有好的performance和generalization。
* *LEARNING WHAT YOU CAN DO BEFORE DOING ANYTHING， ICLR 19'* 想办法从录像中获得一种embedding。比较值得借鉴的想法是利用互信息去衡量两个完全不同的表达方式其中一个embedding另外一个的效果，作为目标函数。
* *Multiagent Cooperation and Competition with Deep
Reinforcement Learning*
这篇文章好像就是调了一下DQN的reward然后说明可以合作/竞争。没有什么太大价值，应用也不存在的。
* *Intrinsic motivation and
automatic curricula via asymmetric self-play*
* *A Structured Prediction Approach of Generalization in Cooperative Multi-Agent Reinforcement Learning*
* *Neural Logic Reinforcement Learning, ICML 19'* NLRL:少见的符号主义和连接主义的结合。用一阶逻辑表示policy。实际上早在世纪初就有用一阶逻辑表示state的尝试；但是这个需要agent对state和reward的逻辑形式有所了解(known dynamics)。
说实话有点看不懂；它基于prolog这个逻辑型程序设计语言。感觉它就是一个从输入到输出的二值神经网络（向量只有0/1）？最后实验也比较弱，大概就是很小地图的cliff walking和砖块的放上放下。
它的运算可以看成一组**clause**对输入的连续处理。它的优点应该是：可解释（policy可以被一些从句解释出来——其实稍微大一点就不human readable了）、不依赖于background knowledge、可移植性强（其实RL已经有很多在注意这个问题了？）。
* *Probability Functional Descent: A Unifying Perspective on GANs, Variational Inference, and Reinforcement Learning*
* *Variational information maximisation for intrinsically motivated reinforcement learning, NIPS 15’* 除了提出了empowerment之外，这篇文章的一个重要可借鉴的地方是：如果函数本身难以优化，就尝试推导一个下界然后去优化它的下界。在凸优化中，我们有时会优化envelope function，找proximal mapping，也就是这个道理。
* *Are security experts useful? Bayesian Nash equilibria for network security games with limited information* 
这篇文章从博弈论的角度告诉我们，杀毒软件不能装太多。原文abstract：“expert users can be not only invaluable contirbutors, but also free-riders, defectors and narcissistic opportunists.”
### Evolutionary
* *Competitive coevolution through evolutionary complexification*
进化算法。
* *Evolutionary Population Curriculum for Scaling Multi-agent Reinforcement Learning, ICLR 2020* 非常好的文章，在agent间沟通的权重选择、变异与进化方法上都有亮点。

### Monte-Carlo Based
* *Mastering the game of go without human knowledge*
AlphaGo。
* *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm 17'*
这一篇和上一篇基本上没有太大区别，只是上一篇应用的一个扩展。
* *Emergent Complexity via Multi-Agent Competition ICLR 18'*
这篇文章的核心思想是competition introduces a natural learning curriculum。
一般来说避免overfit的方法包括随机抽取一个集合作为对手（同样的思路也用在了MADDPG里），给轨迹的熵加正则项（一般用在机器人控制里面，假设移动服从某种概率分布，如高斯分布）。
这篇文章用分布式PPO训练。总的来说很吃数据量（想来也是，它完全不采取centralize的方法进行训练。）。
这篇文章的一个小trick是它使用自己的过去作为sample，另一个是开始阶段手动设计的curriculum。

### Credit Assignment
* *Credit Assignment For Collective Multiagent RL With Global Rewards*
* *Shapley Q-value: A Local Reward Approach to Solve Global Reward Games*

## Game Theory
### Differentiable Games
Differentiable Games是一类特殊的游戏，它要求每个人的reward函数都已知并且由每个人的action(原文为theta)完全决定且对theta可微。
* *N-player Diffentiable Games*
* *Consensus Optimization* 
* *Stable Opponent Shaping* LOLA的改进。
* *Learning with Opponent Learning Awareness(LOLA)* 把对手当成naive learner，预测对手的update。不过这样的假设在对手也是LOLA Agent的时候会导致“arrogant”的行为，即假设对手会遵从自己对我方有利的动作调整（具体阐述见上面的Stable Opponent shaping）。
### Classic MARL
* *Deep Q-Learning for Nash Equilibria: Nash-DQN 19’* 用线性/二阶展开逼近去求Advatange等。
* *Coco-Q: Learning in stochastic games with side payments 13’* 一种新的solution concept，利用“给reward”的形式达成某种类似“契约”的状态。实际上，这个或许能够给reward assignment一点insight？


### Fictitious Play
Fictitious Play是一种寻找双人博弈中Nash均衡的方法。
* *Deep Reinforcement Learning from Self-Play in Imperfect-Information Games 16'*
简介见下面Game theory一节。
* *On the Convergence of Fictitious Play 98'*
对一般的General sum游戏来说，NFSP是不收敛的；实际上，不收敛是一种常态。（但是也许会收敛到cyclic equilibrium？）
CFP almost never converges cyclically to a mixed
strategy equilibrium in which both players use more than two pure strategies. Thus, Shapley's example of nonconvergence is the norm rather than the exception. Mixed strategy equilibria appear to be generally unstable with respect to cyclical fictitious play processes.
In a recent paper, Hofbauer (1994) has made a related conjecture: if CFP converges to a regular mixed strategy equilibrium, then the game is zero-sum.
* *On the Global Convergence of Stochastic Fictitious Play* 揭示出有四种游戏是可以保证全局收敛的：games with an interior ESS（内部进化稳定点，即作为完全对称的游戏，在一个mixed-strategy邻域内是极优策略）, zero sum games, potential games（即所有人获得的reward始终相同）, and supermodular games：在超模博弈中，每个参与者增加其策略所引起的边际效用随着对手策略的递增而增加。博弈里最优反应的对应是递增的，所以参与者的策略是“策略互补的”。
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

### Multiple Payoffs
* *Games with multiple payoffs, 75'*
这里面的设定是虽然multiple，但是实际上根据加权最后还是要有一个唯一的优化目标。这个加权也不是唯一固定的，它在against nature游戏中由玩家决定以最大化总收益，而在双人博弈中似乎不是固定的。


## Blogs and Slides
* *https://www.freecodecamp.org/news/an-introduction-to-policy-gradients-with-cartpole-and-doom-495b5ef2207f/* 为什么需要policy gradient？随机策略和确定性策略比起来有什么好处？
* *http://www.cs.umd.edu/~hajiagha/474GT13/Lecture10152013.pdf* Multiple Payoff的相关课件。
* *https://vincentherrmann.github.io/blog/wasserstein/* 关于Wassenstein GAN的一些内容，或许有助于了解Wassenstein距离。
### Counterfactual
目前只能用在双人零和博弈里面。CFR的一个核心在于“反事实导致的遗憾”，即如果在这个时候选择了其他动作，那么其regret相当于选这个动作所额外带来的reward。
CFR determines an iteration’s strategy by applying any of
several regret minimization algorithms to each infoset (Littlestone & Warmuth, 1994; Chaudhuri et al., 2009). Typically, regret matching (RM) is used as the regret minimization algorithm within CFR due to RM’s simplicity and lack of parameters.
RM大概就是一种在正regret中根据比例选动作的简单算法。
* *Using Counterfactual Regret Minimization to Create
Competitive Multiplayer Poker Agents 10'*

* *Regret Minimization in Games with Incomplete Information 07’* 
* *An Introduction to Counterfactual Regret Minimization* 一篇很好的入门教程。
* *Regret Minimization in Non-Zero-Sum Games with Applications to Building Champion Multiplayer Computer Poker Agents 13’*
这文章其实多少有一点玄学：证明了一个仍然是发散的界（regret与根号倍迭代轮数成正比），然后实际上训了一个还行的结果。
* *Deep Counterfactual Regret Minimization ICML 19'*
The goal of Deep CFR is to approximate the behavior of CFR without calculating and accumulating regrets at each infoset, by generalizing across similar infosets using function approximation
via deep neural networks.
 One method to combat this is Monte Carlo CFR (MCCFR), in which only a portion of the game tree is traversed on each iteration (Lanctot et al.,2009). In MCCFR, a subset of nodes Qt in the game tree is traversed at each iteration, where Qt is sampled from some distribution. 
* *Single Deep Counterfactual Regret Minimization 19'*
* *Efficient Monte Carlo Counterfactual Regret Minimization in Games with Many Player Actions*
说到底，提高采样效率用到的还是老一套，比如Monte Carlo，连稍微高级一点的Gibbs/Hasting-Metropolis采样都没用到。很多算法从tabular classical向deep搬运的过程，说白了就解决了两个问题：1）大规模计算如何用function approximator等工具代替 2）如何更好地采样以代替概率。

### Regret Matching
* *A SIMPLE ADAPTIVE PROCEDURE LEADING TO CORRELATED EQUILIBRIUM 00'* CFR的基础是regret matching，本质上CFR就是regret matching延展到extensive form game的产物（至少主流的CFR是这样）。

### Nash Equilibrium
Nash这种东西在计算意义上的本质，就是反复针对对方做最优决策，然后不断迭代，希望能够达到稳定。
* *Nash Q-Learning for General-Sum Stochastic Games 03'*  就是把Q-learning的最优下一步Q换成下一步大家都按Nash均衡行事。
* *Actor-Critic Algorithms for Learning Nash Equilibria in N-player General-Sum Games*
two desirable properties of any multi-agent learning algorithm
are as follows:
(a) Rationality: Learn to play optimally when other agents follow stationary strategies; and
(b) Self-play convergence: Converge to a Nash equilibrium assuming all agents are using the same learning algorithm
文章是几个印度人写的。
为了绕过前人已经做出的结论（任何value-based方法，试图只用Q-learning不搞其他骚操作的方法，对general-sum game无法保证收敛到纳什均衡），作者写了一句“We avoid this impossibility result by searching for both values and policies instead of just values, in our proposed algorithms”。
这个放在小字里面就很灵性……反正我没看懂什么意思，总不会是“我们这个算法本质爆搜”的意思吧？
* *Learning Nash Equilibrium for General-Sum Markov Games from Batch Data* Markov Game（或者说Stochastic Game）是一种特殊的MDP，或者也可以理解为“回合制的”MDP。特点是决策完全由当前状态决定。它也有对应的部分可见版本，叫POSG。
* *Markov games as a framework for multi-agent reinforcement learning* Littman的经典文章。虽然idea在现在看来都很基本，但它却是博弈论与MARL结合的先驱。
* *Cyclic Equilibria in Markov Games* 这篇文章证明了：但凡使用Q值的值迭代算法（所以也包括DQN及其任意变种）都没法算出任意general sum game的**静态**Nash均衡。不过，作者提出一个新概念叫循环均衡——它满足任何一方单独改变策略都无法优化的条件，但是它并不满足无名氏定理，而是在一组静态策略之间循环。很多双人双状态双动作游戏都无法在value-based方法下收敛，但在几乎所有的游戏之中它们都达到了“循环均衡”。可以理解为剪刀石头布限定纯策略情况下双方在三种策略之间来回震荡，但是总的来说满足均衡条件。需要注意的是：cyclic equilibrium是一种correlated equilibrium，所以它对于competitive game还是……emmm。
**多说一句：** 从动力系统的观点来看，循环均衡正是由每方策略优化方向决定的向量场中互相可达且（在允许无穷小误差意义下）常返的“旋涡”。这本质与alpharank的MCC类似。
* *Actor-Critic Fictitious Play in Simultaneous Move
Multistage Games* 一个NFSP的变种（？从年代上看和NFSP差不多，用去中心化的actor-critic方法解决了2-player 0-sum game。）
### Robust(Minimax) Optimization
* *Handling uncertainty of resource division in multi-agent system using game against nature*
这篇文章是一篇很老的文章，主要就是对未知情景采用minimax来保证表现。文章解决了一类机器人合作收集物品问题。

## Reward Shaping
* *Policy Invariance Under Reward Transformations： Theory and Application to Reward Shaping, ICML 99'*
注意区分reward shaping和正则项。这二者都是改变reward函数，但reward shaping是不改变最优策略的；而正则项是会改变最优策略的。
reward shaping的优点在于完全不会改变最优策略，缺点在于其形式必须满足一个特殊要求：对于任何(s,a,s')的转移，函数都可以写成cf(s')-f(s)，其中c为一小于1的常数。[对于只有位置具有重要性的gridworld就很有用了]
正则项的优点在于可以加入任何东西，但它不能保证最优解。

## Deception
* *Designing Deception in Adversarial Reinforcement Learning*
在传统框架下设计policy让agent学会骗人。这里面提到了一个观点：欺骗是一种能够引诱对手进入某种特殊policy的技巧，有利于把对手拉进自己熟悉的子游戏并战胜之。（类似的MADDPG和18年OpenAI一篇分布式PPO的文章也做了和欺骗相关的multi-agent实验）
* *Finding Friend and Foe in Multi-Agent Games 19’*
有大量的prior knowledge，但是不失为好的尝试。使用CFR（见上面的Counterfactual一节）。

* *Learning Existing Social Conventions via Observationally Augmented Self-Play AIES 19’(?)*
学习民俗，以便更好地融入agent社会中（？）定义了一个偏序关系用来描述policy的相似性，这个比较有意思。总的来说是一篇比较有意思的文章。

## Active Learning
* *Active Classification based on Value of Classifier*
* *Learning how to Active Learn: A Deep Reinforcement Learning Approach*
active learning本来是一种通过分类器主动将未标记文本选择并送给专家标记的方式提高学习效率的方法。本来是将active learning用于NLP，这里把它建模成一个RL选样本作为policy的问题。而且是先在一个语言上学习policy再迁移到另一个语言上。把语料库打乱，然后认为面对一个句子有两个action：接受或不接受。如果接受，则update当前的classifier。注意到他们把当前classifier的状态建模成了一个state，所以可以认为训练是off-policy的。

## Experimental
* *Deep Reinforcement Learning and the Deadly Triad* 证明了DQN没有那么容易陷入死亡三角。另一个值得注意的结论是，如果Q-value异乎寻常的大，那么performance多半不会好。
* *Deep Reinforcement Learning that Matters* 实验做的很充分，但结果却很悲观：甚至连不同的代码实现都会对同一算法的表现带来很大影响。
下面是两篇专门研究PPO和TRPO的文章：
* *Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO*
不加技巧的PPO实际上performance并不好。几个重要的结论有：
1.clip（对于advantage在\[1-epsilon,1+epsilon\]之间的截断）如果没有的话，surrogate loss里的KL散度会大一些——尽管如此，差距并不显著，甚至还基本保持在TRPO的硬性限制之内。从最终reward的差距来看也并不算显著。（或者说，允许的KL散度和似乎有一定关系）
2.加了代码优化的PPO和TRPO都能保持某个平均KL，但不加代码优化的PPO似乎不能保持住trust region的KL差距。

在humanoid2d-v2和walker2d-v2里，anneal_lr（把Adam的lr拿来退火——尽管Adam本身就是自适应算法）似乎有显著的reward提升。归一化reward有显著的reward提升。除此之外，value loss的clip和好的初始化（正交而不是xavier）可以带来微弱的reward提升。

* *Are Deep Policy Gradient ALgorithms Truly Policy Gradient Algorithms?* 这篇文章对DPG算法从原理层面提出了一定的质疑，并且呼吁评价RL算法应该多角度评价（比如，能不能很好地对问题进行真实的建模）。文章表明我们平常的算法采样的数量，下降的梯度以及整个value function的landscape与真实情况实际上相去甚远，而且重复跑多次之后，其梯度差异很大——至少要多三个数量级才能做到这些。

## Application
###  Recommending Systems
* *Generative Adversarial User Model for Reinforcement Learning Based Recommendation System， ICML 19'*
* *Deep Reinforcement Learning for List-wise Recommendations*
### Packet Switching
* *Neural Packet Classification 19’* 用RL（好像还是MARL？）做packet classification。认为生成决策树的每一步是一个action，目的是在每一步最小化时间和空间综合而成的一个loss。使用actor-critic算法。
* *A Deep Reinforcement Learning Perspective on Internet Congestion Control, ICML 19'*
### Network Intrusion Detection
intrusion detection可以分为host-based（基于主机的日志文件）和network-based（基于流量或包的内容）；也可以分为signature-based（固定规则）和anomaly-based。
* *An Overview of Flow-based and Packet-based Intrusion Detection Performance in High-Speed Networks*
* *A Flow-based Method for Abnormal Network Traffic Detection*
* *PHY-layer Spoofing Detection with Reinforcement Learning in Wireless Networks* 这篇本质上是个security games，实际上和博弈论结合更紧密。
### Traffic Control
* *Multi-agent Deep Reinforcement Learning for Large-Scale Traffic Signal Control* 这个是Independent RL。
* *Integrating Independnet and Centralized Milti-Agent Reinforcement Learning For Traffic Signal Network Optimization* 这个在independent RL的基础上做了改进，针对的是交通灯这样一种既可以方便地找到局部reward又可以找到全局reward的问题：他额外计算了一个全局Q值，这个Q值是全局reward，但是policy是每个个体最大化自己的局部reward。然后梯度下降的时候同时考虑全局Q和局部Q。不过，如果只是这样的话，那么可以想见全局Q也不过是在局部最大化这个policy下局部Q的平均，那么整个算法就是IRL；但是他还搞了一个regularizer，就是要求每个Q都不能离平均的Q太远，这里做了一个惩罚项。这里其实有一点assumption，就是没有一个agent是特别突出的。
### Public Health
* *A Microscopic Epidemic Model and Pandemic Prediction Using Multi-Agent Reinforcement Learning*
* *A brief review of synthetic population generation practices in agent-based social simulation* 把这篇放在这里是因为它讲述了生成模拟人口数据的常见方法，而模拟人口数据在传染病研究里很常见。
* *Generating Realistic Synthetic Population Datasets*
### Cloud Computing (Resources Allocation)
* *A Hierarchical Framework of Cloud Resource Allocation and Power Management Using Deep Reinforcement Learning* Autoencoder+全局/机器内部建模。
* *Energy-Efficient Virtual Machines Consolidation in Cloud Data Centers Using Reinforcement Learning*
* *Resource Management with Deep Reinforcement Learning* MIT的Hongzi Mao在这方面有一些工作，最早的就是这篇。这篇实际上是基于REINFORCE with baseline的任务调度，其中“冻结时间”把一步拆成好几步来降低动作空间的思路比较有趣。

## Overfitting Prevention
* *Protecting against evaluation overfitting in empirical
reinforcement learning, AAAI 11'*
* *Improved Empirical Methods in Reinforcement Learning Evaluation, 15'*
* *A Unified Game-Theoretic Approach to
Multiagent Reinforcement Learning, 17'*
g. 
When the model is fully known and the setting is strictly with two players, there are policy
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
* *Quantifying Generalization in Reinforcement Learning* OPENAI的COINRUN。考虑了很多因素，给出了一个平台，证明很多agent都会overfit。

* *Exploration by random network distillation ICLR 19'* 一个鼓励exploration的文章。基本想法是拿一个预测网络，如果结果不好预测说明是未知的，就给一些奖励。文章还提到了一个经典的noisy-TV场景：一个agent在迷宫里走，遇到随机播放频道的电视就会停下来无限探索。

## Novel Architectures
* *Value Propagation Networks, ICLR 19'*  VIN的改进。一个奇妙的observation在于，二维gridworld上的value iteration可以看成是图卷积神经网络的卷积过程。

* *Asynchronous Methods for Deep Reinforcement Learning* 异步DQN，类似于A3C之于A2C的改进（实际上，文章提出的是一个框架）。不过，考虑到A3C其实对于A2C也没什么改进，异步版本DQN能有什么样的性能提升也很难说。
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

### Variance Reduction and Overestimation
* *Reward Estimation for Variance Reduction in Deep Reinforcement Learning* 
给DQN降低方差的技巧： average DQN和ensemble DQN。前者是用过去几次的参数做平均数得到的Q去用在bellman方程里更新，后者是希望解决前者在方差减小的同时导致计算代价增大的问题——同时维护和更新k套参数。
* *Issues in using function approximation for reinforcement learning.* 很老的一篇文章，提出了为什么Q-value经常会overestimate的理由——因为bellman方程中的max operator会捕捉和放大那些偏大的Q值估计；这样会导致花费很多时间去探索那些实际上不好的(s,a)pair。一个关键的observation是，在reward固定且deterministic转移的情况下，假定Q(s,a)-Q*(s,a)期望为0，其所造成的误差是gamma*(max_a Q - max_a Q*)，这个值的期望往往是大于0的（不严谨地说，当所有Q一样且落到0两侧的概率相等时，误差为正的概率是1-(1/2)^|a|；Q不一样但落到0两侧概率相等时，那些Q\*值小一点的贡献也会小一些。）
gamma太高的时候容易导致q-learning fail。
Stochastic Variance Reduction for Deep Q-learning, AAMAS 19’ 把SVRG用在了DQN中。

## Safe RL
### RL with Constraint
* *First order Constrained Optimization in Policy Space*是下面那篇projection-based的改进，它可以不用计算二阶导数。
* *Projection-based Constrainted Policy Optimization*

## Evolutionary RL
* *Evolution-guided policy gradient in reinforcement learning* NIPS 18'
* *Proximal Distilled Evolutionary Reinforcement Learning* AAAI 20' 

