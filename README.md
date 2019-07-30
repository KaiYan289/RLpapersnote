# RL Papers Note
这是一篇阅读文献的简记。
OpenAI spinning up：https://spinningup.openai.com/en/latest/index.html
## Surveys and Books
 * *Deep Reinforcement Learning for Cyber Security, 19'*
 一篇讲述DRL在网络安全中应用的综述。
 * *Autonomous Agents Modelling Other Agents:
A Comprehensive Survey and Open Problems, 17'*
 一篇非常全面地讲述agent modeling的文章。实际上直到这一年，agent modeling一直没有什么很大的进展，停留在提取特征和对对手过去行为做统计（fictitious learning也算），比较依赖于环境本身的信息。另一个比较新颖的思路是把对手的policy看成自动机；很不幸的是，这样不能建模非常复杂的对手，因为问题关于状态是非多项式可解的。
 * *multi-agent systems algorithmic game-theoretic and logical foundations*
 一本涵盖了多智能体与算法、博弈论、分布式推理、强化学习、拍卖机制、社会选择等交集的作品。
 前面提到了一些关于**异步DP，multi-agent的ABT搜索**等内容。
 这里面提到了一些多人博弈、时序博弈中基本的概念，比如**extensive form 和normal form**。对于时序博弈，存在一个“不可信威胁”概念，就是说如果整个Nash均衡，在第一步一方打破Nash均衡后，另一方采取反制措施会让自己的reward收到损失，那么这就是“不可信”的，所以说这样的Nash均衡是不稳定的。于是提出**子游戏精炼纳什均衡**。还有**颤抖手精炼纳什均衡**，大概就是指在假设一定犯错概率的情况下达到纳什均衡。另外还有一个有意思的**无名氏定理**：如果无限次重复进行的游戏具有合适的贴现因子，同时所有人针对一个人时，会给这个人带来额外的损失，那么agent之间是可以合作的。
## Security Games
### Abstract Security Games
  * *Improving learning and adaptation in security games by exploiting information asymmetry, INFOCOM 15'*
  考虑了在抽象security game且状态部分可见的情况下如何利用tabular minmax Q-learning（Q-learning变种）去学习。
### Network Security (Games)
  * *Defending against distributed denial-of-service attacks with max-min fair server-centric router throttles*
  网络攻击的经典模型之一：攻击者控制了几个初始节点（叶子）。向目标服务器发动进攻，而防守者控制着到目标服务器（树根）上某些必经之路的路由器，需要调整丢包比例让保证安全的同时最大限度让合法请求通过。这个设定被用在之后大量的DDOS相关论文中，也包括一些用MARL解决该问题的文章。
  * *Reinforcement Learning for Autonomous Defence in Software-Defined
Networking, 18'*
  网络攻击的经典模型之二：攻击者控制了几个初始节点，并且向目标服务器发动进攻，每回合感染一些与上一回合节点相邻的服务器。防守者可以选择断开某些服务器连接、重连某些服务器、转移目标服务器数据到其他指定位置。
### Green Security Games
  * *Deep Reinforcement Learning for Green Security Games with Real-Time Information, AAAI 19'*
 对Green Security Games这种特殊的安全游戏引入了一种DRL解法。Green Security Game是一个面对偷猎行为建模设计的游戏，在2D gridworld上进行。游戏分为两方，一个是偷猎者，另一个是巡逻者。偷猎者可以四处移动，或是放下偷猎工具，它每回合有一定概率爆炸，若爆炸则收获正reward（巡逻者收获负reward），并消失；巡逻者可以拆除偷猎工具，或者是抓到偷猎者以得到正reward（对应的，偷猎者收获负reward）。游戏是partial observation的。游戏在巡逻者抓住偷猎者且场上没有偷猎工具时结束。DRL本身似乎没有什么特别的。
 
## Classical DRL
 * *DQN*
  Q网络的拟合目标是用Q网络自己的早期版本（即target net）用Bellman方程作为结果。另外Experience Replay把时序过程中的步骤拆分出来作为训练集也是一个经典操作。
 * *TRPO* Trust Region Policy Optimization 15' 
  思路：要保证策略梯度得到的结果单调不降----->
 * *PPO* Proximal Policy Optimization 17'
  简单实用的正则项动态系数调整法（系数动态调整+clip），加正则项的方法都可以借鉴它。
 * *TRPO/PPO for POMDP*
 * *DDPG*
 * *AC*
  Actor-Critic从本质上说是Policy Iteration的升级版。
 * *A2C*
 * *SAC*
 
## MARL
 * *MADDPG*
   经典的Multi-agent算法。本质上说，是DDPG的扩展；它利用centralized training在训练时为critic网络给出了额外的信息，而actor则不直接利用这些信息；最后测试时只使用actor网络决策。另外它为了防止competitive情况下的overfit，训练了一堆平行的参数每次均匀随机选择。
 * *COMA*
 * *DPIQN*
 
## Policy Gradient

## Distance of Distribution
  * *Wassenstein Reinforcement Learning*
  不可无一不可有二的文章。作者对把RL推广到这一量度上做了很多非常用心的理论推导；但是其真的推广到这一领域能比传统RL的表现好多少是存疑的。
  

## IRL
 * *Inverse Reinforcement Learning 00'*
  IRL开山之作，用线性规划解决问题。首次提出reward是由一些“feature”的未知系数的线性组合构成的这个概念，在后面直到deep maxent的一系列IRL工作中发挥了非常重要的作用。
 * *Apprenticeship Learning via Inverse Reinforcement Learning 04'*
  需要明确的一点是：IRL的终极目标不是得到和原来完全一样的reward，而是可以在原reward上“表现得和原来一样好的”policy。
 * *Bayes Inverse Reinforcement
 Learning 08'*
  不需要每次求“最优”Policy了，比较优的就可以。
 * *Maxent 08'*
  两个假设：一个是reward是手动设计的一些特征的线性组合，另一个是认为轨迹的概率分布（这是一个重要概念！）出现的概率是和e^reward成正比。这一篇我复现过，实际效果嘛……emmm。
 * *Deep Maxent 10'*
  “Deep”是用来解决上一篇中特征提取问题的。上一篇认为reward是手动设计的一些特征的线性组合，这里就变成了网络自动从地图里提取特征做组合。
 * *Guided Cost Learning 15'*
   
 * *A Connection Between Generative Adversarial
Networks, Inverse Reinforcement Learning, and
Energy-Based Models*
  GAN，能量模型和Guided Cost Learning是相通的。
 * *Generative Adversarial Imitation Learning 16'*
  它是GAN在强化学习（实际上是模仿学习）领域的推广。分为两个不断重复的阶段：其中一个阶段是固定Generator优化Discriminator；另一个阶段是固定Discriminator以分类结果的对数似然作为Reward去训练Generator。注意学习architecture要画清楚数据流！
## Behavior Cloning

## Agent Modeling
### Classical Modeling：Feature Engineering
### Divergence-based Policy Representation
### Theory of Mind

## Game Theory
### Fictitious Play
