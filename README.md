# RL Papers Note
这是一篇阅读文献的简记。

OpenAI spinning up：https://spinningup.openai.com/en/latest/index.html

## Surveys and Books
 * *Deep Reinforcement Learning for Cyber Security, 19'*
 一篇讲述DRL在网络安全中应用的综述。
 * *Deep Reinforcement Learning for Multi-Agent Systems: A Review of Challenges, Solutions and Applications,18'*
 一篇DRL在MARL中应用的文章。
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
  * *Reinforcement Learning for Autonomous Defence in Software-Defined Networking, 18'*
  网络攻击的经典模型之二：攻击者控制了几个初始节点，并且向目标服务器发动进攻，每回合感染一些与上一回合节点相邻的服务器。防守者可以选择断开某些服务器连接、重连某些服务器、转移目标服务器数据到其他指定位置。
  * *Adversarial Reinforcement Learning for Observer Design in Autonomous Systems under Cyber Attacks*
  固定control policy，用self play同时训练adversarial和观察者。注意到这里controller是fixed的；训练的可以说是一个“输入扭曲器”和“输入矫正器”。文章以经典的pendulum做实验。另外，为了保证self play能收敛，文章使用了TRPO。
### Green Security Games
  * *Deep Reinforcement Learning for Green Security Games with Real-Time Information, AAAI 19'*
 对Green Security Games这种特殊的安全游戏引入了一种DRL解法。Green Security Game是一个面对偷猎行为建模设计的游戏，在2D gridworld上进行。游戏分为两方，一个是偷猎者，另一个是巡逻者。偷猎者可以四处移动，或是放下偷猎工具，它每回合有一定概率爆炸，若爆炸则收获正reward（巡逻者收获负reward），并消失；巡逻者可以拆除偷猎工具，或者是抓到偷猎者以得到正reward（对应的，偷猎者收获负reward）。游戏是partial observation的。游戏在巡逻者抓住偷猎者且场上没有偷猎工具时结束。DRL本身似乎没有什么特别的。
 
## Classical DRL
 * *DQN*
  Q网络的拟合目标是用Q网络自己的早期版本（即target net）用Bellman方程作为结果。另外Experience Replay把时序过程中的步骤拆分出来作为训练集也是一个经典操作。
 * *TRPO* Trust Region Policy Optimization 15' 
  思路：要保证策略梯度得到的结果单调不降----->只要满足advantage>=0---->需要近似计算所以不能离的太远----->对KL散度有限制---->用KL散度的二阶展开近似KL散度变成凸优化问题。
 * *PPO* Proximal Policy Optimization 17'
  简单实用的正则项动态系数调整法（系数动态调整+clip），加正则项的方法都可以借鉴它。
 * *TRPO/PPO for POMDP*
 * *DDPG*
 * *AC*
  Actor-Critic从本质上说是Policy Iteration的升级版。
 * *A2C*
 advantage Actor-critic
 * *A3C*
 在之前的基础上加入了一个异步更新。
 * *SAC*
### Partly Observable RL
* *DRQN* 把第一个全连接层换成了LSTM，其他的和DQN 完 全 一 致。
* *DDRQN*是一个用来解决MARL中团队合作__交流__的网络结构。第一个D代表distributed。文章中提到了三个主要修改：第一个是把上一个agent的动作作为下一个时间步的输入；第二个是所有agent参数共享；第三个是**对于不稳定环境不使用experience replay**。使用soft更新（就是说加权更新target network参数而不是直接复制）。另外实验比较有意思，把帽子和开关的智力题建模成了MARL问题。
* *RIAL&DIAL* 目标是设计一个end-to-end的学习协议。RIAL是建立在DRQN的基础上的。
architecture
独立的Q-learning指agent互相都把对方当成环境的一部分。
the space of protocols is extremely high-dimensional. 
“during learning, agents can
backpropagate error derivatives through (noisy) communication channels”
color-digit MNIST
中心化训练与去中心化执行。
## MARL
 * *MADDPG*
   经典的Multi-agent算法。本质上说，是DDPG的扩展；它利用centralized training在训练时为critic网络给出了额外的信息，而actor则不直接利用这些信息；最后测试时只使用actor网络决策。另外它为了防止competitive情况下的overfit，训练了一堆平行的参数每次均匀随机选择。
 * *R-MADDPG*
   R-MADDPG是MADDPG的一个拓展，它在Actor和Critic（主要是Critic）上增加了一个循环结构，用以储存过去接收的信息，处理依赖于时序的任务。实验结果表明，在critic上安装循环结构效果显著，而在actor上安装循环结构几乎没有效果。
 * *COMA*
 * *DPIQN*
 * *Learning with opponent-learning awareness*


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
