https://www.promptingguide.ai/

https://github.com/rxlqn/awesome-llm-self-reflection

UPRISE: Universal Prompt Retrieval for Improving Zero-Shot Evaluation



# New Papers

## Many-shot

## Self-train

## Math

## Reflection

**Many-shot In-Context Learning** (2404.11018)

"self-training": model generated

What is the fundamental limits of LLM self-training?


我们要做的这个东西好像带一点many-shot ICL （但是不是example而是template）又带一点reflection（因为要把题目加入训练）和self-train（因为要self-train）

其实或许可以理解为一种active learning，就是说绝大部分情况下可以自己learn，但是少部分情况下需求人工标注













首先从方法上，偏向于finetuning（学校里面太缺计算资源了，希望能做一些学校里做不到的事情）
就现在来说，我比较关心三个问题：第一个是multi-turn LLM，第二个是understanding real numbers，第三个是LLM直接解决传统的mujoco问题。multi-turn LLM这里是以archer这篇文章为代表。
understanding real numbers这篇是以https://arxiv.org/abs/2401.03735这篇文章为代表的。
今天聊天的时候提出了一个LLM建立了“model-based”的概念，那具体是什么意思呢……？

# Papers Summary  
**LDB: Large Model Debugger (2402.16906)**
Script generate control flow graph; analyze in breakpoints between flow graph nodes  
**Don't Trust: Verify (2403.18120)**
Generate formal languages and verify  
**Can LLM Infer Causation from Correlation? (2306.05836)**
new problems of inference, new dataset  
**CREATOR: ... Disentangling Abstract and Concrete Reasoning of LLM (2305.14318)**
let an agent to generate high-level plans as decision; execute and rectify using compiler feedback; create tools to address this  
**DeepSeekMath (2402.03300)**
group relative policy optimization (grpo) instead of PPO  
**MUSR: Multi-Step Soft Reasoning (2310.16049)**
new dataset of building logic trees  
**Let's Verify Step by Step (2305.20050)**
train a good reward model; use "feedback in the middle" / "supervise in the proecess" 
**Solving Math Word Problems (2211.14275)**  
**Scaling Relationship with LLM (2308.01825)**
rejection sampling fine-tuning; data augmentation  
**LM Understand Numbers, at Least Partially (2401.03735)**
empirical study on the internal embedding with numbers as input in LM; uses a single linear layer to "decode"  
**AlphaMath Almost Zero (2405.03553)**
MCTS evaluation to get a value function as reward model; do process supervision with reward assigned by the reward model  
**Best Practices and Lessons Learned in Synthetic Data (2404.07503)**  
an introduction to variant synthetic data for LLM papers

**Fine-Tuning LVLM ... Using RL (2405.10292)**  
RL+VLM  
**Controlling LLM Agents with Entropic Activation Steering (2406.00244)**  
to increase diversity of LLM output; train a "steering vector"  
**InterCode: Standardizing and Benchmarking Interactive Code with Execution Feedback (2306.14898)**  

**ToolAlpaca (2306.05301)**  

**ToolFormer (2302.04761)**  

**In-Context AE for Context Compression in LLM (2307.06945)**  

**AutoAct (2401.05268)**  

**AutoGen (2308.08155)**
  
**LLM can Strategically Deceive their Users (2311.07590)**  

The first work that shows LLM cheats even when not instructed to do so but under high pressure from human instructors

**Executable Code Actions Elicit Better LLM Agents (2402.01030)**  

use code as action, makes trajectory more compact and minimizes the number of steps

**Data-Copilot (2306.07209)**  
**WavCraft: Audio Editing & Generation with LLMs (2403.09527)**  

这篇paper是整合了声音输入和编辑的multimodal model，他的编辑是利用了tool API（利用提前整合好的instruction template），输入利用了专门的audio analysis module。实际上外接了大量的已有处理声音的模型。

**SAGE: Semantic and Actionable Parts for Generalizable Manipulation of Articulated Objects (2312.01307)**



**Simulating Opinion Dynamics with Networks of LLM-based Agents (2311.09618)**


**Towards Unified Alignment Between Agents, Humans and Environment (2402.07744)**

大部分篇幅描述了一种设计LLM agent的原则，即要认识到人类的目的、要对环境dynamics有认知和要满足经济性等约束。他也提出了一种方法，从过去成功的traj里抽取关键动作，然后根据关键动作的走向匹配最接近的成功案例这样。

**Self-Training Language Models in Arithmetic Reasoning (ICLR 2024 workshop)**
proposes calcX, a new dataset; use calculator API and self-training with preference optimization.

**Reinforced Self-Training (REST) for LLM (2308.08998)**
从本质上是一种curriculum learning（逐步提高接受data的bar）+rejection sampling（只有return足够高的data才会被接受进入训练）

**LLM can self-improve (2023 emnlp)**
跟我们的想法有一点像，他通过voting选出来的最好的reasoning path会被加入到之后的training samples里面

**Agents: An Open-Source Framework for Autonomous Language Agents (2309.07870)**
multi-agent，是一种“基础模型”，它整合了tool use、multi-agent、HCI、symbolic等内容，可以被用于后续的训练中。
**LEAGUE++ (ICLR 2024)**
**Agent LUMOS: Unified and Modualr Training for Open-Source Language Agents**
**R2E: Turning any Github Repository into a Programming Agent Environment**
**Beyond A\*: Better Planning with Transformers via Search Dynamics Bootstrapping**
**FINMEM: LLM trading agent (2311.13743)**
**Travel Planner: A Benchmark for Real-World Planning with Language Agents**
基本上就是一个新的testbed，没有提出新方法。
**Towards General Computer Control: RDR II**
整个pipeline大概分为self-reflection，task inference，然后coding得到对应的操作，再根据操作做planning得到最终要做的动作。总的来说就是一种比较高级的以code作为action的react。实际上和elicity一文（2402.01030）有点类似。
**Mobile Agent: Autonomous Multi-Modal Mobile Device Agent with Visual Perception**
**LLF-Bench: Interactive Learning from Language Feedback**
**AgentOhana: Design Unified Data and Training Pipeline for Effective Agent Learning**
**Can Large Language Models be Good path planners?**
**Human-inspired reading agent with Gist Memory**
**The ART of LLM Refinement: Ask, Refine, and Trust**
**Do LLM Agents have Regret? A case study in online learning and games**
有些情况下是无regret的，但是也有很简单的情况会有regret
**If LLM is the Wizard, then code is the wand**
**AgentBoard: An analytical evaluation board of Multi-turn LLM  Agents**
A testbed parallel to agentbench
**Is it possible to edit LLM robustly?**


----------------------------------------------------------------------------------------------------------------------------------------

# Chain of Thought

Vanilla: Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. In NeurIPS, 2022.

thought-observation-action: ReAct: Synergizing Reasoning and Acting in Language Models. In ICLR, 2023.

in plan and out-of-plan: AdaPlanner: Adaptive Planning from Feedback with Language Models. In ArXiv, 2023.

long-term memory by self-reflection: Reflexion: Language Agents with Verbal Reinforcement Learning. In ArXiv, 2023.

majority vote: Complexity-Based Prompting for Multi-Step Reasoning. In ICLR, 2023.

Self-consistency: Self-Consistency Improves Chain of Thought Reasoning in Language Models. In ICLR, 2023.

Tree of thought: Tree of Thoughts: Deliberate Problem Solving with Large Language Models. In ArXiv, 2023.

Graph of thought: Graph of Thoughts: Solving Elaborate Problems with Large Language Model. In ArXiv, 2023.

Algorithm-of-Thought: Algorithm of Thoughts: Enhancing Exploration of Ideas in Large Language Models. In ArXiv, 2023.

Skeleton-of-thought: 



Cumulative Reasoning With Large Language Models. In ArXiv, 2023.



# RL for LLM

## Policy Gradient and Actor-Critic

PPO + nucleus sampling to narrow down possible actions: Is Reinforcement Learning (Not) for Natural Language Processing: Benchmarks, Baselines, and Building Blocks for Natural Language Policy Optimization. In ICLR, 2023.

explore in the vicinity: Semi-Offline Reinforcement Learning for Optimized Text Generation. In ICML, 2023.

## MCTS

Reasoning with Language Model is Planning with World Model. In ArXiv, 2023.

## In-context RL

In-context Reinforcement Learning with Algorithm Distillation. In NeurIPS, 2022.

Transformers as Decision Makers: Provable In-Context Reinforcement Learning via Supervised Pretraining

Emergent agentic transformer from chain of hindsight experience

Emergence of In-Context Reinforcement Learning from Noise Distillation

AMAGO: Scalable In-Context Reinforcement Learning for Adaptive Agents

In-Context Reinforcement Learning for Variable Action Spaces

Supervised Pretraining Can Learn In-Context Reinforcement Learning, in NeurIPS, 2023.

In-context Exploration-Exploitation for Reinforcement Learning
