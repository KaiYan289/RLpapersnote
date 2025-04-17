https://www.promptingguide.ai/

https://github.com/rxlqn/awesome-llm-self-reflection

UPRISE: Universal Prompt Retrieval for Improving Zero-Shot Evaluation



# New Papers

**What’s Behind PPO’s Collapse in Long-CoT? Value Optimization Holds the Secret (2503.01491)**

**The Evolution of LLM Adoption in Industry Data Curation Practices (2503.01491)**

**Demystifying Long Chain-of-Thought Reasoning in LLMs (2405.09798)**

**Magnet: Multi-turn Tool-use Data Synthesis and Distillation via Graph Translation (2503.07826)**

**The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization (2403.17031)**

## Many-shot

## Self-train

## Math

## Reflection

**Many-shot In-Context Learning (2404.11018)**

放了最多2048个example到context里面。不同的顺序对不同的子任务performance不一样，但平均起来看没有什么太大区别。

**Buffer-of-Thoughts** 超高的game of 24成功率
基本的想法是生成一些“template”然后在解决问题时取用。

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

Early attempt of solving math problem. Basically standard RLHF. Tried by-step and overall feedback. It seems that RM-weighted return is much greater than majority or greedy answers. 

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

Another paper which uses agent code command as action and interpreter as environment to interact.

**ToolAlpaca (2306.05301)**  

经典的framework，它在自己的语料库上训练，其中包含了大量的tool use。

**ToolFormer (2302.04761)**  

和toolalpaca类似，但是是更早的工作，没有那么成熟。

**In-Context AE for Context Compression in LLM (2307.06945)**  

实际上和activation beacon或者recurrent transformer都很像， 总的来说就是一些特殊的、在上一个chunk结尾继承了大部分信息的token。

**AutoAct (2401.05268)**  

autoact同样是分为几个agent：meta-agent, plan-agent，tool-agent和reflect-agent。meta-agent会负责选择合适的工具，然后还会用react生成一些成功的trajectory作为training source。在这个基础上用lora finetune其他三个agent。比起之前诸如fireact和lumos这样的工作，autoact不需要使用GPT-4。autoact在Tab.1里测试了大量LLM agent，可以参考。

**AutoGen (2308.08155)**

同样是分为几个agent：assistant， proxy和groupchat。user和assistant两个agent会互相对话，proxy会register一些函数，而assistant则会运行这些函数来返回结果。proxy同时也有python interpreter作为assistant的feedback。 group manager会动态地添加agent/维持一组agent的交流。

**LLM can Strategically Deceive their Users (2311.07590)**  

The first work that shows LLM cheats even when not instructed to do so but under high pressure from human instructors

**Executable Code Actions Elicit Better LLM Agents (2402.01030)**  

use code as action, makes trajectory more compact and minimizes the number of steps

**Data-Copilot (2306.07209)**  

data-copilot是一个用来实时处理大规模信息的LLM agent。它会自动生成代码来处理这些消息，然后调用一些之前设计好的处理接口。这些接口本身也是在搜索数据时LLM agent自己建立起来的。

**WavCraft: Audio Editing & Generation with LLMs (2403.09527)**  

这篇paper是整合了声音输入和编辑的multimodal model，他的编辑是利用了tool API（利用提前整合好的instruction template），输入利用了专门的audio analysis module。实际上外接了大量的已有处理声音的模型。

**SAGE: Semantic and Actionable Parts for Generalizable Manipulation of Articulated Objects (2312.01307)**

实际上是一个visual input的agent，它自己可以描述场景。它调用3D segmentation模型以获知在一个物体上哪些部位是可以动的，这样就可以把现在可以做的action补充到描述里。然后LLM作为一个不断接受feedback的agent来输出policy，使用别的经典算法来估测需要移动的幅度作为反馈。输出是一个tuple，包含目标、动作和参数，这些东西会被送进motion planner。

**Simulating Opinion Dynamics with Networks of LLM-based Agents (2311.09618)**

实际上是一个偏向社会学的工作，试图通过LLM的自然语言来描述人类的心态进而考虑其行为，而不是简单地用先验的公式来描述人类行为。LLM population里每个agent的想法写出来实际上就是模拟每个人内心的想法。当然，所有的信息交换也是通过模拟的twitter实现的。

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

LLM robotic agent，使用形式化的语言描述plan，生成reward，同时调用一个semantic skills library。这个library会被rollout中余弦相似度最像的traj不断更新。

**Agent LUMOS: Unified and Modular Training for Open-Source Language Agents**

LUMOS features a learnable, unified and modular architecture with a planning module that learns highlevel subgoal generation, and a grounding module trained to translate these into the actions using various tools in the execution module.

仍然是同一个套路：一个planning-agent自然语言生成high-level plan，再用grounding agent细化成标准化动作，最后用各种API实现。

**R2E: Turning any Github Repository into a Programming Agent Environment**

对LLM的自动化testbed生成器。对于每个repo，首先找出一些“有意思的函数”（注意并不是整个repo），然后收集它们的context；利用prompted 程序分析生成testing harness，即自动化测试框架。注意到这里并不是简单地生成输入输出，而是包含了一整套外在的config。

**Beyond A\*: Better Planning with Transformers via Search Dynamics Bootstrapping (2402.14083)**

训练一个transformer来预测search trace。测试了两种情况，一种是先输出trace后输出plan，一种是直接输出plan。显然前者的正确率更高。

**FINMEM: LLM trading agent (2311.13743)**

一个llm trading agent，能够处理不同类型、特别是时效性不同的金融数据。它分为profiling，memory和decision-making三个模块。其实profiling大概就是一个prompt，描述了一些过去的信息。然后有不同agent扮演不同风险爱好的决策者。memory模块会结合每天的新闻、股价、公司报告等选出最相关的信息（综合自不同时效），并且根据时效其重要性会指数衰减。同时，agent还会做reflection。decision-making就是简单地从buy/sell/hold里选一个动作。

**Travel Planner: A Benchmark for Real-World Planning with Language Agents**
基本上就是一个新的testbed，没有提出新方法。

**Towards General Computer Control: RDR II (2402.01030)**

整个pipeline大概分为self-reflection，task inference，然后coding得到对应的操作，再根据操作做planning得到最终要做的动作。总的来说就是一种比较高级的以code作为action的react。实际上和elicity一文有点类似。

**Mobile Agent: Autonomous Multi-Modal Mobile Device Agent with Visual Perception**

一个基于GPT-4V和一些辅助工具生成的手机助手task。不过没有测试baseline。

**LLF-Bench: Interactive Learning from Language Feedback(2312.06853)**

一个新的benchmark，在一般agent环境的基础上添加了语言feedback。

**AgentOhana: Design Unified Data and Training Pipeline for Effective Agent Learning**

它把来自不同环境的trajectory综合成一种恒定的格式，从而生成一种新的dataloader。与此同时利用这个dataloader训练了一个新模型xLAM-v0.1。

**Can Large Language Models be Good path planners? (2310.03249)**

测试了一系列LLM在gridworld navigation上的performance。总的来说，LLM的表现并不好——需要situated spatial info和持续不断的反馈。如果是finetune的LLM，则generalizability不太好。相对来说，ReAct效果不错。不过需要注意到这篇文章比较早了，现在可能有更好的方法。

**Human-inspired reading agent with Gist Memory**

基本上就是让llm缩写context之后把缩写结果扔到context里面然后再RAG。需要把一个很长的文章用LLM分段，然后对每一段做缩写。在提问的时候，prompt llm问他需不需要再仔细看某一页的内容。这篇文章里提到了许多context compression的方法。

**The ART of LLM Refinement: Ask, Refine, and Trust**

首先生成一个初始的答案，然后让一个提问题的LLM基于这个问题分解出一系列小问题，如果都能回答上来则直接输出，否则基于这些小问题修正答案。如果修正了答案，那么让一个truster LLM来评价一下是初始答案更好还是后面的答案更好。

**Do LLM Agents have Regret? A case study in online learning and games**

研究了一些简单的bandit环境。有些情况下是无regret的，但是也有很简单的情况会有regret

**If LLM is the Wizard, then code is the wand (2401.00812)**

是一篇survey，提到了code是如何让LLM变得更好的。具体地说，LLM可以直接写code、评价code，可以做program-of-thought，可以辅助CoT做更好的task decomposition，可以建立reasoning graph，辅助vision input，使用工具，从interpreter那里得到反馈，做env的感知和planning，作为action，组织memory或者是自我改进。

**AgentBoard: An analytical evaluation board of Multi-turn LLM  Agents**

A testbed parallel to agentbench

**Is it possible to edit LLM robustly?**

研究是否在finetune改变llm的某个认知之后会被用户几句话给拐回来。事实表明，越是接近基础的认知越不容易被robustly edited，也就是越容易被用户拐回来。


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
