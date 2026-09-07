# Q12 什么时候该用微调（Fine-tuning），什么时候该用 RAG？

`模块: 四 模型优化与训练` `难度: ★★★☆☆` `掌握度: 🟡待巩固` `首答: 2026-09-06` `复习: 0 次` `上次复习: —` `基线核查: 随首答 2026-09-07`

> **判定**（维护方，教师 Claude 停用后第一题）：首答**从机制答起而不是背口号**——「就算调了，某一年的财报多少钱，模型照样是概率输出」「匹配到的塞入的是原文」，这是把模块三的「经不经过模型」直接拿到新题上用，没人提醒。及格线那句「风格靠微调、准确知识靠 RAG」是推出来的，不是背的。
> 三处待补：① 出题人埋的五个判据只推出两个（改得快 / 数据变了要重来），回滚、出处、权限一个没碰，**也没做「答完对一遍」这个动作**；② 被点名后三个方向都推对，但**每题只答了改权重那一边**；③ 「几乎不能撤回」会被面试官抓住——旧权重换回去可以，撤不了的是**一条**。
> 📌 **本题最有价值的两步都是用户自己走的**：把「私有知识」从「数据」推到「做法」（第 2.5 节第三轮），以及在基线核查后判出「这题在当下的语境不太成立」（第七轮）——后者正是 2026 版答案的第一句，也是本仓第一次由用户抓出维护方的基线落后。
> ⚠️ **维护方两处流程失误**：生产级第一版停在 2024–2025 的材料，用户「你不能拿 2024 年的东西来糊弄我」之后才做基线核查（第 7.6 节）；讲解两轮过载，用户「难以收敛」「我混乱了」，最后收成三句才落地。两条都记入 [CLAUDE.md](../CLAUDE.md) ③。

---

## ⚡ 30 秒速览（复习时只看这一块）

- **一句话回答**：三句。① 一次调用只有两样东西决定输出：权重、输入。微调改权重，RAG 改输入。② **权重里没有「一条」这个单位，上下文里有**——所以按条管的（事实、会变的、要撤要引要分权限的）走上下文，写不出来、只能拿例子示范的「怎么做」才进权重。③ 2026 年多一句前置：**先问有没有权重**。没有（用别人的 API），题只剩「上下文怎么组织」；有（自己部署的开源模型），才谈微调，而且只管行为。
- **必须覆盖的关键词**：`权重 vs 输入` `没有「一条」` `撤回的最小单位` `出处 / 权限按条` `写得出来吗` `50 条例子先试` `塞事实进权重会带坏别的` `先问有没有权重` `后训练工具链（LoRA / 蒸馏）` `持续学习是研究前沿` `1M 标准定价 + 缓存` `长度本身伤` `即时拿 / 上下文工程`
- **上次的失分点**（首答 2026-09-06）：五个判据推了两个，没对账；「几乎不能撤回」——撤回的最小单位是「整个版本」不是「不能撤」；三个判据只答改权重那一边；「私有知识」举了 SOP，而 SOP 写得出来、Q11 自己说过它是提示词；「吧」又出现一次。
- **答题骨架**：① 先判题：你有没有权重 → ② 一句话定义：微调改权重，RAG 改输入 → ③ 判据表，底下一个原因「没有一条」 → ④ 什么时候 RAG / 什么时候微调：写得出来吗 → ⑤ 证据：两篇论文 + 「内部文档训进权重」那张表 → ⑥ 2026 现状：闭源门在关 / 开源后训练在长 / 持续学习是研究前沿 / 窗口 1M 但全塞不是答案。

---

## 1. 题目原文

> 第十二题
> 什么时候该用微调（Fine-tuning），什么时候该用 RAG？

**出题人的框定**（教师 Claude，2026-09-05，原话）：

> 先说清楚这题为什么值得认真答：它是这份面经里最容易背〔模板化答案〕、也最容易被追问穿的一道。
> 标准答案人人会背："RAG 管知识，微调管风格"。但面试官只要追一句"为什么"，八成人就答不下去了。
> 你答的时候，我要你想清楚一件事——这两个东西，在物理上到底改的是什么？
> 回想我们前面推过的：一次模型调用，只有两样东西决定输出。一样是权重，一样是那一坨输入的 token。
> 微调改前者，RAG 改后者。
> 从这个区分出发，几乎所有判据都能自己推出来：哪个改得快、哪个改完能回滚、哪个能引用出处、哪个数据一变就得重来、哪个能做权限控制。
> 你先答，不用长，把你能推出来的推出来。 你有基础：你知道 SFT 是几百上千条问答对改变模型行为，你也真做过 RAG。

> 📌 上一段引文中〔〕内一词按[规范](../CLAUDE.md)第 5.5 节校正，出题人原文见[归档](../_原始记录/2026-09-05-Q11全过程与模块三总结.md)。

**出题人自陈的待查项**（原话）：

> 一是 2026 年 8 月微调这块的现状（LoRA 之后的做法、还有没有人真的做全量微调），二是"上下文窗口都这么大了、还需不需要微调"这个当下的真问题。

**来源**：题面见 [2026-09-05 归档](../_原始记录/2026-09-05-Q11全过程与模块三总结.md)；作答全过程见 [2026-09-06 归档](../_原始记录/2026-09-06-Q12-微调vsRAG作答.md)（八轮，跨 09-06 与 09-07）。本题起由维护方在本会话出题与讲解（[CLAUDE.md](../CLAUDE.md) 第 6 节第 4 条）。

**考察意图**：出题人自己点破了——它是**最容易背、也最容易被追问穿**的一题。及格线是那句口号；分水岭在「为什么」：能不能从「改的是权重还是输入」把判据推出来。**2026 年的分水岭又往前挪了一格**：这题默认两样都摆在你面前，而对用闭源前沿 API 的人，权重那一边已经基本没得选——先判题再答题，才是这题的最高一层（第 4.5 节）。

---

## 2. 我的原始回答（不修饰，保留原样）

> 微调涉及到的很多，当然也很重，而且如果知识更新很快，微调的速度还达不到知识更新的频率，而且就算调了，那也不是模型能准确答出来的东西，某一年的财报多少钱，模型照样是概率输出。SFT监督微调，能让模型学到回答的风格，但是准确的知识，这一点又是在看运气了。
> RAG不太一样，它依赖的是外部的向量数据库，虽然也是语义匹配，但是可以加混合检索，匹配到的塞入的是原文，是准确的信息进入到上下文，还是财报的例子，匹配对了之后，模型拿到的不是猜的数据，回答会更准确。

---

## 2.5 我的追问与推导（原文，共七轮）

**第二轮**（三个判据的推导；维护方问的是「改权重 / 改输入」两边，各一句）：

> 我怎么觉得微调是伪命题，除非你数据对特别多，而且不是准确知识，是私有知识，用微调没毛病。但是微调了之后，几乎不能撤回了吧？数字从哪来他也不知道，没有事实来源。改了权重，所有人只要用这个模型，就能得到答案。

**第三轮**（追问「私有知识指哪一种」）：

> 私有知识，我可能更愿意说的是，一些SOP的流程这种，比如说，我们公司在做的是什么，这种长期的，特定私有的知识，而非准确的数据。

**第四轮**（2026-09-07）：

> 你要不给我一份完整的内容，我觉得现在有点难以收敛

**第五轮**（维护方给出完整版之后）：

> 你不能拿2024年的东西来糊弄我

**第六轮**（维护方按 2026 材料重讲之后）：

> 那么这个问题，听上去不是特别成立了，在当下的语境下

**第七轮**：

> 现在信息太多了，我混乱了

**第八轮**（收尾，用户自己的理解）：

> 这样吧，我来说我的理解。
> 微调，Lora我不知道在干嘛，SFT在定义模型的回答风格，是去定义模型的表现行为，你倾向于什么样子的回复风格，全面？温和？笃定？等等，后训练是在改变模型的权重，这些目前都是开源模型可以做的事情。
>
> RAG，就是额外的提示词，并不改变模型的权重，本质上就是上下文工程，让模型看到内容，并不改变模型的行为风格（除非提示词告知，但是也是概率的）

---

## 3. 诊断

### 3.1 首答

| 维度 | 我的表现 | 差距在哪 |
| --- | --- | --- |
| 概念准确性 | 机制层对：权重里的知识是概率输出，上下文里的原文不用猜；SFT 学的是回答风格 | 「微调很重」「速度跟不上」是结果，没说到原因（改权重 = 要重训、没有条目单位） |
| 深度（有没有到原理层） | **用了模块三的判据**「经过模型的都是概率」，比题库口号深一层 | 停在「准确 / 不准确」，没往下推到「按条管理」——回滚、出处、权限三个判据一个没碰 |
| 结构 | 微调一段、RAG 一段，对照清楚 | 出题人列了五个判据，**答完没对账**；作答前自查里写着「逐项数一数」，题目档也提醒过 |
| 工程落地感 | 财报数字这个例子贴切；「可以加混合检索」是 Q07 的资产 | 「匹配对了之后」是个前提，没说匹配错了怎么办（RAG 的风险从生成侧挪到了检索侧） |

**最致命的一条**：**钥匙在手上却只开了一把锁。** 「微调改权重、RAG 改输入」是出题人白送的原理，首答只用它推了两个判据；被点名后三个方向都推对了，却每题只推改权重那一边。原理会用和把一个原理推到底，是两回事——这与 [[Q09-主子Agent通信与异常|Q09]]「已经推出的判断没拿去用」是同一个形状。

### 3.2 推导与追问质量

| 轮次 | 判定 | 说明 |
| --- | --- | --- |
| 第二轮：三判据推导 | ✅ 方向全对，⚠️ 只答一半 | 撤回 / 出处 / 权限三个方向都从「改权重」推出来了；**改输入那边一句没答**，由维护方补齐（及格线到此才齐）。「几乎不能撤回了吧」两处问题：撤回的最小单位说错（整个版本能撤、一条撤不了）；疑问句包装 |
| 第三轮：私有知识 = SOP | ✅ 方向对，❌ 例子反了 | 从「数据」推到「做法」，正是「是什么 / 怎么做」的分界。但 SOP 写得出来，**Q11 里用户自己说过 skills「本质还是提示词」「像一套 SOP」**——写得出来的 SOP 现在就是靠改输入在做。**「已经推出的判断没拿去修正相邻结论」第四次**（前三次 Q09、Q11、Q10 复习），这次是跨题的 |
| 第四轮：要完整版 | ✅ 合理要求 | 出题人说「你先答，不用长」，讲解应当在此收敛。维护方给的完整版 **停在 2024–2025**（见下一行） |
| 第五轮：「不能拿 2024 年的东西」 | ⭐ **用户抓出维护方的基线落后** | 出题人的待查项明写「2026 年 8 月的现状」，维护方第一版引的是 2023–2025 的论文与博客。规范第 4.6 节的「基线核查」本该由维护方主动做，这次是用户反向执行了护航员的职责。记为维护方教训，用户加分 |
| 第六轮：「这题不太成立了」 | ⭐ **判题能力**，⚠️ 语气 | 从 2026 材料自己推出「二选一对没权重的人不成立」，这是答案的第一句。判断用了「听上去不是特别」的包装——同一习惯（[[Q06-FunctionCalling与RAG\|Q06]]、[[Q08-Agent自主运行\|Q08]]） |
| 第七轮：「我混乱了」 | — | 维护方过载第二次（第一次 2026-09-05 Q11 追问）。收成三句后落地 |
| 第八轮：收尾理解 | ✅ 收题 | 三个「为什么」都在：改不改权重、行为还是内容、有权重才谈微调；「RAG 本质上就是上下文工程」把 2026 的上位词接上了。一处校准：**微调后的行为同样是概率**（第 4.3 节）。「Lora我不知道在干嘛」照实说了——进待查清单，存在但未学 |

---

## 4. 完整答案（面试可直接复述）

### 4.1 一句话定义

一次调用只有两样东西决定输出：**权重**和**输入**。微调改权重，RAG 改输入。所有判据都从这一句推；2026 年再加一句前置——**先问你有没有权重**。

### 4.2 判据表，底下只有一个原因

| 判据 | 改权重（微调） | 改输入（RAG / 上下文） |
| --- | --- | --- |
| 改得快 | 要重训 | 换文档，下一次调用就生效 |
| **撤回的最小单位** | **整个版本**：旧权重换回去可以，撤一条不行 | **一条** |
| 出处 | 没有，数字是回忆出来的 | 有，原文和来源一起进上下文 |
| 数据变了 | 重来 | 换文档 |
| 权限 | 一份权重谁用都一样 | 检索时按人过滤 |
| 每次调用的花费 | 训练时付清，调用不多花 | 每次都把资料塞进上下文，每次付；缓存读取只收一成（Anthropic 现行定价） |

底下只有一个原因：**权重里没有「一条」这个单位，上下文里有。** 训练是把所有样本的梯度加到同一组参数上，那条财报数字和几百上千条别的一起揉进去了，没有地址；上下文是有位置的 token 序列，每条可寻址、可删、可引、可过滤。撤不了一条、指不出一条、限不住一条，都是它。

同一个性质换个方向看就是微调的长处：**风格、格式、判断习惯本来就不是「一条一条」的东西**，整体改正好合适。

### 4.3 什么时候 RAG，什么时候微调

**什么时候 RAG**：事实、会变的、要撤要引要分权限的，全走上下文。

**什么时候微调**：写不出来、只能拿几百个例子示范的「怎么做」——口吻、格式的一致性、一个领域里的判断习惯。提示词写十条它还飘，给几百个例子它就稳。还有一种：大模型带着长提示词能做的事，想让小模型便宜、快地做同一件事，就把「怎么做」从每次付钱的上下文搬进一次付清的权重（蒸馏）。

**一把尺子**：**能写成一份文档交给模型的，进上下文；只能示范、写不出来的，进权重。** 用户举的「SOP 流程」「公司在做什么」都写得出来——前者正是 Q11 里 skills 的形态，后者是一段会变的事实——所以都走上下文；「私有」不是轴，「写不写得出来」才是。

**门槛不是数据多**：OpenAI 的微调文档让你先拿 50 条精心写的例子试，「If 50 examples have no impact, rethink your task or prompt before adding training data」。

**一处校准**（用户收尾时说「提示词改风格也是概率的」）：**微调后的行为同样是概率。** 权重只是让那种风格更可能出现，不是保证。真正的差别是三点：更稳、不占上下文、不会像 [[Q10-上下文漂移与工具幻觉|Q10]] 说的那样被后面的 token 压过去。面试官追「微调后就确定了吗」，答「不，还是采样，只是分布挪了」。

### 4.4 证据：塞事实进权重不只学不牢，还带坏别的

- **2023-12（Ovadia 等）**：直接对比无监督微调与 RAG 的知识注入，「RAG consistently outperforms it, both for existing knowledge encountered during training and entirely new knowledge」；并且「LLMs struggle to learn new factual information through unsupervised fine-tuning」，要「numerous variations of the same fact」才记得住。
- **2024-05（Gekhman 等）**：微调时引入新知识的样本「learned significantly slower」；一旦学会，「linearly increase the model's tendency to hallucinate」。
- **2026-04（Kaplan、Gekhman 等，8 月底修订）**：把微调致幻定性为**新旧事实语义重叠互相干扰**（「forgetting grows with the overlap between new and stored facts」）；缓解手段是自蒸馏式 SFT，不需要新知识时冻住相应参数组。

这就是首答那句「准确的知识看运气」的三代数据版：不只学不牢，学进去会把别的带坏。

### 4.5 2026 年的前置判据：先问有没有权重

用闭源前沿 API 的人，2026 年权重那一边基本没得选（自助入口在关，企业渠道只剩零星几条）：

| 厂商 | 2026-09 状态（官方页面） |
| --- | --- |
| OpenAI | 微调指南顶部：「OpenAI is winding down the fine-tuning platform. The platform is no longer accessible to new users」；弃用页时间线：**2026-05-07** 起没做过微调的组织不能新建任务，**2026-07-02** 起 60 天内没用过微调模型的也不能，**2027-01-06** 起老客户也不能新建。能微调的只有 4.1 系（SFT / DPO）、o4-mini（RFT）、gpt-4o（视觉），GPT-5 系一个都没有 |
| Anthropic | 官方术语表：「The Claude API does not currently offer fine-tuning」；唯一渠道是 2024-11 起 Bedrock 上的 Claude 3 Haiku |
| Google | Gemini API 文档：「With the deprecation of Gemini 1.5 Flash-001 in May 2025, we no longer have a model available which supports fine-tuning in the Gemini API or AI Studio」；微调收进企业云平台（Vertex 上的 Gemini 3 系细节见待查） |

所以对用 API 的人，「什么时候微调」的答案是**你没得选**：题塌成「上下文怎么组织」。这不是这题作废，是它的第一句变了——**先判题，再答题**。

### 4.6 有权重的人：微调搬进了「后训练」工具链

开源权重私有化部署的团队（Qwen、DeepSeek 一类），权重在自己手里，微调照做，工具链反而在长：

- **LoRA 是默认**（新词，一句话：不动原来的权重，旁边加一小块可拆的权重，训练只动这一小块）。Thinking Machines 2025-09：后训练量级的数据上「LoRA learns with the same sample efficiency as FullFT and achieves the same ultimate performance」，强化学习里「even with ranks as low as 1」；他们 2025-10 上线的训练服务干脆只提供 LoRA，「so that we can share the same pool of compute between multiple training runs」。这顺带修正判据表两格：撤回的最小单位从「整个版本」缩到「拔掉那一层」，权限能做到「一个客户一层、一台服务器同时挂很多层」。**仍然不是「一条」，表的结论不变。**
- **On-policy 蒸馏**（Thinking Machines 2025-10-27）：让大模型给小模型的每个 token 打分来训，博客口径比强化学习省 50–100 倍总算力。他们真做了「把内部文档训进权重」：

| 阶段 | 内部知识题 | 指令遵循 |
| --- | --- | --- |
| 原模型（Qwen3-8B） | 18% | 85% |
| 只喂文档 | 43% | 45% |
| 七成文档三成对话 | 36% | 79% |
| 再用原模型蒸馏拉回 | 41% | 83% |

原话：「Training both at once is generally difficult, and light-weight finetunes are often insufficient」。**私有知识进权重能做，代价是两段训练，知识题只到四成，指令遵循先塌再救**；把原文放进上下文，同一道题是九成以上。

- **全量微调没消失**，但对用模型的人已不是默认选项；做模型的一侧（后训练）仍在用。

### 4.7 长上下文 2026：门槛推高五倍，但全塞不是答案

- Anthropic 当前三款主力（Fable 5.1 / Opus 5 / Sonnet 5）全是 **1M** 窗口；2026-03-13 起 1M 按标准定价，「There's no multiplier: a 900K-token request is billed at the same per-token rate as a 9K one」；官方给的用法是「an entire codebase, thousands of pages of contracts, or the full trace of a long-running agent」。缓存读取收一成（Fable 5.1 收 2.5%）。2024-09 那条「知识库小于 20 万 token 就整个放进提示词」的门槛，被推高了五倍。
- **两条反向证据**：① 2025-10（Du 等）：「even when models can perfectly retrieve all relevant information, their performance still degrades substantially (13.9%--85%) as input length increases」——长度本身伤成绩；② Anthropic 2025-09 的做法是「just in time」：「maintain lightweight identifiers (file paths, stored queries, web links, etc.) and use these references to dynamically load data into context at runtime using tools」，Claude Code 靠 glob / grep 即时拿，CLAUDE.md 才是「naively dropped into context up front」；他们 2026 年发的 9 篇工程文章没有一篇讲 RAG 或微调。
- 所以「改输入」这边 2026 年的主流词不是 RAG，是**上下文工程**（[[Q06-FunctionCalling与RAG|Q06]] 的四柱）。窗口变大挤掉的是「预先切块塞向量库」这一种做法，不是检索本身，更挤不到微调。2024 年 Google 团队那篇对比的结论也还在：资源够时全塞平均更好，「RAG's significantly lower cost remains a distinct advantage」，生产上按问题路由。

### 4.8 研究前沿在攻「权重里没有一条」

2025-10 到 2026-08 三条线冲着同一个目标：

- **Meta，2025-10**（稀疏记忆微调）：塞新事实后旧知识「drops by 89% after full finetuning on new facts and 71% with LoRA, sparse memory finetuning yields only an 11% drop」——做法就是给权重分出一格一格的记忆槽，只改被这条新知识点亮的那几格。
- **Google Research，2025-11**（嵌套学习）：原话「knowledge is confined to either the immediate context of their input window or the static information that they learn during pre-training」，要造能持续学的架构（Hope）。
- **Kaplan、Gekhman 等，2026-04**：见 4.4。

方向很清楚：让权重长出「一条」这个单位。**但没有一家进了 API。** 面试里的说法：**持续学习是研究前沿，不是生产选项。**

### 4.9 追问兜底：RAG 有出处，模型就不会编出处了吗

会。引用本身也经过模型（[[Q10-上下文漂移与工具幻觉|Q10]] 的判据）。所以做法是让软件把引用锁到上下文里真实存在的原句上——Anthropic 的引用功能：「ground its answers in source documents」，给出「detailed references to the exact sentences and passages」。不经过模型的那一步才可靠。

---

## 5. 第一性原理（往下再挖一层）

- **权重是一团，上下文是一条条。** 训练把每条样本的梯度加到同一组参数上，学到的东西没有地址；上下文是有位置的 token 序列，每个 token 知道自己在哪。于是「按条」的三件事（撤、引、限）只能在上下文里做，「整体」的事（风格、格式、判断习惯）在权重里做才划算。持续学习研究做的，就是给权重造地址（记忆槽）。
- **经过模型的都是概率**（[[Q10-上下文漂移与工具幻觉|Q10]]）：权重里的事实靠**回忆**，上下文里的靠**抄写**。抄比回忆可靠得多，但仍经过模型——所以 RAG 之后数字也不是 100%，引用也会编，要靠不经过模型的那一步锁定。
- **每次付 vs 一次付清**：上下文每次调用都付 token，权重训练时付清。缓存把前者压到一成，1M 把「全塞」的门槛推高，但长度本身仍伤成绩——所以「即时拿」而不是「全塞」成了默认。
- **不这么做会怎样**：把事实训进权重——学得慢、学会后幻觉率线性上升、撤不了、指不出、限不住；把风格写进提示词——每次付钱、被后面的 token 压过去、换个软件就没了。两边的失败形态互为镜像，这就是判据表的来历。

---

## 6. 追问链（面试官大概率会往下问）

| # | 追问 | 答法要点 | 自查 |
| --- | --- | --- | --- |
| 1 | 为什么「RAG 管知识、微调管风格」？ | **微调改权重，RAG 改输入**；权重里没有「一条」，上下文里有。事实要按条管（撤 / 引 / 限），风格本来就不是一条一条的 | ☑ |
| 2 | 微调后能撤回吗？ | **能整体撤，不能撤一条。** 旧权重换回去和任何上线回滚一样；那条事实和几百条别的揉在一起，拿不出来单独那条。判据是撤回的最小单位 | ☐ |
| 3 | 微调后的模型能说出数字从哪来吗？ | 不能，数字是回忆出来的。RAG 的原文和来源一起进上下文，答的时候指得出来 | ☑（权重侧） |
| 4 | 同一份资料，A 能看 B 不能看，怎么做？ | 权重做不到——一份权重谁用都一样；RAG 在检索时按人过滤。LoRA 能做到「一个客户一层」，仍不是按条 | ☑（权重侧） |
| 5 | 私有知识该微调吗？ | **「私有」不是轴，「写不写得出来」才是。** 写得出来的（SOP、公司介绍、财报）进上下文；只能示范的（口吻、格式、判断习惯）进权重 | ☐ |
| 6 | 微调要多少数据？ | 门槛不是多。OpenAI：先 50 条精心写的试，没效果回头改任务或提示词 | ☐ |
| 7 | 微调塞知识为什么不行？ | 不只学不牢，还带坏别的：新知识学得慢，学会后幻觉线性上升（2024）；主因是新旧事实语义重叠干扰（2026） | ☑（首答「看运气」+ 概率输出） |
| 8 | 那 Thinking Machines 不是把内部文档训进去了？ | 能做，代价是两段训练：知识题 18%→43%，指令遵循 85%→45%，再蒸馏拉回 83% / 41%。知识只到四成，原文进上下文是九成以上 | ☐ |
| 9 | 2026 年还能微调吗？ | **先问有没有权重。** 闭源 API：OpenAI 2026-05 起收口、2027-01 关；Anthropic 不提供；Google 收进云。开源权重：LoRA 默认 + 蒸馏，后训练工具链在长 | ☑（「开源模型可以做」） |
| 10 | 上下文窗口都 1M 了，还要 RAG 吗？ | 窗口变大挤掉的是「预先切块塞向量库」，不是检索本身。长度本身伤成绩（13.9%–85%），主流做法是即时拿：留标识、运行时用工具取 | ☐ |
| 11 | 那还要微调吗？ | 窗口再大也只是能塞更多「是什么」，写不出来的「怎么做」它塞不进去。窗口变化在改输入这一边，挤不到微调 | ☐ |
| 12 | 微调后的行为就确定了吗？ | 不，还是采样，只是分布挪了。差别是更稳、不占上下文、不会被后面的 token 压过去 | ☐ |
| 13 | RAG 有出处，模型就不会编出处了？ | 会，引用也经过模型。让软件把引用锁到上下文里真实存在的原句上，不经过模型的那一步才可靠 | ☐ |
| 14 | 持续学习是不是快解决了？ | 研究前沿：稀疏记忆微调（旧知识只掉 11%）、嵌套学习、自蒸馏 SFT。没有一家进 API。**研究前沿，不是生产选项** | ☐ |
| 15 | 这题在 2026 年还成立吗？ | 表面的二选一对没权重的人不成立；题底下考的「改的是什么」不变，反而更能筛人。先判题再答题 | ☑（用户第六轮自己判出） |

---

## 7. 待查清单（我自己挂起的问题）

- [ ] **LoRA 到底在干嘛**——用户原话「Lora我不知道在干嘛」。**存在但未学**：本记录只给了一句话（不动原权重，旁边加一小块可拆的权重）。Q13 会正面碰到，届时讲透。
- [ ] **Google Vertex 上 Gemini 3 系微调的细节**——搜索摘要称 gemini-3.1-flash-lite / 3.5-flash 的 SFT 处于预览、调后推理价为基础模型 1.5 倍、Gemini 3 起建议训练时把思考档位设为最低。**官方页面抓不到正文，三条均未核实。**
- [ ] **GPT-5 系与 Gemini 3.5 Pro 的窗口**——博客称 GPT-5.5 / 5.4 为 1M、Gemini 3.5 Pro 为 2M（2026-05 预览）。Google 文档只见 3.x 为 1M，**2M 未见一手来源**。
- [ ] **国内私有化部署团队做微调的比例**——正文说「你面的公司如果是私有化部署，这题在它那里就是活的」，这是推断，**没有一手来源**。
- [ ] **Gekhman 2024 论文对「为什么幻觉上升」的解释原文**——正文引的是摘要（学得慢、幻觉线性上升）；维护方在对话里说的「它学到的是『不知道也照样答』」是转述，**待读原文核对**。
- [ ] **Thinking Machines 工具链「2026-06 更新到 Qwen3.5」**——来自搜索摘要，博客页面本身未逐字核实。

---

## 7.5 查证记录（2026-09-06 / 07）

### ✅ 核实通过

| 说法 | 核实结果 |
| --- | --- |
| 「一次调用只有权重和输入两样决定输出」 | 与 [[Q08-Agent自主运行\|Q08]]「上下文是伪造的记忆」同一句话：模型每轮只看到权重与本次输入。概念性论断，无需外部来源 |
| Ovadia 等《Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs》 | arXiv:2312.05934，2023-12-10 提交、v3 2024-01-30。摘要原句：「RAG consistently outperforms it, both for existing knowledge encountered during training and entirely new knowledge」「LLMs struggle to learn new factual information through unsupervised fine-tuning, and that exposing them to numerous variations of the same fact during training could alleviate this problem」 |
| Gekhman 等《Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?》 | arXiv:2405.05904，2024-05-09 提交、2024-10-01 修订。摘要原句：「fine-tuning examples that introduce new knowledge are learned significantly slower」「as the examples with new knowledge are eventually learned, they linearly increase the model's tendency to hallucinate」 |
| Kaplan、Gekhman 等《Why Fine-Tuning Encourages Hallucinations and How to Fix It》 | arXiv:2604.15574，v1 2026-04-16、v2 2026-08-31。摘要原句：「a self-distillation-based SFT method…regularizing output-distribution drift」「suppressing factual plasticity by freezing parameter groups」「forgetting grows with the overlap between new and stored facts」 |
| OpenAI 微调文档：50 条起步 | 监督微调指南原句：「We recommend starting with 50 well-crafted demonstrations」「If 50 examples have no impact, rethink your task or prompt before adding training data」；模型优化页：SFT 适用于 classification / nuanced translation / generating content in specific formats / correcting instruction-following failures，DPO 适用于「the right tone and style」，RFT 用「a programmable grader that scores every candidate response」 |
| OpenAI 关闭自助微调 | 微调指南顶部原句：「OpenAI is winding down the fine-tuning platform. The platform is no longer accessible to new users, but existing users of the fine-tuning platform will be able to create training jobs for the coming months」；弃用页三条：2026-05-07「Creating fine-tuning jobs or training is not available to organizations that have not previously run fine-tuning」、2026-07-02「…not run inference on a fine-tuned model in the past 60 days」、2027-01-06「Active existing customers will no longer be able to create new fine-tuning jobs on this date」；可微调模型仅 gpt-4.1 系（SFT / DPO）、o4-mini（RFT）、gpt-4o（视觉） |
| Anthropic 不提供微调 | 官方术语表原句：「The Claude API does not currently offer fine-tuning, but ask your Anthropic contact if you are interested in exploring this option」；AWS 公告：Claude 3 Haiku 微调 2024-11 在 Bedrock 正式可用 |
| Google 公开 API 无可微调模型 | Gemini API 模型微调页原句：「With the deprecation of Gemini 1.5 Flash-001 in May 2025, we no longer have a model available which supports fine-tuning in the Gemini API or AI Studio」，指向企业平台 |
| Thinking Machines《LoRA Without Regret》 | 2025-09-29，Schulman 等。原句：「LoRA learns with the same sample efficiency as FullFT and achieves the same ultimate performance」（后训练量级、中小数据）；「For datasets that exceed LoRA capacity, LoRA underperforms FullFT」；RL「even with ranks as low as 1」；「LoRA performs better when applied to all weight matrices, especially MLP and MoE layers」；多租户「a single inference server can keep many adapters in memory」 |
| Tinker 只提供 LoRA | 2025-10-01 公告原句：「We use LoRA so that we can share the same pool of compute between multiple training runs, lowering costs」 |
| Thinking Machines《On-Policy Distillation》 | 2025-10-27，Kevin Lu 等。「sample trajectories from the student model and use a high-performing teacher to grade each token」；比 RL 快约 7–10 倍、总算力少 50–100 倍（博客口径）；内部文档实验表：Qwen3-8B 18% / 85% → 只喂文档 43% / 45% → 七三混合 36% / 79% → 蒸馏后 41% / 83%；「Training both at once is generally difficult, and light-weight finetunes are often insufficient」；LoRA 在大规模 SFT 下更差、「learns less and forgets less」 |
| Anthropic Contextual Retrieval 的「20 万以内全塞」 | 2024-09-19 原句：「If your knowledge base is smaller than 200,000 tokens (about 500 pages of material), you can just include the entire knowledge base in the prompt」；缓存「reducing latency by over 2x and costs up to 90%」 |
| Anthropic 1M 标准定价 | 2026-03-13 原句：「Claude Opus 4.6 and Sonnet 4.6 now include the full 1M context window at standard pricing」「There's no multiplier: a 900K-token request is billed at the same per-token rate as a 9K one」；模型总览表：Fable 5.1 / Opus 5 / Sonnet 5 窗口 1M、Haiku 4.5 200K；「prompt cache reads cost 10% of the base input price (2.5% on Claude Fable 5.1 and Claude Mythos 5.1)」 |
| Du 等《Context Length Alone Hurts LLM Performance Despite Perfect Retrieval》 | arXiv:2510.05381，2025-10-06。原句：「even when models can perfectly retrieve all relevant information, their performance still degrades substantially (13.9%--85%) as input length increases」 |
| Li 等《Retrieval Augmented Generation or Long-Context LLMs?》 | arXiv:2407.16833，2024-07-23、2024-10-17 修订。原句：「when resourced sufficiently, LC consistently outperforms RAG in terms of average performance. However, RAG's significantly lower cost remains a distinct advantage」；Self-Route 按自评路由 |
| Anthropic《Effective context engineering for AI agents》 | 2025-09-29。原句：「maintain lightweight identifiers (file paths, stored queries, web links, etc.) and use these references to dynamically load data into context at runtime using tools」「Claude Code is an agent that employs this hybrid model: CLAUDE.md files are naively dropped into context up front, while primitives like glob and grep allow it to navigate its environment and retrieve files just-in-time」；工程博客 2026 年 9 篇（01-09 至 04-23）均为 agent / 评测主题，无检索或微调 |
| Lin 等《Continual Learning via Sparse Memory Finetuning》 | arXiv:2510.15103，2025-10-16，作者 Lin、Zettlemoyer、Ghosh、Yih 等（按作者所属推断为 Meta FAIR，页面未列机构）。摘要原句：「NaturalQuestions F1 drops by 89% after full finetuning on new facts and 71% with LoRA, sparse memory finetuning yields only an 11% drop with the same level of new knowledge acquisition」 |
| Google Research《Introducing Nested Learning》 | 2025-11-07，Behrouz & Mirrokni。原句：「knowledge is confined to either the immediate context of their input window or the static information that they learn during pre-training」；Hope 为「a self-modifying recurrent architecture」 |
| Anthropic 引用功能 | claude.com 博客（页面日期 2025-06-23）：「ground its answers in source documents」「detailed references to the exact sentences and passages it uses to generate responses」 |
| 「撤一条」= 机器遗忘仍是开放问题 | 综述 arXiv:2503.01854（2025-03）、arXiv:2601.13264（2026-01）：LLM 级遗忘「remains an open research problem」，有效遗忘常损伤整体能力 |

### ➕ 本次新增（能直接拉高答案深度）

- **「权重里没有一条」**：把五个判据收成一个原因，也解释了为什么同一性质在风格上是长处。
- **「写得出来吗」这把尺子**：替代「知识 / 风格」的二分，直接判 SOP、公司介绍这类边界例子。
- **「先问有没有权重」**：2026 年的前置判据，三家官方页面撑着。
- **内部文档训进权重那张表**：四行数字说清「能做、代价、上限」。
- **长度本身伤成绩 + 即时拿**：回答「窗口都 1M 了」时不落进「全塞」。
- **持续学习是研究前沿，不是生产选项**：三条线一句收。

### 📌 用词校准 / 最该记住的修正

1. **「几乎不能撤回」→「撤回的最小单位是整个版本」**。旧权重还在，整体回滚和任何上线回滚一样；撤不了的是一条。
2. **「私有知识用微调没毛病」→「私有不是轴，写不写得出来才是」**。SOP 写得出来，走上下文（Q11 的 skills 就是它）。
3. **「提示词改风格也是概率的」→ 补另一半「微调后的行为同样是概率」**。差别是更稳、不占上下文、不被后面的 token 压过去。
4. **「权重那扇门关了」要加量词**：只对**闭源前沿 API 的自助入口**成立；企业渠道仍有零星（Bedrock 上的 Claude 3 Haiku、Vertex、OpenAI 老客户到 2027-01），开源权重一侧相反在扩张。
5. **维护方对话中的两处口径**：「2024 年 Google 的对比」——论文页未显示机构，记录里只写作者与年份；「论文的解释是它学到了『不知道也照样答』」——那是维护方转述，摘要只说幻觉倾向线性上升，已进待查。
6. **「听上去不是特别成立」→ 判断句**：「这题在 2026 年要先问一句，你有没有权重。」

### ⚠️ 存疑不背书

- 「LoRA 达到全量微调 95–99% 的效果」「LoRA / QLoRA 是 2026 年唯一现实路径」「典型栈 base → SFT → DPO」——行业博客，无基准出处。
- 「RULER 测出有效窗口只有标称的 50–65%」——博客转述，未见该数字的一手来源；本仓 [[Q03-Token与上下文窗口与上下文腐化\|Q03]] 引 RULER 用的是论文原始口径。
- 「2026 年 6 月已有 13 款 1M 以上窗口的前沿模型」「GPT-5.5 / 5.4 为 1M」「Gemini 3.5 Pro 2M」——博客汇总，未逐一核实。
- 「RFT 在 o4-mini 正式可用、GPT-5 私测」——来自社区帖与搜索摘要；官方页面只列了 o4-mini。
- 「Claude 微调唯一路径是 Bedrock 上的 Claude 3 Haiku」——第三方博客的措辞；本记录改引官方术语表与 AWS 公告两条一手来源。

---

## 7.6 基线核查（随首答 2026-09-07）

> 触发：用户第五轮「你不能拿 2024 年的东西来糊弄我」。出题人的待查项明写「2026 年 8 月的现状」，维护方第一版完整答案引的是 2023–2025 材料——**这是本仓第一次由用户反向执行基线核查**，记入台账。

按四种落后形态查，本题**四种全命中**，外加一条题目集合漂移：

**① 实现层翻代**
- **原文说了什么**：微调 = SFT + LoRA；RAG = 切块入向量库、混合检索。
- **现在是什么**：微调一侧长出 on-policy 蒸馏（2025-10）、稀疏记忆微调（2025-10）、自蒸馏 SFT（2026-04）；RAG 一侧的主流做法是即时拿（2025-09），Anthropic 2026 年的工程文章已不谈 RAG。

**② 被当成常量的变量**
- **原文说了什么**：「知识库 20 万 token 以内直接全塞」（2024-09）。
- **现在是什么**：主力模型 1M 标准定价（2026-03）、缓存读取一成，门槛推高五倍；但长度本身伤成绩（2025-10），所以门槛变了、「全塞不是默认」这个结论没变。

**③ 缺上位概念**
- **原文说了什么**：「RAG」。
- **现在是什么**：上下文工程（[[Q06-FunctionCalling与RAG|Q06]] 四柱），RAG 只是「检索」那一根柱子里的一种做法；用户第八轮自己用了这个词。

**④ 范围外推**
- **差点犯的**：「2026 年权重那扇门关了」——只对闭源前沿 API 的自助入口成立，开源权重一侧相反在扩张。正文已限定为「用闭源前沿 API 的人」。

**⑤ 题目集合漂移**（登记进[台账](../00-索引/基线核查台账.md)「题目集合本身的漂移」与[补充议题](../07-补充议题/README.md)）
- **题面**：「什么时候微调、什么时候 RAG」默认两样都可选。
- **现在**：对用闭源 API 的人二选一不成立，题塌成「上下文怎么组织」；对有权重的人仍成立且更活。**题眼没变（改的是什么），第一句变了（你有没有权重）。** 用户第六轮自己判出。

**出题人两条待查项的结论**：
- 「LoRA 之后的做法、还有没有人做全量微调」→ LoRA 之后是 on-policy 蒸馏与稀疏记忆微调（研究）；全量微调没消失，但对用模型的人不是默认（Thinking Machines 的服务只给 LoRA）。
- 「窗口这么大还要不要微调」→ 问错了对象：窗口变化在改输入这一边，挤掉的是「预先切块塞向量库」，挤不到微调；微调要不要，看有没有权重、是不是写不出来的「怎么做」。

---

## 8. 关联

- **直接父题**：[[Q08-Agent自主运行|Q08]] —— 「一次调用只有权重和输入」就是「上下文是伪造的记忆」的另一种说法；本题把「输入」那一半展开成判据。
- **同一条判据的第四次出现**：[[Q10-上下文漂移与工具幻觉|Q10]] —— 「经不经过模型」。首答用它判「财报数字照样是概率」；校准「微调后的行为同样是概率」；4.9 节引用也要靠不经过模型的锁定。「被压过去」正是提示词改风格与权重改风格的差别。
- **钥匙已在手上（反例）**：[[Q11-MCP与Skill|Q11]] —— 用户在 Q11 说过 skills「像一套 SOP」「本质是提示词」，本题却把 SOP 当成该进权重的私有知识。写得出来的「怎么做」走上下文，Q11 是它的现成例子。
- **上位概念**：[[Q06-FunctionCalling与RAG|Q06]] —— 上下文工程四柱；「经典 RAG 因可预测仍是强默认」在 2026 要加一句「主流做法是即时拿」。
- **检索侧质量**：[[Q07-Embedding与检索策略|Q07]] —— 首答的「可以加混合检索」；「匹配对了之后」这个前提就是 recall@k 天花板。
- **物理底座**：[[Q03-Token与上下文窗口与上下文腐化|Q03]] —— 1M 是第一个数字，缓存改的是第二个，长度本身伤成绩说的是第三个；「全塞」的账要三个数字一起算。
- **幻觉侧**：[[Q04-幻觉成因与治理|Q04]] —— 塞新事实进权重让幻觉线性上升，是四道工序之外的第五种来源（训练阶段引入）。
- **延伸**：[[Q13-SFT与RLHF的破局点|Q13]]（后训练工具链：SFT / RL / 蒸馏；LoRA 在那里讲透）、[[Q14-推理降本与多Agent权衡|Q14]]（蒸馏小模型、缓存、按问题路由）、[[Q15-窗口内放什么与记忆压缩|Q15]]（全塞 vs 即时拿）、[[Q16-混合路由与限流|Q16]]（Self-Route 是路由题的一个实例）。

---

## 9. 核心结论

> **一次调用只有两样东西决定输出：权重和输入。微调改权重，RAG 改输入。权重里没有「一条」这个单位，上下文里有——所以事实、会变的、要撤要引要分权限的全走上下文，只有写不出来、只能拿例子示范的「怎么做」才值得进权重。2026 年先问一句你有没有权重：没有，题只剩「上下文怎么组织」；有，微调是后训练手段，管行为不管知识；知识进权重是研究前沿，不是生产选项。**

> **用户版**（第八轮原话压缩）：微调改权重，定义模型的表现行为，开源模型才能做；RAG 是额外的提示词，本质是上下文工程，让模型看到内容，不改权重。补一句：两边的行为都是概率，权重只是把分布挪了。

---

## 10. 复习记录

| 日期 | 方式 | 掌握度 | 备注 |
| --- | --- | --- | --- |
| 2026-09-06 / 07 | 首答 + 七轮推导与讲解（维护方出题，教师 Claude 停用后第一题） | 🟡 | 首答从机制答起，五判据推二；三判据方向全对但只答改权重一边；SOP 例子与 Q11 自己的结论冲突；用户抓出维护方基线落后（09-07），自己判出「题面在 2026 不成立」；收尾理解正确，一处校准（微调后的行为同样是概率）。原话见 [归档](../_原始记录/2026-09-06-Q12-微调vsRAG作答.md) |
