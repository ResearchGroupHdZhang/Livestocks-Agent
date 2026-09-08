# 基于 PPO–MILP 实证结果的审稿回复策略

更新时间：2026-09-08

## 策略结论

审稿回复应采用“承认 MILP 在离线目标质量上的优势，同时证明 PPO 生成了更稀疏、更节制、更高单位移动效率的方案，并将 PPO 的核心价值落在可扩展/可复用决策上”的叙事。当前数据不支持“PPO 的综合目标优于 MILP”，也暂时不能支持“PPO 严格满足与 MILP 完全一致的 1 kg source-N guard”。

## 结果驱动的主张边界

### 可以立即主张

1. 在同输入 EU/美国上，0.1% gap-certified MILP 为全局目标 $J$ 提供强参照。
2. MILP 的 $J$、N 消解率、环境得分和构成代理均高于 PPO。
3. PPO 使用更少移动量与路线，单位动物 N 消解更高，真实剩余构成 share-L1 更低。
4. 二者 native objective 不同；PPO 的 $J$ 是 post-hoc common rescore。
5. 当前 MILP 的构成代理与真实构成可分离，因此不能只以 $J$ 判断现实方案质量。

### 补齐证据后才能主张

1. PPO 更快：必须获得可复现的单次推理、局部 LP 解码和端到端耗时。
2. PPO 可适应输入变化：必须用冻结 checkpoint 在新情景上验证，不得把重新训练写成泛化。
3. PPO 严格可行：先修复 EU 10 行、美国 88 行的 source-N 小残差，或预注册所有方法统一容差。
4. PPO 在现实综合目标上更优：需要把移动成本/路线复杂度/真实构成纳入预注册评价，而不是事后改变 $J$ 来保证 PPO 获胜。

## 建议回复结构

1. 感谢审稿人提出 exact/optimization baseline 的要求。
2. 说明新增 normalized weighted MILP，并明确它与论文 action-local reward 的差异。
3. 报告 EU/美国认证结果和同输入独立重算表。
4. 坦诚说明 MILP 在其目标 $J$ 上更优。
5. 说明 PPO 的互补优势：更少移动、更少路线、更高 N/animal、更低真实 share-L1。
6. 把结论收缩为“quality certificate versus scalable solution generation”。
7. 说明已识别并将处理 PPO 的 sub-1-kg source-N residuals；修复前不声称严格等价可行。
8. 列出论文中新增的方法、结果、运行时间和限制内容。

## 英文回复初稿

> **Response.** We thank the reviewer for requesting a mathematical-programming reference. We implemented a normalized weighted MILP and compared it with the archived paper-method PPO outputs on exactly matched first-stage national inputs for the EU and the United States. We independently recomputed all outcomes from the serialized final livestock layouts and route records rather than comparing the methods by their logged rewards.
>
> The two methods do not optimize identical native objectives. PPO uses an action-local reward, $4R_{NH_3}+2R_{sens}+R_{PM2.5}$, when selecting a source–destination pair, while a local linear allocator determines the livestock quantities. The MILP directly optimizes the global objective $J=4\hat N-2\hat L+\hat E$, where $\hat N$ is normalized source-N resolution, $\hat L$ is a source-composition proxy loss, and $\hat E$ is the normalized destination environmental score. We therefore report PPO's value of $J$ only as a post-hoc common rescore, not as its training reward.
>
> As expected for a method directly optimizing this objective, the 0.1%-gap-certified MILP achieved a higher $J$ than PPO: 4.7069 versus 3.9641 in the EU and 4.6199 versus 4.2439 in the United States. It also achieved higher source-N resolution (99.9995% versus 93.7148% in the EU; 99.9968% versus 98.3531% in the United States) and higher normalized environmental scores. However, PPO generated substantially sparser and more movement-efficient layouts. It moved 26.1% and 29.4% fewer animals, used 59.3% and 46.9% fewer nonzero routes, and resolved 26.8% and 39.3% more N per moved animal in the EU and United States, respectively. PPO also produced lower actual source-composition share changes, whereas the MILP's optimized linear composition proxy did not always track this physical metric near source depletion.
>
> Our independent audit verified exact species conservation and route-to-inventory consistency for both methods. The archived PPO layouts satisfy destination N and ammonia limits under the common scale-aware tolerance, but 10 EU and 88 U.S. source rows exceed the MILP's explicit 1-kg source-N safety guard; the maximum residual is below 1 kg in both cases. We now report this distinction explicitly and do not label the archived PPO layouts as strictly equivalent-feasible until the residuals are repaired or a method-independent numerical tolerance is preregistered.
>
> These results refine rather than invalidate our contribution. We no longer claim that PPO universally outperforms a converged MILP in objective quality. Instead, the MILP provides certified offline quality references on tractable instances, while PPO produces a different, sparser class of solutions and is intended for reusable low-latency decision generation at scales where dense global certification is expensive. We have added the formulation, solver gaps, independent feasibility checks, movement/route-efficiency metrics, and this limitation to the revised manuscript. We will report training, inference, local-decoding, and end-to-end times separately to avoid comparing one PPO forward pass with the complete MILP solve.

## 中文回复骨架

> 感谢审稿人建议加入传统优化方法作为解质量参照。我们补充实现了归一化加权 MILP，并在 EU 与美国的完全相同第一阶段输入上，与论文 PPO 的最终方案进行了独立物理重算。需要澄清的是，PPO 的逐动作奖励与 MILP 的全局目标并不相同，因此我们没有直接比较原始 reward，而是将两个最终布局统一重算为 N 消解、环境得分、构成变化、移动规模和 MILP 的 post-hoc $J$。
>
> 在 MILP 直接优化的 $J$ 上，达到 0.1% MIP gap 证书的 MILP 优于 PPO；这一结果符合预期。与此同时，PPO 的方案更稀疏：在 EU/美国分别少移动 26.1%/29.4% 动物、少使用 59.3%/46.9% 路线，单位动物 N 消解高 26.8%/39.3%，真实源区域构成变化也更小。因此，MILP 提供离线目标质量证书，而 PPO 展示的是另一类更节制的方案。我们据此删除“PPO 在解质量上普遍优于 MILP”的表述，并将贡献定位为可扩展、可复用的方案生成。

## 尚未写死到回复中的部分

- PPO 时间优势：等待严格计时。
- 澳大利亚：作为 v8 版本限定补充结果，而非与 v9 混成单一主表。
- 中国/巴西：等待 MILP 证书；未认证结果只用于规模讨论。
- PPO source-N 修复后的具体数值：修复完成前不填。
