# 上游代码审计与引用记录 (UPSTREAM)

## 1. 上游来源信息

- **项目名称**：NV-CoT (Thinking with Images as Continuous Actions: Numerical Visual Chain-of-Thought)
- **上游仓库**：[https://github.com/kesenzhao/NV-CoT](https://github.com/kesenzhao/NV-CoT)
- **基准提交 Hash**：`0d36f9c515fc40864c3c4497f5de73148b77b8ac`
- **上游论文**：*Thinking with Images as Continuous Actions: Numerical Visual Chain-of-Thought* ([arXiv:2602.23959](https://arxiv.org/abs/2602.23959))
- **开源许可证**：Apache-2.0 License

## 2. 关键架构审查与修正

在对上游代码库进行深入审查后，确认了以下关键实现事实：

| 文件位置 | 审查发现的事实 | 本项目 (Coord-OPD) 策略 |
|---|---|---|
| `policy.py:load_policy` | 模型初始化时断言 `hidden_size == 4096`（适配 Qwen3-VL-8B/Qwen2.5-VL-7B 体系） | 若后续进行 2B/3B 学生模型轻量化迁移，必须修改此维度检查并适配 projection 层 |
| `policy.py:action_distribution` | 返回对角高斯分布 `Normal(loc, scale)`，其中 `scale` 固定为 0.15 | 第一阶段采用固定方差的高斯 Reverse KL；当 $\sigma_S = \sigma_T = 0.15$ 时，KL 等价于均值 MSE 的恒等缩放 |
| `rollout.py:sample` | VLM 首先自回归生成粗框文本，随后连续头输出残差 $z$，经由非可逆截断得到最终裁剪框 $b_{final}$ | 明确保留混合动作协议。在计算坐标 OPD 时，必须在未截断的 raw action 空间计算高斯 KL |
| `train.py` | 采用组内优势（Group Relative Policy Optimization），若组内奖励全部相同（方差为 0）会触发重采样 | OPD 是纯分布匹配，不依赖组内奖励方差，因此在组内答案全对或全错时仍可提供有效更新 |

## 3. 教师与学生约定

1. **同构约束**：教师模型必须具备经过微调训练的同结构连续头，不能使用未对齐的随机初始化头作为教师。
2. **粗框条件一致性**：计算坐标 OPD 时，教师连续头的条件输入必须是**学生采样出的粗框**，禁止替换为教师自己偏好的粗框。
