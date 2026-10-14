# 实验信息模板

`experiment.template.json` 记录实验身份、模型、数据、状态、硬件与指标。它不是训练器配置，不能直接传给尚未实现的 train.py。

复制到外部运行目录后填写；也可以将去掉私有路径和敏感信息的完成版保存到 results/ 对应运行目录。null 表示尚未确定或测量，false 表示尚未验证。

`student_rollouts_per_prompt=1` 是 OPD 起步建议。RL 基线应使用其有意义的组采样配置，并记录相应预算，不能为强行对齐而令 GRPO 的组优势恒为零。正式运行时必须写入实际使用值。

以后增加可运行训练配置时另建文件，并在 README 明确区分“记录模板”和“训练配置”。

## 已有 YAML 草稿

[research/pilot_crop_opd.yaml](research/pilot_crop_opd.yaml) 随远程研究代码保留；没有已接通的训练入口。模型 ID、数据字段、硬件映射和动作/显存参数均需核验。已知问题见 [INTEGRATION_NOTES.md](../docs/INTEGRATION_NOTES.md)，不能把 YAML 存在当成可以开始训练。
