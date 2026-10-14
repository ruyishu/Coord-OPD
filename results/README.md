# 实验结果

目前没有实验结果。`metrics.csv` 只有表头，空单元格表示未测量，不能用 0 或示例分数冒充结果。

一行记录一个 run、method、seed 与评测集合。accuracy 使用 0—1；GPU 小时按 A800 与 L20 等型号分别记录，其他型号写在对应 manifest。`a800_gpu_hours`、`l20_gpu_hours` 是本次训练总成本，包含采样、学生更新、teacher 评分与训练期评测；`teacher_a800_gpu_hours`、`teacher_l20_gpu_hours` 是其中的教师评分子项，不要再次相加。教师预训练/强化学习准备成本另记在 manifest，不能隐去。

`report_path` 指向仓库内的小型报告或获准分享的存储链接；不要提交 token、机器私有凭证或未经授权公开的数据。大图、完整轨迹、模型文件放仓库外。

比较协议见 [EXPERIMENTS.md](../docs/EXPERIMENTS.md)。失败/中止实验也应有 run 记录和原因，不只保存成功运行。
