# 松开夹爪被持物状态拦截的修复

现场日志显示：抓取后 PLACE 轨迹失败（-5），执行器保持 holding=true。网页只在整次任务成功时同步 holding，因此自己的 Session 仍为 false。之后的“松开”正确解析成 release，却被“手上本来就没有东西”的网页判断挡住；last_result 还保留着前一轮“仍在持物，拒绝执行新的抓取计划”的错误。

修复：

- release 不再依赖网页缓存中的 holding。夹爪闭合但没持物、网页刚重启或上一轮失败，都允许显式松开。
- 接受 release 时清除旧结果；忙碌时仍拒绝重复提交。
- 抓取、放置、释放任务均按执行器返回的 holding 同步状态，包括失败结果。缺失反馈保留已有状态；失败发现持物时不把本轮尝试的新目标当成已持物身份。

12 项回归测试通过；使用假的 ROS bridge，无真实夹爪或手臂动作。覆盖本地解析、空缓存/已持物时释放、忙碌拦截、抓取后放置失败、释放后撤退失败和释放失败保留持物信息。

只重启网页后端，执行器、控制器、相机和夹爪未重启，未发送任何运动命令。重启后从执行器重新同步实际 holding。修改前备份在 `/tmp/x5a_release_fix_20261003/`。

```bash
source scripts/stack_env.sh
PYTHONPATH="src/x5a_web_agent:$PYTHONPATH" /usr/bin/python3 -m pytest src/x5a_web_agent/test -q
```
