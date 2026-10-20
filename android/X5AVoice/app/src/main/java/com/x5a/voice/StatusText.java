package com.x5a.voice;

import org.json.JSONObject;

/** Short display labels. Full diagnostics remain available on the host. */
final class StatusText {
    private StatusText() {}

    static boolean busy(JSONObject state) {
        return state.optBoolean("busy", false) || "BUSY".equals(state.optString("robot_state"));
    }

    static String title(JSONObject state) {
        String fault = state.optString("arm_fault", "");
        if (!fault.isEmpty()) return fault(fault, state.optJSONObject("arm_health"));
        switch (state.optString("stage", "IDLE")) {
            case "IDLE": return "待准备";
            case "READY": return "就绪";
            case "HOME": return "已归位";
            case "GO_HOME": return "归位中";
            case "ACCEPTED": return "任务已接收";
            case "MOVE_INITIAL_OBSERVATION":
            case "INITIAL_OBSERVATION": return "准备中";
            case "WAITING_VISION":
            case "TARGET_FOUND":
            case "REOBSERVE_UNCOVERED":
            case "REOBSERVING":
            case "VERIFYING_CORRECTION": return "目标定位";
            case "SELECTING_BUFFER": return "暂放点确认";
            case "CLEARING_BLOCKER": return "移开遮挡物";
            case "MOVE_READY":
            case "MOVE_PRE_GRASP":
            case "CORRECTING_PRE_GRASP":
            case "APPROACH":
            case "GRASPING": return "抓取中";
            case "LIFTING": return "抬升中";
            case "HOLDING_ABOVE_SOURCE":
            case "BUFFER_PLACEMENT":
            case "MOVE_PRE_PLACE": return "转移中";
            case "DESCENDING":
            case "RELEASING": return "放置中";
            case "RETREATING":
            case "RETURN_HOME": return "返回中";
            case "WAITING_PLACE": return "待放置";
            case "SUCCESS": return "任务完成";
            case "FAILED": return "任务失败";
            case "DRY_RUN": return "演练完成";
            case "RECOVERING_STATE": return "状态恢复中";
            default:
                if (busy(state)) return "执行中";
                return state.optBoolean("prepared", false) ? "就绪" : "待准备";
        }
    }

    static String detail(JSONObject state) {
        if (!state.optString("arm_fault", "").isEmpty()) return "请检查设备";
        if (busy(state)) return "";
        boolean holding = state.optBoolean("holding", false);
        String stage = state.optString("stage", "IDLE");
        if ("FAILED".equals(stage)) {
            JSONObject last = state.optJSONObject("last_result");
            String reason = failure(last == null ? "" : last.optString("message", ""));
            return holding ? reason + " · 保持夹持" : reason;
        }
        if (holding) return "已夹持" + color(state.optString("holding_color", "")) + "方块";
        if ("HOME".equals(stage) || "IDLE".equals(stage)) return "请执行准备";
        if ("READY".equals(stage)) return "等待指令";
        return "";
    }

    static String fault(String message, JSONObject health) {
        if (health != null) {
            if (!health.optBoolean("online", true)) return "通信异常";
            if (health.optBoolean("protected", false)) return "电机保护";
        }
        if (message.contains("保护")) return "电机保护";
        if (message.contains("反馈") || message.contains("中断") || message.contains("过期")) return "通信异常";
        if (message.contains("重复") || message.contains("控制来源")) return "控制冲突";
        return "设备异常";
    }

    static String failure(String message) {
        if (message.contains("保护")) return "电机保护";
        if (message.contains("电机") || message.contains("反馈")) return "设备状态异常";
        if (message.contains("取消") || message.contains("已停止") || message.contains("中断")) return "任务已中止";
        if (message.contains("正在执行") || message.contains("正在运行") || message.contains("正在移动")) return "设备正在执行任务";
        if (message.contains("先说") && message.contains("准备") || message.contains("首次准备") || message.contains("请先准备")) return "请先执行准备";
        if (message.contains("暂放") || message.contains("候选空位")) {
            if (message.contains("占用")) return "暂放区域被占用";
            if (message.contains("超时")) return "暂放点确认超时";
            if (message.contains("不可达")) return "暂放位置不可达";
            return "暂放位置不可用";
        }
        if (message.contains("顶面") || message.contains("遮挡") || message.contains("被压")) return "目标被遮挡";
        if (message.contains("夹持状态") || message.contains("夹爪")) return "请检查夹持状态";
        if (message.contains("盒") && (message.contains("定位") || message.contains("深度"))) return "盒子定位失败";
        if (message.contains("定位") || message.contains("深度") || message.contains("坐标")) return "目标定位失败";
        if (message.contains("规划") || message.contains("不可达") || message.contains("planning")) return "路径规划失败";
        if (message.contains("指令") || message.contains("支持")) return "指令未识别";
        if (message.contains("连接") || message.contains("connect") || message.contains("主机")) return "主机连接异常";
        if (message.contains("超时") || message.contains("timeout")) return "任务响应超时";
        return "请检查任务条件";
    }

    private static String color(String color) {
        switch (color) {
            case "red": return "红色";
            case "white": return "白色";
            case "orange": return "橙色";
            default: return "";
        }
    }
}
