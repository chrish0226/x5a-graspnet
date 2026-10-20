package com.x5a.voice;

import android.Manifest;
import android.content.pm.PackageManager;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.view.MotionEvent;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.json.JSONObject;

import java.io.File;


public class MainActivity extends AppCompatActivity {
    private static final int REQ_MIC = 17;

    private final RobotClient client = new RobotClient();
    private final Handler handler = new Handler(Looper.getMainLooper());

    private TextView linkBadge;
    private TextView hostText;
    private TextView busyBadge;
    private TextView stageText;
    private TextView resultText;
    private TextView transcriptText;
    private TextView parsedText;
    private TextView speechStatus;
    private TextView holdBtn;
    private TextView observationSideText;
    private boolean destroyed;

    private MediaRecorder recorder;
    private File recordFile;
    private long holdStart;
    private boolean holding;
    private final Runnable pollTask = new Runnable() {
        @Override
        public void run() {
            if (destroyed) return;
            client.pollState(new RobotClient.Callback() {
                @Override
                public void onOk(JSONObject json) {
                    if (destroyed) return;
                    applyState(json);
                    handler.postDelayed(pollTask, 800);
                }

                @Override
                public void onError(String message) {
                    if (destroyed) return;
                    linkBadge.setText("已断开");
                    stageText.setText("连接中断");
                    busyBadge.setText("离线");
                    busyBadge.setTextColor(getColor(R.color.bad));
                    showStatusDetail("请检查网络连接", true);
                    handler.postDelayed(pollTask, 1500);
                }
            });
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        linkBadge = findViewById(R.id.linkBadge);
        hostText = findViewById(R.id.hostText);
        busyBadge = findViewById(R.id.busyBadge);
        stageText = findViewById(R.id.stageText);
        resultText = findViewById(R.id.resultText);
        transcriptText = findViewById(R.id.transcriptText);
        parsedText = findViewById(R.id.parsedText);
        speechStatus = findViewById(R.id.speechStatus);
        holdBtn = findViewById(R.id.holdBtn);
        observationSideText = findViewById(R.id.observationSideText);

        findViewById(R.id.btnPrepare).setOnClickListener(v -> sendCommand("当前侧准备"));
        findViewById(R.id.btnPrepareMirror).setOnClickListener(v -> sendCommand("镜像侧准备"));
        findViewById(R.id.btnHome).setOnClickListener(v -> sendCommand("回原点"));
        findViewById(R.id.btnStop).setOnClickListener(v -> client.cancel(toastCb("已请求停止")));

        holdBtn.setOnTouchListener((v, event) -> {
            if (event.getAction() == MotionEvent.ACTION_DOWN) {
                v.performClick();
                startHold();
                return true;
            }
            if (event.getAction() == MotionEvent.ACTION_UP
                    || event.getAction() == MotionEvent.ACTION_CANCEL) {
                endHold();
                return true;
            }
            return false;
        });

        connect();
    }

    private void connect() {
        linkBadge.setText("连接中");
        stageText.setText("连接中");
        showStatusDetail("", false);
        client.discover(new RobotClient.Callback() {
            @Override
            public void onOk(JSONObject json) {
                if (destroyed) return;
                String base = json.optString("base", client.baseUrl());
                hostText.setText("App " + BuildConfig.VERSION_NAME + " · " + base);
                linkBadge.setText("已连接");
                stageText.setText(json.optBoolean("task_server", true) ? "待准备" : "服务未就绪");
                handler.removeCallbacks(pollTask);
                handler.post(pollTask);
            }

            @Override
            public void onError(String message) {
                if (destroyed) return;
                linkBadge.setText("未连接");
                stageText.setText("连接异常");
                busyBadge.setText("离线");
                busyBadge.setTextColor(getColor(R.color.bad));
                showStatusDetail("请检查网络连接", true);
                handler.postDelayed(MainActivity.this::connect, 2000);
            }
        });
    }

    private void applyState(JSONObject state) {
        linkBadge.setText("已连接");
        String stage = state.optString("stage", "IDLE");
        stageText.setText(StatusText.title(state));
        boolean busy = StatusText.busy(state);
        busyBadge.setText(busy ? "执行中" : "待命");
        busyBadge.setTextColor(getColor(busy ? R.color.accent : R.color.ok));
        String armFault = state.optString("arm_fault", "");
        observationSideText.setText("观察方向：" + state.optString("observation_label", "当前侧"));
        if (!armFault.isEmpty()) {
            busyBadge.setText("异常");
            busyBadge.setTextColor(getColor(R.color.bad));
        }
        String transcript = state.optString("transcript", "");
        if (!transcript.isEmpty() && !holding) {
            transcriptText.setText(transcript);
        }
        JSONObject parsed = state.optJSONObject("parsed");
        if (parsed != null) {
            parsedText.setText(parsed.toString());
        }
        showStatusDetail(StatusText.detail(state), !armFault.isEmpty() || "FAILED".equals(stage));
    }

    private void showStatusDetail(String text, boolean error) {
        resultText.setText(text);
        resultText.setTextColor(getColor(error ? R.color.bad : R.color.muted));
        resultText.setVisibility(text.isEmpty() ? View.GONE : View.VISIBLE);
    }

    private void startHold() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.RECORD_AUDIO}, REQ_MIC);
            return;
        }
        holding = true;
        holdStart = SystemClock.elapsedRealtime();
        holdBtn.setBackgroundResource(R.drawable.hold_btn_hot);
        holdBtn.setText("松开发送");
        holdBtn.setTextColor(0xFFFFFFFF);
        speechStatus.setText("正在录音…");
        speechStatus.setTextColor(getColor(R.color.bad));
        transcriptText.setText("正在录音…");
        try {
            recordFile = new File(getCacheDir(), "speech.m4a");
            if (recordFile.exists()) {
                //noinspection ResultOfMethodCallIgnored
                recordFile.delete();
            }
            recorder = new MediaRecorder();
            recorder.setAudioSource(MediaRecorder.AudioSource.MIC);
            recorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
            recorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC);
            recorder.setAudioSamplingRate(16000);
            recorder.setAudioEncodingBitRate(64000);
            recorder.setOutputFile(recordFile.getAbsolutePath());
            recorder.prepare();
            recorder.start();
        } catch (Exception e) {
            holding = false;
            resetHold();
            speechStatus.setText("无法开始录音：" + e.getMessage());
        }
    }

    private void endHold() {
        if (!holding) {
            return;
        }
        holding = false;
        long elapsed = SystemClock.elapsedRealtime() - holdStart;
        try {
            if (recorder != null) {
                recorder.stop();
                recorder.release();
            }
        } catch (Exception ignored) {
        }
        recorder = null;
        resetHold();
        if (elapsed < 500 || recordFile == null || !recordFile.exists() || recordFile.length() < 800) {
            speechStatus.setText("按太短了，请按住说完整一句");
            transcriptText.setText("按住说话，松手发送");
            return;
        }
        speechStatus.setText("正在用主机模型识别…");
        transcriptText.setText("正在识别…");
        client.uploadAudio(recordFile, new RobotClient.Callback() {
            @Override
            public void onOk(JSONObject json) {
                String text = json.optString("text", "").trim();
                if (text.isEmpty()) {
                    speechStatus.setText(json.optString("message", "没有听清，请再说一次"));
                    transcriptText.setText("没有听清");
                    return;
                }
                transcriptText.setText(text);
                speechStatus.setText("识别到：" + text);
                speechStatus.setTextColor(getColor(R.color.ok));
                client.sendText(text, resultCb());
            }

            @Override
            public void onError(String message) {
                speechStatus.setText("识别失败：" + message);
            }
        });
    }

    private void resetHold() {
        holdBtn.setBackgroundResource(R.drawable.hold_btn);
        holdBtn.setText("按住 说话");
        holdBtn.setTextColor(0xFF1A1204);
        speechStatus.setTextColor(getColor(R.color.muted));
    }

    private void sendCommand(String command) {
        transcriptText.setText("手动：" + command);
        client.sendText(command, resultCb());
    }

    private RobotClient.Callback resultCb() {
        return new RobotClient.Callback() {
            @Override
            public void onOk(JSONObject json) {
                if (json.has("parsed")) {
                    parsedText.setText(String.valueOf(json.opt("parsed")));
                }
                if (json.has("message")) {
                    boolean ok = json.optBoolean("ok", json.optBoolean("success", false));
                    showStatusDetail(ok ? "指令已接收" : StatusText.failure(json.optString("message")), !ok);
                }
            }

            @Override
            public void onError(String message) {
                showStatusDetail(StatusText.failure(message), true);
            }
        };
    }

    private RobotClient.Callback toastCb(String fallback) {
        return new RobotClient.Callback() {
            @Override
            public void onOk(JSONObject json) {
                Toast.makeText(MainActivity.this, json.optString("message", fallback), Toast.LENGTH_SHORT).show();
            }

            @Override
            public void onError(String message) {
                Toast.makeText(MainActivity.this, message, Toast.LENGTH_SHORT).show();
            }
        };
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQ_MIC && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(this, "已授权，再按住说话", Toast.LENGTH_SHORT).show();
        }
    }

    @Override
    protected void onDestroy() {
        destroyed = true;
        handler.removeCallbacksAndMessages(null);
        if (recorder != null) {
            try {
                recorder.release();
            } catch (Exception ignored) {
            }
        }
        super.onDestroy();
    }
}
