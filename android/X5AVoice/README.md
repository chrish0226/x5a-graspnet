# X5A 语音抓取 Android 客户端

包名 `com.x5a.voice`，版本 `1.3-wrist`，versionCode 4。原生布局提供文字/语音、准备、镜像准备、回零和停止，状态请求完成后再调度下一次轮询。

默认连接主机 `192.168.0.50:8000` / HTTPS 8443。换主机需修改 `app/src/main/java/com/x5a/voice/MainActivity.java` 的地址。按钮对应能力由服务端决定：SAM 不支持镜像准备，不能因为手机上存在按钮就认为后端已实现。

使用 Android SDK 34、兼容的 JDK / Gradle 7.6.4，配置本机 SDK（`local.properties` 不提交）后构建：

```bash
cd android/X5AVoice
gradle --no-daemon assembleDebug
```

输出为 `app/build/outputs/apk/debug/app-debug.apk`。仓库不含 Gradle 分发包、APK 或原设备签名密钥。新环境默认 debug 签名与原设备不同，不能保证覆盖安装；需要自行安排安装和配置。浏览器页面无需编译 Android。

松爪、回零等指令会调用已启动服务的真实动作入口；以服务端执行模式和当前持物状态为准。
