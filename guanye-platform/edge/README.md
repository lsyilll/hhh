# 采集终端

在平台设备中心申请、批准后下载本设备的 TLS 凭证，解压到设备私有目录。
编辑 `device.json` 的主机名和 API 地址，必须与服务证书 SAN 一致。默认平台只绑定本机回环地址；跨设备接入前由管理员配置受控局域网地址、主机防火墙和可信 TLS 证书，不要使用 `verify=False`。

树莓派安装 `python3-picamera2`、`python3-gpiozero`；USB 摄像头使用 OpenCV 回退。
GPIO17 接采集按钮，GPIO27 接继电器/驱动模块控制补光，按硬件规范接线，不能用 GPIO 直接驱动大功率灯。

`python agent.py --config /private/guanye/device.json --gpio`

单次测试：`python agent.py --config /private/guanye/device.json --once`。
图片先落盘进 SQLite 队列，HTTPS 成功接收后才删除；失败采用指数退避。MQTT 5 通过双向 TLS 上报心跳，主题按设备证书 CN 限制。暂停设备后业务服务拒绝上传，证书最长24小时；需要立即阻断已有 MQTT 连接时在受控网络撤销连接。

手机可继续使用原工作台“拍一张”和样本编号。二维码只包含配对编号，不包含私钥。当前电脑没有经过真实摄像头、麦克风或树莓派硬件验收。
