# SIT-VTON

深度学习SIT项目

linux上通过此方式运行：

```bash
git clone https://github.com/Qalxry/SIT-VTON.git
cd ./SIT-VTON
sudo chmod 777 ./webui.sh
./webui.sh
```

然后通过本地端口5000访问：

```bash
curl -X POST -H "Content-Type: application/json" -d '{"image_path": "$(pwd)/tmp/person/", "cloth_path": "$(pwd)/tmp/cloth/", "output_path": "$(pwd)/tmp/"}' http://127.0.0.1:5000/run
```

初次运行需要对输入的人物、衣物图像进行预处理，同时还要加载模型，时间较久，第二次POST运行较快。
运行结束后可以在 `SIT-VTON/tmp/result` 查看到结果
