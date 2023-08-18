# SIT-VTON

深度学习SIT项目

2023/8/18  v0.0.1
 - CIHP_PGN的效果有点差，存在BUG
 - 暂时没有编写webui，将在未来几天进行编写，现在可用server.sh运行
 - windows运行脚本server.bat暂时未编写
 - 目前只支持linux
 - 直接运行server.sh可自动下载所需的所有文件，同时自动创建conda环境

linux上通过此方式运行：

```bash
git clone https://github.com/Qalxry/SIT-VTON.git
cd ./SIT-VTON
sudo chmod 777 ./server.sh
./server.sh
```

然后通过本地端口5000访问：

```bash
curl -X POST -H "Content-Type: application/json" -d '{"image_path": "$(pwd)/tmp/person/", "cloth_path": "$(pwd)/tmp/cloth/", "output_path": "$(pwd)/tmp/"}' http://127.0.0.1:5000/run
```

初次运行需要对输入的人物、衣物图像进行预处理，同时还要加载模型，时间较久，第二次POST运行较快。
运行结束后可以在 `SIT-VTON/tmp/result` 查看到结果

目前程序所接受json的key：
 - `image_path`: 768×1024人像图片路径
 - `cloth_path`: 768×1024衣物图片路径
 - `output_path`: 预处理数据和生成图片保存路径
 - `return_image`: `bool`值，是否返回生成的图片的字符流

返回的json字段：
 - `status`: `'success'`
 - `image`: base64图片字符流
