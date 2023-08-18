# SIT-VTON

深度学习SIT项目

linux上通过此方式运行：

```bash
git clone 
cd ./SIT-VTON
sudo chmod 777 ./webui.sh
./webui.sh
```

然后可以通过本地端口5000访问：

```bash
curl -X POST -H "Content-Type: application/json" -d '{"image_path": "$(pwd)/tmp/person/", "cloth_path": "$(pwd)/tmp/cloth/", "output_path": "$(pwd)/tmp/"}' http://127.0.0.1:5000/run
```

之后可以在 `SIT-VTON/tmp/result` 查看到结果
