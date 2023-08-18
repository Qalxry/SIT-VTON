#!/bin/bash
### DO NOT MODIFY THIS FILE ###


conda_available=false


if command -v conda &>/dev/null; then
    conda_available=true
    echo "conda is available"
else
    echo "conda is not installed"
    # Install miniconda3
    if [ -d "$(pwd)/venv/miniconda3" ]; then
        echo "Miniconda3 exists."
    else
        echo "Miniconda3 does not exist. Install miniconda3."
        # check the miniconda.sh file exists
        if [ ! -f "$(pwd)/venv/miniconda.sh" ]; then
            echo "Miniconda.sh does not exist. Downloading miniconda.sh."
            wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $(pwd)/venv/miniconda.sh
        fi
        # install miniconda
        bash $(pwd)/venv/miniconda.sh -b -p $(pwd)/venv/miniconda3
    fi
fi


# Create conda environment: env1
if [ -d "$(pwd)/venv/env1" ]; then
    echo "Conda environment 1 exists."
else
    echo "Conda environment 1 does not exist. Auto create conda environment."
    if $conda_available; then
        conda env create -f "$(pwd)/venv/env1.yml" -p $(pwd)/venv/env1
    else
        $(pwd)/venv/miniconda3/bin/conda env create -f "$(pwd)/venv/env1.yml" -p $(pwd)/venv/env1
    fi
fi


# Create conda environment: env2
if [ -d "$(pwd)/venv/env2" ]; then
    echo "Conda environment 2 exists."
else
    echo "Conda environment 2 does not exist. Auto create conda environment."
    if $conda_available; then
        conda env create -f "$(pwd)/venv/env2.yml" -p $(pwd)/venv/env2
    else
        $(pwd)/venv/miniconda3/bin/conda env create -f "$(pwd)/venv/env2.yml" -p $(pwd)/venv/env2
    fi
fi


# 设置信号处理程序，当脚本接收到终止信号时，结束后台运行的 getparse.py
cleanup() {
    echo "Cleaning up..."
    pkill -f "python $(pwd)/annotator/CIHP_PGN/getparse.py"
    exit 1
}
# 捕获终止信号，调用 cleanup 函数
trap cleanup SIGINT SIGTERM


# 确保 getparse.py 没有在后台运行
pkill -f "python $(pwd)/annotator/CIHP_PGN/getparse.py"

# 在 env2 环境后台运行 getparse.py
nohup "$(pwd)/venv/env2/bin/python" $(pwd)/annotator/CIHP_PGN/getparse.py &

# 等待一段时间，以确保 getparse.py 已经在后台运行
sleep 1

# 在 env1 环境运行 launch.py
"$(pwd)/venv/env1/bin/python" $(pwd)/launch.py

# 在 app1.py 执行完毕后，结束后台运行的 app2.py 进程
pkill -f "python $(pwd)/annotator/CIHP_PGN/getparse.py"


