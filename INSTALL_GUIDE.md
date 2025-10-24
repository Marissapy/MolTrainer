# MolTrainer 安装与更新指南

## 📥 下载与安装

### 方法 1：从GitHub克隆（推荐）

这种方法允许您轻松更新和修改代码。

```bash
# 1. 克隆仓库
git clone https://github.com/Marissapy/MolTrainer.git
cd MolTrainer

# 2. 安装RDKit（必需）
conda install -c conda-forge rdkit
# 或使用pip: pip install rdkit-pypi

# 3. 安装其他依赖
pip install -r requirements.txt

# 4. 以可编辑模式安装MolTrainer
pip install -e .
```

**优点**：
- ✅ 代码修改后无需重新安装
- ✅ 可以轻松更新到最新版本
- ✅ 方便开发和调试

### 方法 2：直接从GitHub安装

如果您只想使用，不需要修改代码：

```bash
# 1. 安装RDKit（必需）
conda install -c conda-forge rdkit

# 2. 直接安装MolTrainer
pip install git+https://github.com/Marissapy/MolTrainer.git
```

**优点**：
- ✅ 一步安装
- ✅ 不占用额外磁盘空间存储源码

### 方法 3：下载ZIP包

```bash
# 1. 访问 https://github.com/Marissapy/MolTrainer
# 2. 点击 "Code" -> "Download ZIP"
# 3. 解压到本地目录

# 4. 进入目录
cd MolTrainer-main

# 5. 安装RDKit
conda install -c conda-forge rdkit

# 6. 安装依赖和MolTrainer
pip install -r requirements.txt
pip install -e .
```

---

## 🔄 更新MolTrainer

### 如果使用方法1（Git克隆）

```bash
# 进入MolTrainer目录
cd MolTrainer

# 拉取最新代码
git pull origin main

# 更新依赖（如果有新依赖）
pip install -r requirements.txt --upgrade

# 重新安装（如果setup.py有变化）
pip install -e . --upgrade
```

### 如果使用方法2（直接安装）

```bash
# 强制重新安装最新版本
pip install --upgrade --force-reinstall git+https://github.com/Marissapy/MolTrainer.git
```

### 如果使用方法3（ZIP包）

```bash
# 1. 重新下载最新的ZIP包
# 2. 解压并覆盖旧文件
# 3. 重新安装依赖
cd MolTrainer-main
pip install -r requirements.txt --upgrade
pip install -e . --upgrade
```

---

## ✅ 验证安装

安装完成后，运行以下命令验证：

```bash
# 查看帮助（应该显示MolTrainer logo和选项）
moltrainer -h

# 查看版本信息
moltrainer -h | head -10
```

如果看到类似以下输出，说明安装成功：

```
:------------------------------------------------------:
: ___  ___       _  _____               _               :
:|  \/  |      | ||_   _|             (_)              :
:| .  . |  ___ | |  | |  _ __  __ _  _  _ __    ___  _ __ :
:| |\/| | / _ \| |  | | | '__|/ _` || || '_ \  / _ \| '__|:
:| |  | || (_) | |  | | | |  | (_| || || | | ||  __/| |   :
:\_|  |_/ \___/|_|  \_/ |_|   \__,_||_||_| |_| \___||_|   :
:                                                          :
: Machine Learning for Molecular Data :
: Version 0.1.0 :
:------------------------------------------------------:
```

---

## 🐛 常见问题

### 1. RDKit安装失败

**问题**：`conda install rdkit` 或 `pip install rdkit-pypi` 失败

**解决方案**：
```bash
# 推荐使用conda安装
conda create -n moltrainer python=3.9
conda activate moltrainer
conda install -c conda-forge rdkit

# 然后在这个环境中安装MolTrainer
pip install -e .
```

### 2. XGBoost/LightGBM安装失败

**问题**：`pip install xgboost` 或 `pip install lightgbm` 失败

**解决方案**：
```bash
# 使用conda安装（推荐）
conda install -c conda-forge xgboost lightgbm

# 或者跳过安装（使用Random Forest和SVM仍可工作）
# 编辑requirements.txt，注释掉xgboost和lightgbm行
```

### 3. 找不到moltrainer命令 ⭐ 常见问题

**问题**：安装后运行 `moltrainer` 提示 `command not found`

**原因**：用户模式安装时，可执行文件在 `~/.local/bin` (Linux/Mac) 或 `%APPDATA%\Python\Scripts` (Windows)，但这些目录可能不在PATH中。

**解决方案A：直接使用Python模块（最简单）**
```bash
# 直接运行
python -m moltrainer -h

# 或创建别名（Linux/Mac）
echo 'alias moltrainer="python -m moltrainer"' >> ~/.bashrc
source ~/.bashrc
moltrainer -h

# Windows PowerShell创建别名
echo 'function moltrainer { python -m moltrainer $args }' >> $PROFILE
```

**解决方案B：添加到PATH（推荐）**
```bash
# Linux/Mac
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Windows (以管理员身份运行PowerShell)
[Environment]::SetEnvironmentVariable("Path", "$env:Path;$env:APPDATA\Python\Scripts", "User")
```

**解决方案C：在conda环境中安装**
```bash
conda create -n moltrainer python=3.9
conda activate moltrainer
pip install git+https://github.com/Marissapy/MolTrainer.git
# 现在可以直接使用 moltrainer 命令
```

**解决方案D：使用完整路径**
```bash
# Linux/Mac
~/.local/bin/moltrainer -h

# Windows
%APPDATA%\Python\Scripts\moltrainer.exe -h
```

### 4. 权限错误

**问题**：`Permission denied` 或 `Access denied`

**解决方案**：
```bash
# Linux/Mac：使用用户安装
pip install -e . --user

# Windows：以管理员身份运行PowerShell或CMD
```

### 5. Git推送失败（开发者）

**问题**：`git push` 被拒绝

**解决方案**：
```bash
# 先拉取远程更改
git pull origin main --rebase

# 解决冲突后再推送
git push origin main
```

---

## 📦 完整依赖列表

### 必需依赖
```
python >= 3.8
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
rdkit >= 2022.03.1
matplotlib >= 3.4.0
seaborn >= 0.11.0
art >= 5.7
colorama >= 0.4.4
```

### 可选依赖
```
pyyaml >= 6.0          # 用于配置文件支持
xgboost >= 2.0.0       # 用于XGBoost模型
lightgbm >= 4.0.0      # 用于LightGBM模型
```

---

## 🚀 快速开始

安装完成后，尝试以下命令：

```bash
# 1. 查看帮助
moltrainer -h

# 2. 查看完整文档
# 英文：help.md
# 中文：help_Chinese.md

# 3. 运行测试（可选）
cd debug
pwsh run_tests.ps1       # Windows PowerShell
bash run_tests.sh        # Linux/Mac (需要先创建)

# 4. 创建配置文件示例
moltrainer -create_config my_config.yaml

# 5. 开始您的第一次分析
moltrainer -i your_data.csv -desc_stats
```

---

## 📞 获取帮助

- **完整文档**：[help.md](help.md) (英文) / [help_Chinese.md](help_Chinese.md) (中文)
- **快速开始**：[QUICKSTART.md](QUICKSTART.md)
- **GitHub Issues**：https://github.com/Marissapy/MolTrainer/issues
- **示例配置**：查看 `debug/` 文件夹中的示例脚本

---

## 💡 开发者模式

如果您想修改代码并立即测试：

```bash
# 1. 克隆仓库
git clone https://github.com/Marissapy/MolTrainer.git
cd MolTrainer

# 2. 创建开发环境
conda create -n moltrainer-dev python=3.9
conda activate moltrainer-dev

# 3. 安装所有依赖
conda install -c conda-forge rdkit xgboost lightgbm
pip install -r requirements.txt

# 4. 以可编辑模式安装
pip install -e .

# 5. 修改代码后直接测试（无需重新安装）
moltrainer -i test.csv -desc_stats

# 6. 运行测试套件
cd debug
python test_feature_engineering.py
pwsh run_tests.ps1
pwsh run_advanced_tests.ps1
```

**注意**：使用 `-e` 参数安装后，代码修改会立即生效，无需重新安装！

---

祝使用愉快！🎉

