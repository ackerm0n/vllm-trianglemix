# 将修改的vLLM上传到GitHub仓库指南

## 方法1：创建新的仓库（推荐用于第一次上传）

### 步骤1：在GitHub上创建新仓库

1. 访问 https://github.com/new
2. 输入仓库名称，例如：`vllm-trianglemix` 或 `vllm-qwen3-npu`
3. 选择 **Public** 或 **Private**
4. **不要初始化** README、.gitignore 或 LICENSE（因为我们要上传已有的代码）
5. 点击 "Create repository"

### 步骤2：初始化本地Git仓库

```bash
# 进入vllm目录
cd /Users/tsy/Downloads/vllm-0.11.0

# 初始化git仓库
git init

# 查看修改的文件
git status
```

### 步骤3：配置Git用户信息

```bash
# 全局配置（如果还没配置过）
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# 或仅为此仓库配置
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

### 步骤4：添加所有文件到暂存区

```bash
# 添加所有修改的文件
git add .

# 查看即将提交的文件（可选）
git status
```

### 步骤5：创建首次提交

```bash
git commit -m "Initial commit: TriangleMix attention implementation for vLLM with Ascend NPU support"
```

### 步骤6：添加远程仓库并推送

```bash
# 替换 your-username 和 your-repo-name
git remote add origin https://github.com/your-username/your-repo-name.git

# 推送到GitHub（首次）
git branch -M main
git push -u origin main
```

---

## 方法2：Fork现有vLLM仓库并推送修改（如果想与原仓库保持关联）

### 步骤1：Fork原仓库

1. 访问 https://github.com/vllm-project/vllm
2. 点击右上角 "Fork" 按钮
3. 在你的账户下创建fork

### 步骤2：克隆你的fork

```bash
git clone https://github.com/your-username/vllm.git
cd vllm
```

### 步骤3：添加upstream远程（保持与原仓库同步）

```bash
git remote add upstream https://github.com/vllm-project/vllm.git
```

### 步骤4：创建功能分支

```bash
git checkout -b feature/trianglemix-ascend-npu
```

### 步骤5：复制你的修改文件

```bash
# 将修改的文件复制到克隆的仓库中
cp -r /Users/tsy/Downloads/vllm-0.11.0/vllm/attention/trianglemix*.* ./vllm/attention/
cp -r /Users/tsy/Downloads/vllm-0.11.0/vllm/attention/ascend_trianglemix.py ./vllm/attention/
cp /Users/tsy/Downloads/vllm-0.11.0/vllm/model_executor/models/qwen3.py ./vllm/model_executor/models/
```

### 步骤6：提交并推送

```bash
git add .
git commit -m "feat: Add TriangleMix attention pattern for efficient long-context prefilling with Ascend NPU support"
git push origin feature/trianglemix-ascend-npu
```

### 步骤7：创建Pull Request

1. 访问你的fork仓库
2. 点击 "Compare & pull request"
3. 填写PR描述
4. 提交PR到原vLLM仓库

---

## 方法3：快速推送整个修改后的仓库

如果你想快速上传整个修改后的文件夹，使用此脚本：

### 创建上传脚本

```bash
# 创建脚本文件
cat > /tmp/upload_to_github.sh << 'EOF'
#!/bin/bash

# 配置
REPO_URL="${1:-}"
COMMIT_MSG="${2:-Initial commit with TriangleMix implementation}"

if [ -z "$REPO_URL" ]; then
    echo "Usage: ./upload_to_github.sh <repository-url> [commit-message]"
    echo "Example: ./upload_to_github.sh https://github.com/username/vllm-trianglemix.git"
    exit 1
fi

cd /Users/tsy/Downloads/vllm-0.11.0

# 初始化仓库
git init
git config user.name "Your Name"
git config user.email "your.email@example.com"

# 添加文件
git add .
git commit -m "$COMMIT_MSG"

# 添加远程仓库
git remote add origin "$REPO_URL"

# 推送
git branch -M main
git push -u origin main

echo "✓ 上传完成！"
EOF

chmod +x /tmp/upload_to_github.sh
```

### 运行上传脚本

```bash
/tmp/upload_to_github.sh https://github.com/your-username/your-repo-name.git
```

---

## 详细步骤指南（推荐新手）

### 1. 准备GitHub仓库

```bash
# 在GitHub.com网页上创建仓库后，你会看到类似的命令

# 进入你的vllm目录
cd /Users/tsy/Downloads/vllm-0.11.0
```

### 2. 初始化并配置

```bash
# 初始化Git
git init

# 配置用户
git config user.name "Your GitHub Username"
git config user.email "your-email@github.com"

# 检查修改的文件
git status
```

### 3. 提交修改

```bash
# 暂存所有文件
git add .

# 创建第一个提交
git commit -m "feat: Implement TriangleMix attention for vLLM

- Add TriangleMix sparse attention pattern
- Support Streaming, Last Q-K, and Middle sections
- Implement Triangle mask generation
- Add Ascend NPU optimizations
- Integrate with Qwen3 model
- Add comprehensive tests and documentation"
```

### 4. 推送到GitHub

```bash
# 添加远程仓库（替换your-username和repo-name）
git remote add origin https://github.com/your-username/your-repo-name.git

# 重命名分支为main
git branch -M main

# 推送代码
git push -u origin main
```

---

## 使用SSH密钥（如果GitHub配置了SSH）

如果你已经在GitHub上配置了SSH密钥：

```bash
# 使用SSH URL而不是HTTPS
git remote add origin git@github.com:your-username/your-repo-name.git

# 其余步骤相同
git push -u origin main
```

---

## 完整的一键上传脚本

```bash
#!/bin/bash

# 用户配置
GITHUB_USERNAME="your-username"
REPO_NAME="vllm-trianglemix"
REPO_URL="https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}vLLM TriangleMix 上传脚本${NC}"
echo -e "${YELLOW}========================================${NC}"

# 检查目录
if [ ! -d "/Users/tsy/Downloads/vllm-0.11.0" ]; then
    echo -e "${RED}❌ 找不到vllm目录${NC}"
    exit 1
fi

cd /Users/tsy/Downloads/vllm-0.11.0

# 初始化
echo -e "${YELLOW}初始化Git仓库...${NC}"
git init

# 配置用户
echo -e "${YELLOW}配置Git用户...${NC}"
git config user.name "$GITHUB_USERNAME"
git config user.email "your-email@github.com"

# 查看修改
echo -e "${YELLOW}修改的文件:${NC}"
git status --short | head -20

# 提交
echo -e "${YELLOW}创建提交...${NC}"
git add .
git commit -m "feat: TriangleMix attention implementation with Ascend NPU support"

# 添加远程
echo -e "${YELLOW}添加远程仓库...${NC}"
git remote add origin "$REPO_URL"

# 推送
echo -e "${YELLOW}推送到GitHub...${NC}"
git branch -M main
git push -u origin main

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 上传成功！${NC}"
    echo -e "${GREEN}仓库地址: ${REPO_URL}${NC}"
else
    echo -e "${RED}❌ 上传失败，请检查:${NC}"
    echo -e "${RED}1. GitHub仓库URL是否正确${NC}"
    echo -e "${RED}2. GitHub凭据是否已配置${NC}"
    echo -e "${RED}3. 网络连接是否正常${NC}"
    exit 1
fi
```

保存为 `upload.sh` 并运行：

```bash
chmod +x upload.sh
./upload.sh
```

---

## 常见问题

### Q1: 提示"远程仓库已存在"

```bash
# 移除已有的远程
git remote remove origin

# 添加新的远程
git remote add origin https://github.com/your-username/your-repo-name.git
```

### Q2: 提示"没有权限"

```bash
# 确保你已登录GitHub
# 1. 生成personal access token
# 2. 使用token作为密码

# 或配置SSH密钥（推荐）
ssh-keygen -t ed25519 -C "your-email@github.com"
# 然后在GitHub Settings -> SSH Keys 中添加公钥
```

### Q3: 提示"大文件"

vLLM仓库包含一些较大的文件。如果遇到限制：

```bash
# 1. 使用Git LFS（Large File Storage）
git lfs install

# 2. 追踪大文件
git lfs track "*.bin"
git lfs track "*.so"

# 3. 重新提交
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

### Q4: 想要合并上游更新

```bash
# 如果使用了fork方式
git fetch upstream
git merge upstream/main

# 或rebase以保持线性历史
git rebase upstream/main
```

---

## 推荐的提交信息格式

```
feat: 新增功能
fix: 修复bug
docs: 文档更新
style: 代码格式
refactor: 代码重构
test: 测试相关
chore: 构建、依赖等

示例:
feat: Add TriangleMix attention pattern
- Implement streaming mask generation
- Add last Q-K section support
- Create triangle attention masks
- Integrate with Ascend NPU

fix: Handle -inf values in attention mask for NPU

docs: Add comprehensive TriangleMix documentation

test: Add unit tests for mask generation
```

---

## 下一步

上传完成后：

1. **在README中说明修改**
   - 描述TriangleMix的改动
   - 说明Ascend NPU支持
   - 提供使用示例

2. **创建release版本**
   ```bash
   git tag -a v0.11.0-trianglemix -m "Version 0.11.0 with TriangleMix"
   git push origin v0.11.0-trianglemix
   ```

3. **设置GitHub Pages文档**
   - 将文档放在 `docs/` 目录
   - 启用GitHub Pages

4. **配置CI/CD**
   - 添加GitHub Actions自动测试
   - 自动运行单元测试

---

## 需要帮助？

如果遇到问题，可以：

1. 检查GitHub的SSH/HTTPS配置
2. 验证仓库URL: `git remote -v`
3. 查看推送日志: `git push -v`
4. 阅读GitHub官方文档: https://docs.github.com/

祝上传顺利！🚀
