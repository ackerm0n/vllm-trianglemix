# 快速上传到GitHub（5分钟指南）

## 前置准备

1. **创建GitHub仓库**（如果还没有）
   - 访问 https://github.com/new
   - 输入仓库名（例如：`vllm-trianglemix`）
   - 选择 Public 或 Private
   - **不要初始化任何文件**
   - 点击 "Create repository"

2. **配置GitHub登录**
   ```bash
   # 使用Personal Access Token或SSH密钥
   # 方式1: HTTPS (输入token或密码)
   # 方式2: SSH (需要配置SSH密钥)
   ```

## 最快上传方法

### 方法A：使用自动脚本（推荐）

```bash
# 1. 进入脚本目录
cd /Users/tsy/Downloads/vllm-0.11.0

# 2. 运行上传脚本
chmod +x upload_to_github.sh
./upload_to_github.sh your-github-username your-repo-name your-email@example.com
```

示例：
```bash
./upload_to_github.sh john-doe vllm-trianglemix john@example.com
```

**就这样！脚本会自动处理所有步骤。**

### 方法B：手动上传（4步）

```bash
# 1. 进入目录
cd /Users/tsy/Downloads/vllm-0.11.0

# 2. 初始化并提交
git init
git config user.name "Your Name"
git config user.email "your@email.com"
git add .
git commit -m "Initial commit: TriangleMix attention implementation"

# 3. 添加远程仓库 (替换your-username和your-repo)
git remote add origin https://github.com/your-username/your-repo.git

# 4. 推送到GitHub
git branch -M main
git push -u origin main
```

## 如果遇到问题

### 问题：提示"fatal: 'origin' already exists"
```bash
git remote remove origin
git remote add origin https://github.com/your-username/your-repo.git
```

### 问题：提示"Authentication failed"
- **HTTPS方式**: 使用 GitHub Personal Access Token 作为密码
  - 生成token: https://github.com/settings/tokens
  - 当输入密码时，粘贴token而不是密码

- **SSH方式**: 配置SSH密钥（推荐）
  ```bash
  ssh-keygen -t ed25519 -C "your@email.com"
  # 将公钥添加到 https://github.com/settings/keys
  
  # 使用SSH URL推送
  git remote set-url origin git@github.com:your-username/your-repo.git
  ```

### 问题：推送超时
```bash
# 设置更大的缓冲和超时
git config http.postBuffer 524288000
git config http.lowSpeedTime 1200
git config http.lowSpeedLimit 0

# 重试推送
git push -u origin main
```

### 问题：文件过大
如果有超过100MB的文件会被GitHub拒绝，可以：

```bash
# 1. 检查大文件
find . -size +100M -type f

# 2. 选项A: 删除不必要的大文件
# 3. 选项B: 使用Git LFS
git lfs install
git lfs track "*.bin"
git add .gitattributes
git add .
git commit -m "Add Git LFS tracking"
git push
```

## 验证上传成功

推送完成后，访问你的仓库：
```
https://github.com/your-username/your-repo
```

应该能看到：
- ✓ 所有文件都被上传
- ✓ commit历史显示你的提交
- ✓ 分支为 `main`

## 下一步建议

1. **添加README**
   ```bash
   # 编辑README.md，说明TriangleMix功能
   git add README.md
   git commit -m "docs: Add TriangleMix documentation"
   git push
   ```

2. **创建发布版本**
   ```bash
   git tag -a v1.0-trianglemix -m "TriangleMix implementation v1.0"
   git push origin v1.0-trianglemix
   ```

3. **设置分支保护**（在GitHub Settings中）
   - 保护main分支
   - 需要Pull Request审查

## 使用SSH的完整步骤

如果已配置SSH密钥：

```bash
cd /Users/tsy/Downloads/vllm-0.11.0

git init
git config user.name "Your Name"
git config user.email "your@email.com"

git add .
git commit -m "Initial commit: TriangleMix implementation"

# 使用SSH URL
git remote add origin git@github.com:your-username/your-repo.git

git branch -M main
git push -u origin main
```

## 常用命令参考

```bash
# 查看远程仓库
git remote -v

# 修改远程URL
git remote set-url origin https://github.com/your-username/your-repo.git

# 查看提交历史
git log --oneline

# 查看当前分支
git branch

# 查看本地文件状态
git status

# 查看修改内容
git diff

# 撤销文件修改
git checkout -- filename

# 撤销commit（保留改动）
git reset --soft HEAD~1
```

## 验证清单

- [ ] GitHub仓库已创建
- [ ] 已安装Git
- [ ] Git用户名和邮箱已配置
- [ ] 已运行上传脚本或执行手动步骤
- [ ] 推送成功（无错误消息）
- [ ] 访问GitHub仓库URL确认文件已上传
- [ ] 所有TriangleMix相关文件都在 `vllm/attention/` 中

---

**需要更详细的帮助？** 参考 `GITHUB_UPLOAD_GUIDE.md`
