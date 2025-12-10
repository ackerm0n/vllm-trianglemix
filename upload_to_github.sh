#!/bin/bash

# TriangleMix vLLM - 一键上传到GitHub脚本
# 使用方法: ./upload_to_github.sh <github-username> <repo-name>

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 欢迎信息
print_header() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║     vLLM TriangleMix - GitHub上传工具                      ║"
    echo "║     Upload vLLM with TriangleMix to GitHub                ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# 错误处理
error_exit() {
    echo -e "${RED}❌ 错误: $1${NC}"
    exit 1
}

# 成功提示
success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# 信息提示
info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

print_header

# 检查参数
if [ $# -lt 2 ]; then
    echo -e "${YELLOW}用法: $0 <github-username> <repo-name> [email]${NC}"
    echo ""
    echo "示例:"
    echo "  $0 myusername vllm-trianglemix"
    echo "  $0 myusername vllm-trianglemix myemail@example.com"
    echo ""
    exit 1
fi

GITHUB_USERNAME="$1"
REPO_NAME="$2"
EMAIL="${3:-your-email@github.com}"
VLLM_DIR="/Users/tsy/Downloads/vllm-0.11.0"
REPO_URL="https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"

# 验证目录
if [ ! -d "$VLLM_DIR" ]; then
    error_exit "找不到vLLM目录: $VLLM_DIR"
fi

echo ""
info "配置信息:"
echo "  GitHub用户名: $GITHUB_USERNAME"
echo "  仓库名称: $REPO_NAME"
echo "  邮箱: $EMAIL"
echo "  仓库地址: $REPO_URL"
echo "  本地目录: $VLLM_DIR"
echo ""

# 确认信息
read -p "$(echo -e ${YELLOW}确认继续? [y/N]${NC}) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

echo ""
info "步骤 1/5: 检查Git安装"
if ! command -v git &> /dev/null; then
    error_exit "Git未安装，请先安装Git"
fi
success "Git已安装"

echo ""
info "步骤 2/5: 初始化本地仓库"
cd "$VLLM_DIR"

if [ -d ".git" ]; then
    info "Git仓库已初始化，跳过初始化步骤"
else
    git init
    success "Git仓库初始化完成"
fi

# 配置用户
git config user.name "$GITHUB_USERNAME"
git config user.email "$EMAIL"
success "用户配置完成"

echo ""
info "步骤 3/5: 检查修改的文件"
echo ""

# 获取统计信息
TOTAL_FILES=$(find . -type f ! -path './.git/*' | wc -l)
MODIFIED_FILES=$(git diff --name-only 2>/dev/null | wc -l)
UNTRACKED_FILES=$(git ls-files --others --exclude-standard | wc -l)

echo "  总文件数: $TOTAL_FILES"
echo "  修改的文件: $MODIFIED_FILES"
echo "  未追踪的文件: $UNTRACKED_FILES"
echo ""

# 显示修改的文件列表
if [ $MODIFIED_FILES -gt 0 ]; then
    echo "  修改的文件:"
    git diff --name-only 2>/dev/null | sed 's/^/    - /'
fi

if [ $UNTRACKED_FILES -gt 0 ]; then
    echo ""
    echo "  新增的文件 (部分):"
    git ls-files --others --exclude-standard | head -10 | sed 's/^/    - /'
    if [ $UNTRACKED_FILES -gt 10 ]; then
        echo "    ... 和其他 $((UNTRACKED_FILES - 10)) 个文件"
    fi
fi

echo ""
info "步骤 4/5: 提交修改"
echo ""

# 添加所有文件
git add .
success "所有文件已暂存"

# 获取提交统计
CHANGES=$(git diff --cached --stat | tail -1)
echo "  提交统计: $CHANGES"
echo ""

# 创建提交
COMMIT_MSG="feat: TriangleMix attention implementation for efficient long-context prefilling

- Implement TriangleMix sparse attention pattern
- Add Streaming, Last Q-K, and Middle Q-K section masks
- Create Triangle mask generation for reduced complexity from O(N²) to O(N)
- Add Ascend NPU specific optimizations
- Integrate with Qwen3 model architecture
- Implement gradient-based layer analysis for automatic Triangle layer selection
- Add comprehensive test suite and documentation
- Include quick reference and usage guides

Features:
- Static sparse patterns (no dynamic kernel prediction needed)
- Block-wise optimization for Ascend NPU execution
- Support for custom layer selection strategies
- Cache mechanism for repeated mask generation
- Compatible with long-context inference scenarios"

git commit -m "$COMMIT_MSG"
success "提交完成"

echo ""
info "步骤 5/5: 上传到GitHub"
echo ""

# 检查远程是否已存在
if git remote get-url origin &>/dev/null; then
    info "移除已有的远程仓库"
    git remote remove origin
fi

# 添加远程
git remote add origin "$REPO_URL"
success "远程仓库已添加"

# 重命名分支
git branch -M main
info "分支已重命名为 main"

# 推送
echo ""
info "推送代码到GitHub (这可能需要几分钟)..."
echo ""

if git push -u origin main 2>&1; then
    echo ""
    success "代码上传成功！"
    echo ""
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✓ 上传完成！${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "仓库信息:"
    echo "  URL: $REPO_URL"
    echo "  分支: main"
    echo ""
    echo "下一步:"
    echo "  1. 访问 $REPO_URL 查看仓库"
    echo "  2. 更新README.md说明TriangleMix实现"
    echo "  3. 添加GitHub Actions CI/CD配置"
    echo "  4. (可选) 创建release版本"
    echo ""
else
    error_exit "推送失败，请检查:
    1. GitHub仓库是否已创建
    2. GitHub登录凭据是否正确
    3. 网络连接是否正常
    
    你可以手动执行:
    cd $VLLM_DIR
    git push -u origin main --force"
fi

echo ""
info "脚本执行完成！"
echo ""
