# GitHub Pages 设置说明

本目录包含用于 GitHub Pages 的文档文件。

## 文件说明

- `index.md`: API Reference 主文档（从 `doc/API_REFERENCE.md` 复制）
- `_config.yml`: Jekyll 主题配置

## 设置步骤

1. **提交并推送文件**：
   ```bash
   git add docs/
   git commit -m "Add docs for GitHub Pages"
   git push
   ```

2. **在 GitHub 上启用 Pages**：
   - 打开仓库，点击右上角的 **Settings**
   - 在左侧菜单中找到 **Pages**
   - 在 _Source_ 部分选择：
     - Branch: `main` 或 `dev`（根据你的主分支）
     - Folder: `/docs`
   - 点击 **Save**

3. **等待部署**：
   - GitHub 会在几分钟内构建并部署你的站点
   - 部署完成后，你会看到一个链接，例如：
     ```
     https://yourusername.github.io/OrcaGym/
     ```

## 注意事项

- 文档中的链接已调整为从 `docs/` 目录访问 `doc/` 目录的相对路径
- 如果修改了 `doc/` 目录下的文件，需要手动更新 `docs/index.md` 中的链接
- Jekyll 主题 `jekyll-theme-cayman` 会自动渲染 Markdown 文件

