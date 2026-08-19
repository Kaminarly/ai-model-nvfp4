# Issue tracker: Local Markdown

本项目的议题和规格保存在 `.scratch/` 下的 Markdown 文件中。

## Conventions

- 每个功能使用一个目录：`.scratch/<feature-slug>/`。
- 规格文件为 `.scratch/<feature-slug>/spec.md`。
- 实施议题每项一个文件，路径为 `.scratch/<feature-slug>/issues/<NN>-<slug>.md`，编号从 `01` 开始；不得合并为单一事项文件。
- 分流状态记录在议题文件顶部附近的 `Status:` 行；角色字符串见 `triage-labels.md`。
- 评论与对话记录追加在文件末尾的 `## Comments` 标题下。

## 当技能要求发布到议题追踪器时

在 `.scratch/<feature-slug>/` 下创建相应文件；不存在时一并创建目录。

## 当技能要求读取相关议题时

读取用户提供的文件路径或议题编号所对应的 Markdown 文件。

## Wayfinding operations

供 `/wayfinder` 使用。地图文件与每个子议题分别保存。

- 地图：`.scratch/<effort>/map.md`，记录 Notes、Decisions-so-far 与 Fog。
- 子议题：`.scratch/<effort>/issues/NN-<slug>.md`，编号从 `01` 开始；文件顶部使用 `Type:` 记录 `research`、`prototype`、`grilling` 或 `task`，使用 `Status:` 记录 `claimed` 或 `resolved`。
- 依赖：在文件顶部附近使用 `Blocked by: NN, NN`。列出的所有议题均为 `resolved` 时，该议题解除阻塞。
- 前沿：扫描 `.scratch/<effort>/issues/`，选择未关闭、未阻塞且未认领的议题，编号最小者优先。
- 认领：先将 `Status:` 写为 `claimed` 并保存，再开始处理。
- 解决：在 `## Answer` 下追加结论，设置 `Status: resolved`，并在 `map.md` 的 Decisions-so-far 中追加简要链接。
