# Agent Skills

- 不要用长 heredoc + ssh 写远程文件，容易 timeout。
- 不要用 shell heredoc 写包含 shebang 的文件。
- 本地文件创建/编辑用 str_replace_editor。
- 远程文件传输用 scp。
- 复杂转换先写成脚本文件再运行。
- bash 命令保持短小，避免超过 30 秒。
