# CodexMemory

CodexMemory 是一个 Windows 本地记忆库。它按需归档 Codex 对话，在 SQLite 中保存可回查的正文，并通过 MCP 提供关键词检索、语义检索和结论写入。向量由本机 CPU 生成。

当前版本为 0.1.0。部署需要 Windows x64、支持 AVX2 的 CPU、64 位 Python 3.12，以及可运行的 Codex CLI。程序、模型、数据库和归档日志位于安装目录。

## 1. 安装 Python 依赖

在一个可长期使用的目录打开 PowerShell，执行以下命令。已有源码时，从包含 `run.py` 的目录开始。也可以从 GitHub 的 Code 菜单下载源码 ZIP 并解压。

~~~powershell
git clone https://github.com/nellnoob/codex-memory.git
Set-Location -LiteralPath .\codex-memory
py -3.12 -m venv .venv
$pythonExe = (Resolve-Path -LiteralPath '.\.venv\Scripts\python.exe').Path
$launcher = (Resolve-Path -LiteralPath '.\run.py').Path
& $pythonExe -B -m pip install -r .\requirements.txt
& $pythonExe -B -m pip check
& $pythonExe -B $launcher --version
~~~

版本输出应为 `0.1.0`。如果 Windows 没有 `py` 启动器，可用已经安装的 64 位 `python.exe` 的完整路径运行 `-m venv .venv`。 [Python Windows 下载](https://www.python.org/downloads/windows/) · [Codex CLI](https://learn.chatgpt.com/docs/cli)

## 2. 自行部署 embedding 小模型

部署者需单独下载 IBM Granite Embedding 97M Multilingual R2 的 ONNX INT8 AVX2 权重和分词器。两份文件合计约 124 MB。下面的命令从 IBM 的发布仓库下载固定版本，校验 SHA-256，并存到 `data\granite`。下载完成后，ONNX Runtime 会直接在本机 CPU 上生成 384 维向量。

~~~powershell
& $pythonExe -B $launcher download-model
~~~

如果目标机无法直连 Hugging Face，可以先在能联网的电脑下载以下两个官方文件，再复制到安装目录的 `data\granite`。两个文件都放在 `granite` 目录下。复制完成后，再运行 `download-model` 校验。完整文件会直接通过校验。

| 本地文件 | 官方下载 |
| --- | --- |
| `data/granite/model_quint8_avx2.onnx` | [INT8 权重](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2/resolve/c61e626/onnx/model_quint8_avx2.onnx) |
| `data/granite/tokenizer.json` | [分词器](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2/resolve/c61e626/tokenizer.json) |

0.1.0 使用上述 Granite 版本和分词器。使用其他 embedding 模型需要适配分词、推理和索引，并重新生成向量。 [IBM 模型卡](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2)

## 3. 用示例试跑

下列命令只读取仓库里的四条模拟消息，并在 `data\demo.sqlite3` 中验证索引与检索。

~~~powershell
$demoDb = Join-Path (Get-Location).Path 'data\demo.sqlite3'
& $pythonExe -B $launcher --db $demoDb sync --source .\demo\rollout-demo.jsonl
& $pythonExe -B $launcher --db $demoDb index
& $pythonExe -B $launcher --db $demoDb search 'model_fit.R' --mode keyword
& $pythonExe -B $launcher --db $demoDb search '怎样按意思查找以前的聊天记录？' --mode vector
~~~

关键词查询应命中包含 `model_fit.R` 的消息。语义查询可找到有关检索旧聊天的英文片段。用结果中的消息 ID 执行 `read` 可以回看原文。

~~~powershell
& $pythonExe -B $launcher --db $demoDb read '填入实际的 m_ 消息 ID'
~~~

## 4. 归档本机 Codex 会话

`sync` 读取当前 Codex 用户目录下 `sessions` 和 `archived_sessions` 中已落盘的用户及助手文本，写入 `data\memory.sqlite3`。脚本按文件偏移增量读取，数据库和向量留在本机。全新安装可先完成一轮 Codex 对话，生成会话日志。

~~~powershell
$codexRoot = if ($env:CODEX_HOME) { $env:CODEX_HOME } else { Join-Path $env:USERPROFILE '.codex' }
& $pythonExe -B $launcher sync --codex-home $codexRoot
& $pythonExe -B $launcher index
& $pythonExe -B $launcher status
~~~

首轮 `index` 的用时取决于会话量和 CPU。`pending_embeddings` 表示待生成向量的片段数。正文入库后即可用 `--mode keyword` 查询。

## 5. 接入 Codex MCP

用 `codex mcp list` 检查同名配置。已有 `codex-memory` 时，先用 `codex mcp get codex-memory` 核对它指向的安装目录。首次注册执行下列命令。

~~~powershell
codex mcp add codex-memory -- $pythonExe '-B' $launcher serve
codex mcp get codex-memory
~~~

打开新聊天后，可以让 Codex 查询以前的对话。如果工具列表尚未刷新，重开 Codex。 [Codex MCP 配置说明](https://learn.chatgpt.com/docs/extend/mcp?surface=cli)

| 工具 | 用途 |
| --- | --- |
| `memory_search` | 按关键词、向量或两者结合查找对话片段和结论 |
| `memory_read` | 按 ID 分页展开原文或结论 |
| `memory_thread` | 按顺序查看同一会话的消息预览 |
| `memory_write` | 保存可跨对话复用的结论，可关联原始消息 |

## 6. 启动持续归档

首轮 `index` 完成后，在 PowerShell 中启动 `watch`。它默认每轮结束后等 30 秒再检查新消息，按批次补齐向量。关闭窗口或按 Ctrl+C 会结束前台归档。

~~~powershell
& $pythonExe -B $launcher watch --codex-home $codexRoot
~~~

### 可选的 Windows 登录自启动

用实际运行 Codex 的 Windows 用户执行以下命令。它将项目中的 `watch-start.pyw` 注册到当前用户的 `Run` 项。登录后，启动器会静默运行并在 `logs` 中记录输出。若使用自定义 `CODEX_HOME`，它需要在登录后的环境中可用。 [Microsoft Run 注册表项说明](https://learn.microsoft.com/zh-cn/windows/win32/setupapi/run-and-runonce-registry-keys)

~~~powershell
$pythonw = (Resolve-Path -LiteralPath '.\.venv\Scripts\pythonw.exe').Path
$startup = (Resolve-Path -LiteralPath '.\watch-start.pyw').Path
$commandLine = '"{0}" -B "{1}"' -f $pythonw, $startup
if ($commandLine.Length -gt 260) { throw '请将项目放到更短的路径。' }
$key = [Microsoft.Win32.Registry]::CurrentUser.CreateSubKey('Software\Microsoft\Windows\CurrentVersion\Run')
try {
    $current = $key.GetValue('CodexMemoryWatch')
    if ($null -eq $current) {
        $key.SetValue('CodexMemoryWatch', $commandLine, [Microsoft.Win32.RegistryValueKind]::String)
    } elseif ($current -cne $commandLine) {
        throw '已有一个指向其他安装的 CodexMemoryWatch。请先核对旧目录。'
    }
    $key.GetValue('CodexMemoryWatch')
} finally {
    $key.Dispose()
}
~~~

停用时移除当前用户的 `CodexMemoryWatch` Run 值。解除 MCP 注册可执行 `codex mcp remove codex-memory`。这些设置与本地的 `data\memory.sqlite3` 分开管理。

## 验证

~~~powershell
$env:CODEX_MEMORY_TEST_MODEL = (Resolve-Path -LiteralPath '.\data\granite').Path
& $pythonExe -B -m unittest discover -s tests -v
~~~

测试覆盖增量归档、索引、检索、原文回查和 MCP 写入。仓库只分发程序、文档、测试和合成会话。实际聊天记录、数据库、虚拟环境和模型由部署机器自行保管。
