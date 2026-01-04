# YOLO-UniOW：使用 UV 進行環境設置

本指南使用 **uv** 作為包管理器和虛擬環境管理工具，替代傳統的 conda 和 pip。

## 前置要求

- **Python 3.9+**（已安裝在系統中）
- **uv**（已安裝，版本 0.9.21+）
- **CUDA 11.8**（GPU 支持）
- **PyTorch 2.1.2**

## 快速安裝步驟

### 1. 建立虛擬環境

使用 uv 在項目目錄建立虛擬環境：

```bash
cd C:\Users\dachen\YOLO-UniOW
uv venv .venv --python 3.9
```

**說明：**
- `.venv` 是虛擬環境的目錄名稱（建立在項目根目錄）
- `--python 3.9` 指定 Python 版本

### 2. 激活虛擬環境

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
.\.venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

### 3. 安裝 PyTorch（CUDA 11.8）

```bash
uv pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
```

### 4. 安裝 mmcv

```bash
uv pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
```

### 5. 安裝項目依賴

```bash
uv pip install -r requirements.txt
```

### 6. 安裝項目本身

```bash
uv pip install -e .
```

## 完整命令列表

如果您想一次執行所有步驟，可以使用以下命令序列：

```bash
# 進入項目目錄
cd C:\Users\dachen\YOLO-UniOW

# 建立虛擬環境
uv venv .venv --python 3.9

# 激活環境（Windows PowerShell）
.\.venv\Scripts\Activate.ps1

# 安裝所有依賴
uv pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
uv pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
uv pip install -r requirements.txt
uv pip install -e .

# 驗證安裝
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import mmcv; print(f'mmcv {mmcv.__version__}')"
```

## UV 常用命令

| 命令 | 說明 |
|------|------|
| `uv venv .venv` | 建立虛擬環境 |
| `uv pip install <package>` | 安裝單個包 |
| `uv pip install -r requirements.txt` | 安裝所有依賴 |
| `uv pip list` | 列出已安裝的包 |
| `uv pip uninstall <package>` | 卸載包 |
| `uv pip sync requirements.txt` | 同步環境與 requirements.txt |

## 環境位置

✅ **虛擬環境位置：** `C:\Users\dachen\YOLO-UniOW\.venv`

相比 conda，uv 的優點：
- 環境建立在項目目錄，便於管理
- 包安裝速度更快（使用 Rust 編寫）
- 內存占用更少
- 環境隔離更清晰

## 驗證安裝

安裝完成後，驗證環境：

```bash
# 檢查 Python 版本
python --version

# 檢查 PyTorch
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# 檢查 mmcv
python -c "import mmcv; print(mmcv.__version__)"

# 檢查 yolo_world
python -c "import yolo_world; print('YOLO-UniOW ready!')"
```

## 停用虛擬環境

```bash
deactivate
```

## 常見問題

**Q：如何刪除虛擬環境重新開始？**
```bash
rm -r .venv  # Linux/Mac
rmdir /s .venv  # Windows cmd
Remove-Item -Recurse .venv  # Windows PowerShell
```

**Q：如何更新已安裝的包？**
```bash
uv pip install --upgrade <package>
```

**Q：如何在項目中添加新依賴？**
```bash
uv pip install <new-package>
# 然後更新 requirements.txt
uv pip freeze > requirements.txt
```
