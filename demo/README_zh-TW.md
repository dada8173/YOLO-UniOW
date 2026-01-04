## YOLO-World 示範

### 開始使用

將 `PYTHONPATH` 設置為 `YOLO-World` 的路徑並執行：

```bash
PYTHONPATH=/xxxx/YOLO-World python demo/yyyy_demo.py
# 或直接使用
PYTHONPATH=./ python demo/yyyy_demo.py
```

#### Gradio 示範

我們為本地設備提供了 [Gradio](https://www.gradio.app/) 示範：

```bash
pip install gradio==4.16.0
python demo/demo.py path/to/config path/to/weights
```

此外，您可以使用 Dockerfile 構建帶有 gradio 的映像。作為先決條件，請確保您已安裝相應的驅動程序以及 [nvidia-container-runtime](https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime)。將 MODEL_NAME 和 WEIGHT_NAME 替換為相應的值，或省略此步驟並使用 [Dockerfile](Dockerfile#3) 中的默認值。

```bash
docker build --build-arg="MODEL=MODEL_NAME" --build-arg="WEIGHT=WEIGHT_NAME" -t yolo_demo .
docker run --runtime nvidia -p 8080:8080
```

#### 圖像示範

我們提供了一個簡單的圖像示範，用於對圖像進行推理並輸出視覺化結果。

```bash
python demo/image_demo.py path/to/config path/to/weights image/path/directory 'person,dog,cat' --topk 100 --threshold 0.005 --output-dir demo_outputs
```

**注意事項：**
* `image` 可以是一個目錄或單個圖像。
* `texts` 可以是用逗號分隔的類別字符串（名詞短語）。我們也支持 `txt` 文件，其中每行包含一個類別（名詞短語）。
* `topk` 和 `threshold` 控制預測數量和置信度閾值。


#### 視頻示範

`video_demo` 具有與 `image_demo` 類似的超參數。

```bash
python demo/video_demo.py path/to/config path/to/weights video_path 'person,dog' --out out_video_path
```

### 常見問題

> 1. `Failed to custom import!`
```bash
  File "simple_demo.py", line 37, in <module>
    cfg = Config.fromfile(config_file)
  File "/data/miniconda3/envs/det/lib/python3.8/site-packages/mmengine/config/config.py", line 183, in fromfile
    raise ImportError('Failed to custom import!') from e
ImportError: Failed to custom import!
```
**解決方案：**

```bash
PYTHONPATH=/xxxx/YOLO-World python demo/simple_demo.py
```
