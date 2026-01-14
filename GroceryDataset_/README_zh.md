# 雜貨店數據集 (Grocery Dataset)

此存儲庫包含雜貨店數據集 (Grocery Dataset) 的標註檔案及圖像下載連結。

雜貨店數據集是一個圖像數據集：
* 收集自約 40 間雜貨店，使用了 4 種攝影機。
* 包含 10 個產品類別。
* 於 2014 年春季由土耳其伊斯坦堡的 Idea Teknoloji 建立。

本數據集僅供研究使用，不得用於商業用途。

## 下載連結
以下是下載圖像檔案的連結：[第一部分 (part1)](https://github.com/gulvarol/grocerydataset/releases/download/1.0/GroceryDataset_part1.tar.gz) 和 [第二部分 (part2)](https://github.com/gulvarol/grocerydataset/releases/download/1.0/GroceryDataset_part2.tar.gz)。總磁碟空間約為 3GB。你也可以在此存儲庫的 Releases 頁面找到這些連結。

## 資料夾結構

### ShelfImages (貨架圖像)
此資料夾包含 354 張雜貨店貨架圖像。

命名規則如下：
```
			"C<c>_P<p>_N<n>_S<s>_<i>.JPG"
			其中
				<c> := 攝影機 ID (1: iPhone5S, 2: iPhone4, 3: Sony Cybershot, 4: Nikon Coolpix)
				<p> := 平面圖 (planogram) ID
				<n> := 圖像上頂層貨架根據平面圖的排名
				<s> := 圖像中的貨架數量
				<i> := 副本編號
```

### ProductImages (產品圖像)
此資料夾包含 10 個不同產品類別的圖像（編號 1-10）。每個類別都有獨立的目錄。

命名規則如下：
```
		"B<b>_N<N>.JPG"
			其中
				<b> := 品牌 ID
				<n> := 副本編號
```

### BrandImages (品牌圖像)
此資料夾包含 `ProductImages` 目錄的裁切版本。圖像經過裁切，僅保留品牌標誌。

### ProductImagesFromShelves (貨架上的產品圖像)
此資料夾包含從貨架圖像中裁切出的產品圖像。它們被分為 10 個產品類別，外加一個負樣類別（收集不屬於這 10 類的產品）。

命名規則如下：
```
			"<貨架圖像名稱>_<x>_<y>_<w>_<h>.png"
			其中
			<貨架圖像名稱>   := 裁切來源的原始圖像名稱
			<x>              := 圖像左上角在原始圖像上的 x 座標
			<y>              := 圖像左上角在原始圖像上的 y 座標
			<w>              := 圖像在原始圖像上的寬度
			<h>              := 圖像在原始圖像上的高度
```

### BrandImagesFromShelves (貨架上的品牌圖像)
此資料夾包含 `ProductImagesFromShelves` 目錄的裁切版本。圖像經過裁切，僅保留品牌標誌。

*******************************
## 標註檔案 (Annotation files)

### annotation.txt
此檔案將貨架圖像的標註資訊彙整為一個文字檔。每一行代表一張貨架圖像。

單行格式如下：
```
			<貨架圖像名稱> <n> <x_1> <y_1> <w_1> <h_1> <b_1> <x_2> <y_2> <w_2> <h_2> <b_2> ... <x_n> <y_n> <w_n> <h_n> <b_n>
			其中
			<貨架圖像名稱>   := 貨架圖像名稱
			<n>              := 該貨架圖像上的產品數量
			<x_i>            := 第 i 個產品圖像的 x 座標
			<y_i>            := 第 i 個產品圖像的 y 座標
			<w_i>            := 第 i 個產品圖像的寬度
			<h_i>            := 第 i 個產品圖像的高度
			<b_i>            := 第 i 個產品圖像的品牌
```

### subset.txt
此檔案列出了用於訓練和測試的圖像檔案名稱。它是 `BrandImages` 和 `BrandImagesFromShelves` 內容的一個子集。

## 引用
如果您使用此數據集，請引用以下內容：
> @article{varol16a,  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;TITLE = {{Toward Retail Product Recognition on Grocery Shelves}},  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;AUTHOR = {Varol, G{\"u}l and Kuzu, Ridvan S.},  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;JOURNAL =  {ICIVC},  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;YEAR = {2014}  
}

## 致謝
本數據集是 Idea Teknoloji 執行之 TUBITAK 資助項目的一部分。

## 開源貢獻
[sayakpaul](https://github.com/sayakpaul) 開發了 [Grocery-Product-Detection](https://github.com/sayakpaul/Grocery-Product-Detection) 存儲庫，展示了如何使用此數據集訓練物件偵測模型。
