PageRank Web RWD Project
===

🎉 PageRank應用於高中棒球排名的動態網站專案！
---

## 專案簡介
這是一個 **動態網站** ，結合了 **PageRank 演算法** 與 **RWD (響應式設計)**，專注於台灣高中棒球賽事數據分析及展示。
本專案旨在呈現球隊實力分布及發展趨勢，主要針對 **中信盃黑豹旗** 等具代表性的棒球賽事，
透過數據視覺化與動態後端串接，讓用戶直觀理解數據背後的價值。

## 專案特色
- 數據驅動的動態網站
使用 **PageRank 演算法** 處理並展示高中棒球賽事數據，揭示排名與趨勢。

+ 響應式設計 (RWD)
採用 HTML、CSS、Bootstrap 與 Flask 等技術，確保網站在各裝置上都有順暢的瀏覽體驗。

* 前後端整合
使用 **Flask (Python) **進行後端處理，資料庫使用** MySQL**進行數據存取與管理。

功能展示
本網站分為多個功能頁面，提供完整的數據展示體驗：

1. Home
- 平台概覽與主要功能介紹。
2. Data
+ 高中棒球數據分析與排名視圖，展示 PageRank 演算法結果。
3. School
* 探索各學校與球隊的詳細資訊。
Visual
- 可互動的動態數據圖表，揭示數據背後的趨勢與深意。
## 使用技術
### 前端技術
- **HTML**：構建網站結構與語義化標籤。
+ **CSS**：自定義樣式，提供視覺吸引力與 RWD 設計。
* **Bootstrap**：加速響應式設計開發，提升用戶體驗。
### 後端技術
- **Flask** (Python)：作為輕量級 Web 框架，負責後端處理與 API 開發。
+ **MySQL**：資料庫連接與數據管理。
### 其他工具
- **Visual Studio Code**：編寫前後端程式碼。
+ **MySQL Workben***：設計並管理資料庫。
* **GitHub**：版本控制與專案部署。
## 如何運行專案？
1. 複製專案到本地
```
git clone https://github.com/lucapow/PageRankWeb_RWD-.git
cd PageRankWeb_RWD-
```
3. 安裝所需套件
確保 Python 已安裝，並安裝 requirements.txt 中的依賴項：
```
pip install -r requirements.txt
```
4. 設置 MySQL 資料庫
- 在 MySQL Workbench 建立 sports_data 資料庫。
+ 確保 app.py 中的資料庫連接資訊正確：
```
self.db = mysql.connector.connect(
    host="127.0.0.1",
    user="your_username",
    password="your_password",
    database="sports_data"
)
```
4. 啟動 Flask 應用
```
python app.py
```
5. 在瀏覽器中打開

* 開啟 http://127.0.0.1:5000，即可查看網站。
## 網站預覽
你可以在這裡查看完整的網站：
🔗 PageRank Web RWD Demo

## 關於作者
👋 你好！
--
我是即將畢業的大四學生，專注於前後端技術開發，擅長**靜態網站設計**、**數據處理** 及 **資料庫串接**。
我對數據分析、演算法應用充滿熱情，致力於提供創新的數據視覺化體驗。

📧 聯絡方式：luca19111215@gmail.com
🔗 其他專案：歡迎參考我的 GitHub 個人專頁！

