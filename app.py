from flask import Flask, request, jsonify, render_template, send_file
import os
import logging
from datetime import datetime
import numpy as np
from scipy.sparse import csr_matrix
from scipy import linalg
import matplotlib
matplotlib.use('Agg')  # 適用於非 GUI 環境
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import pandas as pd
import networkx as nx
from sqlalchemy import create_engine
import mysql.connector
from mysql.connector import Error
import geopandas as gpd
font_path = os.path.join("static", "fonts", "TaipeiSansTCBeta-Bold.ttf")
prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.sans-serif'] = prop.get_name()
plt.rcParams['axes.unicode_minus'] = False
# SQLAlchemy 引擎配置engine = create_engine(DB_URI)

# Flask 應用初始化
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# 資料庫連接上下文管理器
class DBConnection:
    def __init__(self):
        self.db = None
        self.cursor = None

    def __enter__(self):
        try:
            self.db = mysql.connector.connect(
                host="127.0.0.1",
                user="luca1911",
                password="19111215",
                database="sports_data"
            )
            self.cursor = self.db.cursor(dictionary=True)
            return self.cursor
        except mysql.connector.Error as err:
            logging.error(f"資料庫連線失敗: {err}")
            raise RuntimeError(f"資料庫連線失敗: {err}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cursor:
            self.cursor.close()
        if self.db:
            self.db.close()
        if exc_type or exc_val or exc_tb:
            logging.error(f"資料庫操作中發生異常: {exc_val}")

# 節點類，用於建模每支球隊的數據
class Node:
    def __init__(self, name):
        self.name = name
        self.losses = {}
        self.games = []
        self.wins_count = 0
        self.losses_count = 0
        self.region = None

    def add_game(self, opponent, point_diff, date, is_win):
        self.games.append({'opponent': opponent, 'point_diff': point_diff, 'date': date, 'is_win': is_win})
        if is_win:
            self.wins_count += 1
        else:
            self.losses_count += 1
            self.losses[opponent] = self.losses.get(opponent, 0) + 1

# 從資料庫中加載比賽數據
def load_data_from_db(year):
    try:
        with DBConnection() as cursor:
            query = """
            SELECT 
                m.Match_id, m.Year, m.Visitor, m.Home, m.Month, m.Date, 
                m.PTS_Visit, m.PTS_Home, 
                v.HighSchool AS Visitor_School_Name, COALESCE(v.County, "未知地區") AS Visitor_Region,
                h.HighSchool AS Home_School_Name, COALESCE(h.County, "未知地區") AS Home_Region
            FROM 
                match_data m
            JOIN 
                highschool v ON m.Visitor_id = v.HighSchool_id
            JOIN 
                highschool h ON m.Home_id = h.HighSchool_id
            WHERE 
                m.Year = %s
            """
            cursor.execute(query, (year,))
            games = cursor.fetchall()
            games_df = pd.DataFrame(games)

            if games_df.empty:
                logging.warning(f"未找到年份 {year} 的比賽數據")
            else:
                logging.info(f"成功加載 {len(games_df)} 條比賽數據")
            return games_df

    except mysql.connector.Error as err:
        logging.error(f"資料庫查詢失敗: {err}")
        return pd.DataFrame()

def build_graph(games):
    nodes = {}
    for _, game in games.iterrows():
        loser = game['Visitor_School_Name'] if game['PTS_Home'] > game['PTS_Visit'] else game['Home_School_Name']
        winner = game['Home_School_Name'] if game['PTS_Home'] > game['PTS_Visit'] else game['Visitor_School_Name']
        point_diff = abs(game['PTS_Home'] - game['PTS_Visit'])

        if loser not in nodes:
            nodes[loser] = Node(loser)
            nodes[loser].region = game['Visitor_Region']  # 修正地區設置
        if winner not in nodes:
            nodes[winner] = Node(winner)
            nodes[winner].region = game['Home_Region']  # 修正地區設置

        nodes[loser].add_game(winner, point_diff, game['Year'], is_win=False)
        nodes[winner].add_game(loser, point_diff, game['Year'], is_win=True)

    logging.info(f"構建節點圖完成，共有 {len(nodes)} 個節點")
    return nodes

# PageRank 算法
def pageRank(H, damping_factor=0.85, max_iterations=100, tol=1e-6):
    num_teams = H.shape[0]
    pr = np.ones(num_teams) / num_teams

    for i in range(max_iterations):
        pr_new = damping_factor * H.T.dot(pr) + (1 - damping_factor) / num_teams
        if np.linalg.norm(pr_new - pr, 1) < tol:
            logging.info(f"PageRank 在第 {i + 1} 次迭代後收斂")
            break
        pr = pr_new

    return pr / np.sum(pr)

def get_custom_years(start_year, end_year):
    """
    根據用戶提供的起始和結束年份生成年份範圍列表。
    """
    try:
        start_year = int(start_year)
        end_year = int(end_year)

        if start_year > end_year:
            raise ValueError("起始年份不能大於結束年份")
        if start_year < 2013 or end_year > 2023:
            raise ValueError("年份範圍必須在 2013 到 2023 之間")

        return list(range(start_year, end_year + 1))
    except ValueError as e:
        logging.error(f"年份範圍錯誤: {e}")
        return []
    
def get_school_regions_and_ranks(teams, year):
    """
    查詢學校的地區名稱與官方排名，並記錄匹配日誌。
    """
    ranks = {}
    regions = {}
    pairing_logs = []

    try:
        with DBConnection() as cursor:
            # 查詢所有學校的地區
            cursor.execute("""
            SELECT HighSchool_id, HighSchool, County
            FROM highschool
            """)
            school_data = cursor.fetchall()
            school_map = {row["HighSchool"]: row for row in school_data}

            # 查詢指定年份的官方排名
            cursor.execute("""
            SELECT o.HighSchool_id, o.ranking, h.HighSchool
            FROM office_rank o
            JOIN highschool h ON o.HighSchool_id = h.HighSchool_id
            WHERE o.year = %s
            """, (year,))
            rank_data = cursor.fetchall()
            rank_map = {row["HighSchool"]: row["ranking"] for row in rank_data}

        # 為隊伍匹配地區和排名
        for team in teams:
            if team in school_map:
                school_info = school_map[team]
                region = school_info["County"] or "未知地區"
                ranks[team] = rank_map.get(team, "無")
                regions[team] = region
                pairing_logs.append(f"成功匹配：{team} => 地區: {region}, 排名: {ranks[team]}")
            else:
                ranks[team] = "無"
                regions[team] = "未知地區"
                pairing_logs.append(f"匹配失敗：{team} => 無法找到對應的學校信息")

    except mysql.connector.Error as err:
        logging.error(f"查詢地區和排名失敗: {err}")

    for log in pairing_logs:
        logging.info(log)

    return ranks, regions, pairing_logs
def calculate_rankings(year):
    """
    計算指定年份的學校排名，並匹配地區和官方排名。
    """
    try:
        games = load_data_from_db(year)
        if games.empty:
            logging.info(f"{year} 年無比賽數據")
            return {"rankings": [], "logs": []}

        nodes = build_graph(games)
        teams = sorted(nodes.keys())
        ranks, regions, logs = get_school_regions_and_ranks(teams, year)

        team_index = {team: i for i, team in enumerate(teams)}
        row, col, data = [], [], []

        for team, node in nodes.items():
            total_losses = sum(node.losses.values())
            for opponent, losses in node.losses.items():
                if opponent in team_index:
                    row.append(team_index[team])
                    col.append(team_index[opponent])
                    data.append(losses / total_losses if total_losses > 0 else 0)

        H_sparse = csr_matrix((data, (row, col)), shape=(len(teams), len(teams)))
        pr_scores = pageRank(H_sparse)

        ranked_teams = [
            {
                "隊伍": team,
                "PR分數": round(pr_scores[index], 4),
                "勝場": nodes[team].wins_count,
                "敗場": nodes[team].losses_count,
                "地區": regions.get(team, "未知地區"),
                "官方排名": ranks.get(team, "無"),
            }
            for team, index in team_index.items()
        ]

        ranked_teams.sort(key=lambda x: x["PR分數"], reverse=True)
        return {"rankings": ranked_teams, "logs": logs}

    except Exception as e:
        logging.error(f"計算排名失敗: {e}", exc_info=True)
        return {"rankings": [], "logs": []}

# API 路由
@app.route('/api/calculate_pr_rank', methods=['GET'])
def calculate_pr_rank():
    year = request.args.get('year', type=int)
    if not year:
        return jsonify({"error": "請提供有效的年份參數"}), 400

    rankings = calculate_rankings(year)

    if rankings:
        # 返回排名結果與配對成功/失敗日誌
        return jsonify(rankings)
    else:
        return jsonify({"message": f"指定年份 {year} 沒有可用的比賽資料"}), 404

import os


def clean_name(name):
    """清理學校名稱：去除前後空格、全形空格和換行符"""
    if name:
        return name.replace('\r', '').replace('\n', '').replace('\u3000', '').strip()
    return ''
# 清理名稱
def clean_name(name):
    """清理學校名稱：去除前後空格、全形空格和換行符"""
    if name:
        return name.replace('\r', '').replace('\n', '').replace('\u3000', '').strip()
    return ''


# 獲取學校詳細資訊
def get_team_details(cursor, team_name):
    team_name = clean_name(team_name)
    print(f"查詢的學校名稱: {team_name}")

    if not team_name:
        return None

    details = {
        'name': team_name,
        'image_path': '/static/image/default.png',
        'highest_rank': '無',
        'championships': 0,
        'avg_score': 0,
        'win_rate': '無比賽紀錄'
    }

    # 查詢勝率
    cursor.execute("""
        SELECT 
            SUM(CASE WHEN (Home = %s AND PTS_Home > PTS_Visit) 
                     OR (Visitor = %s AND PTS_Visit > PTS_Home) THEN 1 ELSE 0 END) AS wins,
            COUNT(*) AS total_games
        FROM match_data
        WHERE Home = %s OR Visitor = %s
    """, (team_name, team_name, team_name, team_name))
    result = cursor.fetchone()
    if result and result['total_games'] > 0:
        details['win_rate'] = f"{round((result['wins'] / result['total_games']) * 100, 2)}%"

    # 查詢學校圖片
    cursor.execute("""
        SELECT HighSchool, image_id
        FROM highschool
        WHERE TRIM(REPLACE(HighSchool, '\u3000', '')) = %s
    """, (team_name,))
    team_data = cursor.fetchone()
    if team_data and team_data['image_id']:
        details['image_path'] = f"/static/image/{team_data['image_id']}.png"

    # 查詢最高排名和冠軍數
    cursor.execute("""
        SELECT 
            MIN(ranking) AS highest_rank,
            COUNT(CASE WHEN ranking = 1 THEN 1 END) AS championships
        FROM office_rank
        WHERE team = %s
    """, (team_name,))
    rank_data = cursor.fetchone()
    if rank_data:
        details['highest_rank'] = rank_data['highest_rank'] if rank_data['highest_rank'] else '無'
        details['championships'] = rank_data['championships']

    # 查詢平均得分
    cursor.execute("""
        SELECT AVG(
            CASE 
                WHEN Home = %s THEN PTS_Home
                ELSE PTS_Visit
            END
        ) AS avg_score
        FROM match_data
        WHERE Home = %s OR Visitor = %s
    """, (team_name, team_name, team_name))
    score_data = cursor.fetchone()
    if score_data and score_data['avg_score'] is not None:
        details['avg_score'] = round(score_data['avg_score'], 2)

    print(f"學校詳細資訊: {details}")
    return details


@app.route('/school', methods=['GET', 'POST'])
def school_page():
    home_team = clean_name(request.args.get('home_team', ''))
    away_team = clean_name(request.args.get('away_team', ''))

    logging.info(f"Cleaned Home Team: '{home_team}', Cleaned Away Team: '{away_team}'")
    print(f"收到的主隊: {home_team}, 客隊: {away_team}")

    try:
        with DBConnection() as cursor:
            cursor.execute("SELECT HighSchool FROM highschool ORDER BY HighSchool")
            schools = [row['HighSchool'] for row in cursor.fetchall()]
            print(f"學校列表: {schools}")

            # 主客隊詳細資訊
            home_team_details = get_team_details(cursor, home_team) if home_team else None
            away_team_details = get_team_details(cursor, away_team) if away_team else None

            # 查詢最近比賽
            cursor.execute("""
                SELECT 
                    Year, Month, Date, 
                    Home, PTS_Home, Visitor, PTS_Visit, type
                FROM match_data
                WHERE (Home = %s AND Visitor = %s) OR (Home = %s AND Visitor = %s)
                ORDER BY Year DESC, Month DESC, Date DESC
                LIMIT 5
            """, (home_team, away_team, away_team, home_team))
            recent_matches = cursor.fetchall()

            # 格式化比賽日期
            month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                         'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
            formatted_matches = []
            for match in recent_matches:
                date_str = f"{match['Year']}/{month_map.get(match['Month'], '未知')}/{match['Date']}"
                formatted_matches.append({
                    'date': date_str,
                    'Home_School_Name': match['Home'],
                    'PTS_Home': match['PTS_Home'],
                    'Visitor_School_Name': match['Visitor'],
                    'PTS_Visit': match['PTS_Visit'],
                    'match_type': match['type']
                })

            print(f"格式化後的最近比賽記錄: {formatted_matches}")

            return render_template(
                'school.html',
                schools=schools,
                home_team_details=home_team_details,
                away_team_details=away_team_details,
                matches=formatted_matches,
                show_results=bool(home_team and away_team)
            )

    except Exception as e:
        logging.error(f"發生異常: {e}")
        print(f"系統異常: {e}")
        return "系統發生錯誤，請稍後再試。", 500






# 通用函數：保存並檢查圖像
def save_and_check_plot(filepath, title):
    """
    保存圖像並檢查其成功生成，防止圖表重疊
    """
    plt.title(title, fontproperties=prop)
    plt.savefig(filepath)
    plt.close()
    if not os.path.exists(filepath):
        print(f"Error: Failed to save plot at {filepath}")
        return False
    return True

def generate_score_trend_chart(filepath, start_year=None, end_year=None, rank=None):
    """生成指定年份範圍內前 N 名學校的得分趨勢圖。"""
    try:
        # 確認年份參數有效
        if not start_year or not end_year:
            logging.warning("起始年份或結束年份未提供")
            return

        years_to_display = list(range(int(start_year), int(end_year) + 1))

        query = """
            WITH TopTeams AS (
                SELECT Home AS team_name, SUM(PTS_Home) AS total_score
                FROM match_data
                WHERE Year BETWEEN %s AND %s
                GROUP BY Home
                ORDER BY total_score DESC
                LIMIT %s
            )
            SELECT Year, team_name, SUM(PTS_Home + PTS_Visit) AS total_score
            FROM match_data
            JOIN TopTeams ON match_data.Home = TopTeams.team_name
            WHERE Year BETWEEN %s AND %s
            GROUP BY Year, team_name
            ORDER BY Year
        """

        with DBConnection() as cursor:
            cursor.execute(query, (start_year, end_year, int(rank), start_year, end_year))
            data = cursor.fetchall()

        # 檢查是否有數據
        if not data:
            logging.warning("沒有可用的比賽數據")
            return

        # 繪製圖表
        plt.figure(figsize=(10, 6))
        schools = set(row["team_name"] for row in data)
        for school in schools:
            school_data = [row for row in data if row["team_name"] == school]
            years = [row["Year"] for row in school_data]
            total_scores = [row["total_score"] for row in school_data]
            plt.plot(years, total_scores, marker='o', label=school)

        plt.xticks(years_to_display, rotation=45)
        plt.legend(title="學校", prop=prop)
        plt.xlabel("年份", fontproperties=prop)
        plt.ylabel("總得分", fontproperties=prop)
        save_and_check_plot(filepath, f"得分趨勢 - {start_year} 至 {end_year} 前 {rank} 名學校")

    except Exception as e:
        logging.error(f"生成得分趨勢圖失敗: {e}")
@app.route('/generate_score_trend_chart')
def generate_score_trend_chart_route():
    """生成得分趨勢圖的 API。"""
    start_year = request.args.get('startYear')
    end_year = request.args.get('endYear')
    rank = request.args.get('rank')
    directory = os.path.join("static", "image", "score_trend")
    filename = f"score_trend_{start_year}_{end_year}_{rank}.png"
    filepath = os.path.join(directory, filename)

    if not os.path.exists(directory):
        os.makedirs(directory)

    generate_score_trend_chart(filepath, start_year, end_year, rank)
    return send_file(filepath, mimetype='image/png')

def get_school_regions_and_ranks_for_years(teams, years):
    """
    查詢學校的地區名稱與多年度的官方排名，並記錄匹配日誌。
    """
    ranks = {year: {} for year in years}  # 初始化每年度的排名字典
    regions = {}
    pairing_logs = []

    try:
        with DBConnection() as cursor:
            # 查詢所有學校的地區
            cursor.execute("""
                SELECT HighSchool_id, HighSchool, County
                FROM highschool
            """)
            school_data = cursor.fetchall()
            school_map = {row["HighSchool"]: row for row in school_data}

            # 查詢指定多個年份的官方排名
            cursor.execute("""
                SELECT o.year, o.HighSchool_id, o.ranking, o.team AS HighSchool
                FROM office_rank o
                WHERE o.year IN (%s)
            """ % ', '.join(['%s'] * len(years)), tuple(years))
            rank_data = cursor.fetchall()

            # 將排名數據組織成字典，方便查詢
            rank_map = {}
            for row in rank_data:
                year = row["year"]
                high_school = row["HighSchool"]
                ranking = row["ranking"]
                if year not in rank_map:
                    rank_map[year] = {}
                rank_map[year][high_school] = ranking

        # 為隊伍匹配地區和排名
        for team in teams:
            if team in school_map:
                school_info = school_map[team]
                region = school_info["County"] or "未知地區"
                regions[team] = region
                for year in years:
                    ranks[year][team] = rank_map.get(year, {}).get(team, "無")
                    pairing_logs.append(f"成功匹配：{team} => 年份: {year}, 地區: {region}, 排名: {ranks[year][team]}")
            else:
                regions[team] = "未知地區"
                for year in years:
                    ranks[year][team] = "無"
                    pairing_logs.append(f"匹配失敗：{team} => 年份: {year}, 無法找到對應的學校信息")

    except mysql.connector.Error as err:
        logging.error(f"查詢地區和排名失敗: {err}")

    for log in pairing_logs:
        logging.info(log)

    return ranks, regions, pairing_logs


def load_all_teams():
    """
    加載所有參加比賽的球隊。
    """
    try:
        with DBConnection() as cursor:
            cursor.execute("""
                SELECT DISTINCT team
                FROM office_rank
            """)
            teams = [row["team"] for row in cursor.fetchall()]
        return teams
    except mysql.connector.Error as err:
        logging.error(f"加載球隊失敗: {err}")
        return []
    
def generate_official_rank_trend_chart(filepath, start_year, end_year, rank):
    """
    生成官方排名變化圖，支持自定義起始和結束年份。
    """
    years_to_display = list(range(start_year, end_year + 1))
    if not years_to_display:
        logging.warning("無效的年份範圍參數")
        return

    try:
        # 獲取所有參賽隊伍
        teams = load_all_teams()
        print(f"Teams: {teams}")  # 調適打印球隊數據

        # 獲取多年度的學校地區和排名數據
        ranks, regions, logs = get_school_regions_and_ranks_for_years(teams, years_to_display)
        print(f"Ranks: {ranks}")  # 調適打印排名數據

        if not ranks:
            logging.warning("無可用數據生成官方排名變化圖")
            return

        # 繪製圖表
        plt.figure(figsize=(10, 6))
        for team in teams:
            # 過濾掉沒有排名或者排名超出前 'rank' 名的學校
            rankings = [
                ranks[year].get(team) for year in years_to_display
                if ranks[year].get(team) != "無" and ranks[year].get(team) <= rank
            ]
            # 只保留有排名的年份
            years = [
                year for year in years_to_display
                if ranks[year].get(team) != "無" and ranks[year].get(team) <= rank
            ]

            print(f"Team: {team}, Rankings: {rankings}, Years: {years}")  # 調適打印每個球隊的數據

            # 繪製每支球隊的排名變化曲線
            if rankings:
                plt.plot(years, rankings, marker='o', label=team)

        plt.xticks(years_to_display, rotation=45)
        plt.ylim(rank + 1, 0)  # 排名越高數值越小
        plt.legend(title="學校")
        plt.xlabel("年份")
        plt.ylabel("排名")
        plt.title(f"官方排名變化圖 - {start_year} 到 {end_year} 前 {rank} 名學校")

        plt.savefig(filepath)
        plt.close()

    except Exception as e:
        logging.error(f"生成官方排名變化圖失敗: {e}", exc_info=True)

# 路由處理函數
@app.route('/generate_official_rank_trend_chart')
def generate_official_rank_trend_chart_route():
    """
    生成官方排名變化圖的 API，支持自定義年份範圍。
    """
    start_year = request.args.get('startYear')
    end_year = request.args.get('endYear')
    
    try:
        rank = int(request.args.get('rank', 10))  # 默認前 10 名
    except ValueError:
        return jsonify({"error": "無效的排名參數，請提供整數值"}), 400

    if not (start_year.isdigit() and end_year.isdigit()):
        return jsonify({"error": "無效的年份參數，請提供有效的年份"}), 400

    start_year = int(start_year)
    end_year = int(end_year)

    directory = os.path.join("static", "image", "official_rank")
    filename = f"official_rank_trend_{start_year}_{end_year}_{rank}.png"
    filepath = os.path.join(directory, filename)

    if not os.path.exists(directory):
        os.makedirs(directory)

    try:
        # 調用生成圖表的核心函數
        generate_official_rank_trend_chart(filepath, start_year, end_year, rank)
        return send_file(filepath, mimetype='image/png')
    except Exception as e:
        logging.error(f"生成官方排名變化圖失敗: {e}", exc_info=True)
        return jsonify({"error": "生成圖表失敗，請稍後再試"}), 500


def calculate_county_scores(start_year, end_year):
    """計算指定年份範圍內的縣市總得分"""
    query = """
    SELECT h.County, SUM(m.PTS_Home + m.PTS_Visit) as total_score
    FROM match_data m
    JOIN highschool h ON (m.Visitor_id = h.HighSchool_id OR m.Home_id = h.HighSchool_id)
    WHERE m.Year BETWEEN %s AND %s
    GROUP BY h.County
    """
    try:
        with DBConnection() as cursor:
            cursor.execute(query, (start_year, end_year))
            result = cursor.fetchall()
            print(f"Query Result: {result}")  # 打印數據庫查詢結果
        return pd.DataFrame(result)
    except Exception as e:
        logging.error(f"計算縣市得分失敗: {e}")
        return pd.DataFrame()

def generate_heatmap_chart(filepath, start_year, end_year):
    """生成台灣縣市總得分熱區圖，支持自定義年份範圍"""
    try:
        county_scores = calculate_county_scores(start_year, end_year)
        print(f"County Scores DataFrame:\n{county_scores}")  # 打印 DataFrame 數據
        if county_scores.empty:
            logging.warning("無可用數據生成熱區圖")
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'Error: No data available', fontsize=20, ha='center')
            plt.savefig(filepath)
            plt.close()
            return

        geojson_path = 'static/geojson/twCounty2010.geo.json'
        gdf = gpd.read_file(geojson_path)
        print(f"GeoJSON Data:\n{gdf.head()}")  # 打印 GeoJSON 文件的前幾行數據

        # 確保合併資料格式一致
        county_scores['County'] = county_scores['County'].str.strip()  # 去除縣市名稱空格
        gdf = gdf.merge(county_scores, left_on='COUNTYNAME', right_on='County', how='left')
        print(f"Merged GeoDataFrame:\n{gdf[['COUNTYNAME', 'total_score']].head()}")  # 打印合併後的關鍵數據

        if gdf['total_score'].isnull().all():
            logging.warning("無匹配數據生成熱區圖")
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'Error: No matching data', fontsize=20, ha='center')
            plt.savefig(filepath)
            plt.close()
            return

        # 繪製熱區圖
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        gdf.plot(column='total_score', cmap='OrRd', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
        ax.set_title(f"台灣縣市 {start_year}-{end_year} 總得分熱區圖")
        plt.savefig(filepath)
        plt.close()
    except Exception as e:
        logging.error(f"生成熱區圖失敗: {e}")

@app.route('/generate_region_heatmap_chart')
def generate_region_heatmap_chart_route():
    """生成台灣縣市總得分熱區圖的 API 路由，支持年份範圍"""
    start_year = request.args.get('startYear', type=int)
    end_year = request.args.get('endYear', type=int)

    # 驗證年份範圍
    if not start_year or not end_year or start_year > end_year:
        return jsonify({"error": "請提供有效的起始年份和結束年份"}), 400

    directory = os.path.join("static", "image", "heatmap")
    filename = f"region_heatmap_chart_{start_year}_{end_year}.png"
    filepath = os.path.join(directory, filename)

    if not os.path.exists(directory):
        os.makedirs(directory)

    generate_heatmap_chart(filepath, start_year, end_year)
    return send_file(filepath, mimetype='image/png')


def generate_school_count_trend_chart(filepath, start_year, end_year):
    """
    生成學校數量趨勢圖
    """
    try:
        # 改進的 SQL 查詢，統計主場和客場學校數量
        query = """
            SELECT Year, COUNT(DISTINCT School) AS school_count
            FROM (
                SELECT Year, Home AS School FROM match_data
                UNION
                SELECT Year, Visitor AS School FROM match_data
            ) AS combined
            WHERE Year BETWEEN %s AND %s
            GROUP BY Year
            ORDER BY Year
        """
        with DBConnection() as cursor:
            cursor.execute(query, (start_year, end_year))
            data = cursor.fetchall()

        # 調試：打印查詢結果
        print(f"Query Result: {data}")

        if not data:
            logging.warning(f"無法獲取年份範圍 {start_year}-{end_year} 的學校數量數據")
            return

        # 提取年份和學校數量數據
        years = [row["Year"] for row in data]
        school_counts = [row["school_count"] for row in data]

        # 調試：打印年份和學校數量
        print(f"Years: {years}")
        print(f"School Counts: {school_counts}")

        # 繪製學校數量趨勢圖
        plt.figure(figsize=(10, 6))
        plt.plot(years, school_counts, marker='o', label="學校數量")
        plt.xticks(years, rotation=45)
        plt.xlabel("年份", fontproperties=prop)
        plt.ylabel("學校數量", fontproperties=prop)
        plt.title(f"學校數量趨勢 - {start_year} 至 {end_year}")
        plt.legend()
        plt.savefig(filepath)
        plt.close()

    except Exception as e:
        logging.error(f"生成學校數量趨勢圖失敗: {e}")

@app.route('/generate_school_count_trend_chart')
def generate_school_count_trend_chart_route():
    """
    生成學校數量趨勢圖的 API 路由
    """
    start_year = request.args.get('startYear', type=int)
    end_year = request.args.get('endYear', type=int)

    # 驗證年份參數
    if not start_year or not end_year or start_year > end_year:
        return jsonify({"error": "請提供有效的起始年份和結束年份"}), 400

    directory = os.path.join("static", "image", "school_count")
    filename = f"school_count_trend_{start_year}_{end_year}.png"
    filepath = os.path.join(directory, filename)

    if not os.path.exists(directory):
        os.makedirs(directory)

    generate_school_count_trend_chart(filepath, start_year, end_year)
    return send_file(filepath, mimetype='image/png')


def load_team_specific_data(start_year, end_year, team_name=None):
    """
    加載指定年份範圍內特定隊伍的比賽數據。
    """
    query = """
        SELECT m.*, 
               h1.HighSchool AS Home_School_Name, 
               h2.HighSchool AS Visitor_School_Name,
               h1.County AS Home_Region, 
               h2.County AS Visitor_Region
        FROM match_data m
        JOIN highschool h1 ON m.Home_id = h1.HighSchool_id
        JOIN highschool h2 ON m.Visitor_id = h2.HighSchool_id
        WHERE m.Year BETWEEN %s AND %s
    """
    params = [start_year, end_year]

    if team_name:
        query += """
            AND (h1.HighSchool = %s OR h2.HighSchool = %s)
        """
        params.extend([team_name, team_name])

    try:
        logging.info(f"執行 SQL 查詢，起始年份: {start_year}, 結束年份: {end_year}, 隊伍名稱: {team_name}")
        with DBConnection() as cursor:
            cursor.execute(query, params)
            result = cursor.fetchall()
            logging.info(f"查詢結果數量: {len(result)}")
            return pd.DataFrame(result)
    except Exception as e:
        logging.error(f"加載特定隊伍比賽數據失敗: {e}")
        return pd.DataFrame()

def validate_params(required_params):
    """
    檢查必要參數是否缺失或無效。

    Args:
        required_params (dict): 必要參數的名稱和值。

    Returns:
        Response 或 None: 如果參數缺失，返回錯誤響應；否則返回 None。
    """
    missing_or_invalid = [key for key, value in required_params.items() if not value]
    if missing_or_invalid:
        error_message = f"缺失或無效的參數: {', '.join(missing_or_invalid)}"
        logging.error(error_message)
        return jsonify({"error": error_message}), 400
    return None

def build_graph(games):
    """
    從比賽數據構建節點圖。
    """
    nodes = {}
    for _, game in games.iterrows():
        loser = game['Visitor_School_Name'] if game['PTS_Home'] > game['PTS_Visit'] else game['Home_School_Name']
        winner = game['Home_School_Name'] if game['PTS_Home'] > game['PTS_Visit'] else game['Visitor_School_Name']
        point_diff = abs(game['PTS_Home'] - game['PTS_Visit'])

        if loser not in nodes:
            nodes[loser] = Node(loser)
            nodes[loser].region = game['Visitor_Region']
        if winner not in nodes:
            nodes[winner] = Node(winner)
            nodes[winner].region = game['Home_Region']

        nodes[loser].add_game(winner, point_diff, game['Year'], is_win=False)
        nodes[winner].add_game(loser, point_diff, game['Year'], is_win=True)

    logging.info(f"構建節點圖完成，共有 {len(nodes)} 個節點")
    return nodes
def build_adjacency_matrix(nodes):
    """
    構建比賽的鄰接矩陣。
    """
    teams = list(nodes.keys())
    team_index = {team: idx for idx, team in enumerate(teams)}
    num_teams = len(teams)

    adjacency_matrix = np.zeros((num_teams, num_teams))
    for team, node in nodes.items():
        team_idx = team_index[team]
        for opponent, weight in node.losses.items():
            opponent_idx = team_index[opponent]
            adjacency_matrix[team_idx, opponent_idx] = weight

    logging.info("成功構建鄰接矩陣")
    return adjacency_matrix, team_index
def build_edges(nodes):
    """
    根據節點數據構建比賽網絡中的邊，從輸球隊指向贏球隊，權重為分差。
    
    Args:
        nodes (dict): 節點數據，每個節點是 Node 類型的對象。
    
    Returns:
        list: 邊列表，格式為 [(loser, winner, {"weight": point_diff}), ...]
    """
    edges = []
    for loser, details in nodes.items():
        if hasattr(details, "losses"):  # 確保有 losses 屬性
            for winner, point_diff in details.losses.items():
                edges.append((loser, winner, {"weight": point_diff}))  # 從輸球隊指向贏球隊
    return edges


@app.route('/generate_team_pr_relationship_chart')
def generate_team_pr_relationship_chart_route():
    try:
        # 收集參數
        start_year = request.args.get('startYear')
        end_year = request.args.get('endYear')
        rank = request.args.get('rank')

        # 驗證參數
        if not start_year or not end_year or not rank:
            return jsonify({"error": "缺失必要參數: startYear, endYear 或 rank"}), 400

        # 轉換參數類型
        start_year, end_year, rank = int(start_year), int(end_year), int(rank)
        logging.info(f"處理請求: startYear={start_year}, endYear={end_year}, rank={rank}")

        # 加載比賽數據
        games = load_team_specific_data(start_year, end_year)
        if games.empty:
            logging.warning(f"在年份範圍 {start_year}-{end_year} 內無數據")
            return jsonify({"error": "無數據生成關係圖"}), 404

        # 構建節點圖並計算 PR 值
        nodes = build_graph(games)
        adjacency_matrix, team_index = build_adjacency_matrix(nodes)
        pr_scores = pageRank(adjacency_matrix)

        # 匹配學校名稱和 PR 值
        team_names = list(team_index.keys())
        pr_data = pd.DataFrame({"Team": team_names, "PR_Score": pr_scores})
        pr_data = pr_data.sort_values(by="PR_Score", ascending=False).head(rank)

        # 構建關係網絡
        data_network = nx.DiGraph()
        edges = build_edges(nodes)
        data_network.add_edges_from(edges)

        # 將 PR 值添加到節點屬性中
        for _, row in pr_data.iterrows():
            team = row["Team"]
            pr_score = row["PR_Score"]
            if team in data_network.nodes:
                data_network.nodes[team]["score"] = pr_score

        # 計算節點大小和顏色
        node_sizes = []
        node_colors = []

        scores = [data.get("score", 0) for _, data in data_network.nodes(data=True)]

        max_score = max(scores) if scores else 0
        min_score = min(scores) if scores else 0

        for node, data in data_network.nodes(data=True):
            if "score" in data:
                node_sizes.append(data["score"] * 5000)
                normalized_score = (data["score"] - min_score) / (max_score - min_score) if max_score != min_score else 0
                node_colors.append(normalized_score)
            else:
                node_sizes.append(100)  # 默認節點大小
                node_colors.append(0.5)  # 默認顏色

        # 設定字體的路徑
        plt.rcParams["font.family"] = "Taipei Sans TC Beta"
        plt.rcParams["font.sans-serif"] = ["Taipei Sans TC Beta"]
        plt.rcParams["axes.unicode_minus"] = False  # 正常顯示負號

        # 設定節點標籤
        labels = {node: f"{node}\n{data_network.nodes[node].get('score', 0):.3f}" for node in data_network.nodes()}

        # 繪製圖形
        plt.figure(figsize=(20, 20))
        pos = nx.spring_layout(data_network, k=0.8)  # 使用 spring layout 布局
        nx.draw(
            data_network,
            pos,
            with_labels=False,
            node_size=node_sizes,
            node_color=node_colors,
            cmap=plt.cm.YlOrRd,
            edge_color="grey",
            arrowsize=10,
        )

        # 添加標籤位置
        label_pos = {k: (v[0], v[1] + 0.01) for k, v in pos.items()}
        nx.draw_networkx_labels(data_network, label_pos, labels, font_size=10)

        # 保存圖片
        directory = os.path.join("static", "image", "team_pr")
        if not os.path.exists(directory):
            os.makedirs(directory)

        filepath = os.path.join(directory, f"team_pr_relationship_chart_{start_year}_{end_year}_top{rank}.png")
        plt.savefig(filepath)
        plt.close()

        return send_file(filepath, mimetype="image/png")

    except Exception as e:
        logging.error(f"生成球隊 PR 值關係圖失敗: {e}", exc_info=True)
        return jsonify({"error": "內部服務器錯誤"}), 500



@app.route('/generate_pr_value_chart')
def generate_pr_value_chart_route():
    try:
        # 收集參數
        start_year = request.args.get("startYear")
        end_year = request.args.get("endYear")
        rank = request.args.get("rank")

        # 驗證參數
        if not start_year or not end_year or not rank:
            return jsonify({"error": "缺失必要參數: startYear, endYear 或 rank"}), 400

        # 轉換參數類型
        start_year, end_year, rank = int(start_year), int(end_year), int(rank)
        logging.info(f"處理請求: startYear={start_year}, endYear={end_year}, rank={rank}")

        # 設置圖表保存路徑
        directory = os.path.join("static", "image", "pr_value")
        filename = f"pr_value_chart_{start_year}_{end_year}_top{rank}.png"
        filepath = os.path.join(directory, filename)

        if not os.path.exists(directory):
            os.makedirs(directory)

        # 初始化 PR 值數據
        pr_data_all_years = pd.DataFrame()

        # 遍歷年份範圍，逐年計算 PR 值
        for year in range(start_year, end_year + 1):
            games = load_team_specific_data(year, year)
            if games.empty:
                logging.warning(f"年份 {year} 無數據，跳過")
                continue

            # 構建節點圖並計算 PR 值
            nodes = build_graph(games)
            adjacency_matrix, team_index = build_adjacency_matrix(nodes)
            pr_scores = pageRank(adjacency_matrix)

            # 匹配學校名稱和 PR 值
            team_names = list(team_index.keys())
            pr_data_year = pd.DataFrame({"Team": team_names, "PR_Score": pr_scores, "Year": year})
            pr_data_all_years = pd.concat([pr_data_all_years, pr_data_year], ignore_index=True)

        if pr_data_all_years.empty:
            logging.warning(f"在年份範圍 {start_year}-{end_year} 內無數據")
            return jsonify({"error": "無數據生成 PR 值圖表"}), 404

        # 計算 Top N 學校
        top_teams = (
            pr_data_all_years.groupby("Team")["PR_Score"]
            .mean()
            .sort_values(ascending=False)
            .head(rank)
            .index
        )

        # 繪製 PR 值變化折線圖
        plt.figure(figsize=(10, 6))

        for team in top_teams:
            team_data = pr_data_all_years[pr_data_all_years["Team"] == team]
            plt.plot(
                team_data["Year"],
                team_data["PR_Score"],
                marker="o",
                label=team
            )

        # 設定圖表標題和軸標籤
        plt.xlabel("年份", fontsize=12)
        plt.ylabel("PR 值", fontsize=12)
        plt.title(f"Top {rank} 學校 PR 值變化 ({start_year}-{end_year})", fontsize=14)

        # 添加圖例
        plt.legend(title="學校名稱", fontsize=10)

        # 顯示網格
        plt.grid(alpha=0.5)

        # 保存折線圖
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()

        return send_file(filepath, mimetype='image/png')

    except Exception as e:
        logging.error(f"生成圖表失敗: {e}", exc_info=True)
        return jsonify({"error": "內部服務器錯誤"}), 500
    


@app.route('/visual')
def visual():
    return render_template('visual.html')
@app.route('/')
def home():
    return render_template('home.html')  # Render the homepage template
@app.route('/data')
def data():
    return render_template('data.html')

@app.route('/school')
def school():
    return render_template('school.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)