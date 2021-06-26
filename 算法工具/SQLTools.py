import requests
from bs4 import BeautifulSoup
from sqlalchemy import create_engine,Column,String,Integer,Float
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from icecream import ic

# 初始化数据库连接
engine = create_engine('mysql+pymysql://root:1qa2ws3ed@localhost:3306/flask')

# 创建DBSession类型
DBSession = sessionmaker(bind=engine)
# 创建session对象
session = DBSession()

# 通过url返回BS对象
def get_bs_object(url):
    # ic(url)
    html = requests.get(url,timeout=10)
    content = html.text
    soup = BeautifulSoup(content,'html.parser',from_encoding='utf8')
    # ic(soup)
    return soup

# 数据处理
def analyze_score(spans):
    spanList = spans.find_all('span')
    z_value = spanList[0].text
    attr = spanList[1].text
    k_value = spanList[2].text
    return z_value,attr,k_value

def add_data(hero_name,game_id,player_name,kda_k,kda_d,kda_a,money,damage_output,damage_input,winner):
    # 原生sql模式
    insert_stmt = 'INSERT IGNORE INTO hero_play(hero_name,game_id,player_name,kda_k,kda_d,kda_a,money,damage_output,damage_input,winner) VALUES (:hero_name,:game_id,:player_name,:kda_k,:kda_d,:kda_a,:money,:damage_output,:damage_input,:winner)'
    session.execute(insert_stmt,{
        'hero_name':hero_name,
        'game_id':game_id,
        'player_name':player_name,
        'kda_k':int(kda_k),
        'kda_d':int(kda_d),
        'kda_a':int(kda_a),
        'money':int(money),
        'damage_output':int(damage_output),
        'damage_input':int(damage_input),
        'winner':winner})

# 请求url
start_page = 65200
end_page = 65324
for game_id in range(start_page,end_page+1):
    url = f'https://www.wanplus.com/match/{game_id}.html#data'
    soup = get_bs_object(url)
    # 获取比赛类型
    try:
        game = soup.find('div', class_='matching_intro').text
    except:
        continue
    if 'KPL' not in game:
        continue
    z_list = soup.find_all('div',class_='bans_l')
    score_list = soup.find_all('div',class_='bans_m')
    k_list = soup.find_all('div',class_='bans_r')
    for z_hero,score,k_hero in zip(z_list,score_list,k_list):
        # 赢家
        k_team = soup.find('span',class_='tr bssj_tt3').text
        if '胜' in k_team:
            winner = 0
        else:
            winner = 1
        # 主队名字
        temp_z = z_hero.find('div',class_='bans_tx fl').find_all('a',limit=2)
        z_player_name = temp_z[0].text
        z_hero_name = temp_z[1].text
        # 客队名字
        temp_k = k_hero.find('div',class_='bans_tx fl').find_all('a',limit=2)
        k_player_name = temp_k[0].text
        k_hero_name = temp_k[1].text
        # 成绩
        temp = score.find_all('li')
        # ic(temp_z)
        # ic(temp_k)
        # ic(temp)
        # KDA
        z_value,attr,k_value = analyze_score(temp[0])
        [z_k,z_d,z_a] = z_value.split('/')
        [k_k,k_d,k_a] = k_value.split('/')
        # 金钱
        z_money,attr,k_money = analyze_score(temp[1])
        # 输出
        z_damage_output,attr,k_damage_output = analyze_score(temp[2])
        # 承受伤害
        z_damage_input,attr,k_damage_input = analyze_score(temp[3])
        # 添加到数据库
        # ic(z_hero_name,game_id,z_player_name,z_k,z_d,z_a,z_money,z_damage_output,z_damage_input,winner)
        add_data(z_hero_name,game_id,z_player_name,z_k,z_d,z_a,z_money,z_damage_output,z_damage_input,winner)
        add_data(k_hero_name,game_id,k_player_name,k_k,k_d,k_a,k_money,k_damage_output,k_damage_input,1-winner)
session.commit()
session.close()
