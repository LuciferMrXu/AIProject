#-*- conding:utf-8 -*-
import requests
import json
import time
import re
from pymongo import MongoClient
client = MongoClient('localhost', 27017)
db = client.mydb
collection=db.歌单信息


startUrl ='https://c.y.qq.com/soso/fcgi-bin/client_search_cp?ct=24&qqmusic_ver=1298&new_json=1&remoteplace=txt.yqq.center&searchid=40031448914336763&t=0&aggr=1&cr=1&catZhida=1&lossless=0&flag_qc=0&p={1}&n=20&w={0}&g_tk=5381&jsonpCallback=MusicJsonCallback1559302090762067&loginUin=0&hostUin=0&format=jsonp&inCharset=utf8&outCharset=utf-8&notice=0&platform=yqq&needNewCode=0'
headers = {
    'referer' : 'https://y.qq.com/portal/search.html',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
}

def download(word):
    n=1
    p=1
    m=20
    while True:
        response1 = requests.get(startUrl.format(word,p),headers=headers).text
        response1 = json.loads(response1.strip('MusicJsonCallback1559302090762067()'))

        for j in response1['data']['song']['list']:
            songname=j['name']
            singer=j['singer'][0]['name']
            subtitle=j['subtitle']
            album=j['album']['name']
            time=j['time_public']
            lyid = j['id']


            vid=j['mv']['vid']

            songmid=j['mid']
            strMediaMid=j['file']['strMediaMid']
            filename = 'C400' + strMediaMid + '.m4a'

            lyricUrl='https://c.y.qq.com/lyric/fcgi-bin/fcg_query_lyric.fcg?nobase64=1&musicid={0}&callback=jsonp1&g_tk=5381&jsonpCallback=jsonp1&loginUin=0&hostUin=0&format=jsonp&inCharset=utf8&outCharset=utf-8&notice=0&platform=yqq&needNewCode=0'
            headers['referer'] = 'https://y.qq.com/n/yqq/song/{0}.html'.format(songmid)
            response2 = requests.get(lyricUrl.format(lyid), headers=headers).text
            response2 = json.loads(response2.strip('jsonp1()'))
            try:
                lyc=response2['lyric']
                pattern = re.compile('[\u4e00-\u9fa5]+|[\u30a0-\u30ff\u3040-\u309f\u30fc-\u30ff\u4e00-\u9faf\u3400-\u4dbf]+|[a-zA-Z]+')   # 根据个人爱好只做了英文、中文和日文的正则，其他语言正则的utf-8编码可自行百度添加
                seq = re.findall(pattern, lyc)
                lyc=' '.join(seq)
            except:
                lyc='暂无歌词'

            data={
                '序号':n,'专辑名':album,'歌名':songname,'歌手':singer,'发行时间':time,'介绍':subtitle,'歌词':lyc
            }
            print(data)
            collection.insert_one(data)

            VkeyUrl ='https://u.y.qq.com/cgi-bin/musicu.fcg?callback=getplaysongvkey5836548152671817&g_tk=5381&jsonpCallback=getplaysongvkey9748072731754585&loginUin=0&hostUin=0&format=jsonp&inCharset=utf8&outCharset=utf-8&notice=0&platform=yqq&needNewCode=0&data='
            data={"req":{"module":"CDN.SrfCdnDispatchServer","method":"GetCdnDispatch","param":{"guid":"1205259272","calltype":0,"userip":""}},"req_0":{"module":"vkey.GetVkeyServer","method":"CgiGetVkey","param":{"guid":"1205259272","songmid":["0018pn540iWgm0"],"songtype":[0],"uin":"0","loginflag":1,"platform":"20"}},"comm":{"uin":0,"format":"json","ct":20,"cv":0}}
            # data={                                                                                                                             "req_0":{"module":"vkey.GetVkeyServer","method":"CgiGetVkey","param":{"guid":"1205259272","songmid":["0010pp8O428q7Y"],"songtype":[0],"uin":"0","loginflag":1,"platform":"20"}},"comm":{"uin":0,"format":"json","ct":20,"cv":0}}
            # data={                                                                                                                             "req_0":{"module":"vkey.GetVkeyServer","method":"CgiGetVkey","param":{"guid":"1205259272","songmid":["003jjoM94WLiTf"],"songtype":[0],"uin":"0","loginflag":1,"platform":"20"}},"comm":{"uin":0,"format":"json","ct":20,"cv":0}}
            # data={                                                                                                                             "req_0":{"module":"vkey.GetVkeyServer","method":"CgiGetVkey","param":{"guid":"1205259272","songmid":["002E3MtF0IAMMY"],"songtype":[0],"uin":"0","loginflag":1,"platform":"20"}},"comm":{"uin":0,"format":"json","ct":20,"cv":0}}
            data["req_0"]["param"]["songmid"]=[songmid]
            a = json.dumps(data)
            headers['referer'] ='https://y.qq.com/portal/player.html'
            response3=requests.get(VkeyUrl+a, headers=headers).text
            response3=json.loads(response3.strip('getplaysongvkey5836548152671817()'))

            vkey= response3['req_0']['data']['midurlinfo'][0]['vkey']

            MusicUrl = 'http://isure.stream.qqmusic.qq.com/{0}?guid=1205259272&vkey={1}&uin=0&fromtag=66'
            del headers['referer']
            Music = requests.get(MusicUrl.format(filename, vkey), headers=headers, stream=True)

            try:
                with open('QQmusic/' + singer+'的'+songname + '.mp3', 'wb+') as file:
                    for music in Music.iter_content(1024*100):
                        file.write(music)
            except Exception as e:
                print(e)
            else:
                print('已下载第{0}首歌，歌手{1}的：{2}'.format(n,singer,songname))
                n += 1
            if vid!='':
                MVvkeyurl='https://u.y.qq.com/cgi-bin/musicu.fcg?data={0}&g_tk=5381&callback=jQuery112306832906624967148_1543156554450&loginUin=0&hostUin=0&format=jsonp&inCharset=utf8&outCharset=GB2312&notice=0&platform=yqq&needNewCode=0'
                data={"getMvUrl": {"module": "gosrf.Stream.MvUrlProxy", "method": "GetMvUrls",
                                    "param": {"vids": ["c00200si22o"], "request_typet": 10001}}}
                data['getMvUrl']['param']['vids']=[vid]
                b = json.dumps(data)
                headers['referer'] = 'https://y.qq.com/n/yqq/mv/v/{0}.html'.format(vid)
                response4 = requests.get(MVvkeyurl.format(b), headers=headers).text
                response4 = json.loads(response4.strip('jQuery112306832906624967148_1543156554450()'))

                for vk in response4['getMvUrl']['data'][vid]['mp4']:
                    MVvkey = vk['vkey']
                    cn=vk['cn']
                    grade=vk['filetype']
                    MVUrl = 'http://112.29.199.143/vcloud1049.tc.qq.com/{0}?vkey={1}'

                    MV = requests.get(MVUrl.format(cn, MVvkey), headers=headers, stream=True)
                    try:
                        with open('QQmusic/' + singer + '的' + songname+'清晰度'+str(grade)+'.mp4', 'wb+') as file:
                            for mv in MV.iter_content(1024 * 100):
                                file.write(mv)
                    except Exception as e:
                        print(e)
                    else:
                        print('已下载歌曲{0}的MV，清晰度为{1}'.format(songname,grade))

        if m < response1['data']['song']['totalnum']:
            p += 1
            m+=20
        else:
            break

if __name__ == '__main__':
    word=input('请输入关键词：')
    download(word)