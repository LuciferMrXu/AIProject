import re
contents='句:琴曲唯留古，书多半是经。（见《周氏涉笔》）横裁桑节杖，直剪竹皮巾。鹤警琴亭夜，莺啼酒瓮春。颜回唯乐道，原宪岂伤贫。（被召谢病，见《西清诗话》）寄身千载下，聊游万物初。欲令无作有，翻觉实成虚。（《独坐》）双关防易断，只眼畏难全。鱼鳞张九拒，鹤翅拥三边。（《围棋长篇》。见《韵语阳秋》）'
re_complile=re.compile('(（.*?）)')
contents=re.sub(re_complile,'',contents)
print(contents)