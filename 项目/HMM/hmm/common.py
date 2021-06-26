# -- encoding:utf-8 --
"""
Create by ibf on 2018/10/18
"""


def convert_obs_seq_2_index(Q, index=None):
    """
    根据传入的黑白文字序列转换为对应的索引值，如果是黑转换为1.如果是白转换为0.
    :param Q:
    :param index:
    :return:
    """
    if index is not None:
        cht = Q[index]
        if cht == '黑':
            return 1
        else:
            return 0
    else:
        result = []
        for q in Q:
            if q == '黑':
                result.append(1)
            else:
                result.append(0)
        return result
