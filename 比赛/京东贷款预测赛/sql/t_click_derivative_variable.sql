-- 1. 创建对象的表
create table t_click_feature(
uid int(10), -- 用户id
Velocity_3mon_ClkNum_per_uid int(10), -- 用户三个月内的点击次数
Velocity_3mon_pid_per_uid int(10), -- 用户三个月内点击的页面数
Velocity_3mon_pidParam_per_uid int(10), -- 用户三个月内点击的页面参数的个数
Velocity_LtoN_intervalPerClk_per_uid int(10), -- 用户最近一次点击离现在的天数
Velocity_3mon_uMax_upid_per_uid double, -- 用户点击最多的页面的被该用户点击次数占该用户点击次数的比例
Velocity_3mon_Max_pid_per_uid double, -- 用户点击最多的页面的被点击次数占总被点击次数的比例
Velocity_3mon_uMax_upidParam_per_uid double, -- 用户点击最多的页面参数的被该用户点击次数占该用户点击次数的比例
Velocity_3mon_Max_pidParam_per_uid double, -- 用户点击最多的页面参数的被点击次数占总被点击次数的比例
Velocity_EtoNavg_intervalPerClk_per_uid double -- 用户各次点击离现在的天数的平均
);

-- 2. 插入基本数据
-- 因为现在所有数据均是在三个月内的
insert into t_click_feature
    select t1.uid, c1, c2, c3, c4, c5, c6, c7, c8, c9
    from
        (
            select
                uid, count(uid) as c1, count(distinct(pid)) as c2, count(distinct(param)) as c3,
                to_days(date_format('2016-11-30', '%Y-%m-%d')) - to_days(max(click_time)) as c4,
                avg(to_days(date_format('2016-11-30', '%Y-%m-%d')) - to_days(click_time)) as c9
            from t_click
            where click_time >= '2016-08-01' and click_time < '2016-11-01'
            group by uid
        ) t1
        join
        (
            select
                uid, max(pidc)/sum(pidc) as c5,max(pidc)*1000/pidsum as c6
            from
                (select uid,pid,count(*) pidc from t_click where click_time >= '2016-08-01' and click_time < '2016-11-01' group by uid,pid) a
                join
                (select pid,count(*) pidsum from t_click where click_time >= '2016-08-01' and click_time < '2016-11-01' group by pid) b
                on a.pid = b.pid
            group by uid
        ) t2
        on t1.uid = t2.uid
        join
        (
            select
                uid, max(paramc)/sum(paramc) as c7,max(paramc)*1000/paramsum as c8
            from
                (select uid,param,count(*) paramc from t_click where click_time >= '2016-08-01' and click_time < '2016-11-01' group by uid,param) c
                join
                (select param,count(*) paramsum from t_click where click_time >= '2016-08-01' and click_time < '2016-11-01' group by param) d
                on c.param = d.param
            group by uid
        ) t3
        on t1.uid = t3.uid;