-- 1. 创建表结构
create table t_user_feature(
uid int(10), -- 用户id
age int(10), -- 用户年龄
sex int(10), -- 用户性别
`limit` double, -- 用户初始贷款额度
User_age_range int(10), -- 用户年龄区间
User_interdate int(10) -- 用户第一次激活时间间隔
);

-- 2. 插入数据
insert into t_user_feature
  select
    uid, age, sex, `limit`,
    case
	    when age < 30 then 1
	    when age < 35 then 2
	    when age < 40 then 3
	    else 4
    end User_age_range,
    to_days(date_format('2016-11-30', '%Y-%m-%d')) - to_days(active_date) as User_interdate
  from t_user;
