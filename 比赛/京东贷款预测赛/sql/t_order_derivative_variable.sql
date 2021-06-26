-- 1. 创建基础衍生表结构
create table t_order_full(
uid INT(10), -- 用户id
buy_time DATE, -- 购买时间
price DECIMAL(50,30), -- 购买价格
qty INT(10), -- 购买数量
cate_id INT(10), -- 品类id
discount DECIMAL(50,30), -- 折扣金额
bllamt DECIMAL(50,30), -- 交易金额
ratio_discount DECIMAL(50,30) -- 折扣占比
);

-- 2. 插入基础衍生数据
insert into t_order_full
  select
    *, round(price*qty - discount, 10) as bllamt, round(discount/qty/price, 10) as ratio_discount
  from t_order;


--3. 创建额外的扩展特征表
CREATE TABLE t_order_feature
(
uid INT(10), -- 用户id
Velocity_3mon_bllamt_per_uid DOUBLE, -- 用户A三个月内的购买总额
Velocity_3mon_buynum_per_uid INT, -- 用户A三个月内的购买次数
Velocity_3mon_bllamtAvg_per_uid DOUBLE, -- 用户A三个月内的购买均值1
Velocity_3mon_qtyMax_per_uid DOUBLE, -- 用户A三个月内的购买max pty
Velocity_3mon_qtyMin_per_uid DOUBLE, -- 用户A三个月内的购买min pty
Velocity_3mon_qty_per_uid INT, -- 用户A三个月内的购买数量
Velocity_3mon_qtyAvg_per_uid INT, -- 用户A三个月内的购买均值2
Velocity_3mon_discountNum_per_uid INT, -- 用户A三个月内使用的discount的次数
Velocity_3mon_ratio_discount_per_uid INT, -- 用户A三个月内使用的discount的占比
Velocity_3mon_cateId_per_uid INT, -- 用户A三个月内购买的产品类别数
Velocity_3mon_price_per_cateId DOUBLE, -- 用户A三个月内购买单价最贵的产品的花费
Velocity_3mon_qty_per_cateId INT, -- 用户A三个月内购买数量最多的一次花费
Velocity_3mon_priceMaxbllamt_per_price_cateid DOUBLE, -- 用户A三个月内单价最贵的产品的花费占三个月内花费的比
Velocity_3mon_priceMaxqty_per_qty_cateid DOUBLE, -- 用户A三个月内单价最贵的产品的销量占三个月内销量的比
Velocity_3mon_priceMaxCount_per_Count_cateid DOUBLE, -- 用户A三个月内单价最贵的产品的次数占该产品被购买次数的比
Velocity_3mon_priceMaxCount_per_cateId INT, -- 用户A对单价最贵的产品的购买次数
Velocity_eved_BuyBetweenNow_per_cateId INT -- 用户各次购买离现在的天数的平均
);

-- 4. 插入衍生数据
insert into t_order_feature
	select
		t1_uid as uid,
		o1 AS  Velocity_3mon_bllamt_per_uid,
		o2 AS  Velocity_3mon_buynum_per_uid,
		o3 AS Velocity_3mon_bllamtAvg_per_uid,
		o4 AS Velocity_3mon_qtyMax_per_uid,
		o5 AS Velocity_3mon_qtyMin_per_uid,
		o6 AS Velocity_3mon_qty_per_uid,
		o7 AS Velocity_3mon_qtyAvg_per_uid,
		o8 AS Velocity_3mon_discountNum_per_uid,
		o9 AS Velocity_3mon_ratio_discount_per_uid,
		o10 AS Velocity_3mon_cateId_per_uid,
		o11 AS Velocity_3mon_price_per_cateId,
		o12 AS Velocity_3mon_qty_per_cateId,
		Velocity_3mon_priceMaxbllamt_per_price_cateid,
		Velocity_3mon_priceMaxqty_per_qty_cateid,
		Velocity_3mon_priceMaxCount_per_Count_cateid,
		Velocity_3mon_priceMaxCount_per_cateId,
		Velocity_eved_BuyBetweenNow_per_cateId
	from
		(
			SELECT
				uid as t1_uid,
				SUM(bllamt) as o1,COUNT(*) as o2,AVG(bllamt) as o3,
				MAX(bllamt) as o4,MIN(bllamt) as o5,SUM(qty) as o6,
				AVG(qty) as o7,COUNT(IF (discount = 0,NULL,1)) as o8,
				COUNT(IF (discount = 0,NULL,1))/COUNT(*) as o9,
				COUNT(DISTINCT cate_id) as o10,
				MAX(price) as o11,MAX(qty) as o12
			from t_order_full
			where buy_time >= '2016-08-01' and buy_time < '2016-11-01'
			GROUP BY uid
		) t1
		join
		(
			SELECT
				uid as t2_uid,
				SUM(IF(price = price_max,bllamt,0))/SUM(bllamt) AS Velocity_3mon_priceMaxbllamt_per_price_cateid,
				SUM(IF(price = price_max,qty,0))/count_qty AS Velocity_3mon_priceMaxqty_per_qty_cateid,
				COUNT(IF(price = price_max,1,NULL))/count_num AS Velocity_3mon_priceMaxCount_per_Count_cateid,
				COUNT(IF(price = price_max,1,NULL)) AS  Velocity_3mon_priceMaxCount_per_cateId
			FROM
				(
					SELECT
						a.uid,a.cate_id,count_qty,qty,count_num,price_max,a.price,a.bllamt
					FROM
						`t_order_full` a
					 	JOIN (
							SELECT
								cate_id,SUM(qty) AS count_qty,COUNT(*) AS count_num,price
							FROM `t_order_full`
							where buy_time >= '2016-08-01' and buy_time < '2016-11-01'
							GROUP BY cate_id
						) b ON a.cate_id = b.cate_id
						JOIN (
							SELECT
								uid AS c_uid,MAX(price) AS price_max,price
							FROM `t_order_full`
							where buy_time >= '2016-08-01' and buy_time < '2016-11-01'
							GROUP BY uid
						) c ON a.uid = c_uid
					where buy_time >= '2016-08-01' and buy_time < '2016-11-01'
				) d
			GROUP BY uid
		) t2
		on t1.t1_uid = t2.t2_uid
		join
		(
			SELECT
				uid as t3_uid,
				AVG(Velocity_eved_BuyBetweenNow_per_cateId) AS Velocity_eved_BuyBetweenNow_per_cateId
			FROM
				(
					SELECT
						*,
						TO_DAYS(DATE_FORMAT('2016-11-30','%Y-%m-%d'))-TO_DAYS(buy_time) AS  Velocity_eved_BuyBetweenNow_per_cateId
					FROM `t_order_full`
					where buy_time >= '2016-08-01' and buy_time < '2016-11-01'
				) a
			GROUP BY uid
		) t3
		ON t3.t3_uid = t3.t3_uid;
