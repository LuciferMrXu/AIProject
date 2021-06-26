-- 1. 创建表
create table t_loan_feature (
    uid int(10),
    Velocity_f_loanAmt_per_uid  double  COMMENT '用户A最近一次贷款的金额',
    Velocity_f_plannum_per_uid  double  COMMENT '用户A最近一次贷款的分期数',
    Velocity_f_amtPerPlan_per_uid double  COMMENT '用户A最近一次贷款的每期金额',
    Velocity_3mon_loanAmt_per_uid double  COMMENT '用户A三个月内总贷款额',
    Velocity_3mon_loanNum_per_uid double  COMMENT '用户A三个月内总贷款次数',
    Velocity_3mon_loanAvg_per_uid double  COMMENT '用户A三个月内贷款均值',
    Velocity_3mon_loanMax_per_uid double  COMMENT '用户A三个月内的贷款max ',
    Velocity_3mon_loanMin_per_uid double  COMMENT '用户A三个月内的贷款min',
    Velocity_3mon_plannum_per_uid double  COMMENT '用户A三个月内总贷款期数',
    Velocity_3mon_planAvg_per_uid double  COMMENT '用户A三个月内贷款期数均值',
    Velocity_3mon_planMax_per_uid double  COMMENT '用户A三个月内的贷款期数的max',
    Velocity_3mon_planMin_per_uid double  COMMENT '用户A三个月内的贷款期数的min',
    Velocity_3mon_amtPerPlanAvg_per_uid double  COMMENT '用户A三个月内每期贷款额均值',
    Velocity_3mon_amtPerPlanMax_per_uid double  COMMENT '用户A三个月内的每期贷款额max',
    Velocity_3mon_amtPerPlanMin_per_uid double  COMMENT '用户A三个月内的每期贷款额min',
    Velocity_days_LtoNow_per_uid  double  COMMENT '用户A最近一次贷款行为离现在的天数',
    Velocity_days_FtoNow_per_uid  double  COMMENT '用户A第一次贷款离现在的天数'
);


-- 2. 插入扩展数据
insert into t_loan_feature(
		uid, Velocity_f_loanAmt_per_uid, Velocity_f_plannum_per_uid,
		Velocity_f_amtPerPlan_per_uid, Velocity_3mon_loanAmt_per_uid,
		Velocity_3mon_loanNum_per_uid, Velocity_3mon_loanAvg_per_uid,
		Velocity_3mon_loanMax_per_uid, Velocity_3mon_loanMin_per_uid,
		Velocity_3mon_plannum_per_uid, Velocity_3mon_planAvg_per_uid,
		Velocity_3mon_planMax_per_uid, Velocity_3mon_planMin_per_uid,
		Velocity_3mon_amtPerPlanAvg_per_uid, Velocity_3mon_amtPerPlanMax_per_uid,
		Velocity_3mon_amtPerPlanMin_per_uid, Velocity_days_LtoNow_per_uid,
		Velocity_days_FtoNow_per_uid
	)
	select
		f.uid,
		f.loan_amount as Velocity_f_loanAmt_per_uid,
		f.plannum as Velocity_f_plannum_per_uid,
		f.loan_amount/f.plannum as Velocity_f_amtPerPlan_per_uid,
		sum_amount as Velocity_3mon_loanAmt_per_uid,
		count_uid as Velocity_3mon_loanNum_per_uid,
		avg_amount as Velocity_3mon_loanAvg_per_uid,
		max_amount as Velocity_3mon_loanMax_per_uid,
		min_amount as Velocity_3mon_loanMin_per_uid,
		sum_num as Velocity_3mon_plannum_per_uid,
		sum_num/count_uid as Velocity_3mon_planAvg_per_uid,
		max_num as Velocity_3mon_planMax_per_uid,
		min_num as Velocity_3mon_planMin_per_uid,
		sum_planavg/count_uid as Velocity_3mon_amtPerPlanAvg_per_uid,
		max_planavg as Velocity_3mon_amtPerPlanMax_per_uid,
		min_planavg as Velocity_3mon_amtPerPlanMin_per_uid,
		max_now as Velocity_days_LtoNow_per_uid,
		min_now as Velocity_days_FtoNow_per_uid
	from
		(
			select
				uid,
				loan_amount as l_amount,
				plannum as plan,
				max(loan_time) as loan_time,
				sum(loan_amount) sum_amount,
				count(uid) count_uid,
				avg(loan_amount) avg_amount,
				max(loan_amount) max_amount,
				min(loan_amount) min_amount,
				sum(plannum) sum_num,
				max(plannum) max_num,
				min(plannum) min_num,
				sum(loan_amount/plannum) sum_planavg,
				max(loan_amount/plannum) max_planavg,
				min(loan_amount/plannum) min_planavg,
				TO_DAYS('2016-11-30') - TO_DAYS(substring( max(loan_time) ,1,10 )) max_now,
				TO_DAYS('2016-11-30') - TO_DAYS(substring( min(loan_time) ,1,10 )) min_now
			from t_loan
			where loan_time >= '2016-08-01' and loan_time < '2016-11-01'
			group by uid
		) as t
		INNER JOIN t_loan as f ON t.uid=f.uid AND t.loan_time=f.loan_time;