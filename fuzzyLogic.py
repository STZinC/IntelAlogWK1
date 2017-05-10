from math import sqrt, pi, exp, log, e
from numpy import spacing
#Use Gaussian function to fuzzy
def myGaussian(x,u,d):
	x = float(x);
	u = float(u);
	d = float(d);
	GaussianMax = float(1)/(d * (sqrt(2 * pi)));  #To stretch the function output to [0,1]
	ans = float(1)*exp(-(pow(x-u,2)/(2 * pow(d,2))))/(d * (sqrt(2 * pi)));
	ans = ans/GaussianMax;
	return ans

#Use the inverse of Gaussian function to defuzzy
#Argument 'positive' is to decide which output to be used
def myInvGaussian(y,u,d,positive):
	eps = spacing(1)
	GaussianMax = float(1)/(d * (sqrt(2 * pi)));  #To stretch the function input to [0,1]
	y = y * GaussianMax;
	anspos = u + sqrt(-2 * pow(d,2) * log(eps + y * d * sqrt(2 * pi),e));
	ansneg = u - sqrt(-2 * pow(d,2) * log(eps + y * d * sqrt(2 * pi),e));
	if(anspos>1):
		anspos = 1
	if(anspos<0):
		anspos = 0
	if(ansneg>1):
		ansneg = 1
	if(ansneg<0):
		ansneg = 0

	if(positive):
		return anspos
	else:
		return ansneg

def isFeasible(x):
	g = [0] * 9;
	g[0] = 2 * x[0] + 2 * x[2] + x[9] + x[10] - 10;
	g[1] = 2 * x[0] + 2 * x[2] + x[9] + x[11] - 10;
	g[2] = 2 * x[1] + 2 * x[2] + x[10] + x[11] - 10;
	g[3] = -8 * x[0] + x[9];
	g[4] = -8 * x[1] + x[10];
	g[5] = -8 * x[2] + x[11];
	g[6] = -2 * x[3] - x[4] +x[9];
	g[7] = -2 * x[5] - x[6] +x[10];
	g[8] = -2 * x[7] - x[8] +x[11];

	for i in range(0,9):
		if(g[i] > 0):
			#print i
			return 0

	return 1

#The function to calculate v(x)
#Constrains are applied here
#x is a vector x[1],x[2]....x[n]
def violation(x):
	g = [0] * 9;
	g[0] = 2 * x[0] + 2 * x[2] + x[9] + x[10] - 10;
	g[1] = 2 * x[0] + 2 * x[2] + x[9] + x[11] - 10;
	g[2] = 2 * x[1] + 2 * x[2] + x[10] + x[11] - 10;
	g[3] = -8 * x[0] + x[9];
	g[4] = -8 * x[1] + x[10];
	g[5] = -8 * x[2] + x[11];
	g[6] = -2 * x[3] - x[4] +x[9];
	g[7] = -2 * x[5] - x[6] +x[10];
	g[8] = -2 * x[7] - x[8] +x[11];

	ans = 0;
	for i in range(0,9):
		if(g[i]>0):
			ans += g[i];

	return ans

#The function to calculate f(x)
#Which is fitness
#x is a vector x[1],x[2]....x[n]
def fitness(x):
	ans1 = 0;
	for i in range(0,4):
		ans1 += 5 * x[i];

	ans2 = 0;
	for i in range(0,4):
		ans2 += -5 * pow(x[i],2);

	ans3 = 0;
	for i in range(4,13):
		ans3 += -1 * x[i];

	ans = ans1 + ans2 + ans3;
	return ans

#The functions to calculate vnorm and fnorm
def vnorm(x,vmin,vmax):
	ans = (violation(x) - vmin)/(vmax - vmin)
	# if(ans < 0):
	# 	print("vnorm:"+str(ans))
	return ans

def fnorm(x,fmin,fmax):
	ans = (fitness(x) - fmin)/(fmax - fmin)
	# if(ans < 0):
	# 	print("fnorm:"+str(ans))
	return ans



#The function to calculate penalty
def p(x,rf,vmin,vmax,fmin,fmax):
	v = vnorm(x,vmin,vmax);
	f = fnorm(x,fmin,fmax);
	membership_table_v = {};
	membership_table_f = {};
	membership_table_rf = {};

	membership_table_v['low'] = myGaussian(v,0.0,0.27);
	membership_table_v['high'] = myGaussian(v,1.0,0.27);
	membership_table_f['low'] = myGaussian(f,0.0,0.27);
	membership_table_f['high'] = myGaussian(f,1.0,0.27);
	membership_table_rf['low'] = myGaussian(rf,0.0,0.27);
	membership_table_rf['high'] = myGaussian(rf,1.0,0.27);

	m_l_l_l = min([membership_table_v['low'],membership_table_f['low'],membership_table_rf['low']]);
	m_l_l_h = min([membership_table_v['low'],membership_table_f['low'],membership_table_rf['high']]);
	m_x_h_l = min([membership_table_f['high'],membership_table_rf['low']]);
	m_x_h_h = min([membership_table_f['high'],membership_table_rf['high']]);
	m_h_l_x = min([membership_table_v['high'],membership_table_f['low']]);

	penalty_table = {};
	penalty_table['mid'] = max([m_l_l_l,m_x_h_h,m_h_l_x]);
	penalty_table['high'] = m_x_h_l;
	penalty_table['low'] = m_l_l_h;
	m_low = penalty_table['low'];
	m_mid = penalty_table['mid'];
	m_high = penalty_table['high'];

	p_low = myInvGaussian(penalty_table['low'],0.0,0.18,1);
	p_mid = myInvGaussian(penalty_table['mid'],0.5,0.18,1);
	p_high = myInvGaussian(penalty_table['high'],1.0,0.18,0);
	if(max([m_low,m_mid,m_high])==m_low):
		ans = p_low
	elif(max([m_low,m_mid,m_high])==m_mid):
		ans = p_mid
	else: 
		ans = p_high
	#ans = p_low + p_mid + p_high;
	#ans = (p_low + p_high + p_mid)/(penalty_table['low'] + penalty_table['mid'] + penalty_table['high']);
	if(ans<0):
		print("-----------------")
		print("vnorm:" + str(v))
		print("fnorm:" + str(f))
		print("rf:" + str(rf))
		print("v_low:" + str(membership_table_v['low']))
		print("v_high:" + str(membership_table_v['high']))
		print("f_low:" + str(membership_table_f['low']))
		print("f_high:" + str(membership_table_f['high']))
		print("rf_low:" + str(membership_table_rf['low']))
		print("rf_high:" + str(membership_table_rf['high']))
		print("m_low:" + str(penalty_table['low']))
		print("m_mid:" + str(penalty_table['mid']))
		print("m_high:" + str(penalty_table['high']))
		print("p_low:" + str(p_low))
		print("p_mid:" + str(p_mid))
		print("p_high:" + str(p_high))
		print("penalty:" + str(ans))

	return ans

#The function to calculate the F(x) = p(x) + d(x)
def F(x,rf,vmin,vmax,fmin,fmax):
	dx = rf * fnorm(x,fmin,fmax) + (1 - rf) * vnorm(x,vmin,vmax);
	px = p(x,rf,vmin,vmax,fmin,fmax);
	ans = px + dx
	if(dx<0 or px<0):
		print ("F:" + str(px + dx))
		print ("px:" + str(px))
		print ("dx:" + str(dx))
		print ("fnorm:"+str(fnorm(x,fmin,fmax)))
		print ("vnorm:"+str(vnorm(x,vmin,vmax)))
	return px + dx

# x = [1,1,1,1,1,1,1,1,1,3,3,3,1]
# print fitness(x)