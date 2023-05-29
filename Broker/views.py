from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib import messages
from django.contrib.auth.models import User, auth
from Broker.models import *
from django.contrib.auth.decorators import login_required
from django.contrib.staticfiles.storage import staticfiles_storage
from sklearn.metrics import r2_score,mean_squared_error,precision_score,accuracy_score,recall_score,roc_auc_score
import datetime
from datetime import date
import numpy as np 
import yfinance as yf
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from nsetools import Nse
from django.templatetags.static import static
from scipy.optimize import linprog
from django.db.models import F
import json

# Create your views here.

@login_required(login_url='/login/')
def userpage(request):
	p = request.POST 
	sector = p.get("sector",False)
	today_date = str(datetime.date.today())
	today_ = datetime.date.today()
	first_name = request.user.first_name
	if sector:
		stocks = Stock.objects.filter(Sector=sector)
	else:
		stocks = Stock.objects.all()

	for stock in stocks:
		try:
			symbol = str(stock.Symbol)
			df = yf.download(tickers=symbol+'.NS',period='1d',interval='1m')
			last_price = np.round(list(df['Close'])[-1],2)
			low_price = np.round(df['Low'].min(),2)
			high_price = np.round(df['High'].max(),2)
			stock.EOD_Price = last_price
			stock.low_price = low_price
			stock.high_price = high_price
			stock.save()
		except:
			continue

	total_assets = 0
	unrealized_profit_loss = 0 
	total_investment = 0
	portfolios = Stock_Portfolio.objects.filter(Trader=request.user)

	for portfolio in portfolios:
		portfolio.Current_Value = round(portfolio.Stock.EOD_Price*portfolio.Units,2)
		portfolio.Invested = round(portfolio.Invested,2) 
		portfolio.save()
		total_assets+=portfolio.Current_Value
		unrealized_profit_loss+=portfolio.Current_Value - portfolio.Invested
		total_investment += portfolio.Invested
	if total_investment > 0:
		unrealized_profit_pct = round((unrealized_profit_loss/total_investment)*100,2)
	else:
		unrealized_profit_pct = 0
    
	profit_loss_statement = Stock_Profit_Loss.objects.filter(Trader=request.user)

	if profit_loss_statement:
		realized_profit_loss = profit_loss_statement[0].realized_profit_loss
	else:
		profit_loss_statement = Stock_Profit_Loss(Trader=request.user,realized_profit_loss=0)
		profit_loss_statement.save()
		realized_profit_loss = profit_loss_statement.realized_profit_loss


	####Option Portfolio###########################################
	opt_profit_loss_statement = Option_Profit_Loss.objects.filter(Trader=request.user)
	if opt_profit_loss_statement:
		opt_realized_profit_loss = opt_profit_loss_statement[0].realized_profit_loss
	else:
		opt_profit_loss_statement = Option_Profit_Loss(Trader=request.user,realized_profit_loss=0)
		opt_profit_loss_statement.save()
		opt_realized_profit_loss = opt_profit_loss_statement.realized_profit_loss
	opt_profit_loss_statement = Option_Profit_Loss.objects.get(Trader=request.user)
	total_opt_assets = 0
	unrealized_opt_profit_loss = 0 
	opt_total_investment = 0
	opt_portfolios = Option_Portfolio.objects.filter(Trader=request.user)

	for portfolio in opt_portfolios:
		if portfolio.Option.Expiry >= today_:
			portfolio.Current_Value = round(portfolio.Option.EOD_Price*portfolio.Units,2)
			portfolio.Invested = round(portfolio.Invested,2) 
			portfolio.save()
			total_opt_assets+=portfolio.Current_Value
			unrealized_opt_profit_loss+=portfolio.Current_Value - portfolio.Invested
			opt_total_investment += portfolio.Invested
		else:
			portfolio.Current_Value = portfolio.Option.EOD_Price*portfolio.Units
			opt_profit_loss_statement.realized_profit_loss += portfolio.Current_Value - portfolio.Invested
			portfolio.delete()
	opt_profit_loss_statement.save()
	opt_realized_profit_loss = opt_profit_loss_statement.realized_profit_loss
	opt_portfolios = Option_Portfolio.objects.filter(Trader=request.user)
	if opt_total_investment > 0:
		unrealized_opt_profit_pct = round((unrealized_opt_profit_loss/opt_total_investment)*100,2)
	else:
		unrealized_opt_profit_pct = 0

	
	return render(request, "Broker/userpage.html", 
		context={"today":today_date,"first_name":first_name ,"stocks":stocks,"sector":sector,
		"portfolios":portfolios,"realized_profit_loss":round(realized_profit_loss,2),
		"total_investment":round(total_investment,2),
		"total_assets":round(total_assets,2),
		"unrealized_profit_loss":round(unrealized_profit_loss,2),
		"unrealized_profit_pct":unrealized_profit_pct,
		"opt_portfolios":opt_portfolios,"opt_realized_profit_loss":round(opt_realized_profit_loss,2),
		"opt_total_investment":round(opt_total_investment,2),
		"total_opt_assets":round(total_opt_assets,2),
		"unrealized_opt_profit_loss":round(unrealized_opt_profit_loss,2),
		"unrealized_opt_profit_pct":unrealized_opt_profit_pct
		})


@login_required(login_url='/login/')
def stockrecommend(request):
	stocks = Stock.objects.all()
	trader = request.user 
	sell_stocks = [portfolio.Stock for portfolio in Stock_Portfolio.objects.filter(Trader=trader) if portfolio.Stock.EOD_Price > portfolio.Stock.Expected_Price]
	p=request.POST
	try:
		money_invest = float(p.get("invest"))
	except:
		money_invest = False
	try:
		sectors = p.getlist('sectors')
	except:
		sectors = False
	try:
		min_return = float(p.get("min_return"))
	except:
		min_return = False
	try:
		max_return = float(p.get("max_return"))
	except:
		max_return = False
	try:
		exclude_stocks = p.getlist('Exclude')
	except:
		exclude_stocks = False 
	try:
		include_stocks = p.getlist('Include')
	except:
		exclude_stocks = False 

	if sectors:
		buy_stocks = list(Stock.objects.filter(EOD_Price__lt = F('Expected_Price'),Sector__in=sectors))
	else:
		buy_stocks = list(Stock.objects.filter(EOD_Price__lt = F('Expected_Price')))
	print("Filter1: ",len(buy_stocks))
	if exclude_stocks:
		buy_stocks = [stock for stock in buy_stocks if stock.Symbol not in list(exclude_stocks)]
	print("Filter2: ",len(buy_stocks))
	 
	if include_stocks:
		buy_stocks = [stock for stock in buy_stocks if stock.Symbol in list(include_stocks)]
	print("Filter3: ",len(buy_stocks))
	
	if min_return:
		buy_stocks = [stock for stock in buy_stocks if (stock.Expected_Price/stock.EOD_Price-1)*100 >= min_return]
	print("Filter4: ",len(buy_stocks))

	if max_return:
		buy_stocks = [stock for stock in buy_stocks if (stock.Expected_Price/stock.EOD_Price-1)*100 <= max_return]
	print("Filter5: ",len(buy_stocks))

	if len(buy_stocks)==0:
		messages.info(request, 'Above Criterias could not be met , we have selected stocks with positive returns')
		buy_stocks = list(Stock.objects.filter(EOD_Price__lt = F('Expected_Price')))


	if money_invest:
		symbols = [stock.Symbol for stock in buy_stocks]
		obj = [stock.EOD_Price-stock.Expected_Price for stock in buy_stocks]
		lhs_ineq = [[stock.EOD_Price for stock in buy_stocks]]
		rhs_ineq =[money_invest]
		bnd = [(0,int(money_invest/stock.EOD_Price)) for stock in buy_stocks]
		opt = linprog(c=obj,A_ub=lhs_ineq,b_ub=rhs_ineq,bounds=bnd,method="revised simplex")
		qty = [int(z) for z in opt['x']]
		current_prices = [Stock.objects.get(Symbol=symbol).EOD_Price for symbol in symbols]
		expected_prices = [Stock.objects.get(Symbol=symbol).Expected_Price for symbol in symbols]
		buy_list = list(tuple(zip(symbols,qty,current_prices,expected_prices)))
		buy_list = [x for x in buy_list if x[1] > 0]
	else:
		buy_list = []



	return render(request,"Broker/smarttrade.html",context={'stocks':stocks,'sell_stocks':sell_stocks,'buy_list':buy_list})


@login_required(login_url='/login/')
def smartinvest(request):
	stocks = Stock.objects.all()
	valuations = Valuation.objects.all()
	trader = request.user 
	p=request.POST
	try:
		money_invest = float(p.get("invest"))
	except:
		money_invest = False
	try:
		sectors = p.getlist('sectors')
	except:
		sectors = False
	try:
		min_return = float(p.get("min_return"))
	except:
		min_return = False
	try:
		max_return = float(p.get("max_return"))
	except:
		max_return = False
	try:
		exclude_stocks = p.getlist('Exclude')
	except:
		exclude_stocks = False 
	try:
		include_stocks = p.getlist('Include')
	except:
		exclude_stocks = False 

	if sectors:
		buy_stocks = list(Valuation.objects.filter(Expected_YR_Returns__gt=0))
		buy_stocks = [valuation for valuation in buy_stocks if valuation.Stock.Sector in list(sectors)]
	else:
		buy_stocks = list(Valuation.objects.filter(Expected_YR_Returns__gt=0))
	print("Filter1: ",len(buy_stocks))
	if exclude_stocks:
		buy_stocks = [valuation for valuation in buy_stocks if valuation.Stock.Symbol not in list(exclude_stocks)]
	print("Filter2: ",len(buy_stocks))
	 
	if include_stocks:
		buy_stocks = [valuation for valuation in buy_stocks if valuation.Stock.Symbol in list(include_stocks)]
	print("Filter3: ",len(buy_stocks))
	
	if min_return:
		buy_stocks = [valuation for valuation in buy_stocks if valuation.Expected_YR_Returns*100 >= min_return]
	print("Filter4: ",len(buy_stocks))

	if max_return:
		buy_stocks = [valuation for valuation in buy_stocks if valuation.Expected_YR_Returns*100 <= max_return]
	print("Filter5: ",len(buy_stocks))

	if len(buy_stocks)==0:
		messages.info(request, 'Above Criterias could not be met , we have selected stocks with positive returns')
		buy_stocks = list(Valuation.objects.filter(Expected_YR_Returns__gt=0))


	if money_invest:
		symbols = [valuation.Stock.Symbol for valuation in buy_stocks]
		obj = [-valuation.Stock.EOD_Price*valuation.Expected_YR_Returns for valuation in buy_stocks]
		lhs_ineq = [[valuation.Stock.EOD_Price for valuation in buy_stocks]]
		rhs_ineq =[money_invest]
		bnd = [(0,int(money_invest/valuation.Stock.EOD_Price)) for valuation in buy_stocks]
		opt = linprog(c=obj,A_ub=lhs_ineq,b_ub=rhs_ineq,bounds=bnd,method="revised simplex")
		qty = [int(z) for z in opt['x']]
		current_prices = [Stock.objects.get(Symbol=symbol).EOD_Price for symbol in symbols]
		expected_returns = [np.round(Valuation.objects.get(Stock=Stock.objects.get(Symbol=symbol)).Expected_YR_Returns*100,2) for symbol in symbols]
		buy_list = list(tuple(zip(symbols,qty,current_prices,expected_returns)))
		buy_list = [x for x in buy_list if x[1] > 0]
	else:
		buy_list = []



	return render(request,"Broker/smartinvest.html",context={'stocks':stocks,'buy_list':buy_list})




@login_required(login_url='/login/')
def stock_detail(request,symbol):
	today_date = datetime.date.today()
	stock = Stock.objects.get(Symbol=symbol)
	# nse = Nse()
	# order_dict = nse.get_quote(symbol)
	# buyprice = []
	# sellprice = []
	# buyqty = []
	# sellqty = []

	# for i in range(1,6):
	# 	buyprice.append(str(order_dict[f'buyPrice{i}']))
	# 	buyqty.append(str(order_dict[f'buyQuantity{i}']))
	# 	sellprice.append(str(order_dict[f'sellPrice{i}']))
	# 	sellqty.append(str(order_dict[f'sellQuantity{i}']))

	# orderbook = list(tuple(zip(buyprice,buyqty,sellqty,sellprice)))
	# high52 = str(np.round(order_dict['high52'],2))
	# low52 = str(np.round(order_dict['low52'],2))

	stock_predictions = pd.read_excel(staticfiles_storage.path('stock predictions.xlsx'),header=0)
	last_day_close = np.round(stock_predictions.loc[stock_predictions['Symbol']==symbol,'EOD Price'].values[0],2)
	reason_codes = pd.read_excel(staticfiles_storage.path('reason_codes.xlsx'),header=0)
	macro_cont = np.round((reason_codes.loc[reason_codes['Symbol']==symbol,'Macro'].values[0]-1)*10000,2)
	sector_cont = np.round((reason_codes.loc[reason_codes['Symbol']==symbol,'Sector'].values[0]-1)*10000,2)
	cap_cont = np.round((reason_codes.loc[reason_codes['Symbol']==symbol,'Capital'].values[0]-1)*10000,2)
	comp_cont = np.round((reason_codes.loc[reason_codes['Symbol']==symbol,'Company'].values[0]-1)*100,2)
	market_cont = np.round((reason_codes.loc[reason_codes['Symbol']==symbol,'Market'].values[0]-1)*100,2)

	net_impact = np.round((reason_codes.loc[reason_codes['Symbol']==symbol,'Net Impact'].values[0]-1)*100,2)

	history = pd.read_excel(staticfiles_storage.path('historical_data.xlsx'),header=0)
	history = history[history['Symbol']==symbol]
	history.sort_values(by='Date',inplace=True)

	p = request.POST 
	period = int(p.get("period",False))

	if period:
		pass 
	else:
		period = 5

	history = history.iloc[-period:]
	history.set_index('Date',inplace=True)
	accuracy = np.round(history['correct prediction'].sum()/history.shape[0]*100,0)
	pred_score = np.round(r2_score(history['Actual Price'],history['Predicted Price'])*100,0)

	# fig,ax = plt.subplots(1,1,figsize=(50,20))
	# sns.set(font_scale=5)
	# sns.lineplot(data=history[['Actual Price','Predicted Price']])
	# try:
	# 	ax.set_xticks([history.index[i] for i in range(1,len(list(history.index)),3)])
	# except:
	# 	ax.set_xticks([])
	# ax.set_xlabel('')
	# fig.savefig(staticfiles_storage.path(f'Graphs/{symbol}.png'))

	# img_url = static(f'Graphs/{symbol}.png')

	possible_expiries = sorted([str(x[0].strftime('%Y-%m-%d')) for x in list(Option.objects.filter(Stock=stock,Expiry__gte=today_date).values_list('Expiry').distinct())])

	try:
		p = request.POST 
		expiry = datetime.datetime.strptime(p.get("expiry",True),"%Y-%m-%d").date()
	except:
		expiry = False
	if expiry:
		calls = Option.objects.filter(Stock=stock,Expiry=expiry,Opt_Type="CE").order_by('Strike').values()
		puts = Option.objects.filter(Stock=stock,Expiry=expiry,Opt_Type="PE").order_by('Strike').values()
	else:
		calls = Option.objects.filter(Stock=stock,Opt_Type="CE",Expiry__gte=today_date).order_by('Expiry','Strike').values()
		puts = Option.objects.filter(Stock=stock,Opt_Type="PE",Expiry__gte=today_date).order_by('Expiry','Strike').values()

	try:
		projections = pd.read_excel(staticfiles_storage.path(f'Projections/{symbol}.xlsx'),header=[0,1],index_col=0)
		projections.reset_index(inplace=True)
	except:
		projections = pd.DataFrame()

	try:
		holding_breakup = pd.read_excel(staticfiles_storage.path(f'Holding_Breakup/{symbol}.xlsx'),header=0)
	except:
		holding_breakup = pd.DataFrame()

	try:
		valuation = Valuation.objects.get(Stock=stock)
		p = request.POST 
		holding_period = int(p.get("holding_period",False))

		if holding_period:
			pass 
		else:
			holding_period = 1

		if holding_period==1:
			dcf_value = valuation.Value1
		elif holding_period==2:
			dcf_value = valuation.Value2
		elif holding_period==3:
			dcf_value = valuation.Value3
		elif holding_period==4:
			dcf_value = valuation.Value4
		else:
			dcf_value = valuation.Value5

		dcf_value = np.round(dcf_value,2)


	except:
		valuation = []
		dcf_value = np.nan 

	sector = stock.Sector

	peer_stocks = Stock.objects.filter(Sector=sector).exclude(Symbol=symbol)
	peer_valuations = Valuation.objects.filter(Stock__in=peer_stocks)




	return render(request,"Broker/stock_detail.html",context={'macro_cont':macro_cont,'sector_cont':sector_cont,
		'cap_cont':cap_cont,'comp_cont':comp_cont,'market_cont':market_cont,'net_impact':net_impact,'stock':stock,
		'last_day_close':last_day_close,'accuracy':accuracy,'pred_score':pred_score,'calls':calls,'puts':puts,
		'possible_expiries':possible_expiries,
		#'img_url':img_url,
		# 'orderbook':orderbook,'high52':high52,'low52':low52,
		'projections':projections,
		'holding_breakup':holding_breakup,'valuation':valuation,'peer_valuations':peer_valuations,'dcf_value':dcf_value
		})



@login_required(login_url='/login/')
def tradepage(request):
	today_date = str(datetime.date.today())
	first_name = request.user.first_name
	stocks = Stock.objects.all()
	if request.method=="POST":
		trader = request.user
		stock = Stock.objects.get(Symbol=request.POST["stock"])
		trade_type = request.POST["trade_type"]
		trade_method = request.POST["trade_method"]
		trade_qty = int(request.POST["trade_qty"])
		trade_price = float(request.POST["trade_price"])

		if trade_type == "Sell":
			if Stock_Portfolio.objects.filter(Trader=request.user,Stock=stock):
				portfolio = Stock_Portfolio.objects.get(Trader=request.user,Stock=stock)
				if trade_qty > portfolio.Units:
					messages.info(request,f"Insufficient Shares to sell, can sell upto {portfolio.Units} shares only")
					return render(request,'Broker/tradepage.html',context={'stocks':stocks,'first_name':first_name,"today":today_date})
				else:
					trade = Stock_Trade(Trader=trader,Stock=stock,Trade_Type=trade_type,Trade_Method=trade_method,
						Trade_Qty=trade_qty,Trade_Price=trade_price)
					trade.save()
					investment_considered = portfolio.Invested/portfolio.Units*trade_qty
					sale_receivables = trade_price*trade_qty
					realized_profit = sale_receivables - investment_considered
					if portfolio.Units == trade_qty:
						portfolio.delete()
					else:
						portfolio.Units -= trade_qty
						portfolio.Invested -= investment_considered
						portfolio.Current_Value = stock.EOD_Price*portfolio.Units 
						portfolio.save()
					profit_loss_statement = Stock_Profit_Loss.objects.get(Trader=trader)
					profit_loss_statement.realized_profit_loss += realized_profit
					profit_loss_statement.save()
			else:
				messages.info(request,f"No Shares in account to sell")
				return render(request,'Broker/tradepage.html',context={'stocks':stocks,'first_name':first_name,"today":today_date})
		else:
			if Stock_Portfolio.objects.filter(Trader=request.user,Stock=stock):
				portfolio = Stock_Portfolio.objects.get(Trader=request.user,Stock=stock)
				trade = Stock_Trade(Trader=trader,Stock=stock,Trade_Type=trade_type,Trade_Method=trade_method,
						Trade_Qty=trade_qty,Trade_Price=trade_price)
				trade.save()
				investment_considered = trade_price*trade_qty
				portfolio.Units += trade_qty
				portfolio.Invested += investment_considered
				portfolio.Current_Value = stock.EOD_Price*portfolio.Units 
				portfolio.save()
			else:
				trade = Stock_Trade(Trader=trader,Stock=stock,Trade_Type=trade_type,Trade_Method=trade_method,
						Trade_Qty=trade_qty,Trade_Price=trade_price)
				trade.save()
				investment_considered = trade_price*trade_qty
				portfolio = Stock_Portfolio(Trader=trader,Stock=stock,Units=trade_qty,Invested=investment_considered,
					Current_Value=trade_qty*stock.EOD_Price)
				portfolio.save()

	return render(request,'Broker/tradepage.html',context={'stocks':stocks,'first_name':first_name,"today":today_date})

@login_required(login_url='/login/')
def opttradepage(request,id):
	today_date = str(datetime.date.today())
	first_name = request.user.first_name
	option = Option.objects.get(id=id)
	if request.method=="POST":
		trader = request.user
		trade_type = str(request.POST["trade_type"])
		trade_method = str(request.POST["trade_method"])
		trade_qty = int(request.POST["trade_qty"])
		trade_price = float(request.POST["trade_price"])

		if Option_Profit_Loss.objects.filter(Trader=request.user).exists():
			option_profit_loss = Option_Profit_Loss.objects.get(Trader=request.user)
		else:
			option_profit_loss = Option_Profit_Loss()
			option_profit_loss.Trader = request.user
			option_profit_loss.realized_profit_loss = 0 

		if trade_type == "Sell":
			if Option_Portfolio.objects.filter(Trader=trader,Option=option).exists():
				portfolio = Option_Portfolio.objects.get(Trader=request.user,Option=option)
				trade = Option_Trade(Trader=trader,Option=option,Trade_Type=trade_type,Trade_Method=trade_method,
					Trade_Qty=trade_qty,Trade_Price=trade_price)
				trade.save()
				portfolio.Units -= trade_qty
				portfolio.Invested -= trade_price*trade_qty
				portfolio.Current_Value = option.EOD_Price*portfolio.Units 
				if portfolio.Units == 0:
					option_profit_loss.realized_profit_loss += (portfolio.Current_Value-portfolio.Invested)
					portfolio.delete()
				else:
					portfolio.save()
			else:
				portfolio = Option_Portfolio()
				portfolio.Trader = request.user
				portfolio.Option = option
				portfolio.Units = 0 
				portfolio.Invested = 0 
				portfolio.Current_Value = 0 
				trade = Option_Trade(Trader=trader,Option=option,Trade_Type=trade_type,Trade_Method=trade_method,
					Trade_Qty=trade_qty,Trade_Price=trade_price)
				trade.save()
				portfolio.Units -= trade_qty
				portfolio.Invested -= trade_price*trade_qty
				portfolio.Current_Value = option.EOD_Price*portfolio.Units 
				portfolio.save()
		else:
			if Option_Portfolio.objects.filter(Trader=request.user,Option=option).exists():
				portfolio = Option_Portfolio.objects.get(Trader=request.user,Option=option)
				trade = Option_Trade(Trader=trader,Option=option,Trade_Type=trade_type,Trade_Method=trade_method,
					Trade_Qty=trade_qty,Trade_Price=trade_price)
				trade.save()
				portfolio.Units += trade_qty
				portfolio.Invested += trade_price*trade_qty
				portfolio.Current_Value = option.EOD_Price*portfolio.Units 
				if portfolio.Units == 0:
					option_profit_loss.realized_profit_loss += (portfolio.Current_Value-portfolio.Invested)
					portfolio.delete()
				else:
					portfolio.save()
			else:
				portfolio = Option_Portfolio()
				portfolio.Trader = request.user
				portfolio.Option = option
				portfolio.Units = 0 
				portfolio.Invested = 0 
				portfolio.Current_Value = 0 
				trade = Option_Trade(Trader=trader,Option=option,Trade_Type=trade_type,Trade_Method=trade_method,
					Trade_Qty=trade_qty,Trade_Price=trade_price)
				trade.save()
				portfolio.Units += trade_qty
				portfolio.Invested += trade_price*trade_qty
				portfolio.Current_Value = option.EOD_Price*portfolio.Units 
				portfolio.save()
		option_profit_loss.save()
	return render(request,'Broker/opttradepage.html',context={'id':id,'first_name':first_name,"today":today_date})
	

