from django.db import models

# Create your models here.

from django.contrib.auth.models import User

class Stock(models.Model):
	Sector = models.CharField(max_length=50)
	Company = models.CharField(max_length=50)
	Symbol = models.CharField(max_length=10,primary_key=True)
	EOD_Price = models.FloatField(null=True)
	Expected_Price = models.FloatField(null=True)
	low_price = models.FloatField(null=True)
	high_price = models.FloatField(null=True)
	

	def __str__(self):
		return str(self.Symbol)

Trade_Choices = (("Buy","Buy"),("Sell","Sell"),)
Method_Choices = (("Market","Market"),("Limit","Limit"),)

class Stock_Trade(models.Model):
	Trader = models.ForeignKey(User,on_delete=models.CASCADE)
	Stock = models.ForeignKey(Stock,on_delete=models.CASCADE)
	Trade_Type = models.CharField(max_length=4,choices=Trade_Choices)
	Trade_Method = models.CharField(max_length=10,choices=Method_Choices)
	Trade_Qty = models.IntegerField()
	Trade_Price = models.FloatField()
	Trade_Date = models.DateTimeField(auto_now_add=True)

	def __str__(self):
		return str(self.Trade_Type)+" "+str(self.Trade_Qty)+" "+str(self.Stock)+" @"+str(self.Trade_Price)

class Stock_Portfolio(models.Model):
	Trader = models.ForeignKey(User,on_delete=models.CASCADE)
	Stock = models.ForeignKey(Stock,on_delete=models.CASCADE)
	Units = models.IntegerField()
	Invested = models.FloatField()
	Current_Value = models.FloatField()

	def __str__(self):
		return str(self.Stock)+" gave gain/loss of "+str(round((self.Current_Value/self.Invested-1)*100,2))+" %"

class Stock_Profit_Loss(models.Model):
	Trader = models.OneToOneField(User,on_delete=models.CASCADE)
	realized_profit_loss = models.FloatField(default=0)

	def __str__(self):
		return str(self.Trader.first_name)+" has made gains/loss of "+str(realized_profit)+" Rs."

Option_Choices = (("CE","CE"),("PE","PE"),)

class Option(models.Model):
	Stock = models.ForeignKey(Stock,on_delete=models.CASCADE)
	Expiry = models.DateField()
	Strike = models.FloatField()
	Opt_Type = models.CharField(max_length=2,choices=Option_Choices)
	EOD_Price = models.FloatField(null=True)
	Expected_Price = models.FloatField(null=True)

	def __str__(self):
		return str(self.Stock.Symbol)+" "+str(self.Opt_Type)+":"+str(self.Strike)+" Rs. Expiry:" +str(self.Expiry)


class Option_Trade(models.Model):
	Trader = models.ForeignKey(User,on_delete=models.CASCADE)
	Option = models.ForeignKey(Option,on_delete=models.CASCADE)
	Trade_Type = models.CharField(max_length=4,choices=Trade_Choices)
	Trade_Method = models.CharField(max_length=10,choices=Method_Choices)
	Trade_Qty = models.IntegerField()
	Trade_Price = models.FloatField()
	Trade_Date = models.DateTimeField(auto_now_add=True)

	def __str__(self):
		return str(self.Trade_Type)+" "+str(self.Trade_Qty)+" "+str(self.Option)+" @"+str(self.Trade_Price)


class Option_Portfolio(models.Model):
	Trader = models.ForeignKey(User,on_delete=models.CASCADE)
	Option = models.ForeignKey(Option,on_delete=models.CASCADE)
	Units = models.IntegerField()
	Invested = models.FloatField()
	Current_Value = models.FloatField()

	def __str__(self):
		return str(self.Stock)+" gave gain/loss of "+str(round((self.Current_Value/self.Invested-1)*100,2))+" %"

class Option_Profit_Loss(models.Model):
	Trader = models.OneToOneField(User,on_delete=models.CASCADE)
	realized_profit_loss = models.FloatField(default=0)

	def __str__(self):
		return str(self.Trader.first_name)+" has made gains/loss of "+str(realized_profit)+" Rs."

class Valuation(models.Model):
	Stock = models.ForeignKey(Stock,on_delete=models.CASCADE)
	Shares_Outstanding = models.IntegerField()
	Beta = models.FloatField()
	Cost_Of_Equity = models.FloatField()
	Cost_Of_Debt = models.FloatField()
	Debt_to_Equity = models.FloatField()
	Cost_Of_Capital = models.FloatField()
	Value1 = models.FloatField()
	Value2 = models.FloatField()
	Value3 = models.FloatField()
	Value4 = models.FloatField()
	Value5 = models.FloatField()
	EV_Sales = models.FloatField()
	EV_Ebit = models.FloatField()
	PE_Ratio = models.FloatField()
	PB_Ratio = models.FloatField()
	Expected_YR_Returns = models.FloatField()

	def __str__(self):
		return str(self.Stock.Symbol)















