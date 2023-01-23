from django.urls import path, include

from Broker import views
from django.conf.urls.static import static
from django.conf import settings

app_name = "Broker"

urlpatterns = [

path('userpage/', views.userpage, name="userpage"),
path('tradepage/', views.tradepage, name="tradepage"),
path('opttradepage/<int:id>',views.opttradepage,name="opttradepage"),
path('stockdetail/<str:symbol>/',views.stock_detail,name='stockdetail'),
path('stockrecommend',views.stockrecommend,name='stockrecommend'),
path('smartinvest',views.smartinvest,name='smartinvest'),
#path('description/<int:bid>/', views.bookDescription, name='description'),
#path('delete/<int:bid>/', views.deleteBook, name='deletebook'),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)