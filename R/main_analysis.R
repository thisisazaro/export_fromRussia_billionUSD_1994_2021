# Установка и загрузка необходимых пакетов
# install.packages(c("readxl", "forecast", "ggplot2", "tseries", "tidyverse", "tsibble", "fable", "randomForest"))
library(readxl)
library(forecast)
library(ggplot2)
library(tseries)
library(tidyverse)
library(tsibble)
library(fable)
library(randomForest)

# Загрузка данных
data <- read_excel("data/export_fromRussia_billionUSD.xlsx")
str(data)
# Переименовываем колонку для удобства
colnames(data) <- c("Year", "Month", "Export")

# Преобразуем в временной ряд
data_ts <- ts(data$Export, start = c(min(data$Year), 1), frequency = 12)

# Проверяем временной ряд
autoplot(data_ts) + ggtitle("Общий объем экспорта из РФ (млрд USD)")

# Видно, что объем экспорта имеет выраженный тренд – общий рост с 1994 года, 
# резкие колебания в кризисные периоды (например, 2008, 2014, 2020).
# Также заметны всплески и падения, связанные с экономическими событиями.

# Обычные и сезонные разности
diff_ts <- diff(data_ts)  # Обычная разность
diff_seasonal_ts <- diff(data_ts, lag = 12)  # Сезонная разность

autoplot(diff_ts) + ggtitle("Обычная разность ряда")

# После применения первой разности тренд убирается, но остается высокая изменчивость.
# Временами дисперсия увеличивается, что может указывать на гетероскедастичность 
# (неоднородность дисперсии).
# Наличие больших выбросов предполагает экономические кризисы или значимые 
# макроэкономические изменения.

autoplot(diff_seasonal_ts) + ggtitle("Сезонная разность ряда")

# Сезонная разность показывает значительные колебания, особенно после 2008 года.
# Вероятно, сезонный компонент не является полностью стабильным во времени, что 
# усложняет прогнозирование.

# Компоненты ряда
decomp <- stl(data_ts, s.window = "periodic")
autoplot(decomp)

# ACF и PACF
acf(data_ts, main = "ACF исходного ряда")
pacf(data_ts, main = "PACF исходного ряда")

acf(diff_ts, main = "ACF обычной разности")
pacf(diff_ts, main = "PACF обычной разности")

acf(diff_seasonal_ts, main = "ACF сезонной разности")
pacf(diff_seasonal_ts, main = "PACF сезонной разности")


# 3. Проверка стационарности
adf.test(data_ts)

# 4. Преобразование при необходимости
log_ts <- log(data_ts)  # Логарифмируем ряд для стабилизации дисперсии
adf.test(diff(log_ts))  # Повторная проверка

# 5. Разделение на обучающую и тестовую выборку
train_ts <- window(data_ts, end = c(2018, 12))
test_ts <- window(data_ts, start = c(2019, 1))

# 6. Оценка моделей
naive_model <- naive(train_ts)
ets_model <- ets(train_ts)
sarima_model <- auto.arima(train_ts, seasonal = TRUE)
theta_model <- thetaf(train_ts)
rf_model <- randomForest(y = as.numeric(train_ts), x = time(train_ts))

# Усреднение лучших моделей
avg_forecast <- (forecast(ets_model, h = length(test_ts))$mean + forecast(sarima_model, h = length(test_ts))$mean) / 2

# 7. Выбор наилучшей модели и визуализация остатков
best_model <- sarima_model  # Например, выбираем SARIMA
checkresiduals(best_model)
forecast_best <- forecast(best_model, h = 24)
autoplot(forecast_best) + ggtitle("Прогноз на 2 года вперед")

# 8. Дополнительные техники
# Проверка на аномалии
outliers <- tso(data_ts)
autoplot(outliers)

# 9. Бонус
# Описание проблем прогнозирования временных рядов
cat("Основные проблемы: наличие тренда и сезонности, изменение структурных параметров, влияние внешних факторов.")
